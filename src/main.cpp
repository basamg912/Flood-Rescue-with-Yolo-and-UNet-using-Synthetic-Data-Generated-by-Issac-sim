#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>    // for createInferRuntime
#include <cuda_runtime_api.h>  // for CUDA functions (cudaMalloc, cudaMemcpy)


namespace fs = std::experimental::filesystem;
using Clock = std::chrono::steady_clock;

bool is_image(const std::string& path) {
    static const std::set<std::string> exts = {
        ".jpg",".jpeg",".png",".bmp",".tiff",".tif"
    };
    auto ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return exts.count(ext) > 0;
}

bool is_video(const std::string& path) {
    static const std::set<std::string> exts = {
        ".mp4",".avi",".mov",".mkv",".wmv",".flv"
    };
    auto ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return exts.count(ext) > 0;
}



// =============================
// 1) 환경 설정 및 클래스 이름 리스트
// =============================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // INFO보다 심각한 로그만 출력
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger; // 클래스 선언과 동시에 전역 객체로 선언

// COCO 클래스 이름 벡터 (80개 클래스)
static const std::vector<std::string> CLASS_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

//==========
// Unet , grid map generation 모델 관련
//==========
class TRTUNet2Head {
public:
    TRTUNet2Head(const std::string& enginePath) {
        std::ifstream f(enginePath, std::ios::binary);
        if(!f) throw std::runtime_error("UNet engine open failed");
        f.seekg(0, f.end); size_t size = f.tellg(); f.seekg(0, f.beg);
        std::vector<char> buf(size); f.read(buf.data(), size);
        runtime_ = nvinfer1::createInferRuntime(gLogger);
        engine_  = runtime_->deserializeCudaEngine(buf.data(), size, nullptr);
        ctx_     = engine_->createExecutionContext();
        cudaStreamCreate(&stream_);
        // 바인딩/버퍼 준비 (여기선 전부 float, 1x3xH xW → 1x1xH xW ×2 라고 가정)
        int nB = engine_->getNbBindings();
        buffers_.resize(nB, nullptr);
        for (int i=0;i<nB;++i){
            auto dims = engine_->getBindingDimensions(i);
            size_t vol=1; for(int d=0; d<dims.nbDims; ++d) vol *= dims.d[d];
            size_t bytes = vol * sizeof(float);
            cudaMalloc(&buffers_[i], bytes);
            if (engine_->bindingIsInput(i)) { inIdx_ = i; inH_ = dims.d[2]; inW_ = dims.d[3]; }
            else outIdxs_.push_back(i);
        }
    }
    ~TRTUNet2Head(){
        for (void* b: buffers_) if(b) cudaFree(b);
        if(ctx_) ctx_->destroy();
        if(engine_) engine_->destroy();
        if(runtime_) runtime_->destroy();
        cudaStreamDestroy(stream_);
    }

    // 입력 BGR → RGB/CHW float [0,1], 출력 2개 맵을 float Mat로 반환(예시)
    void infer(const cv::Mat& bgr, cv::Mat& gridA, cv::Mat& gridB) {
        cv::Mat resized, rgb;
        cv::resize(bgr, resized, {inW_, inH_});
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        std::vector<float> hostIn(3*inH_*inW_);
        for(int y=0;y<inH_;++y){
            for(int x=0;x<inW_;++x){
                cv::Vec3b p = rgb.at<cv::Vec3b>(y,x);
                int idx = y*inW_ + x;
                hostIn[0*inH_*inW_ + idx] = p[0]/255.f;
                hostIn[1*inH_*inW_ + idx] = p[1]/255.f;
                hostIn[2*inH_*inW_ + idx] = p[2]/255.f;
            }
        }
        cudaMemcpyAsync(buffers_[inIdx_], hostIn.data(), hostIn.size()*sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        ctx_->enqueueV2(buffers_.data(), stream_, nullptr);

        // 출력 두 개라고 가정(outIdxs_[0], outIdxs_[1])
        // 각 출력 shape: (1,1,inH_,inW_)
        auto fetch = [&](int outIdx)->cv::Mat {
            auto dims = ctx_->getBindingDimensions(outIdx);
            int h=dims.d[2], w=dims.d[3];
            std::vector<float> hostOut(h*w);
            cudaMemcpyAsync(hostOut.data(), buffers_[outIdx], hostOut.size()*sizeof(float),
                            cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            cv::Mat m(h,w,CV_32F, hostOut.data());
            return m.clone(); // 복사본 반환
        };
        gridA = fetch(outIdxs_[0]);
        gridB = fetch(outIdxs_[1]);
    }

    int inputH() const { return inH_; }
    int inputW() const { return inW_; }
private:
    nvinfer1::IRuntime* runtime_{};
    nvinfer1::ICudaEngine* engine_{};
    nvinfer1::IExecutionContext* ctx_{};
    cudaStream_t stream_{};
    std::vector<void*> buffers_;
    std::vector<int> outIdxs_;
    int inIdx_{-1}, inH_{}, inW_{};
};

// =============================
// 2) TensorRT 추론 클래스 정의
// =============================
// TensorRT 래퍼 클래스 (입력 크기 고정)
class TRTUltralyticsYOLO {
public:
    TRTUltralyticsYOLO(const std::string& enginePath) {
        // 1) 엔진 읽기 & 생성
        std::ifstream file(enginePath, std::ios::binary);
        if (!file) throw std::runtime_error("Engine 파일 열기 실패");
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> buf(size);
        file.read(buf.data(), size);
        auto runtime = nvinfer1::createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(buf.data(), size, nullptr);
        context = engine->createExecutionContext();
        cudaStreamCreate(&stream);

        // 2) 입력/출력 바인딩 인덱스 및 메모리 할당
        int nBindings = engine->getNbBindings();
        buffers.resize(nBindings);
        for (int i = 0; i < nBindings; ++i) {
            auto dims = engine->getBindingDimensions(i);
            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d) vol *= dims.d[d];
            bool isInput = engine->bindingIsInput(i);
            if (isInput) {
                inputIndex = i;
                inputH = dims.d[2];
                inputW = dims.d[3];
            } else {
                outputIndices.push_back(i);
            }
            size_t bytes = vol * sizeof(float);
            cudaMalloc(&buffers[i], bytes);
        }
    }

    ~TRTUltralyticsYOLO() {
        for (auto& b : buffers) cudaFree(b);
        context->destroy();
        engine->destroy();
        cudaStreamDestroy(stream);
    }

    // 추론: resizedRGB는 inputW×inputH의 RGB Mat
    void infer(const cv::Mat& resizedRGB, std::vector<float>& outputHost) {
        int volIn = 3 * inputH * inputW;
        // HWC→CHW, 정규화
        std::vector<float> hostData(volIn);
        for (int y = 0; y < inputH; ++y) {
            for (int x = 0; x < inputW; ++x) {
                auto px = resizedRGB.at<cv::Vec3b>(y, x);
                int idx = y * inputW + x;
                hostData[0 * inputH * inputW + idx] = px[0] / 255.f;
                hostData[1 * inputH * inputW + idx] = px[1] / 255.f;
                hostData[2 * inputH * inputW + idx] = px[2] / 255.f;
            }
        }
        cudaMemcpyAsync(buffers[inputIndex], hostData.data(), volIn * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // 추론
        context->enqueueV2(buffers.data(), stream, nullptr);

        // 출력 크기 계산 (첫 번째 출력 바인딩 기준)
        int outIdx = outputIndices[0];
        auto outDims = context->getBindingDimensions(outIdx);
        size_t volOut = 1;
        for (int d = 0; d < outDims.nbDims; ++d) volOut *= outDims.d[d];

        outputHost.resize(volOut);
        cudaMemcpyAsync(outputHost.data(), buffers[outIdx], volOut * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    int getInputH() const { return inputH; }
    int getInputW() const { return inputW; }

private:
    nvinfer1::ICudaEngine* engine{};
    nvinfer1::IExecutionContext* context{};
    cudaStream_t stream{};
    int inputIndex{}, inputH{}, inputW{};
    std::vector<int> outputIndices;
    std::vector<void*> buffers;
};

// =============================
// 3) 보조 함수: NMS 및 시각화
// =============================

// IoU 계산 함수 (두 박스의 IoU를 구함, 박스는 [x1,y1,x2,y2] 좌표계)
float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float interArea;
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    interArea = w * h;
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float unionArea = areaA + areaB - interArea;
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

// NMS 알고리즘 구현 (scores는 신뢰도, boxes는 cv::Rect2f 형 벡터)
std::vector<int> NMS(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     float scoreThreshold, float nmsThreshold) {
    std::vector<int> indices;
    // 처음에 scoreThreshold보다 큰 박스 인덱스만 고려
    for (int i = 0; i < (int)scores.size(); ++i) {
        if (scores[i] > scoreThreshold) {
            indices.push_back(i);
        }
    }
    // 점수 내림차순 정렬
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    std::vector<int> result;  // 최종 NMS를 통과한 인덱스
    std::vector<int> temp = indices;
    // NMS 루프
    while (!temp.empty()) {
        int current = temp.front();
        result.push_back(current);
        // current와 비교하여 IoU가 높은 인덱스 제거
        std::vector<int> remaining;
        for (size_t j = 1; j < temp.size(); ++j) {
            int idx = temp[j];
            float iou = IoU(boxes[current], boxes[idx]);
            if (iou <= nmsThreshold) {
                remaining.push_back(idx);
            }
        }
        temp = remaining;
    }
    return result;
}

// 결과 이미지에 박스와 라벨 그리기
void visualizeDetections(cv::Mat& image, const std::vector<int>& indices,
                         const std::vector<cv::Rect2f>& boxes,
                         const std::vector<int>& classIds,
                         const std::vector<float>& scores) {
    for (int idx : indices) {
        cv::Rect box_int = cv::Rect(cv::Point(std::round(boxes[idx].x), std::round(boxes[idx].y)),
                                    cv::Point(std::round(boxes[idx].x + boxes[idx].width),
                                              std::round(boxes[idx].y + boxes[idx].height)));
        // 경계 상자 그리기 (파란색)
        cv::rectangle(image, box_int, cv::Scalar(255, 0, 0), 2);
        // 라벨 텍스트 구성 ("class score")
        std::string label = CLASS_NAMES[classIds[idx]] + " " + cv::format("%.2f", scores[idx]);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
        // 텍스트 배경 사각형 (가독성 높이기 위해 옵션으로 사용할 수 있음)
        cv::rectangle(image, cv::Point(box_int.x, box_int.y - textSize.height - 5),
                      cv::Point(box_int.x + textSize.width, box_int.y), cv::Scalar(255, 255, 255), cv::FILLED);
        // 텍스트 그리기 (박스 상단에)
        cv::putText(image, label, cv::Point(box_int.x, box_int.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
    }
}

// =============================
// 4) 메인 함수: 추론 & 시각화
// =============================
void process(cv::Mat& result_image, const cv::Mat& img,
            TRTUltralyticsYOLO& yolo, const float conf_threshold,
            const float nms_threshold)
{
    auto t1 = Clock::now();
    int origW = img.cols, origH = img.rows;
    int IW = yolo.getInputW(), IH = yolo.getInputH();
    result_image = img.clone();
    cv::Mat resized, resizedRGB;
    cv::resize(img, resized, {IW, IH});
    cv::cvtColor(resized, resizedRGB, cv::COLOR_BGR2RGB);
    auto t2 = Clock::now();
    std::vector<float> outputData;
    yolo.infer(resizedRGB, outputData);
    auto t3 = Clock::now();
    // outputData → boxes, classIds, scores 파싱
    const int numClasses = static_cast<int>(CLASS_NAMES.size()); // 80
    const int C = 4 + numClasses;                               // 84
    // outputData 전체 요소 개수에서 C로 나누어 N을 구함
    int total = static_cast<int>(outputData.size());
    if (total % C != 0) {
        std::cerr << "출력 크기가 예상과 다릅니다: total=" << total << ", C=" << C << std::endl;
        return;
    }
    int N = total / C;

    std::vector<cv::Rect2f> boxes;
    std::vector<int> clsIds;
    std::vector<float> scores;
    for (int i = 0; i < N; ++i) {
        float cx = outputData[0 * N + i];
        float cy = outputData[1 * N + i];
        float w  = outputData[2 * N + i];
        float h  = outputData[3 * N + i];
        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < numClasses; ++c) {
            float score = outputData[(4 + c) * N + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }
        if (maxClassId >= 0 && maxScore > conf_threshold) {
            float x1 = (cx - w/2) * origW/IW;
            float y1 = (cy - h/2) * origH/IH;
            float ww = w * origW/IW, hh = h * origH/IH;
            boxes.emplace_back(x1, y1, ww, hh);
            clsIds.push_back(maxClassId);
            scores.push_back(maxScore);
        }
    }
    auto keep = NMS(boxes, scores, conf_threshold, nms_threshold);
    visualizeDetections(result_image, keep, boxes, clsIds, scores);
    auto t4 = Clock::now();
    auto pre_proc = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto infer_proc = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto post_proc = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    std::cout << "pre processing: " << pre_proc << " ms" << std::endl;
    std::cout << "inference: " << infer_proc << " ms" << std::endl;
    std::cout << "post processing: " << post_proc << " ms" << std::endl;
}
void process_image(const std::string& save_path, const std::string& input_path,
                    const std::string& engine_path, const float conf_threshold,
                    const float nms_threshold, const bool display_result)
{
    cv::Mat src_image = cv::imread(input_path);
    if (src_image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }

    TRTUltralyticsYOLO yolo(engine_path);
    
    cv::Mat dst_image;
    process(dst_image, src_image, yolo, conf_threshold, nms_threshold);
    
    if (!save_path.empty()) {
        cv::imwrite(save_path, dst_image);
        std::cout << "Image saved to: " << save_path << std::endl;
    } 
    if (save_path.empty() || display_result) {
        std::cout << "Processed image displayed." << std::endl;
        cv::namedWindow("Processed Image", cv::WINDOW_NORMAL);
        cv::imshow("Processed Image", dst_image);
        cv::waitKey(0);
    }
}

void process_video(const std::string& save_path, const std::string& input_path,
                    const std::string& engine_path, const float conf_threshold,
                    const float nms_threshold, const bool display_result,
                    const std::string& unet_engine, const std::string& out_dir) // unet 관련 파라미터 추가
{
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open or find the video!" << std::endl;
        return;
    }

    cv::VideoWriter writer;
    if (!save_path.empty()) {
        writer.open(save_path, cap.get(cv::CAP_PROP_FOURCC), cap.get(cv::CAP_PROP_FPS), 
                    cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not open the video writer!" << std::endl;
            return;
        }
    }

    cv::Mat frame, processed_frame;
    TRTUltralyticsYOLO yolo(engine_path);
    TRTUNet2Head unet(unet_engine);
    int idx=0;
    if(save_path.empty() || display_result)
        cv::namedWindow("Processed Video", cv::WINDOW_NORMAL);
    while (cap.read(frame)) {
        process(processed_frame, frame, yolo, conf_threshold, nms_threshold);
        
        cv::Mat gridA32, gridB32;
        unet.infer(frame, gridA32, gridB32);
        cv::Mat gridA8, gridB8;
        gridA32.convertTo(gridA8, CV_8U, 255.0);
        gridB32.convertTo(gridB8, CV_8U, 255.0);

        if (!out_dir.empty()) {
            cv::imwrite(out_dir + "/gridA_" + std::to_string(idx) + ".png", gridA8);
            cv::imwrite(out_dir + "/gridB_" + std::to_string(idx) + ".png", gridB8);
        }
        if (!save_path.empty()) {
            writer.write(processed_frame);
        } 
        if (save_path.empty() || display_result) {
            cv::imshow("Processed Video", processed_frame);
            char c = (char)cv::waitKey(1);
            if (c==27)
                break;
        }
        ++idx;
    }

    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    // 명령행 인자 처리
    if (argc < 3) {
        std::cout << "사용법: " << argv[0] << " --engine <엔진파일> --image <이미지파일> [--conf <신뢰도임계값>] [--nms <NMS임계값>] [--display]\n";
        return 1;
    }
    std::string engine_path;
    std::string input_path;
    std::string save_path = "";
    std::string unet_engine_path;
    std::string out_dir ="grid_out";
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    bool display_result = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--engine" || arg == "-e") && i + 1 < argc) {
            engine_path = argv[++i];
        } else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((arg == "--save" || arg == "-o") && i + 1 < argc) {
            save_path = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--display") {
            display_result = true;
        }
        else if ((arg=="--unet-engine"||arg=="-u") && i+1<argc) {
            unet_engine_path = argv[++i];
        }
        else if ((arg=="--grid-out"   ||arg=="-g") && i+1<argc) {
            out_dir = argv[++i];
        }
    }
    if (engine_path.empty() || input_path.empty()) {
        std::cerr << "엔진 경로와 input_path 경로는 필수입니다.\n";
        return 1;
    }
    if (is_image(input_path)) {
        std::cout << "Processing image: " << input_path << std::endl;
        process_image(save_path, input_path, engine_path, conf_threshold, nms_threshold, display_result);
    } 
    else if (is_video(input_path)) {
        std::cout << "Processing Video: " << input_path << std::endl;
        process_video(save_path, input_path, engine_path, conf_threshold, nms_threshold, display_result, unet_engine_path, out_dir);
    } 
    else {
        std::cout << "Error: Unsupported file type!" << std::endl;
    }
    return 0;
}
