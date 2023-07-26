//
// Created by ubuntu on 1/20/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include "uart.hpp"

const std::vector<std::string> CLASS_NAMES = {
    "ambulance", "fire truck", "police car", "baby car", "Off-road vehicles", "race car"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

int main(int argc, char** argv)
{
    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    assert(argc == 3);

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFolder(path)) {
        cv::glob(path + "/*.jpg", imagePathList);
    }

	unsigned char  w_buf[2] = {0x0, 0x00};
	size_t w_len = sizeof(w_buf);
	unsigned char r_buf[10] = {0};
	int ret = -1;
	int fd = -1;

	fd = uart_open("/dev/ttyUSB0");
	uart_setup(fd, B115200);

#if 0
	ret = uart_write(fd,w_buf,w_len);
	if(ret == -1)
	{
		fprintf(stderr,"uart write failed!\n");
		exit(EXIT_FAILURE);
	}

	ret = uart_read(fd,r_buf,w_len);
	if(ret == -1)
	{
		fprintf(stderr,"uart read failed!\n");
		exit(EXIT_FAILURE);
	}
#endif
    cv::Mat             res, image;
    cv::Size            size = cv::Size{640, 640};
    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);
        //cv::VideoCapture cap(0);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
			for (auto& obj : objs) {
				cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
				cv::rectangle(res, obj.rect, color, 2);
				// top: 0, down: 1, left: 2, right: 3
				char text[256];
				int      baseLine   = 0;
				cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 2, &baseLine);
				if (obj.label == 0 || obj.label == 1)
				{
					if (obj.rect.x < 310 ) //left
					{
						w_buf[1]= 0x2; //left
						sprintf(text, "%s %.1f%%, left ", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);
					cv::putText(res, text, cv::Point(20, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 0}, 1);
					}

					if (obj.rect.x > 330) 
					{
						w_buf[1]= 0x3; //right
						sprintf(text, "%s %.1f%%, right ", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);
					cv::putText(res, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 0, 0}, 1);
					}
					if (obj.rect.y < 310) 
					{
						w_buf[1]= 0x0; //top
						sprintf(text, "%s  %.1f%%, top ", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);
					cv::putText(res, text, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 255, 0}, 1);
					}

					if (obj.rect.y > 330) 
					{
						w_buf[1]= 0x1; // down
						sprintf(text, "%s %.1f%%, down ", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);
					cv::putText(res, text, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 0, 255}, 1);
					}

					w_buf[0]= obj.label;
					ret = uart_write(fd,w_buf,2);
					if(ret == -1)
					{
						fprintf(stderr,"uart write failed!\n");
						exit(EXIT_FAILURE);
					}


				}
			}

            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& path : imagePathList) {
            objs.clear();
            image = cv::imread(path);
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }

	uart_close(fd);
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
