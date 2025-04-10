// На С++ реализовать приложение модулятора FM сигнала. Необходимо прочитать сигнал из .wav файла,
// загрузить на устройство, промодулировать, выкачать результат на хост и записать в .bin файл.
// Пример запуска: ./modulator_fm input_file.wav output_file.bin
#include <iostream>
#include "AudioFile.h"
#include <vector>
#include <cmath>
#include "CL/cl.hpp"

# define PI           3.14159265358979323846

int main()
{
    AudioFile<float> in;
    in.load("./input_file.wav");
    /*std::vector<float> A = {1, 2, 3, 4, 5};
    std::vector<float> B = { 10, 20, 30, 40, 50 };
    std::vector<float> C(A.size());*/

    std::vector<float> A = in.samples[0];
    std::vector<float> B(A.size());
    

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device device;
    for (const auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_CPU, &devices);
        if (!devices.empty()) {
            device = devices.front();
            break;
        }
    }
    cl::Context context(device);  // Создаем контекст и очередь команд
    cl::CommandQueue queue(context, device);

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data()); // Выделяем память на GPU
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(float) * B.size());
    
    // Компилируем ядро
    const char* kernel_code = R"( 
     __kernel void fm_mod(__global const float *A, __global const float *B) {
            int id = get_global_id(0);
            B[id] = 1;
        }
    )";
    //B[id] = cos(2 * PI * (1000 + 2000 * A[id]));
    cl::Program program(context, kernel_code);
    program.build("-cl-std=CL1.2");
    //program.build(device);
    // Создаем ядро и устанавливаем аргументы
    cl::Kernel kernel(program, "fm_mod");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    // Запускаем ядро
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(A.size()), cl::NullRange);
    // Копируем результат обратно на CPU
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(float) * B.size(), B.data());
    
    /*for (float b : B) {
        std::cout << b << " ";
    }
    return 0;*/

    std::ofstream out("output_file.bin", std::ios::out | std::ios::binary);
    for (int i = 0; i < B.size(); i++)
    {
        out.write((char*)&B[i], sizeof(float));
    }
}
