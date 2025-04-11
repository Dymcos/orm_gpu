#ifdef __unix__
#include "CL/opencl.hpp"
#endif
#ifdef _WIN32
#include "CL/cl.hpp"
#endif
#include <iostream>
#include <vector>
#include <math.h>
#include "AudioFile.h"

# define PI 3.14159265358979323846

int main(int argc, char* argv[]){
    
    // загрузка аудиофайла
    AudioFile<float> in;
    in.load(argc == 1 ? "../input_file.wav" : argv[1]);
    
    // Исходные данные
    std::vector<float> A = in.samples[0]; // модулирующий сигнал
    std::vector<float> B(A.size()); // модулированный сигнал
    std::cout << in.getSampleRate();

    std::vector<float> C(1024); // определяем массив значений косинуса
    for (int i = 0; i<1024; i++)
    {
      C[i] = cos(2*PI / 1024 * i);
    }
    // Получаем платформы и устройства
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device device;
    for (const auto &platform: platforms){
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if(!devices.empty()){
            device = devices.front();
            break;
        }
    }

    // Создаем контекст и очередь команд
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Выделяем память на GPU
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data());
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(float) * B.size());
    cl::Buffer bufferC(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * C.size(), C.data());

    // Компилируем ядро
    const char* kernel_code = R"(
     __kernel void fm_mod(__global const float *A, __global float *B, __global const float *C) {
            int id = get_global_id(0);
            B[id] = C[((int)((1000 + 500 * A[id] / 44100) * id)) % 1024];
        }
    )";
    cl::Program program(context, kernel_code);
    program.build("-cl-std=CL1.2");

    // Создаем ядро и устанавливаем аргументы
    cl::Kernel kernel(program, "fm_mod");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    
    // Запускаем ядро
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(A.size()), cl::NullRange);

    // Копируем результат обратно на CPU
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(float) * B.size(), B.data());

    // Выводим результат в .bin
    std::ofstream out(argc < 3 ? "../output_file.bin" : argv[2], std::ios::out | std::ios::binary);
    for (int i = 0; i < B.size(); i++)
    {
    out.write((char*)&B[i], sizeof(float));
    }

    // Дополнительное сохранение в .wav
    AudioFile<float> outWav;
    outWav.setNumChannels (1);
    outWav.setNumSamplesPerChannel (44100);
    for (int i = 0; i < outWav.getNumSamplesPerChannel(); i++)
    {
      outWav.samples[0][i] = A[i];
    }
    std::string filePath = "../out_file.wav";
    outWav.save ("../out_file.wav", AudioFileFormat::Wave);
}