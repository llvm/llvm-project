//  Copyright John Maddock 2016.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

// Must be included first
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <exception>
#include <boost/math/distributions/saspoint5.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/relative_difference.hpp>

typedef float float_type;

const char* cuda_kernel = R"(
typedef float float_type;
#include <cuda/std/type_traits>
#include <boost/math/distributions/saspoint5.hpp>
extern "C" __global__ 
void test_saspoint5_kernel(const float_type *in1, const float_type*, float_type *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        out[i] = cdf(boost::math::saspoint5_distribution<float_type>(), in1[i]);
    }
}
)";

void checkCUDAError(cudaError_t result, const char* msg)
{
    if (result != cudaSuccess)
    {
        std::cerr << msg << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCUError(CUresult result, const char* msg)
{
    if (result != CUDA_SUCCESS)
    {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        std::cerr << msg << ": " << errorStr << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkNVRTCError(nvrtcResult result, const char* msg)
{
    if (result != NVRTC_SUCCESS)
    {
        std::cerr << msg << ": " << nvrtcGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() 
{
    try
    {
        // Initialize CUDA driver API
        checkCUError(cuInit(0), "Failed to initialize CUDA");

        // Create CUDA context
        CUcontext context;
        CUdevice device;
        checkCUError(cuDeviceGet(&device, 0), "Failed to get CUDA device");
        checkCUError(cuCtxCreate(&context, 0, device), "Failed to create CUDA context");

        nvrtcProgram prog;
        nvrtcResult res;

        res = nvrtcCreateProgram(&prog, cuda_kernel, "test_saspoint5_kernel.cu", 0, nullptr, nullptr);
        checkNVRTCError(res, "Failed to create NVRTC program");

        nvrtcAddNameExpression(prog, "test_saspoint5_kernel");

        #ifdef BOOST_MATH_NVRTC_CI_RUN
        const char* opts[] = {"--std=c++14", "--gpu-architecture=compute_75", "--include-path=/home/runner/work/cuda-math/boost-root/libs/cuda-math/include/", "-I/usr/local/cuda/include"};
        #else
        const char* opts[] = {"--std=c++14", "--include-path=/home/mborland/Documents/boost/libs/cuda-math/include/", "-I/usr/local/cuda/include"};
        #endif

        // Compile the program
        res = nvrtcCompileProgram(prog, sizeof(opts) / sizeof(const char*), opts);
        if (res != NVRTC_SUCCESS) 
        {
            size_t log_size;
            nvrtcGetProgramLogSize(prog, &log_size);
            char* log = new char[log_size];
            nvrtcGetProgramLog(prog, log);
            std::cerr << "Compilation failed:\n" << log << std::endl;
            delete[] log;
            exit(EXIT_FAILURE);
        }

        // Get PTX from the program
        size_t ptx_size;
        nvrtcGetPTXSize(prog, &ptx_size);
        char* ptx = new char[ptx_size];
        nvrtcGetPTX(prog, ptx);

        // Load PTX into CUDA module
        CUmodule module;
        CUfunction kernel;
        checkCUError(cuModuleLoadDataEx(&module, ptx, 0, 0, 0), "Failed to load module");
        checkCUError(cuModuleGetFunction(&kernel, module, "test_saspoint5_kernel"), "Failed to get kernel function");

        int numElements = 5000;
        float_type *h_in1, *h_in2, *h_out;
        float_type *d_in1, *d_in2, *d_out;

        // Allocate memory on the host
        h_in1 = new float_type[numElements];
        h_in2 = new float_type[numElements];
        h_out = new float_type[numElements];

        // Initialize input arrays
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<float_type> dist(0.0f, 1.0f);
        for (int i = 0; i < numElements; ++i) 
        {
            h_in1[i] = static_cast<float_type>(dist(rng));
            h_in2[i] = static_cast<float_type>(dist(rng));
        }

        checkCUDAError(cudaMalloc(&d_in1, numElements * sizeof(float_type)), "Failed to allocate device memory for d_in1");
        checkCUDAError(cudaMalloc(&d_in2, numElements * sizeof(float_type)), "Failed to allocate device memory for d_in2");
        checkCUDAError(cudaMalloc(&d_out, numElements * sizeof(float_type)), "Failed to allocate device memory for d_out");

        checkCUDAError(cudaMemcpy(d_in1, h_in1, numElements * sizeof(float_type), cudaMemcpyHostToDevice), "Failed to copy data to device for d_in1");
        checkCUDAError(cudaMemcpy(d_in2, h_in2, numElements * sizeof(float_type), cudaMemcpyHostToDevice), "Failed to copy data to device for d_in2");

        int blockSize = 256;
        int numBlocks = (numElements + blockSize - 1) / blockSize;
        void* args[] = { &d_in1, &d_in2, &d_out, &numElements };
        checkCUError(cuLaunchKernel(kernel, numBlocks, 1, 1, blockSize, 1, 1, 0, 0, args, 0), "Kernel launch failed");

        checkCUDAError(cudaMemcpy(h_out, d_out, numElements * sizeof(float_type), cudaMemcpyDeviceToHost), "Failed to copy data back to host for h_out");

        // Verify Result
        for (int i = 0; i < numElements; ++i) 
        {
            auto res = cdf(boost::math::saspoint5_distribution<float_type>(), h_in1[i]);
            
            if (boost::math::isfinite(res))
            {
                if (boost::math::epsilon_difference(res, h_out[i]) > 300)
                {
                    std::cout << "error at line: " << i
                            << "\nParallel: " << h_out[i]
                            << "\n  Serial: " << res
                            << "\n    Dist: " << boost::math::epsilon_difference(res, h_out[i]) << std::endl;
                }
            }
        }

        cudaFree(d_in1);
        cudaFree(d_in2);
        cudaFree(d_out);
        delete[] h_in1;
        delete[] h_in2;
        delete[] h_out;

        nvrtcDestroyProgram(&prog);
        delete[] ptx;

        cuCtxDestroy(context);

        std::cout << "Kernel executed successfully." << std::endl;
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Stopped with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
