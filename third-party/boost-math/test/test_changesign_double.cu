//  Copyright John Maddock 2016.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include "cuda_managed_ptr.hpp"
#include "stopwatch.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

typedef double float_type;

/**
 * CUDA Kernel Device code
 *
 */
__global__ void cuda_test(const float_type *in, float_type *out, int numElements)
{
    using std::cos;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        out[i] = boost::math::changesign(in[i]);
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    std::cout << "[Vector addition of " << numElements << " elements]" << std::endl;

    // Allocate the managed input vector A
    cuda_managed_ptr<float_type> h_A(numElements);

    // Allocate the managed output vector C
    cuda_managed_ptr<float_type> h_C(numElements);

    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float_type)RAND_MAX;
        switch(i % 55)
        {
        case 1:
           h_A[i] = 0;
           break;
        case 2:
           h_A[i] = std::numeric_limits<float_type>::infinity();
           break;
        case 3:
           h_A[i] = -std::numeric_limits<float_type>::infinity();
           break;
        }
        if(i % 1)
           h_A[i] = -h_A[i];
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

    watch w;
    cuda_test<<<blocksPerGrid, threadsPerBlock>>>(h_A.get(), h_C.get(), numElements);
    cudaDeviceSynchronize();
    std::cout << "CUDA kernal done in " << w.elapsed() << "s" << std::endl;

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to launch vectorAdd kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        return EXIT_FAILURE;
    }

    // Verify that the result vector is correct
    std::vector<float_type> results;
    results.reserve(numElements);
    w.reset();
    for(int i = 0; i < numElements; ++i)
       results.push_back(boost::math::changesign(h_A[i]));
    double t = w.elapsed();
    // check the results
    for(int i = 0; i < numElements; ++i)
    {
        if (h_C[i] != results[i])
        {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Test PASSED with calculation time: " << t << "s" << std::endl;
    std::cout << "Done\n";

    return 0;
}
