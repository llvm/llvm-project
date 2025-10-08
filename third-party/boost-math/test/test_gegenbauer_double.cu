
//  Copyright John Maddock 2016.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Gegenbauer prime uses all methods internally so it's the easy choice

#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/gegenbauer.hpp>
#include "cuda_managed_ptr.hpp"
#include "stopwatch.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

typedef double float_type;

/**
 * CUDA Kernel Device code
 *
 */
__global__ void cuda_test(const float_type *in1, const float_type *in2, float_type *out, int numElements)
{
    using std::cos;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        out[i] = boost::math::gegenbauer_prime(2, in1[i], in2[i]);
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
    std::cout << "[Vector operation on " << numElements << " elements]" << std::endl;

    // Allocate the managed input vector A
    cuda_managed_ptr<float_type> input_vector1(numElements);

    // Allocate the managed input vector B
    cuda_managed_ptr<float_type> input_vector2(numElements);

    // Allocate the managed output vector C
    cuda_managed_ptr<float_type> output_vector(numElements);

    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i)
    {
        input_vector1[i] = rand()/(float_type)RAND_MAX;
        input_vector2[i] = rand()/(float_type)RAND_MAX;
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

    watch w;

    cuda_test<<<blocksPerGrid, threadsPerBlock>>>(input_vector1.get(), input_vector2.get(), output_vector.get(), numElements);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernal done in: " << w.elapsed() << "s" << std::endl;

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to launch CUDA kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        return EXIT_FAILURE;
    }

    // Verify that the result vector is correct
    std::vector<float_type> results;
    results.reserve(numElements);
    w.reset();
    for(int i = 0; i < numElements; ++i)
       results[i] = boost::math::gegenbauer_prime(2, input_vector1[i], input_vector2[i]);
    double t = w.elapsed();
    // check the results
    int failure_counter = 0;
    for(int i = 0; i < numElements; ++i)
    {
        if (std::isfinite(results[i]))
        {
            const auto eps = boost::math::epsilon_difference(output_vector[i], results[i]);
            // Most elements are under 50 but extremely small numbers very more greatly
            if (eps > 1000)
            {
                std::cerr << "Result verification failed at element " << i << "!\n"
                          << "Device: " << output_vector[i]
                          << "\n  Host: " << results[i]
                          << "\n   Eps: " << eps << std::endl;
                ++failure_counter;
                if (failure_counter > 100)
                {
                    break;
                }
            }
        }
    }

    if (failure_counter > 0)
    {
        return EXIT_FAILURE;
    }

    std::cout << "Test PASSED, normal calculation time: " << t << "s" << std::endl;
    std::cout << "Done\n";

    return 0;
}
