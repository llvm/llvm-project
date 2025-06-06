
//  Copyright John Maddock 2016.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/precision.hpp>
#include "cuda_managed_ptr.hpp"
#include "stopwatch.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

typedef double float_type;

__host__ __device__ float_type func(float_type x)
{
    BOOST_MATH_STD_USING
    return 1/(1+x*x);
}

/**
 * CUDA Kernel Device code
 *
 */
__global__ void cuda_test(float_type *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float_type tol = boost::math::tools::root_epsilon<float_type>();
    float_type error;
    float_type L1;
    boost::math::size_t levels;

    if (i < numElements)
    {
        out[i] = boost::math::quadrature::exp_sinh_integrate(func, tol, &error, &L1, &levels);
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
    cuda_managed_ptr<float_type> input_vector(numElements);

    // Allocate the managed output vector C
    cuda_managed_ptr<float_type> output_vector(numElements);

    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i)
    {
        input_vector[i] = M_PI * (static_cast<float_type>(i) / numElements);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 512;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

    watch w;

    cuda_test<<<blocksPerGrid, threadsPerBlock>>>(output_vector.get(), numElements);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernal done in: " << w.elapsed() << "s" << std::endl;

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
    float_type tol = boost::math::tools::root_epsilon<float_type>();
    float_type error;
    float_type L1;
    boost::math::quadrature::exp_sinh<float_type> integrator;
    for(int i = 0; i < numElements; ++i)
    {
       results.push_back(integrator.integrate(func, tol, &error, &L1));
    }
    double t = w.elapsed();
    // check the results
    int failed_count = 0;
    for(int i = 0; i < numElements; ++i)
    {
        const auto eps = boost::math::epsilon_difference(output_vector[i], results[i]);
        if (eps > 10)
        {
            std::cerr   << std::setprecision(std::numeric_limits<float_type>::digits10)
                        << "Result verification failed at element " << i << "!\n"
                        << "Device: " << output_vector[i]
                        << "\n  Host: " << results[i]
                        << "\n   Eps: " << eps << "\n";
            failed_count++;
        }
        if (failed_count > 100)
        {
            break;
        }
    }

    if (failed_count != 0)
    {
        std::cout << "Test FAILED" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Test PASSED, normal calculation time: " << t << "s" << std::endl;
    std::cout << "Done\n";

    return 0;
}
