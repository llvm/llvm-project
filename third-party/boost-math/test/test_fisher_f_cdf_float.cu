//  Copyright John Maddock 2016.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "cuda_managed_ptr.hpp"
#include "stopwatch.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

typedef float float_type;

/**
 * CUDA Kernel Device code
 *
 */
__global__ void cuda_test(const float_type *in1, float_type *out, int numElements)
{
    using std::cos;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        out[i] = cdf(boost::math::fisher_f_distribution<float_type>(1, 1), in1[i]);
    }
}

/**
 * Host main routine
 */
int main(void)
{
  try{

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    std::cout << "[Vector operation on " << numElements << " elements]" << std::endl;

    // Allocate the managed input vector A
    cuda_managed_ptr<float_type> input_vector1(numElements);

    // Allocate the managed output vector C
    cuda_managed_ptr<float_type> output_vector(numElements);

    boost::random::mt19937 gen;
    boost::random::uniform_real_distribution<float_type> dist;
    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i)
    {
        input_vector1[i] = dist(gen);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

    watch w;
    cuda_test<<<blocksPerGrid, threadsPerBlock>>>(input_vector1.get(), output_vector.get(), numElements);
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
       results.push_back(cdf(boost::math::fisher_f_distribution<float_type>(1, 1), input_vector1[i]));
    double t = w.elapsed();
    // check the results
    for(int i = 0; i < numElements; ++i)
    {
        if (boost::math::epsilon_difference(output_vector[i], results[i]) > 100.0)
        {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            std::cerr << "Error rate was: " << boost::math::epsilon_difference(output_vector[i], results[i]) << "eps" << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Test PASSED with calculation time: " << t << "s" << std::endl;
    std::cout << "Done\n";
  }
  catch(const std::exception& e)
  {
    std::cerr << "Stopped with exception: " << e.what() << std::endl;
  }
  return 0;
}