//  Copyright John Maddock 2016.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

// floating-point value does not fit in required floating-point type
#pragma nv_diag_suppress 221

#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/array.hpp>
#include "cuda_managed_ptr.hpp"
#include "stopwatch.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

typedef float float_type;

/**
 * CUDA Kernel Device code
 *
 */
__global__ void cuda_test(const float_type *in1, const float_type *in2, const float_type *in3, float_type *out, int numElements)
{
    using std::cos;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        out[i] = boost::math::ibetac_inv(in1[i], in2[i], in3[i]);
    }
}

template <class T> struct table_type { typedef T type; };
typedef float_type T;
#define SC_(x) static_cast<T>(x)

#include "ibeta_data.ipp"
#include "ibeta_small_data.ipp"

/**
 * Host main routine
 */
int main(void)
{
  try{
    // Consolidate the test data:
    std::vector<float_type> v1, v2, v3;

    for(unsigned i = 0; i < ibeta_data.size(); ++i)
    {
       v1.push_back(ibeta_data[i][0]);
       v2.push_back(ibeta_data[i][1]);
       v3.push_back(ibeta_data[i][2]);
    }
    for(unsigned i = 0; i < ibeta_small_data.size(); ++i)
    {
       v1.push_back(ibeta_small_data[i][0]);
       v2.push_back(ibeta_small_data[i][1]);
       v3.push_back(ibeta_small_data[i][2]);
    }

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    std::cout << "[Vector operation on " << numElements << " elements]" << std::endl;

    // Allocate the managed input vector A
    cuda_managed_ptr<float_type> input_vector1(numElements);
    cuda_managed_ptr<float_type> input_vector2(numElements);
    cuda_managed_ptr<float_type> input_vector3(numElements);

    // Allocate the managed output vector C
    cuda_managed_ptr<float_type> output_vector(numElements);

    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i)
    {
        int table_id = i % v1.size();
        input_vector1[i] = v1[table_id];
        input_vector2[i] = v2[table_id];
        input_vector3[i] = v3[table_id];
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

    watch w;
    cuda_test<<<blocksPerGrid, threadsPerBlock>>>(input_vector1.get(), input_vector2.get(), input_vector3.get(), output_vector.get(), numElements);
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
       results.push_back(boost::math::ibetac_inv(input_vector1[i], input_vector2[i], input_vector3[i]));
    double t = w.elapsed();
    bool failed = false;
    // check the results
    for(int i = 0; i < numElements; ++i)
    {
        if (boost::math::isfinite(output_vector[i]))
        {
            if (boost::math::epsilon_difference(output_vector[i], results[i]) > 300)
            {
                std::cerr << "Result verification failed at element " << i << "!" << std::endl;
                std::cerr << "Error rate was: " << boost::math::epsilon_difference(output_vector[i], results[i]) << "eps" << std::endl;
                failed = true;
            }
        }
    }

    if (failed)
    {
        return EXIT_FAILURE;
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


