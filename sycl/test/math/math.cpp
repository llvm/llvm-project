// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl -lm
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--------------- math.cpp - SYCL math test ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

int main() {
  float data_1[10];
  for (size_t i = 0; i < 10; ++i) {
    data_1[i] = 1;
  }
  float data_2[10];
  for (size_t i = 0; i < 10; ++i) {
    data_2[i] = 2;
  }
  float result[10] = {0};

  {
    range<1> numOfItems{10};
    buffer<float, 1> bufferData_1(data_1, numOfItems);
    buffer<float, 1> bufferData_2(data_2, numOfItems);
    buffer<float, 1> resultBuffer(result, numOfItems);
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      accessor<float, 1, access::mode::read, access::target::global_buffer,
               access::placeholder::false_t>
          accessorData_1(bufferData_1, cgh);
      accessor<float, 1, access::mode::read_write,
               access::target::global_buffer, access::placeholder::false_t>
          accessorData_2(bufferData_2, cgh);
      accessor<float, 1, access::mode::read_write,
               access::target::global_buffer, access::placeholder::false_t>
          resultAccessor(resultBuffer, cgh);
      cgh.parallel_for<class MathKernel>(range<1>{10}, [=](id<1> wiID) {
        resultAccessor[wiID.get(0)] = cl::sycl::fmax(
            accessorData_1[wiID.get(0)], accessorData_2[wiID.get(0)]);
        resultAccessor[wiID.get(0)] += cl::sycl::fmin(1.f, 2.f);
        resultAccessor[wiID.get(0)] += cl::sycl::native::exp(2.f);
        resultAccessor[wiID.get(0)] += cl::sycl::fabs(-2.f);
        resultAccessor[wiID.get(0)] += cl::sycl::fabs(1.0);
        resultAccessor[wiID.get(0)] += cl::sycl::floor(-3.4);
        resultAccessor[wiID.get(0)] += cl::sycl::ceil(2.4);
      });
    });
  }
  for (size_t i = 0; i < 10; ++i) {
    /* Result of addition of 2 + 1 + 7.389... + 2 + 1 -4 + 3*/
    assert(result[i] > 12 && result[i] < 13 &&
           "Expected result[i] > 12 &&  result[i] < 13");
  }

  return 0;
}
