// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==---------- subbuffer.cpp --- sub-buffer basic test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {

  bool Failed = false;
  // Basic test case
  {
    const int M = 6;
    const int N = 7;
    int Result[M][N] = {0};
    {
      auto OrigRange = range<2>(M, N);
      buffer<int, 2> Buffer(OrigRange);
      Buffer.set_final_data((int *)Result);
      auto Offset = id<2>(1, 1);
      auto SubRange = range<2>(M - 2, N - 2);
      queue MyQueue;
      buffer<int, 2> SubBuffer(Buffer, Offset, SubRange);
      MyQueue.submit([&](handler &cgh) {
        auto B = SubBuffer.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class Subbuf_test>(SubRange,
                                            [=](id<2> Index) { B[Index] = 1; });
      });
    }

    // Check that we filled correct subset of buffer:
    // 0000000     0000000
    // 0000000     0111110
    // 0000000 --> 0111110
    // 0000000     0111110
    // 0000000     0111110
    // 0000000     0000000

    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t Expected =
            ((i == 0) || (i == M - 1) || (j == 0) || (j == N - 1)) ? 0 : 1;
        if (Result[i][j] != Expected) {
          std::cout << "line: " << __LINE__ << " Result[" << i << "][" << j
                    << "] is " << Result[i][j] << " expected " << Expected
                    << std::endl;
          Failed = true;
        }
      }
    }
  }
  // Try to create subbuffer from subbuffer
  {
    const int M = 10;
    int Data[M] = {0};
    auto OrigRange = range<1>(M);
    buffer<int, 1> Buffer(Data, OrigRange);
    auto Offset = id<1>(1);
    auto SubRange = range<1>(M - 2);
    auto SubSubRange = range<1>(M - 4);
    queue MyQueue;
    buffer<int, 1> SubBuffer(Buffer, Offset, SubRange);
    buffer<int, 1> SubSubBuffer(SubBuffer, Offset, SubSubRange);
    MyQueue.submit([&](handler &cgh) {
      auto B = SubSubBuffer.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class Subsubbuf_test>(SubSubRange,
                                          [=](id<1> Index) { B[Index] = 1; });
    });
    auto Acc = Buffer.get_access<cl::sycl::access::mode::read>();
    for (size_t i = 0; i < M; ++i) {
      size_t Expected = (i > 1 && i < M - 2) ? 1 : 0;
      if (Acc[i] != Expected) {
        std::cout << "line: " << __LINE__ << " Data[" << i << "] is " << Acc[i]
                  << " expected " << Expected << std::endl;
        Failed = true;
      }
    }
  }
  return Failed;
}
