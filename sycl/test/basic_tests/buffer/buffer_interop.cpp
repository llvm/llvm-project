// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------------- buffer.cpp - SYCL buffer basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <memory>

using namespace cl::sycl;

int main() {
  {
    const size_t Size = 32;
    int Init[Size] = {5};
    cl_int Error = CL_SUCCESS;

    queue MyQueue;

    cl_mem OpenCLBuffer = clCreateBuffer(
        MyQueue.get_context().get(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        Size * sizeof(int), Init, &Error);
    CHECK_OCL_CODE(Error);
    buffer<int, 1> Buffer(OpenCLBuffer, MyQueue.get_context());

    MyQueue.submit([&](handler &CGH) {
      auto B = Buffer.get_access<access::mode::write>(CGH);
      CGH.parallel_for<class BufferInterop>(
          range<1>{Size}, [=](id<1> Index) { B[Index] = 10; });
    });

    int Data[Size] = {10};
    std::vector<int> Result(Size, 0);
    {
      buffer<int, 1> BufferData(Data, range<1>(Size),
                                {property::buffer::use_host_ptr()});
      BufferData.set_final_data(Result.begin());
      MyQueue.submit([&](handler &CGH) {
        auto Data = BufferData.get_access<access::mode::write>(CGH);
        auto CLData = Buffer.get_access<access::mode::read>(CGH);
        CGH.parallel_for<class UseMemContent>(range<1>{Size}, [=](id<1> Index) {
          Data[Index] = 2 * CLData[Index];
        });
      });
    }

    Error = clReleaseMemObject(OpenCLBuffer);
    CHECK_OCL_CODE(Error);

    for (size_t i = 0; i < Size; ++i) {
      if (Result[i] != 20) {
        std::cout << " array[" << i << "] is " << Result[i] << " expected "
                  << 20 << std::endl;
        assert(false);
      }
    }
  }
  return 0;
}
