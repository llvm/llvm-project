// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--------------- kernel_interop.cpp - SYCL kernel ocl interop test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

#include <cassert>

using namespace cl::sycl;

// This test checks that SYCL kernel interoperabitily constructor is implemented
// in accordance with SYCL spec:
// - It throws an exception when passed SYCL context doesn't represent the same
//   underlying OpenCL context associated with passed cl_kernel
// - It retains passed cl_kernel so releasing kernel won't produce errors.

int main() {
  queue Queue;
  if (!Queue.is_host()) {

    context Context = Queue.get_context();

    cl_context ClContext = Context.get();

    const size_t CountSources = 1;
    const char *Sources[CountSources] = {
        "kernel void foo1(global float* Array, global int* Value) { *Array = "
        "42; *Value = 1; }\n",
    };

    cl_int Err;
    cl_program ClProgram = clCreateProgramWithSource(ClContext, CountSources,
                                                     Sources, nullptr, &Err);
    CHECK_OCL_CODE(Err);

    Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
    CHECK_OCL_CODE(Err);

    cl_kernel ClKernel = clCreateKernel(ClProgram, "foo1", &Err);
    CHECK_OCL_CODE(Err);

    // Try to create kernel with another context
    bool Pass = false;
    queue Queue1;
    context Context1 = Queue1.get_context();
    try {
      kernel Kernel(ClKernel, Context1);
    } catch (cl::sycl::invalid_parameter_error e) {
      Pass = true;
    }
    assert(Pass);

    kernel Kernel(ClKernel, Context);


    CHECK_OCL_CODE(clReleaseKernel(ClKernel));
    CHECK_OCL_CODE(clReleaseContext(ClContext));
    CHECK_OCL_CODE(clReleaseProgram(ClProgram));

  }
  return 0;
}
