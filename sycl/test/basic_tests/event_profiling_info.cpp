// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
//
// Profiling info is not supported on host device so far.
//
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------------- event_profiling_info.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl;

// The test checks that get_profiling_info waits for command asccociated with
// event to complete execution.
int main() {
  sycl::queue Q{sycl::property::queue::enable_profiling()};
  sycl::event Event = Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class EmptyKernel>([=]() {});
  });

  Event.get_profiling_info<sycl::info::event_profiling::command_start>();

  bool Fail = sycl::info::event_command_status::complete !=
              Event.get_info<sycl::info::event::command_execution_status>();

  return Fail;
}
