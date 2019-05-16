// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------------- BasicSchedulerTests.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl;
using sycl_access_mode = cl::sycl::access::mode;

// Execute functor provided passing a queue created with default device selector
// and async handler then waits for the tasks submitted to queue to finish. If
// async exceptions are caught prints them and throw exception signaling about
// it.
template <class TestFuncT> void runTest(TestFuncT TestFunc) {
  bool AsyncException = false;

  sycl::queue Queue([&AsyncException](sycl::exception_list ExceptionList) {
    AsyncException = true;
    for (sycl::exception_ptr_class ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (sycl::exception &E) {
        std::cerr << E.what();
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  });

  TestFunc(Queue);

  Queue.wait_and_throw();

  if (AsyncException)
    throw "Async exception caught";
}

int main() {
  bool Failed = false;

  // Checks creating of the second host accessor while first one is alive.
  try {
    sycl::range<1> BufSize{1};
    sycl::buffer<int, 1> Buf1(BufSize);
    sycl::buffer<int, 1> Buf2(BufSize);

    runTest([&](sycl::queue Queue) {

      Queue.submit([&](sycl::handler &CGH) {
        auto Buf1Acc = Buf1.get_access<sycl_access_mode::read_write>(CGH);
        auto Buf2Acc = Buf2.get_access<sycl_access_mode::read_write>(CGH);
        CGH.parallel_for<class init_a>(
            BufSize, [=](sycl::id<1> Id) { Buf1Acc[Id] = Buf2Acc[Id]; });
      });

      auto Buf1HostAcc = Buf1.get_access<sycl_access_mode::read>();
      auto Buf2HostAcc = Buf2.get_access<sycl_access_mode::read>();
    });

  } catch (...) {
    std::cerr << "Two host accessors test failed." << std::endl;
    Failed = true;
  }

  return Failed;
}
