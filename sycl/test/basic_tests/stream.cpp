// RUN: %clang -std=c++11 -fsycl -lstdc++ %s -o %t.out -lOpenCL -lsycl
// TEMPORARY_DISABLED_RUNx: env SYCL_DEVICE_TYPE=HOST %t.out | FileCheck %s
// TEMPORARY_DISABLED_RUNx: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// TEMPORARY_DISABLED_RUNx: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// TEMPORARY_DISABLED_RUNx: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
//==------------------ stream.cpp - SYCL stream basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/context.hpp>
#include <cassert>

int main() {
  {
    cl::sycl::default_selector Selector;
    cl::sycl::queue Queue(Selector);

    // Check constructor and getters
    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out(1024, 80, CGH);
      assert(Out.get_size() == 1024);
      assert(Out.get_max_statement_size() == 80);
    });

    // Check common reference semantics
    cl::sycl::hash_class<cl::sycl::stream> Hasher;

    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out1(1024, 80, CGH);
      cl::sycl::stream Out2(Out1);

      assert(Out1 == Out2);
      assert(Hasher(Out1) == Hasher(Out2));

      cl::sycl::stream Out3(std::move(Out1));

      assert(Out2 == Out3);
      assert(Hasher(Out2) == Hasher(Out3));
    });

    // Char type
    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out(1024, 80, CGH);
      CGH.parallel_for<class stream_char>(
          cl::sycl::range<1>(10), [=](cl::sycl::id<1> i) { Out << 'a'; });
    });
    Queue.wait();

    // endl manipulator
    // TODO: support cl::sycl::endl. According to specitification endl should be
    // constant global variable in cl::sycl which is initialized with
    // cl::sycl::stream_manipulator::endl. This approach doesn't currently work,
    // variable is not initialized in the kernel code, it contains some garbage
    // value.
    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out(1024, 80, CGH);
      CGH.single_task<class stream_endl>(
          [=]() { Out << cl::sycl::stream_manipulator::endl; });
    });
    Queue.wait();

    // String type
    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out(1024, 80, CGH);
      CGH.parallel_for<class stream_string>(
          cl::sycl::range<1>(10),
          [=](cl::sycl::id<1> i) { Out << "Hello, World!\n"; });
    });
    Queue.wait();

    // Boolean type
    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out(1024, 80, CGH);
      CGH.single_task<class stream_bool1>([=]() { Out << true; });
    });
    Queue.wait();

    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out(1024, 80, CGH);
      CGH.single_task<class stream_bool2>([=]() { Out << false; });
    });
    Queue.wait();

    // Multiple streams in command group
    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out1(1024, 80, CGH);
      cl::sycl::stream Out2(500, 10, CGH);
      CGH.parallel_for<class multiple_streams>(cl::sycl::range<1>(2),
                                               [=](cl::sycl::id<1> i) {
                                                 Out1 << "Hello, World!\n";
                                                 Out2 << "Hello, World!\n";
                                               });
    });
    Queue.wait();

    // The case when stream buffer is full. To check that there is no problem
    // with end of line symbol when printing out the stream buffer.
    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::stream Out(10, 10, CGH);
      CGH.parallel_for<class full_stream_buffer>(
          cl::sycl::range<1>(2),
          [=](cl::sycl::id<1> i) { Out << "aaaaaaaaa\n"; });
    });
    Queue.wait();
  }
  return 0;
}
// CHECK: aaaaaaaaaa
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: truefalseHello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: Hello, World!
// CHECK-NEXT: aaaaaaaaa
