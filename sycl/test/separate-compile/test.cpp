// >> ---- compile src1
// >> device compilation...
// RUN: %clang -std=c++11 --sycl -Xclang -fsycl-int-header=sycl_ihdr_a.h %s -c -o a_kernel.bc
// >> host compilation...
// RUN: %clang -std=c++11 -include sycl_ihdr_a.h -g -c %s -o a.o
//
// >> ---- compile src2
// >> device compilation...
// RUN: %clang -DB_CPP=1 -std=c++11 --sycl -Xclang -fsycl-int-header=sycl_ihdr_b.h %s -c -o b_kernel.bc
// >> host compilation...
// RUN: %clang -DB_CPP=1 -std=c++11 -include sycl_ihdr_b.h -g -c %s -o b.o
//
// >> ---- bundle .o with .spv
// >> run bundler
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -inputs=a.o,a_kernel.bc -outputs=a_fat.o
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -inputs=b.o,b_kernel.bc -outputs=b_fat.o
//
// >> ---- unbundle fat objects
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -outputs=a.o,a_kernel.bc -inputs=a_fat.o -unbundle
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -outputs=b.o,b_kernel.bc -inputs=b_fat.o -unbundle
//
// >> ---- link device code
// RUN: llvm-link -o=app.bc a_kernel.bc b_kernel.bc
//
// >> convert linked .bc to spirv
// RUN: llvm-spirv -o=app.spv app.bc
//
// >> ---- wrap device binary
// >> produce .bc
// RUN: clang-offload-wrapper -o wrapper.bc -host=x86_64-pc-linux-gnu -kind=sycl app.spv
//
// >> compile .bc to .o
// RUN: llc -filetype=obj wrapper.bc -o wrapper.o
//
// >> ---- link the full hetero app
// RUN: %clang wrapper.o a.o b.o -o app.exe -lstdc++ -lOpenCL -lsycl
// RUN: ./app.exe | FileCheck %s
// CHECK: pass

//==----------- test.cpp - Tests SYCL separate compilation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef B_CPP
// -----------------------------------------------------------------------------
#include <CL/sycl.hpp>

int run_test_b(int v) {
  int arr[] = {v};
  {
    cl::sycl::queue deviceQueue;
    cl::sycl::buffer<int, 1> buf(arr, 1);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.single_task<class kernel_b>([=]() { acc[0] *= 3; });
    });
  }
  return arr[0];
}

#else // !B_CPP

// -----------------------------------------------------------------------------
#include <CL/sycl.hpp>
#include <iostream>

using namespace std;

const int VAL = 10;

extern int run_test_b(int);

int run_test_a(int v) {
  int arr[] = {v};
  {
    cl::sycl::queue deviceQueue;
    cl::sycl::buffer<int, 1> buf(arr, 1);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.single_task<class kernel_a>([=]() { acc[0] *= 2; });
    });
  }
  return arr[0];
}

int main(int argc, char **argv) {
  bool pass = true;

  int test_a = run_test_a(VAL);
  const int GOLD_A = 2 * VAL;

  if (test_a != GOLD_A) {
    std::cout << "FAILD test_a. Expected: " << GOLD_A << ", got: " << test_a
              << "\n";
    pass = false;
  }

  int test_b = run_test_b(VAL);
  const int GOLD_B = 3 * VAL;

  if (test_b != GOLD_B) {
    std::cout << "FAILD test_b. Expected: " << GOLD_B << ", got: " << test_b
              << "\n";
    pass = false;
  }

  if (pass) {
    std::cout << "pass\n";
  }
  return pass ? 0 : 1;
}
#endif // !B_CPP
