// RUN: %clang -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice -c %s -o %t.o
// RUN: %clang -fsycl -fsycl-link-targets=spir64-unknown-linux-sycldevice %t.o -o %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: %clang -fsycl -fsycl-add-targets=binary:%t.bc %t.o -o %t.out -lOpenCL -lsycl -lstdc++
//
// Only CPU supports LLVM IR bitcode as a binary
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// XFAIL: *

//==----- with-llvm-bc.cpp - SYCL kernel with LLVM IR bitcode as binary ----==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <array>
#include <iostream>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC) {
  cl::sycl::queue deviceQueue;
  cl::sycl::range<1> numOfItems{N};
  cl::sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVadd<T>>(numOfItems,
    [=](cl::sycl::id<1> wiID) {
        accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    });
  });
}

int main() {
  const size_t array_size = 4;
  std::array<cl::sycl::cl_int, array_size> A = {{1, 2, 3, 4}},
                                           B = {{1, 2, 3, 4}}, C;
  std::array<cl::sycl::cl_float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                             E = {{1.f, 2.f, 3.f, 4.f}}, F;
  simple_vadd(A, B, C);
  simple_vadd(D, E, F);
  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << F[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
