// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  std::cout << "SYCL language version: " << CL_SYCL_LANGUAGE_VERSION
            << std::endl;
  std::cout << "SYCL compiler version: " << __SYCL_COMPILER_VERSION
            << std::endl;
  return 0;
}
