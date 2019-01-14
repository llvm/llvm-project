// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==-- kernel_name_class.cpp - SYCL kernel naming variants test ------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

#define GOLD 10

namespace nm1 {
namespace nm2 {
class C {};
class KernelName0 : public C {};
} // namespace nm2

class KernelName1;

template <typename T> class KernelName3;
template <typename T> class KernelName4;

template <> class KernelName3<nm1::nm2::KernelName0>;
template <> class KernelName3<KernelName1>;

template <> class KernelName4<nm1::nm2::KernelName0> {};
template <> class KernelName4<KernelName1> {};

} // namespace nm1

static int NumTestCases = 0;

namespace nm3 {
struct Wrapper {

  class KN100 {};
  class KN101;

  int test() {
    int arr[] = {0};
    {
      cl::sycl::queue deviceQueue;
      cl::sycl::buffer<int, 1> buf(arr, 1);
      // Acronyms used to designate a test combination:
      //   Declaration levels: 'T'-translation unit, 'L'-local scope,
      //                       'C'-containing class, 'P'-"in place", '-'-N/A
      //   Class definition:   'I'-incomplete (not defined), 'D' - defined,
      //   '-'-N/A
      // Test combination positional parameters:
      // 0: Kernel class declaration level
      // 1: Kernel class definition
      // 2: Declaration level of the template argument class of the kernel class
      // 3: Definition of the template argument class of the kernel class

      // PI--
      // traditional in-place incomplete type
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<class KernelName>([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

      // TD--
      // a class completely defined within a namespace at
      // translation unit scope
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::nm2::KernelName0>([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

#ifdef LI__
      // TODO unexpected compilation error when host code + integration header
      // is compiled LI-- kernel name is an incomplete class forward-declared in
      // local scope
      class KernelName2;
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<KernelName2>([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;
#endif

#ifdef LD__
      // LD--
      // kernel name is a class defined in local scope
      class KernelName2a {};
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<KernelName2a>([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;
#endif

      // TI--
      // an incomplete class forward-declared in a namespace at
      // translation unit scope
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName1>([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

      // TITD
      // an incomplete template specialization class with defined class as
      // argument declared in a namespace at translation unit scope
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName3<nm1::nm2::KernelName0>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

      // TITI
      // an incomplete template specialization class with incomplete class as
      // argument forward-declared in a namespace at translation unit scope
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName3<nm1::KernelName1>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

      // TDTD
      // a defined template specialization class with defined class as argument
      // declared in a namespace at translation unit scope
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName4<nm1::nm2::KernelName0>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

      // TDTI
      // a defined template specialization class with incomplete class as
      // argument forward-declared in a namespace at translation unit scope
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName4<nm1::KernelName1>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

      // TIPI
      // an incomplete template specialization class with incomplete class as
      // argument forward-declared "in-place"
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName3<class KernelName5>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

#ifdef TILI
      // Expected compilation error
      // TILI
      // an incomplete template specialization class with incomplete class as
      // argument forward-declared locally
      class KernelName6;
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName3<KernelName6>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;
#endif

      // TDPI
      // a defined template specialization class with incomplete class as
      // argument forward-declared "in-place"
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName4<class KernelName7>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

#ifdef TDLI
      // TODO unexpected compilation error when host code + integration header
      // is compiled TDLI a defined template specialization class with
      // incomplete class as argument forward-declared locally
      class KernelName6a;
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName4<KernelName6a>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;
#endif

#ifdef TDLD
      // Expected compilation error
      // TDLD
      // a defined template specialization class with a class as argument
      // defined locally
      class KernelName9 {};
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName4<KernelName9>>(
            [=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;
#endif

#ifdef TICD
      // Expected compilation error
      // TICD
      // an incomplete template specialization class with a defined class as
      // argument declared in the containing class
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName3<KN100>>([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;
#endif

#ifdef TICI
      // Expected compilation error
      // TICI
      // an incomplete template specialization class with an incomplete class as
      // argument declared in the containing class
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<nm1::KernelName3<KN101>>([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;
#endif
    }
    return arr[0];
  }
};
} // namespace nm3

int main() {
  nm3::Wrapper w;
  int res = w.test();
  bool pass = res == GOLD * NumTestCases;
  std::cout << (pass ? "pass" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

