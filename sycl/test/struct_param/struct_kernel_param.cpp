// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==-struct_kernel_param.cpp-Checks passing structs as kernel params--------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: *
#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>

using namespace cl::sycl;

struct MyNestedStruct {
  cl::sycl::cl_char FldArr[1];
  cl::sycl::cl_float FldFloat;
};

struct MyStruct {
  cl::sycl::cl_char FldChar;
  cl::sycl::cl_long FldLong;
  cl::sycl::cl_short FldShort;
  cl::sycl::cl_uint FldUint;
  MyNestedStruct FldStruct;
  cl::sycl::cl_short FldArr[3];
  cl::sycl::cl_int FldInt;
};

MyStruct GlobS;

static void printStruct(const MyStruct &S0) {
  std::cout << "{ " << (int)S0.FldChar << ", " << S0.FldLong << ", "
            << S0.FldShort << ", " << S0.FldUint << " { { "
            << (int)S0.FldStruct.FldArr[0] << " }, " << S0.FldStruct.FldFloat
            << " }, { " << S0.FldArr[0] << ", " << S0.FldArr[1] << ", "
            << S0.FldArr[2] << " }, " << S0.FldInt << " }";
}

bool test0() {
  MyStruct S = GlobS;
  MyStruct S0 = { 0 };
  {
    buffer<MyStruct, 1> Buf(&S0, range<1>(1));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class MyKernel>([=] { B[0] = S; });
    });
  }
  bool Passed = (std::memcmp(&S0, &S, sizeof(MyStruct)) == 0);

  if (!Passed) {
    std::cout << "test0 failed" << std::endl;
    std::cout << "test0 input:" << std::endl;
    printStruct(S);
    std::cout << std::endl;
    std::cout << "test0 result:\n";
    printStruct(S0);
    std::cout << std::endl;
  }
  return Passed;
}

bool test1() {
  range<3> ice(8, 9, 10);
  uint ice2 = 888;
  uint result[4] = { 0 };

  {
    buffer<unsigned int, 1> Buffer((unsigned int *)result, range<1>(4));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buffer.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class bufferByRange_cap>(range<1>{ 4 },
                                                [=](id<1> index) {
        B[index.get(0)] = index.get(0) > 2 ? ice2 : ice.get(index.get(0));
      });
    });
  }

  bool Passed = true;

  for (unsigned long i = 0; i < 4; ++i) {
    if (i <= 2) {
      if (result[i] != ice[i])
        Passed = false;
    } else {
      if (result[i] != ice2)
        Passed = false;
    }
  }
  if (!Passed)
    std::cout << "test1 failed" << std::endl;

  return Passed;
}

int main(int argc, char **argv) {
  cl::sycl::cl_char PartChar = argc;
  cl::sycl::cl_short PartShort = argc << 8;
  cl::sycl::cl_int PartInt = argc << 16;
  cl::sycl::cl_uint PartUint = argc << 16;
  cl::sycl::cl_long PartLong = ((cl::sycl::cl_long)argc) << 32;
  cl::sycl::cl_float PartFloat = argc;

  GlobS = { PartChar,
            PartLong,
            PartShort,
            PartUint,
            { { PartChar }, PartFloat },
            { PartShort, PartShort, PartShort },
            PartInt };

  bool Pass = test0() & test1();

  std::cout << "Test " << (Pass ? "passed" : "FAILED") << std::endl;
  return Pass ? 0 : 1;
}

