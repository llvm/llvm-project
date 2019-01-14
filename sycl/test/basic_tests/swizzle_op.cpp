// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//==------------ swizzle_op.cpp - SYCL SwizzleOp basic test ----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#define SYCL_SIMPLE_SWIZZLES

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

int main() {
  {
    cl::sycl::cl_float results[3] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(3));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_1>([=]() {
          cl::sycl::cl_float2 ab = {4, 2};
          cl::sycl::cl_float c = ab.x() * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c;
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
  }

  {
    cl::sycl::cl_float results[3] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(3));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_2>([=]() {
          cl::sycl::cl_float2 ab = {4, 2};
          cl::sycl::cl_float c = ab.x() * 2;
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c;
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
  }

  {
    cl::sycl::cl_float results[3] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(3));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_3>([=]() {
          cl::sycl::cl_float2 ab = {4, 2};
          cl::sycl::cl_float c = 4 * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c;
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
  }

  {
    cl::sycl::cl_float results[4] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_4>([=]() {
          cl::sycl::cl_float2 ab = {4, 2};
          cl::sycl::cl_float2 c = {0, 0};
          c.x() = ab.x() * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c.x();
          B[4] = c.y();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
    assert(results[3] == 0);
  }

  {
    cl::sycl::cl_float results[4] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_5>([=]() {
          cl::sycl::cl_float2 ab = {4, 2};
          cl::sycl::cl_float2 c = {0, 0};
          c.x() = 4 * ab.y();
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c.x();
          B[4] = c.y();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
    assert(results[3] == 0);
  }

  {
    cl::sycl::cl_float results[4] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_6>([=]() {
          cl::sycl::cl_float2 ab = {4, 2};
          cl::sycl::cl_float2 c = {0, 0};
          c.x() = ab.x() * 2;
          B[0] = ab.x();
          B[1] = ab.y();
          B[2] = c.x();
          B[4] = c.y();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 8);
    assert(results[3] == 0);
  }

  {
    cl::sycl::cl_float results[6] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(6));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_7>([=]() {
          cl::sycl::uchar4 abc = {4, 2, 1, 0};

          cl::sycl::uchar4 c_each;
          c_each.x() = abc.x();
          c_each.y() = abc.y();
          c_each.z() = abc.z();

          cl::sycl::uchar4 c_full;
          c_full = abc;

          B[0] = c_each.x();
          B[1] = c_each.y();
          B[2] = c_each.z();
          B[3] = c_full.x();
          B[4] = c_full.y();
          B[5] = c_full.z();
        });
      });
    }
    assert(results[0] == 4);
    assert(results[1] == 2);
    assert(results[2] == 1);
    assert(results[3] == 4);
    assert(results[4] == 2);
    assert(results[5] == 1);
  }

  {
    cl::sycl::cl_float results[4] = {0};
    {
      buffer<cl::sycl::cl_float, 1> b(results, range<1>(4));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::write>(cgh);
        cgh.single_task<class test_8>([=]() {
          cl::sycl::uchar4 cba;
          cl::sycl::uchar x = 1;
          cl::sycl::uchar y = 2;
          cl::sycl::uchar z = 3;
          cl::sycl::uchar w = 4;
          cba.x() = x;
          cba.y() = y;
          cba.z() = z;
          cba.w() = w;

          cl::sycl::uchar4 abc = {1, 2, 3, 4};
          abc.x() = cba.s0();
          abc.y() = cba.s1();
          abc.z() = cba.s2();
          abc.w() = cba.s3();
          if ((cba.x() == abc.x())) {
            abc.xy() = abc.xy() * 3;

            B[0] = abc.x();
            B[1] = abc.y();
            B[2] = abc.z();
            B[3] = abc.w();
          }
        });
      });
    }
    assert(results[0] == 3);
    assert(results[1] == 6);
    assert(results[2] == 3);
    assert(results[3] == 4);
  }
}
