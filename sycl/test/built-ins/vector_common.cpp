// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>

namespace s = cl::sycl;

int main() {
  // max
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxF2F2>([=]() {
          AccR[0] =
              s::max(s::cl_float2{ 0.5f, 3.4f }, s::cl_float2{ 2.3f, 0.4f });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    assert(r1 == 2.3f);
    assert(r2 == 3.4f);
  }

  // max
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxF2F1>([=]() {
          AccR[0] = s::max(s::cl_float2{ 0.5f, 3.4f }, s::cl_float{ 3.0f });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    assert(r1 == 3.0f);
    assert(r2 == 3.4f);
  }

  return 0;
}