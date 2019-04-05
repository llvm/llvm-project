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
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxF1F1>([=]() {
          AccR[0] = s::max(s::cl_float{ 0.5f }, s::cl_float{ 2.3f });
        });
      });
    }
    assert(r == 2.3f);
  }

  return 0;
}