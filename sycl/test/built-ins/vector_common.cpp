// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>

using namespace cl::sycl;

int main() {
  // max
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxF2F2>([=]() {
          AccR[0] = cl::sycl::max(cl::sycl::cl_float2{0.5f, 3.4f},
                                  cl::sycl::cl_float2{2.3f, 0.4f});
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 2.3f);
    assert(r2 == 3.4f);
  }

  // max
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxF2F1>([=]() {
          AccR[0] = cl::sycl::max(cl::sycl::cl_float2{0.5f, 3.4f},
                                  cl::sycl::cl_float{3.0f});
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 3.0f);
    assert(r2 == 3.4f);
  }

  return 0;
}