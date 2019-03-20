// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>

using namespace cl::sycl;

int main() {
  // dot
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class dotF2F2>([=]() {
          AccR[0] = cl::sycl::dot(
              cl::sycl::cl_float2{
                  1.f,
                  2.f,
              },
              cl::sycl::cl_float2{4.f, 6.f});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 16.f);
  }

  return 0;
}