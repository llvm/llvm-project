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
  // max
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxSI1SI1>([=]() {
          AccR[0] = cl::sycl::max(cl::sycl::cl_int{5}, cl::sycl::cl_int{2});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 5);
  }
  // max
  {
    cl::sycl::cl_uint r{0};
    {
      buffer<cl::sycl::cl_uint, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxUI1UI1>([=]() {
          AccR[0] = cl::sycl::max(cl::sycl::cl_uint{5}, cl::sycl::cl_uint{2});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 5);
  }
  // min
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class minSI1SI1>([=]() {
          AccR[0] = cl::sycl::min(cl::sycl::cl_int{5}, cl::sycl::cl_int{2});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 2);
  }
  // min
  {
    cl::sycl::cl_uint r{0};
    {
      buffer<cl::sycl::cl_uint, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class minUI1UI1>([=]() {
          AccR[0] = cl::sycl::min(cl::sycl::cl_uint{5}, cl::sycl::cl_uint{2});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 2);
  }

  return 0;
}