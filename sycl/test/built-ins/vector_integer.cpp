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
    cl::sycl::cl_int2 r{0};
    {
      buffer<cl::sycl::cl_int2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxSI2SI2>([=]() {
          AccR[0] =
              cl::sycl::max(cl::sycl::cl_int2{5, 3}, cl::sycl::cl_int2{2, 7});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 5);
    assert(r2 == 7);
  }

  // max
  {
    cl::sycl::cl_uint2 r{0};
    {
      buffer<cl::sycl::cl_uint2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxUI2UI2>([=]() {
          AccR[0] =
              cl::sycl::max(cl::sycl::cl_uint2{5, 3}, cl::sycl::cl_uint2{2, 7});
        });
      });
    }
    cl::sycl::cl_uint r1 = r.x();
    cl::sycl::cl_uint r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 5);
    assert(r2 == 7);
  }

  // max
  {
    cl::sycl::cl_int2 r{0};
    {
      buffer<cl::sycl::cl_int2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxSI2SI1>([=]() {
          AccR[0] = cl::sycl::max(cl::sycl::cl_int2{5, 3}, cl::sycl::cl_int{2});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // max
  {
    cl::sycl::cl_uint2 r{0};
    {
      buffer<cl::sycl::cl_uint2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class maxUI2UI1>([=]() {
          AccR[0] =
              cl::sycl::max(cl::sycl::cl_uint2{5, 3}, cl::sycl::cl_uint{2});
        });
      });
    }
    cl::sycl::cl_uint r1 = r.x();
    cl::sycl::cl_uint r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // min
  {
    cl::sycl::cl_int2 r{0};
    {
      buffer<cl::sycl::cl_int2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class minSI2SI2>([=]() {
          AccR[0] =
              cl::sycl::min(cl::sycl::cl_int2{5, 3}, cl::sycl::cl_int2{2, 7});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // min
  {
    cl::sycl::cl_uint2 r{0};
    {
      buffer<cl::sycl::cl_uint2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class minUI2UI2>([=]() {
          AccR[0] =
              cl::sycl::min(cl::sycl::cl_uint2{5, 3}, cl::sycl::cl_uint2{2, 7});
        });
      });
    }
    cl::sycl::cl_uint r1 = r.x();
    cl::sycl::cl_uint r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // min
  {
    cl::sycl::cl_int2 r{0};
    {
      buffer<cl::sycl::cl_int2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class minSI2SI1>([=]() {
          AccR[0] = cl::sycl::min(cl::sycl::cl_int2{5, 3}, cl::sycl::cl_int{2});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 2);
    assert(r2 == 2);
  }

  // min
  {
    cl::sycl::cl_uint2 r{0};
    {
      buffer<cl::sycl::cl_uint2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class minUI2UI1>([=]() {
          AccR[0] =
              cl::sycl::min(cl::sycl::cl_uint2{5, 3}, cl::sycl::cl_uint{2});
        });
      });
    }
    cl::sycl::cl_uint r1 = r.x();
    cl::sycl::cl_uint r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 2);
    assert(r2 == 2);
  }

  return 0;
}