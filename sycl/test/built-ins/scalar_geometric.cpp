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
        cgh.single_task<class dotF1F1>([=]() {
          AccR[0] =
              cl::sycl::dot(cl::sycl::cl_float{0.5}, cl::sycl::cl_float{1.6});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 0.8f);
  }

  // distance
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class distanceF1>([=]() {
          AccR[0] = cl::sycl::distance(cl::sycl::cl_float{1.f},
                                       cl::sycl::cl_float{3.f});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 2.f);
  }

  // length
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class lengthF1>(
            [=]() { AccR[0] = cl::sycl::length(cl::sycl::cl_float{1.f}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 1.f);
  }
  // normalize
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class normalizeF1>(
            [=]() { AccR[0] = cl::sycl::normalize(cl::sycl::cl_float{2.f}); });
      });
    }

    std::cout << "r " << r << std::endl;
    assert(r == 1.f);
  }

  // fast_distance
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fast_distanceF1>([=]() {
          AccR[0] = cl::sycl::fast_distance(cl::sycl::cl_float{1.f},
                                            cl::sycl::cl_float{3.f});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 2.f);
  }
  // fast_length
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fast_lengthF1>([=]() {
          AccR[0] = cl::sycl::fast_length(cl::sycl::cl_float{2.f});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 2.f);
  }
  // fast_normalize
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fast_normalizeF1>([=]() {
          AccR[0] = cl::sycl::fast_normalize(cl::sycl::cl_float{2.f});
        });
      });
    }

    std::cout << "r " << r << std::endl;
    assert(r == 1.f);
  }

  return 0;
}