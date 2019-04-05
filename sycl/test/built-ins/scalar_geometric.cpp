// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>

namespace s = cl::sycl;

int main() {
  // dot
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class dotF1F1>([=]() {
          AccR[0] = s::dot(s::cl_float{ 0.5 }, s::cl_float{ 1.6 });
        });
      });
    }
    assert(r == 0.8f);
  }

  // distance
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class distanceF1>([=]() {
          AccR[0] = s::distance(s::cl_float{ 1.f }, s::cl_float{ 3.f });
        });
      });
    }
    assert(r == 2.f);
  }

  // length
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class lengthF1>([=]() {
          AccR[0] = s::length(s::cl_float{ 1.f });
        });
      });
    }
    assert(r == 1.f);
  }

  // normalize
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class normalizeF1>([=]() {
          AccR[0] = s::normalize(s::cl_float{ 2.f });
        });
      });
    }
    assert(r == 1.f);
  }

  // fast_distance
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_distanceF1>([=]() {
          AccR[0] = s::fast_distance(s::cl_float{ 1.f }, s::cl_float{ 3.f });
        });
      });
    }
    assert(r == 2.f);
  }

  // fast_length
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_lengthF1>([=]() {
          AccR[0] = s::fast_length(s::cl_float{ 2.f });
        });
      });
    }
    assert(r == 2.f);
  }

  // fast_normalize
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_normalizeF1>([=]() {
          AccR[0] = s::fast_normalize(s::cl_float{ 2.f });
        });
      });
    }

    assert(r == 1.f);
  }

  return 0;
}