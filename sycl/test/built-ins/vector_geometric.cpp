// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>
#include <cmath>

namespace s = cl::sycl;

bool isFloatEqualTo(float x, float y, float epsilon = 0.005f) {
  return std::fabs(x - y) <= epsilon;
}

int main() {
  // dot
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class dotF2F2>([=]() {
          AccR[0] = s::dot(s::cl_float2{ 1.f, 2.f, }, s::cl_float2{ 4.f, 6.f });
        });
      });
    }
    assert(r == 16.f);
  }

  // cross
  {
    s::cl_float4 r{ 0 };
    {
      s::buffer<s::cl_float4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class crossF4>([=]() {
          AccR[0] = s::cross(s::cl_float4{ 2.f, 3.f, 4.f, 0.f, },
                             s::cl_float4{ 5.f, 6.f, 7.f, 0.f, });
        });
      });
    }

    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    s::cl_float r3 = r.z();
    s::cl_float r4 = r.w();

    assert(r1 == -3.f);
    assert(r2 == 6.f);
    assert(r3 == -3.f);
    assert(r4 == 0.0f);
  }

  // distance
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class distanceF2>([=]() {
          AccR[0] =
              s::distance(s::cl_float2{ 1.f, 2.f, }, s::cl_float2{ 3.f, 4.f, });
        });
      });
    }
    assert(isFloatEqualTo(r, 2.82843f));
  }

  // length
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class lengthF2>([=]() {
          AccR[0] = s::length(s::cl_float2{ 1.f, 2.f, });
        });
      });
    }
    assert(isFloatEqualTo(r, 2.23607f));
  }

  // normalize
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class normalizeF2>([=]() {
          AccR[0] = s::normalize(s::cl_float2{ 1.f, 2.f, });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();

    assert(isFloatEqualTo(r1, 0.447214f));
    assert(isFloatEqualTo(r2, 0.894427f));
  }

  // fast_distance
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_distanceF2>([=]() {
          AccR[0] = s::fast_distance(s::cl_float2{ 1.f, 2.f, },
                                     s::cl_float2{ 3.f, 4.f, });
        });
      });
    }
    assert(isFloatEqualTo(r, 2.82843f));
  }

  // fast_length
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_lengthF2>([=]() {
          AccR[0] = s::fast_length(s::cl_float2{ 1.f, 2.f, });
        });
      });
    }
    assert(isFloatEqualTo(r, 2.23607f));
  }

  // fast_normalize
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_normalizeF2>([=]() {
          AccR[0] = s::fast_normalize(s::cl_float2{ 1.f, 2.f, });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();

    assert(isFloatEqualTo(r1, 0.447144));
    assert(isFloatEqualTo(r2, 0.894287));
  }

  return 0;
}