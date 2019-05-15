// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>

namespace s = cl::sycl;

int main() {
  // fmin
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fminF2F2>([=]() {
          AccR[0] =
              s::fmin(s::cl_float2{ 0.5f, 3.4f }, s::cl_float2{ 2.3f, 0.4f });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    assert(r1 == 0.5f);
    assert(r2 == 0.4f);
  }

  // fabs
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fabsF2>([=]() {
          AccR[0] = s::fabs(s::cl_float2{ -1.0f, 2.0f });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // floor
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class floorF2>([=]() {
          AccR[0] = s::floor(s::cl_float2{ 1.4f, 2.8f });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // ceil
  {
    s::cl_float2 r{ 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class ceilF2>([=]() {
          AccR[0] = s::ceil(s::cl_float2{ 1.4f, 2.8f });
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // fract with global memory
  {
    s::cl_float2 r{ 0, 0 };
    s::cl_float2 i{ 0, 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_float2, 1> BufI(&i, s::range<1>(1));

      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF2GF2>([=]() {
          s::global_ptr<s::cl_float2> Iptr(AccI);
          AccR[0] = s::fract(s::cl_float2{ 1.5f, 2.5f }, Iptr);
        });
      });
    }

    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    s::cl_float i1 = i.x();
    s::cl_float i2 = i.y();

    assert(r1 == 0.5f);
    assert(r2 == 0.5f);
    assert(i1 == 1.0f);
    assert(i2 == 2.0f);
  }

  // fract with private memory
  {
    s::cl_float2 r{ 0, 0 };
    s::cl_float2 i{ 0, 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_float2, 1> BufI(&i, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF2PF2>([=]() {
          s::cl_float2 temp(0.0);
          s::private_ptr<s::cl_float2> Iptr(&temp);
          AccR[0] = s::fract(s::cl_float2{ 1.5f, 2.5f }, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }

    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    s::cl_float i1 = i.x();
    s::cl_float i2 = i.y();

    assert(r1 == 0.5f);
    assert(r2 == 0.5f);
    assert(i1 == 1.0f);
    assert(i2 == 2.0f);
  }

  // lgamma with private memory
  {
    s::cl_float2 r{ 0, 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgamma_rF2>([=]() {
          AccR[0] = s::lgamma(s::cl_float2{ 10.f, -2.4f });
        });
      });
    }

    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();

    assert(r1 > 12.8017f && r1 < 12.8019f); // ~12.8018
    assert(r2 > 0.1024f && r2 < 0.1026f);   // ~0.102583
  }

  // lgamma_r with private memory
  {
    s::cl_float2 r{ 0, 0 };
    s::cl_int2 i{ 0, 0 };
    {
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_int2, 1> BufI(&i, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgamma_rF2PF2>([=]() {
          s::cl_int2 temp(0.0);
          s::private_ptr<s::cl_int2> Iptr(&temp);
          AccR[0] = s::lgamma_r(s::cl_float2{ 10.f, -2.4f }, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }

    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    s::cl_int i1 = i.x();
    s::cl_int i2 = i.y();

    assert(r1 > 12.8017f && r1 < 12.8019f); // ~12.8018
    assert(r2 > 0.1024f && r2 < 0.1026f);   // ~0.102583
    assert(i1 == 1);                        // tgamma of 10 is ~362880.0
    assert(i2 == -1); // tgamma of -2.4 is ~-1.1080299470333461
  }

  return 0;
}
