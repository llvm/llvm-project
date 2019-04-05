// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>
#include <cmath>

namespace s = cl::sycl;

int main() {
  // acos
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class acosF1>([=]() {
          AccR[0] = s::acos(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 1.047f && r < 1.048f); // ~1.0471975511965979
  }

  // acosh
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class acoshF1>([=]() {
          AccR[0] = s::acosh(s::cl_float{ 2.4 });
        });
      });
    }
    assert(r > 1.522f && r < 1.523f); // ~1.5220793674636532
  }

  // acospi
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class acospiF1>([=]() {
          AccR[0] = s::acospi(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.333f && r < 0.334f); // ~0.33333333333333337
  }

  // asin
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class asinF1>([=]() {
          AccR[0] = s::asin(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.523f && r < 0.524f); // ~0.5235987755982989
  }

  // asinh
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class asinhF1>([=]() {
          AccR[0] = s::asinh(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.481f && r < 0.482f); // ~0.48121182505960347
  }

  // asinpi
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class asinpiF1>([=]() {
          AccR[0] = s::asinpi(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.166f && r < 0.167f); // ~0.16666666666666669
  }

  // atan
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atanF1>([=]() {
          AccR[0] = s::atan(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.463f && r < 0.464f); // ~0.4636476090008061
  }

  // atan2
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atan2F1F1>([=]() {
          AccR[0] = s::atan2(s::cl_float{ 0.5 }, s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.785f && r < 0.786f); // ~0.7853981633974483
  }

  // atanh
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atanhF1>([=]() {
          AccR[0] = s::atanh(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.549f && r < 0.550f); // ~0.5493061443340549
  }

  // atanpi
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atanpiF1>([=]() {
          AccR[0] = s::atanpi(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.147f && r < 0.148f); // ~0.14758361765043326
  }

  // atan2pi
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atan2piF1F1>([=]() {
          AccR[0] = s::atan2pi(s::cl_float{ 0.5 }, s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.249f && r < 0.251f); // ~0.25
  }

  // cbrt
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class cbrtF1>([=]() {
          AccR[0] = s::cbrt(s::cl_float{ 27.0 });
        });
      });
    }
    assert(r == 3.f);
  }

  // ceil
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class ceilF1>([=]() {
          AccR[0] = s::ceil(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r == 1.f);
  }

  // copysign
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class copysignF1F1>([=]() {
          AccR[0] = s::copysign(s::cl_float{ 1 }, s::cl_float{ -0.5 });
        });
      });
    }
    assert(r == -1.f);
  }

  // cos
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class cosF1>([=]() {
          AccR[0] = s::cos(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.877f && r < 0.878f); // ~0.8775825618903728
  }

  // cosh
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class coshF1>([=]() {
          AccR[0] = s::cosh(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 1.127f && r < 1.128f); // ~1.1276259652063807
  }

  // cospi
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class cospiF1>([=]() {
          AccR[0] = s::cospi(s::cl_float{ 0.1 });
        });
      });
    }
    assert(r > 0.951f && r < 0.952f); // ~0.9510565162951535
  }

  // erfc
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class erfcF1>([=]() {
          AccR[0] = s::erfc(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.479f && r < 0.480f); // ~0.4795001221869535
  }

  // erf
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class erfF1>([=]() {
          AccR[0] = s::erf(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.520f && r < 0.521f); // ~0.5204998778130465
  }

  // exp
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class expF1>([=]() {
          AccR[0] = s::exp(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 1.648f && r < 1.649f); // ~1.6487212707001282
  }

  // exp2
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class exp2F1>([=]() {
          AccR[0] = s::exp2(s::cl_float{ 8.0 });
        });
      });
    }
    assert(r == 256.0f);
  }

  // exp10
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class exp10F1>([=]() {
          AccR[0] = s::exp10(s::cl_float{ 2 });
        });
      });
    }
    assert(r == 100.0f);
  }

  // expm1
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class expm1F1>([=]() {
          AccR[0] = s::expm1(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r > 0.648f && r < 0.649f); // ~0.6487212707001282
  }

  // fabs
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fabsF1>([=]() {
          AccR[0] = s::fabs(s::cl_float{ -0.5 });
        });
      });
    }
    assert(r == 0.5f);
  }

  // fdim
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fdimF1F1>([=]() {
          AccR[0] = s::fdim(s::cl_float{ 1.6 }, s::cl_float{ 0.6 });
        });
      });
    }
    assert(r == 1.0f);
  }

  // floor
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class floorF1>([=]() {
          AccR[0] = s::floor(s::cl_float{ 0.5 });
        });
      });
    }
    assert(r == 0.f);
  }

  // fma
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fmaF1F1F1>([=]() {
          AccR[0] = s::fma(s::cl_float{ 0.5 }, s::cl_float{ 10.0 },
                           s::cl_float{ 3.0 });
        });
      });
    }
    assert(r == 8.0f);
  }

  // fmax
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fmaxF1F1>([=]() {
          AccR[0] = s::fmax(s::cl_float{ 0.5 }, s::cl_float{ 0.8 });
        });
      });
    }
    assert(r == 0.8f);
  }

  // fmin
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fminF1F1>([=]() {
          AccR[0] = s::fmin(s::cl_float{ 0.5 }, s::cl_float{ 0.8 });
        });
      });
    }
    assert(r == 0.5f);
  }

  // fmod
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fmodF1F1>([=]() {
          AccR[0] = s::fmod(s::cl_float{ 5.1 }, s::cl_float{ 3.0 });
        });
      });
    }
    assert(r == 2.1f);
  }

  // fract with global memory
  {
    s::cl_float r{ 0 };
    s::cl_float i{ 999 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_float, 1> BufI(&i, s::range<1>(1),
                                     { s::property::buffer::use_host_ptr() });
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF1GF1>([=]() {
          s::global_ptr<s::cl_float> Iptr(AccI);
          AccR[0] = s::fract(s::cl_float{ 1.5 }, Iptr);
        });
      });
    }
    assert(r == 0.5f);
    assert(i == 1.0f);
  }

  // fract with private memory
  {
    s::cl_float r{ 0 };
    s::cl_float i{ 999 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_float, 1> BufI(&i, s::range<1>(1),
                                     { s::property::buffer::use_host_ptr() });
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF1PF1>([=]() {
          s::cl_float temp(0.0);
          s::private_ptr<s::cl_float> Iptr(&temp);
          AccR[0] = s::fract(s::cl_float{ 1.5f }, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }
    assert(r == 0.5f);
    assert(i == 1.0f);
  }

  // nan
  {
    s::cl_double r{ 0 };
    {
      s::buffer<s::cl_double, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class nanIS1>([=]() { AccR[0] = s::nan(1LLU); });
      });
    }
    assert(std::isnan(r));
  }

  return 0;
}
