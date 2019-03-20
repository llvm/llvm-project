// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>
#include <cmath>

using namespace cl::sycl;

int main() {
  // acos
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class acosF1>(
            [=]() { AccR[0] = cl::sycl::acos(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 1.047f && r < 1.048f); // ~1.0471975511965979
  }
  // acosh
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class acoshF1>(
            [=]() { AccR[0] = cl::sycl::acosh(cl::sycl::cl_float{2.4}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 1.522f && r < 1.523f); // ~1.5220793674636532
  }
  // acospi
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class acospiF1>(
            [=]() { AccR[0] = cl::sycl::acospi(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.333f && r < 0.334f); // ~0.33333333333333337
  }

  // todo
  // asin
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class asinF1>(
            [=]() { AccR[0] = cl::sycl::asin(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.523f && r < 0.524f); // ~0.5235987755982989
  }
  // asinh
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class asinhF1>(
            [=]() { AccR[0] = cl::sycl::asinh(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.481f && r < 0.482f); // ~0.48121182505960347
  }
  // asinpi
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class asinpiF1>(
            [=]() { AccR[0] = cl::sycl::asinpi(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.166f && r < 0.167f); // ~0.16666666666666669
  }
  // atan
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class atanF1>(
            [=]() { AccR[0] = cl::sycl::atan(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.463f && r < 0.464f); // ~0.4636476090008061
  }
  // atan2
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class atan2F1F1>([=]() {
          AccR[0] =
              cl::sycl::atan2(cl::sycl::cl_float{0.5}, cl::sycl::cl_float{0.5});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.785f && r < 0.786f); // ~0.7853981633974483
  }
  // atanh
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class atanhF1>(
            [=]() { AccR[0] = cl::sycl::atanh(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.549f && r < 0.550f); // ~0.5493061443340549
  }
  // atanpi
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class atanpiF1>(
            [=]() { AccR[0] = cl::sycl::atanpi(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.147f && r < 0.148f); // ~0.14758361765043326
  }

  // atan2pi
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class atan2piF1F1>([=]() {
          AccR[0] = cl::sycl::atan2pi(cl::sycl::cl_float{0.5},
                                      cl::sycl::cl_float{0.5});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.249f && r < 0.251f); // ~0.25
  }
  // cbrt
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class cbrtF1>(
            [=]() { AccR[0] = cl::sycl::cbrt(cl::sycl::cl_float{27.0}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 3.f);
  }
  // ceil
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class ceilF1>(
            [=]() { AccR[0] = cl::sycl::ceil(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 1.f);
  }
  // copysign
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class copysignF1F1>([=]() {
          AccR[0] = cl::sycl::copysign(cl::sycl::cl_float{1},
                                       cl::sycl::cl_float{-0.5});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == -1.f);
  }
  // cos
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class cosF1>(
            [=]() { AccR[0] = cl::sycl::cos(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.877f && r < 0.878f); // ~0.8775825618903728
  }
  // cosh
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class coshF1>(
            [=]() { AccR[0] = cl::sycl::cosh(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 1.127f && r < 1.128f); // ~1.1276259652063807
  }
  // cospi
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class cospiF1>(
            [=]() { AccR[0] = cl::sycl::cospi(cl::sycl::cl_float{0.1}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.951f && r < 0.952f); // ~0.9510565162951535
  }
  // erfc
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class erfcF1>(
            [=]() { AccR[0] = cl::sycl::erfc(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.479f && r < 0.480f); // ~0.4795001221869535
  }
  // erf
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class erfF1>(
            [=]() { AccR[0] = cl::sycl::erf(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.520f && r < 0.521f); // ~0.5204998778130465
  }
  // exp
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class expF1>(
            [=]() { AccR[0] = cl::sycl::exp(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 1.648f && r < 1.649f); // ~1.6487212707001282
  }
  // exp2
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class exp2F1>(
            [=]() { AccR[0] = cl::sycl::exp2(cl::sycl::cl_float{8.0}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 256.0f);
  }

  // exp10
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class exp10F1>(
            [=]() { AccR[0] = cl::sycl::exp10(cl::sycl::cl_float{2}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 100.0f);
  }
  // expm1
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class expm1F1>(
            [=]() { AccR[0] = cl::sycl::expm1(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 0.648f && r < 0.649f); // ~0.6487212707001282
  }
  // fabs
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fabsF1>(
            [=]() { AccR[0] = cl::sycl::fabs(cl::sycl::cl_float{-0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 0.5f);
  }
  // fdim
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fdimF1F1>([=]() {
          AccR[0] =
              cl::sycl::fdim(cl::sycl::cl_float{1.6}, cl::sycl::cl_float{0.6});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 1.0f);
  }
  // floor
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class floorF1>(
            [=]() { AccR[0] = cl::sycl::floor(cl::sycl::cl_float{0.5}); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 0.f);
  }
  // fma
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fmaF1F1F1>([=]() {
          AccR[0] =
              cl::sycl::fma(cl::sycl::cl_float{0.5}, cl::sycl::cl_float{10.0},
                            cl::sycl::cl_float{3.0});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 8.0f);
  }
  // fmax
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fmaxF1F1>([=]() {
          AccR[0] =
              cl::sycl::fmax(cl::sycl::cl_float{0.5}, cl::sycl::cl_float{0.8});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 0.8f);
  }
  // fmin
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fminF1F1>([=]() {
          AccR[0] =
              cl::sycl::fmin(cl::sycl::cl_float{0.5}, cl::sycl::cl_float{0.8});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 0.5f);
  }
  // fmod
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fmodF1F1>([=]() {
          AccR[0] =
              cl::sycl::fmod(cl::sycl::cl_float{5.1}, cl::sycl::cl_float{3.0});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 2.1f);
  }

  // fract
  /*{
    cl::sycl::cl_float r{0};
    cl::sycl::cl_float i{999};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      buffer<cl::sycl::cl_float, 1> BufI(&i, range<1>(1),
                                         {property::buffer::use_host_ptr()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class fractF1GF1>([=]() {
          global_ptr<cl::sycl::cl_float> Iptr(AccI);
          AccR[0] = cl::sycl::fract(cl::sycl::cl_float{1.5}, Iptr);
        });
      });
    }
    std::cout << "r " << r << " i " << i << std::endl;
    assert(r == 0.5f);
    assert(i == 1.0f);
  }*/

  // nan
  {
    cl::sycl::cl_double r{0};
    {
      buffer<cl::sycl::cl_double, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class nanIS1>([=]() { AccR[0] = cl::sycl::nan(1LLU); });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(std::isnan(r));
  }

  // native exp
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class nexpF1>([=]() {
          AccR[0] = cl::sycl::native::exp(cl::sycl::cl_float{1.0f});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r > 2.718f && r < 2.719f); // ~2.718281828459045
  }

  return 0;
}
