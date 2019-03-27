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
  // isequal
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isequalF4F4>([=]() {
          AccR[0] =
              cl::sycl::isequal(cl::sycl::cl_float4{0.5f, 0.6f, NAN, INFINITY},
                                cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == -1);
    assert(r2 == 0);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isnotequal
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isnotequalF4F4>([=]() {
          AccR[0] = cl::sycl::isnotequal(
              cl::sycl::cl_float4{0.5f, 0.6f, NAN, INFINITY},
              cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == -1);
    assert(r4 == -1);
  }

  // isgreater
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isgreaterF4F4>([=]() {
          AccR[0] = cl::sycl::isgreater(
              cl::sycl::cl_float4{0.5f, 0.6f, NAN, INFINITY},
              cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isgreaterequal
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isgreaterequalF4F4>([=]() {
          AccR[0] = cl::sycl::isgreaterequal(
              cl::sycl::cl_float4{0.5f, 0.6f, NAN, INFINITY},
              cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isless
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class islessF4F4>([=]() {
          AccR[0] =
              cl::sycl::isless(cl::sycl::cl_float4{0.5f, 0.4f, NAN, INFINITY},
                               cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // islessequal
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class islessequalF4F4>([=]() {
          AccR[0] = cl::sycl::islessequal(
              cl::sycl::cl_float4{0.5f, 0.4f, NAN, INFINITY},
              cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // islessgreater
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class islessgreaterF4F4>([=]() {
          AccR[0] = cl::sycl::islessgreater(
              cl::sycl::cl_float4{0.5f, 0.4f, NAN, INFINITY},
              cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, INFINITY});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0); // Infinity is considered as greater than any
                     // other value except Infinity.
  }

  // isfinite : host only
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isfiniteF4F4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::cl_int4{-1, -1, 0, 0};
#else
          AccR[0] = cl::sycl::isfinite(
              cl::sycl::cl_float4{0.5f, 0.4f, NAN, INFINITY});
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isinf : host only
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isinfF4F4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::cl_int4{0, 0, 0, -1};
#else
          AccR[0] =
              cl::sycl::isinf(cl::sycl::cl_float4{0.5f, 0.4f, NAN, INFINITY});
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isnan : host only
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isnanF4F4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::cl_int4{0, 0, -1, 0};
#else
          AccR[0] =
              cl::sycl::isnan(cl::sycl::cl_float4{0.5f, 0.4f, NAN, INFINITY});
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == -1);
    assert(r4 == 0);
  }

  // isnormal : host only
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isnormalF4F4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::cl_int4{-1, -1, 0, 0};
#else
          AccR[0] = cl::sycl::isnormal(
              cl::sycl::cl_float4{0.5f, 0.4f, NAN, INFINITY});
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isordered
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isorderedF4F4>([=]() {
          AccR[0] = cl::sycl::isordered(
              cl::sycl::cl_float4{0.5f, 0.6f, NAN, INFINITY},
              cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1); // infinity is ordered.
  }

  // isunordered
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isunorderedF4F4>([=]() {
          AccR[0] = cl::sycl::isunordered(
              cl::sycl::cl_float4{0.5f, 0.6f, NAN, INFINITY},
              cl::sycl::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == -1);
    assert(r4 == 0);
  }

  // signbit : host only
  {
    cl::sycl::cl_int4 r{0};
    {
      buffer<cl::sycl::cl_int4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class signbitF4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::cl_int4{0, -1, 0, 0};
#else
          AccR[0] = cl::sycl::signbit(
              cl::sycl::cl_float4{0.5f, -12.0f, NAN, INFINITY});
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    cl::sycl::cl_int r3 = r.z();
    cl::sycl::cl_int r4 = r.w();

    std::cout << "sign r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 "
              << r4 << std::endl;
    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // any : host only.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class anyI4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = 1;
#else
          AccR[0] = cl::sycl::any(cl::sycl::cl_int4{-12, -12, 0, 1});
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r;

    std::cout << "Any r1 " << r1 << std::endl;
    assert(r1 == 1);
  }

  // all : host only.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class allI4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = 1;
#else
          AccR[0] = cl::sycl::all(cl::sycl::cl_int4{-12, -12, -12, -12});
          // Infinity (positive or negative) or Nan are not integers.
          // Passing them creates inconsistent results between host and device
          // execution.
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r;

    std::cout << "All change r1 " << r1 << std::endl;
    assert(r1 == 1);
  }

  // bitselect
  {
    cl::sycl::cl_float4 r{0};
    {
      buffer<cl::sycl::cl_float4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class bitselectF4F4F4>([=]() {
          AccR[0] =
              cl::sycl::bitselect(cl::sycl::cl_float4{112.112, 12.12, 0, 0.0},
                                  cl::sycl::cl_float4{34.34, 23.23, 1, 0.0},
                                  cl::sycl::cl_float4{3.3, 6.6, 1, 0.0});
        }); // Using NAN/INFINITY as any float produced consistent results
            // between host and device.
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    cl::sycl::cl_float r3 = r.z();
    cl::sycl::cl_float r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(abs(r1 - 80.5477f) < 0.0001);
    assert(abs(r2 - 18.2322f) < 0.0001);
    assert(abs(r3 - 1.0f) < 0.01);
    assert(abs(r4 - 0.0f) < 0.01);
  }

  // select : host only
  {
    cl::sycl::cl_float4 r{0};
    {
      buffer<cl::sycl::cl_float4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class selectF4F4I4>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::cl_float4{112.112f, 112.112f, 112.112f, 112.112f};
#else
          AccR[0] = cl::sycl::select(
              cl::sycl::cl_float4{112.112f, 34.34f, 112.112f, 34.34f},
              cl::sycl::cl_float4{34.34f, 112.112f, 34.34f, 112.112f},
              cl::sycl::cl_int4{0, -1, 0, -1});
          // Using NAN/infinity as an input, which gets
          // selected by -1, produces a NAN/infinity as expected.
#endif
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    cl::sycl::cl_float r3 = r.z();
    cl::sycl::cl_float r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 112.112f);
    assert(r2 == 112.112f);
    assert(r3 == 112.112f);
    assert(r4 == 112.112f);
  }

  return 0;
}
