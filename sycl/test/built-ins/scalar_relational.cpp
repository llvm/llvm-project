// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>
#include <cmath> // for NAN

using namespace cl::sycl;

int main() {
  // isequal-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isequalF1F1>([=]() {
          AccR[0] = cl::sycl::isequal(cl::sycl::cl_float{10.5f},
                                      cl::sycl::cl_float{10.5f});
        });
      });
    }
    std::cout << "isequal r \t" << r << std::endl;
    assert(r == 1);
  }

  // isnotequal-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isnotequalF1F1>([=]() {
          AccR[0] = cl::sycl::isnotequal(cl::sycl::cl_float{0.4f},
                                         cl::sycl::cl_float{0.5f});
        });
      });
    }
    std::cout << "isnotequal r \t" << r << std::endl;
    assert(r == 1);
  }

  // isgreater-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isgreaterF1F1>([=]() {
          AccR[0] = cl::sycl::isgreater(cl::sycl::cl_float{0.6f},
                                        cl::sycl::cl_float{0.5f});
        });
      });
    }
    std::cout << "isgreater r \t" << r << std::endl;
    assert(r == 1);
  }

  // isgreaterequal-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isgreaterequalF1F1>([=]() {
          AccR[0] = cl::sycl::isgreaterequal(cl::sycl::cl_float{0.5f},
                                             cl::sycl::cl_float{0.5f});
        });
      });
    }
    std::cout << "isgreaterequal r \t" << r << std::endl;
    assert(r == 1);
  }

  // isless-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class islessF1F1>([=]() {
          AccR[0] = cl::sycl::isless(cl::sycl::cl_float{0.4f},
                                     cl::sycl::cl_float{0.5f});
        });
      });
    }
    std::cout << "isless r \t" << r << std::endl;
    assert(r == 1);
  }

  // islessequal-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class islessequalF1F1>([=]() {
          AccR[0] = cl::sycl::islessequal(cl::sycl::cl_float{0.5f},
                                          cl::sycl::cl_float{0.5f});
        });
      });
    }
    std::cout << "islessequal r \t" << r << std::endl;
    assert(r == 1);
  }

  // islessgreater-float
  {
    cl::sycl::cl_int r{1};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class islessgreaterF1F1>([=]() {
          AccR[0] = cl::sycl::islessgreater(cl::sycl::cl_float{0.5f},
                                            cl::sycl::cl_float{0.5f});
        });
      });
    }
    std::cout << "islessgreater r \t" << r << std::endl;
    assert(r == 0);
  }

  // isfinite-float
  {
    cl::sycl::cl_int r{1};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isfiniteF1>(
            [=]() { AccR[0] = cl::sycl::isfinite(cl::sycl::cl_float{NAN}); });
      });
    }
    std::cout << "isfinite r \t" << r << std::endl;
    assert(r == 0);
  }

  // isinf-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isinfF1>(
            [=]() { AccR[0] = cl::sycl::isinf(cl::sycl::cl_float{INFINITY}); });
      });
    }
    std::cout << "isinf r \t" << r << std::endl;
    assert(r == 1);
  }

  // isnan-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isnanF1>(
            [=]() { AccR[0] = cl::sycl::isnan(cl::sycl::cl_float{NAN}); });
      });
    }
    std::cout << "isnan r \t" << r << std::endl;
    assert(r == 1);
  }

  // isnormal-float
  {
    cl::sycl::cl_int r{1};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isnormalF1>([=]() {
          AccR[0] = cl::sycl::isnormal(cl::sycl::cl_float{INFINITY});
        });
      });
    }
    std::cout << "isnormal r \t" << r << std::endl;
    assert(r == 0);
  }

  // isnormal-double
  {
    cl::sycl::cl_int r{1};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isnormalD1>([=]() {
          AccR[0] = cl::sycl::isnormal(cl::sycl::cl_double{INFINITY});
        });
      });
    }
    std::cout << "isnormal r \t" << r << std::endl;
    assert(r == 0);
  }

  // isordered-float
  {
    cl::sycl::cl_int r{1};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isorderedF1F1>([=]() {
          AccR[0] = cl::sycl::isordered(cl::sycl::cl_float{4.0f},
                                        cl::sycl::cl_float{NAN});
        });
      });
    }
    std::cout << "isordered r \t" << r << std::endl;
    assert(r == 0);
  }

  // isunordered-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isunorderedF1F1>([=]() {
          AccR[0] = cl::sycl::isunordered(cl::sycl::cl_float{4.0f},
                                          cl::sycl::cl_float{NAN});
        });
      });
    }
    std::cout << "isunordered r \t" << r << std::endl;
    assert(r == 1);
  }

  // signbit-float
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class signbitF1>(
            [=]() { AccR[0] = cl::sycl::signbit(cl::sycl::cl_float{-12.0f}); });
      });
    }
    std::cout << "signbit r \t" << r << std::endl;
    assert(r == 1);
  }

  // any-integer
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class anyF1positive>(
            [=]() { AccR[0] = cl::sycl::any(cl::sycl::cl_int{12}); });
      });
    }
    std::cout << "any + r \t" << r << std::endl;
    assert(r == 0);
  }
  // any-integer
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class anyF1zero>(
            [=]() { AccR[0] = cl::sycl::any(cl::sycl::cl_int{0}); });
      });
    }
    std::cout << "any 0 r \t" << r << std::endl;
    assert(r == 0);
  }

  // any-integer
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class anyF1negative>(
            [=]() { AccR[0] = cl::sycl::any(cl::sycl::cl_int{-12}); });
      });
    }
    std::cout << "any - r \t" << r << std::endl;
    assert(r == 1);
  }

  // all-integer
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class allF1positive>(
            [=]() { AccR[0] = cl::sycl::all(cl::sycl::cl_int{12}); });
      });
    }
    std::cout << "all + r \t" << r << std::endl;
    assert(r == 0);
  }

  // all-integer
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class allF1zero>(
            [=]() { AccR[0] = cl::sycl::all(cl::sycl::cl_int{0}); });
      });
    }
    std::cout << "all 0 r \t" << r << std::endl;
    assert(r == 0);
  }

  // all-integer
  {
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class allF1negative>(
            [=]() { AccR[0] = cl::sycl::all(cl::sycl::cl_int{-12}); });
      });
    }
    std::cout << "all - r \t" << r << std::endl;
    assert(r == 1);
  }

  // bitselect-float
  {
    cl::sycl::cl_float r{0.0f};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class bitselectF1F1F1>([=]() {
          AccR[0] = cl::sycl::bitselect(cl::sycl::cl_float{112.112},
                                        cl::sycl::cl_float{34.34},
                                        cl::sycl::cl_float{3.3});
        });
      });
    }
    std::cout << "bitselect r \t" << r << std::endl;
    assert(r <= 80.5478 && r >= 80.5476); // r = 80.5477
  }

  // select-float,int
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1positive>([=]() {
          AccR[0] = cl::sycl::select(cl::sycl::cl_float{34.34},
                                     cl::sycl::cl_float{123.123},
                                     cl::sycl::cl_int{1});
        });
      });
    }
    std::cout << "select + r \t" << r << std::endl;
    assert(r <= 123.124 && r >= 123.122); // r = 123.123
  }

  // select-float,int
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1zero>([=]() {
          AccR[0] = cl::sycl::select(cl::sycl::cl_float{34.34},
                                     cl::sycl::cl_float{123.123},
                                     cl::sycl::cl_int{0});
        });
      });
    }
    std::cout << "select 0 r \t" << r << std::endl;
    assert(r <= 34.35 && r >= 34.33); // r = 34.34
  }

  // select-float,int
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1negative>([=]() {
          AccR[0] = cl::sycl::select(cl::sycl::cl_float{34.34},
                                     cl::sycl::cl_float{123.123},
                                     cl::sycl::cl_int{-1});
        });
      });
    }
    std::cout << "select - r \t" << r << std::endl;
    assert(r <= 123.124 && r >= 123.122); // r = 123.123
  }

  return 0;
}
