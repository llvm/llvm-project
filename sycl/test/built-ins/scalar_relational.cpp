// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>
#include <cmath>

namespace s = cl::sycl;

int main() {
  // isequal-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isequalF1F1>([=]() {
          AccR[0] = s::isequal(s::cl_float{ 10.5f }, s::cl_float{ 10.5f });
        });
      });
    }
    assert(r == 1);
  }

  // isnotequal-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnotequalF1F1>([=]() {
          AccR[0] = s::isnotequal(s::cl_float{ 0.4f }, s::cl_float{ 0.5f });
        });
      });
    }
    assert(r == 1);
  }

  // isgreater-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterF1F1>([=]() {
          AccR[0] = s::isgreater(s::cl_float{ 0.6f }, s::cl_float{ 0.5f });
        });
      });
    }
    assert(r == 1);
  }

  // isgreaterequal-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterequalF1F1>([=]() {
          AccR[0] = s::isgreaterequal(s::cl_float{ 0.5f }, s::cl_float{ 0.5f });
        });
      });
    }
    assert(r == 1);
  }

  // isless-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessF1F1>([=]() {
          AccR[0] = s::isless(s::cl_float{ 0.4f }, s::cl_float{ 0.5f });
        });
      });
    }
    assert(r == 1);
  }

  // islessequal-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessequalF1F1>([=]() {
          AccR[0] = s::islessequal(s::cl_float{ 0.5f }, s::cl_float{ 0.5f });
        });
      });
    }
    assert(r == 1);
  }

  // islessgreater-float
  {
    s::cl_int r{ 1 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessgreaterF1F1>([=]() {
          AccR[0] = s::islessgreater(s::cl_float{ 0.5f }, s::cl_float{ 0.5f });
        });
      });
    }
    assert(r == 0);
  }

  // isfinite-float
  {
    s::cl_int r{ 1 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isfiniteF1>([=]() {
          AccR[0] = s::isfinite(s::cl_float{ NAN });
        });
      });
    }
    assert(r == 0);
  }

  // isinf-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isinfF1>([=]() {
          AccR[0] = s::isinf(s::cl_float{ INFINITY });
        });
      });
    }
    assert(r == 1);
  }

  // isnan-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnanF1>([=]() {
          AccR[0] = s::isnan(s::cl_float{ NAN });
        });
      });
    }
    assert(r == 1);
  }

  // isnormal-float
  {
    s::cl_int r{ 1 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnormalF1>([=]() {
          AccR[0] = s::isnormal(s::cl_float{ INFINITY });
        });
      });
    }
    assert(r == 0);
  }

  // isnormal-double
  {
    s::cl_int r{ 1 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnormalD1>([=]() {
          AccR[0] = s::isnormal(s::cl_double{ INFINITY });
        });
      });
    }
    assert(r == 0);
  }

  // isordered-float
  {
    s::cl_int r{ 1 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isorderedF1F1>([=]() {
          AccR[0] = s::isordered(s::cl_float{ 4.0f }, s::cl_float{ NAN });
        });
      });
    }
    assert(r == 0);
  }

  // isunordered-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isunorderedF1F1>([=]() {
          AccR[0] = s::isunordered(s::cl_float{ 4.0f }, s::cl_float{ NAN });
        });
      });
    }
    assert(r == 1);
  }

  // signbit-float
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class signbitF1>([=]() {
          AccR[0] = s::signbit(s::cl_float{ -12.0f });
        });
      });
    }
    assert(r == 1);
  }

  // any-integer
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyF1positive>([=]() {
          AccR[0] = s::any(s::cl_int{ 12 });
        });
      });
    }
    assert(r == 0);
  }
  // any-integer
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyF1zero>([=]() {
          AccR[0] = s::any(s::cl_int{ 0 });
        });
      });
    }
    assert(r == 0);
  }

  // any-integer
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyF1negative>([=]() {
          AccR[0] = s::any(s::cl_int{ -12 });
        });
      });
    }
    assert(r == 1);
  }

  // all-integer
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allF1positive>([=]() {
          AccR[0] = s::all(s::cl_int{ 12 });
        });
      });
    }
    assert(r == 0);
  }

  // all-integer
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allF1zero>([=]() {
          AccR[0] = s::all(s::cl_int{ 0 });
        });
      });
    }
    assert(r == 0);
  }

  // all-integer
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allF1negative>([=]() {
          AccR[0] = s::all(s::cl_int{ -12 });
        });
      });
    }
    assert(r == 1);
  }

  // bitselect-float
  {
    s::cl_float r{ 0.0f };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class bitselectF1F1F1>([=]() {
          AccR[0] = s::bitselect(s::cl_float{ 112.112 }, s::cl_float{ 34.34 },
                                 s::cl_float{ 3.3 });
        });
      });
    }
    assert(r <= 80.5478 && r >= 80.5476); // r = 80.5477
  }

  // select-float,int
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1positive>([=]() {
          AccR[0] = s::select(s::cl_float{ 34.34 }, s::cl_float{ 123.123 },
                              s::cl_int{ 1 });
        });
      });
    }
    assert(r <= 123.124 && r >= 123.122); // r = 123.123
  }

  // select-float,int
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1zero>([=]() {
          AccR[0] = s::select(s::cl_float{ 34.34 }, s::cl_float{ 123.123 },
                              s::cl_int{ 0 });
        });
      });
    }
    assert(r <= 34.35 && r >= 34.33); // r = 34.34
  }

  // select-float,int
  {
    s::cl_float r{ 0 };
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1negative>([=]() {
          AccR[0] = s::select(s::cl_float{ 34.34 }, s::cl_float{ 123.123 },
                              s::cl_int{ -1 });
        });
      });
    }
    assert(r <= 123.124 && r >= 123.122); // r = 123.123
  }

  return 0;
}
