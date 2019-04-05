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
  // max
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxSI2SI2>([=]() {
          AccR[0] = s::max(s::cl_int2{5, 3}, s::cl_int2{2, 7});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 7);
  }

  // max
  {
    s::cl_uint2 r{0};
    {
      s::buffer<s::cl_uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxUI2UI2>([=]() {
          AccR[0] = s::max(s::cl_uint2{5, 3}, s::cl_uint2{2, 7});
        });
      });
    }
    s::cl_uint r1 = r.x();
    s::cl_uint r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 7);
  }

  // max
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxSI2SI1>([=]() {
          AccR[0] = s::max(s::cl_int2{5, 3}, s::cl_int{2});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // max
  {
    s::cl_uint2 r{0};
    {
      s::buffer<s::cl_uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxUI2UI1>([=]() {
          AccR[0] = s::max(s::cl_uint2{5, 3}, s::cl_uint{2});
        });
      });
    }
    s::cl_uint r1 = r.x();
    s::cl_uint r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // min
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSI2SI2>([=]() {
          AccR[0] = s::min(s::cl_int2{5, 3}, s::cl_int2{2, 7});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // min
  {
    s::cl_uint2 r{0};
    {
      s::buffer<s::cl_uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minUI2UI2>([=]() {
          AccR[0] = s::min(s::cl_uint2{5, 3}, s::cl_uint2{2, 7});
        });
      });
    }
    s::cl_uint r1 = r.x();
    s::cl_uint r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // min
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSI2SI1>([=]() {
          AccR[0] = s::min(s::cl_int2{5, 3}, s::cl_int{2});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 2);
  }

  // min
  {
    s::cl_uint2 r{0};
    {
      s::buffer<s::cl_uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minUI2UI1>([=]() {
          AccR[0] = s::min(s::cl_uint2{5, 3}, s::cl_uint{2});
        });
      });
    }
    s::cl_uint r1 = r.x();
    s::cl_uint r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 2);
  }

  // abs
  {
    s::cl_uint2 r{0};
    {
      s::buffer<s::cl_uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class absSI2>([=]() {
          AccR[0] = s::abs(s::cl_int2{-5, -2});
        });
      });
    }
    s::cl_uint r1 = r.x();
    s::cl_uint r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 2);
  }

  // abs_diff
  {
    s::cl_uint2 r{0};
    {
      s::buffer<s::cl_uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class abs_diffSI2SI2>([=]() {
          AccR[0] = s::abs_diff(s::cl_int2{-5, -2}, s::cl_int2{-1, -1});
        });
      });
    }
    s::cl_uint r1 = r.x();
    s::cl_uint r2 = r.y();
    assert(r1 == 4);
    assert(r2 == 1);
  }

  // add_sat
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class add_satSI2SI2>([=]() {
          AccR[0] = s::add_sat(s::cl_int2{0x7FFFFFFF, 0x7FFFFFFF},
                               s::cl_int2{100, 90});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x7FFFFFFF);
    assert(r2 == 0x7FFFFFFF);
  }

  // hadd
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class haddSI2SI2>([=]() {
          AccR[0] = s::hadd(s::cl_int2{0x0000007F, 0x0000007F},
                            s::cl_int2{0x00000020, 0x00000020});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x0000004F);
    assert(r2 == 0x0000004F);
  }

  // rhadd
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rhaddSI2SI2>([=]() {
          AccR[0] = s::rhadd(s::cl_int2{0x0000007F, 0x0000007F},
                             s::cl_int2{0x00000020, 0x00000020});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x00000050);
    assert(r2 == 0x00000050);
  }

  // clamp - 1
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clampSI2SI2SI2>([=]() {
          AccR[0] = s::clamp(s::cl_int2{5, 5}, s::cl_int2{10, 10},
                             s::cl_int2{30, 30});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 10);
    assert(r2 == 10);
  }

  // clamp - 2
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clampSI2SI1SI1>([=]() {
          AccR[0] = s::clamp(s::cl_int2{5, 5}, s::cl_int{10}, s::cl_int{30});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 10);
    assert(r2 == 10);
  }

  // clz
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clzSI2>([=]() {
          AccR[0] = s::clz(s::cl_int2{0x0FFFFFFF, 0x0FFFFFFF});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 4);
    assert(r2 == 4);
  }

  // mad_hi
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_hiSI2SI2SI2>([=]() {
          AccR[0] =
              s::mad_hi(s::cl_int2{0x10000000, 0x10000000},
                        s::cl_int2{0x00000100, 0x00000100}, s::cl_int2{1, 1});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x11);
    assert(r2 == 0x11);
  }

  // mad_sat
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_satSI2SI2SI2>([=]() {
          AccR[0] =
              s::mad_sat(s::cl_int2{0x10000000, 0x10000000},
                         s::cl_int2{0x00000100, 0x00000100}, s::cl_int2{1, 1});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x7FFFFFFF);
    assert(r2 == 0x7FFFFFFF);
  }

  // mul_hi
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI2SI2>([=]() {
          AccR[0] = s::mul_hi(s::cl_int2{0x10000000, 0x10000000},
                              s::cl_int2{0x00000100, 0x00000100});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x10);
    assert(r2 == 0x10);
  }

  // rotate
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rotateSI2SI2>([=]() {
          AccR[0] =
              s::rotate(s::cl_int2{0x11100000, 0x11100000}, s::cl_int2{12, 12});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x00000111);
    assert(r2 == 0x00000111);
  }

  // sub_sat
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class sub_satSI2SI2>([=]() {
          AccR[0] = s::sub_sat(s::cl_int2{10, 10},
                               s::cl_int2{int(0x80000000), int(0x80000000)});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x7FFFFFFF);
    assert(r2 == 0x7FFFFFFF);
  }

  // upsample - 1
  {
    s::cl_ushort2 r{0};
    {
      s::buffer<s::cl_ushort2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUC2UC2>([=]() {
          AccR[0] =
              s::upsample(s::cl_uchar2{0x10, 0x10}, s::cl_uchar2{0x10, 0x10});
        });
      });
    }
    s::cl_ushort r1 = r.x();
    s::cl_ushort r2 = r.y();
    assert(r1 == 0x1010);
    assert(r2 == 0x1010);
  }

  // upsample - 2
  {
    s::cl_short2 r{0};
    {
      s::buffer<s::cl_short2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSC2UC2>([=]() {
          AccR[0] =
              s::upsample(s::cl_char2{0x10, 0x10}, s::cl_uchar2{0x10, 0x10});
        });
      });
    }
    s::cl_short r1 = r.x();
    s::cl_short r2 = r.y();
    assert(r1 == 0x1010);
    assert(r2 == 0x1010);
  }

  // upsample - 3
  {
    s::cl_uint2 r{0};
    {
      s::buffer<s::cl_uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUS2US2>([=]() {
          AccR[0] = s::upsample(s::cl_ushort2{0x0010, 0x0010},
                                s::cl_ushort2{0x0010, 0x0010});
        });
      });
    }
    s::cl_uint r1 = r.x();
    s::cl_uint r2 = r.y();
    assert(r1 == 0x00100010);
    assert(r2 == 0x00100010);
  }

  // upsample - 4
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSS2US2>([=]() {
          AccR[0] = s::upsample(s::cl_short2{0x0010, 0x0010},
                                s::cl_ushort2{0x0010, 0x0010});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0x00100010);
    assert(r2 == 0x00100010);
  }

  // upsample - 5
  {
    s::cl_ulong2 r{0};
    {
      s::buffer<s::cl_ulong2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUI2UI2>([=]() {
          AccR[0] = s::upsample(s::cl_uint2{0x00000010, 0x00000010},
                                s::cl_uint2{0x00000010, 0x00000010});
        });
      });
    }
    s::cl_ulong r1 = r.x();
    s::cl_ulong r2 = r.y();
    assert(r1 == 0x0000001000000010);
    assert(r2 == 0x0000001000000010);
  }

  // upsample - 6
  {
    s::cl_long2 r{0};
    {
      s::buffer<s::cl_long2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSI2UI2>([=]() {
          AccR[0] = s::upsample(s::cl_int2{0x00000010, 0x00000010},
                                s::cl_uint2{0x00000010, 0x00000010});
        });
      });
    }
    s::cl_long r1 = r.x();
    s::cl_long r2 = r.y();
    assert(r1 == 0x0000001000000010);
    assert(r2 == 0x0000001000000010);
  }

  // popcount
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class popcountSI2>([=]() {
          AccR[0] = s::popcount(s::cl_int2{0x000000FF, 0x000000FF});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 8);
    assert(r2 == 8);
  }

  // mad24
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad24SI2SI2SI2>([=]() {
          AccR[0] = s::mad24(s::cl_int2{0xFFFFFFFF, 0xFFFFFFFF},
                             s::cl_int2{20, 20}, s::cl_int2{20, 20});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == 0);
    assert(r2 == 0);
  }

  // mul24
  {
    s::cl_int2 r{0};
    {
      s::buffer<s::cl_int2, 1> BufR(&r, s::range<1>(1));
      s::queue myqueue;
      myqueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul24SI2SI2SI2>([=]() {
          AccR[0] =
              s::mul24(s::cl_int2{0xFFFFFFFF, 0xFFFFFFFF}, s::cl_int2{20, 20});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    assert(r1 == -20);
    assert(r2 == -20);
  }

  return 0;
}
