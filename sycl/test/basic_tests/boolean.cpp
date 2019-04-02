// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>

using namespace cl::sycl;
namespace s = cl::sycl;
namespace d = s::detail;

d::Boolean<3> foo() {
  d::Boolean<3> b3{true, false, true};
  return b3;
}

int main() {
  {
    s::cl_long4 r{0};
    {
      buffer<s::cl_long4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class b4_l4>([=]() {
          d::Boolean<4> b4{false, true, false, false};
          AccR[0] = b4;
        });
      });
    }
    s::cl_long r1 = r.s0();
    s::cl_long r2 = r.s1();
    s::cl_long r3 = r.s2();
    s::cl_long r4 = r.s3();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  {
    s::cl_short3 r{0};
    {
      buffer<s::cl_short3, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class b3_sh3>([=]() { AccR[0] = foo(); });
      });
    }
    s::cl_short r1 = r.s0();
    s::cl_short r2 = r.s1();
    s::cl_short r3 = r.s2();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << std::endl;

    assert(r1 == -1);
    assert(r2 == 0);
    assert(r3 == -1);
  }

  {
    s::cl_int r1[6];
    s::cl_int r2[6];
    {
      buffer<s::cl_int, 1> BufR1(r1, range<1>(6));
      buffer<s::cl_int, 1> BufR2(r2, range<1>(6));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR1 = BufR1.get_access<access::mode::write>(cgh);
        auto AccR2 = BufR2.get_access<access::mode::write>(cgh);
        cgh.single_task<class size_align>([=]() {
          AccR1[0] = sizeof(d::Boolean<1>);
          AccR1[1] = sizeof(d::Boolean<2>);
          AccR1[2] = sizeof(d::Boolean<3>);
          AccR1[3] = sizeof(d::Boolean<4>);
          AccR1[4] = sizeof(d::Boolean<8>);
          AccR1[5] = sizeof(d::Boolean<16>);

          AccR2[0] = alignof(d::Boolean<1>);
          AccR2[1] = alignof(d::Boolean<2>);
          AccR2[2] = alignof(d::Boolean<3>);
          AccR2[3] = alignof(d::Boolean<4>);
          AccR2[4] = alignof(d::Boolean<8>);
          AccR2[5] = alignof(d::Boolean<16>);
        });
      });
    }

    for (size_t I = 0; I < 6; I++) {
      std::cout << " r1[" << I << "] " << r1[I];
    }
    std::cout << std::endl;

    for (size_t I = 0; I < 6; I++) {
      std::cout << " r2[" << I << "] " << r2[I];
    }
    std::cout << std::endl;
    assert(r1[0] == sizeof(d::Boolean<1>));
    assert(r1[1] == sizeof(d::Boolean<2>));
    assert(r1[2] == sizeof(d::Boolean<3>));
    assert(r1[3] == sizeof(d::Boolean<4>));
    assert(r1[4] == sizeof(d::Boolean<8>));
    assert(r1[5] == sizeof(d::Boolean<16>));

    assert(r2[0] == alignof(d::Boolean<1>));
    assert(r2[1] == alignof(d::Boolean<2>));
    assert(r2[2] == alignof(d::Boolean<3>));
    assert(r2[3] == alignof(d::Boolean<4>));
    assert(r2[4] == alignof(d::Boolean<8>));
    assert(r2[5] == alignof(d::Boolean<16>));
  }

  {
    s::cl_int4 i4 = {1, -2, 0, -3};
    d::Boolean<4> b4(i4);
    i4 = b4;

    s::cl_int r1 = i4.s0();
    s::cl_int r2 = i4.s1();
    s::cl_int r3 = i4.s2();
    s::cl_int r4 = i4.s3();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  {
    s::cl_int r1 = d::Boolean<1>(s::cl_int{-1});
    s::cl_int r2 = d::Boolean<1>(s::cl_int{0});
    s::cl_int r3 = d::Boolean<1>(s::cl_int{1});
    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << std::endl;
    assert(r1 == 1);
    assert(r2 == 0);
    assert(r3 == 1);
  }

  return 0;
}
