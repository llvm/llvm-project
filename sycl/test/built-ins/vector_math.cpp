// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>

using namespace cl::sycl;

int main() {
  // fmin
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fminF2F2>([=]() {
          AccR[0] = cl::sycl::fmin(cl::sycl::cl_float2{0.5f, 3.4f},
                                   cl::sycl::cl_float2{2.3f, 0.4f});
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 0.5f);
    assert(r2 == 0.4f);
  }

  // native::exp
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class nexpF2>([=]() {
          AccR[0] = cl::sycl::native::exp(cl::sycl::cl_float2{1.0f, 2.0f});
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 > 2.718 && r1 < 2.719); // ~2.718281828459045
    assert(r2 > 7.389 && r2 < 7.390); // ~7.38905609893065
  }

  // fabs
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fabsF2>([=]() {
          AccR[0] = cl::sycl::fabs(cl::sycl::cl_float2{-1.0f, 2.0f});
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // floor
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class floorF2>([=]() {
          AccR[0] = cl::sycl::floor(cl::sycl::cl_float2{1.4f, 2.8f});
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // ceil
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class ceilF2>([=]() {
          AccR[0] = cl::sycl::ceil(cl::sycl::cl_float2{1.4f, 2.8f});
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // fract with global memory
  /*{
    cl::sycl::cl_float2 r{0, 0};
    cl::sycl::cl_float2 i{0, 0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      buffer<cl::sycl::cl_float2, 1> BufI(&i, range<1>(1));

      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class fractF2GF2>([=]() {
          global_ptr<cl::sycl::cl_float2> Iptr(AccI);
          AccR[0] = cl::sycl::fract(cl::sycl::cl_float2{1.5f, 2.5f}, Iptr);
        });
      });
    }

    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    cl::sycl::cl_float i1 = i.x();
    cl::sycl::cl_float i2 = i.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << " i1 " << i1 << " i2 " << i2
              << std::endl;
    assert(r1 == 0.5f);
    assert(r2 == 0.5f);
    assert(i1 == 1.0f);
    assert(i2 == 2.0f);
  }

  // fract with private memory
  {
    cl::sycl::cl_float2 r{0, 0};
    cl::sycl::cl_float2 i{0, 0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      buffer<cl::sycl::cl_float2, 1> BufI(&i, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class fractF2PF2>([=]() {
          cl::sycl::cl_float2 temp(0.0);
          private_ptr<cl::sycl::cl_float2> Iptr(&temp);
          AccR[0] = cl::sycl::fract(cl::sycl::cl_float2{1.5f, 2.5f}, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }

    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    cl::sycl::cl_float i1 = i.x();
    cl::sycl::cl_float i2 = i.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << " i1 " << i1 << " i2 " << i2
              << std::endl;
    assert(r1 == 0.5f);
    assert(r2 == 0.5f);
    assert(i1 == 1.0f);
    assert(i2 == 2.0f);
  }*/

  return 0;
}
