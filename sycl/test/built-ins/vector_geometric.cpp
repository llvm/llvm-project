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

bool isFloatEqualTo(float x, float y, float epsilon = 0.005f){
  return std::fabs(x - y) <= epsilon;
}

int main() {
  // dot
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class dotF2F2>([=]() {
          AccR[0] = cl::sycl::dot(
              cl::sycl::cl_float2{
                  1.f,
                  2.f,
              },
              cl::sycl::cl_float2{4.f, 6.f});
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 16.f);
  }

  // cross
  {
    cl::sycl::cl_float4 r{0};
    {
      buffer<cl::sycl::cl_float4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class crossF4>([=]() {
          AccR[0] = cl::sycl::cross(
              cl::sycl::cl_float4{
                  2.f,
                  3.f,
                  4.f,
                  0.f,
              },
              cl::sycl::cl_float4{
                  5.f,
                  6.f,
                  7.f,
                  0.f,
              });
        });
      });
    }

    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();
    cl::sycl::cl_float r3 = r.z();
    cl::sycl::cl_float r4 = r.w();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == -3.f);
    assert(r2 == 6.f);
    assert(r3 == -3.f);
    assert(r4 == 0.0f);
  }

  // distance
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class distanceF2>([=]() {
          AccR[0] = cl::sycl::distance(
              cl::sycl::cl_float2{
                  1.f,
                  2.f,
              },
              cl::sycl::cl_float2{
                  3.f,
                  4.f,
              });
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(isFloatEqualTo(r, 2.82843f));
  }

  // length
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class lengthF2>([=]() {
          AccR[0] = cl::sycl::length(cl::sycl::cl_float2{
              1.f,
              2.f,
          });
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(isFloatEqualTo(r, 2.23607f));
  }
  // normalize
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class normalizeF2>([=]() {
          AccR[0] = cl::sycl::normalize(cl::sycl::cl_float2{
              1.f,
              2.f,
          });
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();

    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(isFloatEqualTo(r1, 0.447214f));
    assert(isFloatEqualTo(r2, 0.894427f));
  }

  // fast_distance
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fast_distanceF2>([=]() {
          AccR[0] = cl::sycl::fast_distance(
              cl::sycl::cl_float2{
                  1.f,
                  2.f,
              },
              cl::sycl::cl_float2{
                  3.f,
                  4.f,
              });
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(isFloatEqualTo(r, 2.82843f));
  }

  // fast_length
  {
    cl::sycl::cl_float r{0};
    {
      buffer<cl::sycl::cl_float, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fast_lengthF2>([=]() {
          AccR[0] = cl::sycl::fast_length(cl::sycl::cl_float2{
              1.f,
              2.f,
          });
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(isFloatEqualTo(r, 2.23607f));
  }

  // fast_normalize
  {
    cl::sycl::cl_float2 r{0};
    {
      buffer<cl::sycl::cl_float2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fast_normalizeF2>([=]() {
          AccR[0] = cl::sycl::fast_normalize(cl::sycl::cl_float2{
              1.f,
              2.f,
          });
        });
      });
    }
    cl::sycl::cl_float r1 = r.x();
    cl::sycl::cl_float r2 = r.y();

    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(isFloatEqualTo(r1, 0.447144));
    assert(isFloatEqualTo(r2, 0.894287));
  }

  return 0;
}