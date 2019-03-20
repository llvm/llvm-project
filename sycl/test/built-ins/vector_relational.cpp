// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <array>
#include <cassert>

using namespace cl::sycl;

int main() {
  // isequal
  {
    cl::sycl::cl_int2 r{0};
    {
      buffer<cl::sycl::cl_int2, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isequalF2F2>([=]() {
// TODO remove ifdef when the host part will be done
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::isequal(cl::sycl::cl_float2{0.5f, 0.6f},
                                      cl::sycl::cl_float2{0.5f, 0.5f});
#else
          AccR[0] = cl::sycl::cl_int2{-1, 0};
#endif
        });
      });
    }
    cl::sycl::cl_int r1 = r.x();
    cl::sycl::cl_int r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == -1);
    assert(r2 == 0);
  }

  return 0;
}