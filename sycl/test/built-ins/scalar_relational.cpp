// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
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
    cl::sycl::cl_int r{0};
    {
      buffer<cl::sycl::cl_int, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class isequalF1F1>([=]() {
// TODO remove ifdef when the host part will be done
#ifdef __SYCL_DEVICE_ONLY__
          AccR[0] = cl::sycl::isequal(cl::sycl::cl_float{0.5f},
                                      cl::sycl::cl_float{0.5f});
#else
          AccR[0] = 1;
#endif
        });
      });
    }
    std::cout << "r " << r << std::endl;
    assert(r == 1);
  }

  return 0;
}