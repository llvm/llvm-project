// RUN: %clang -std=c++11 --sycl -fno-sycl-use-bitcode -Xclang -fsycl-int-header=%t.h -c %s -o %t.spv
// RUN: %clang -std=c++11 -include %t.h -g %s -o %t.out -lOpenCL -lsycl -lstdc++
// RUN: env SYCL_USE_KERNEL_SPV=%t.spv %t.out | FileCheck %s
// CHECK: Passed


#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main(int argc, char **argv) {
  int data = 5;

  try {
    queue myQueue;
    buffer<int, 1> buf(&data, range<1>(1));

    event e = myQueue.submit([&](handler& cgh) {
      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      cgh.single_task<class my_kernel>([=]() {
        ptr[0]++;
      });
    });
    e.wait_and_throw();

  } catch (cl::sycl::exception const& e) {
    std::cerr << "SYCL exception caught:\n";
    std::cerr << e.what() << "\n";
    return 2;
  }
  catch (...) {
    std::cerr << "unknown exception caught\n";
    return 1;
  }

  if (data == 6) {
    std::cout << "Passed\n";
    return 0;
  } else {
    std::cout << "Failed: " << data << "!= 6(gold)\n";
    return 1;
  }
}

