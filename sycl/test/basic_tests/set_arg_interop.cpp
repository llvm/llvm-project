// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl -O3
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>

using namespace cl::sycl;

int main() {
  queue Queue;
  if (!Queue.is_host()) {
    context Context = Queue.get_context();

    cl_context ClContext = Context.get();

    const size_t CountSources = 2;
    const char *Sources[CountSources] = {
        "kernel void foo1(global float* Array, global int* Value) { *Array = "
        "42; *Value = 1; }\n",
        "kernel void foo2(global float* Array) { int id = get_global_id(0); "
        "Array[id] = id; }\n",
    };

    cl_int Err;
    cl_program ClProgram = clCreateProgramWithSource(ClContext, CountSources,
                                                     Sources, nullptr, &Err);
    assert(Err == CL_SUCCESS);

    Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
    assert(Err == CL_SUCCESS);

    cl_kernel FirstCLKernel = clCreateKernel(ClProgram, "foo1", &Err);
    assert(Err == CL_SUCCESS);

    cl_kernel SecondCLKernel = clCreateKernel(ClProgram, "foo2", &Err);
    assert(Err == CL_SUCCESS);

    const size_t Count = 100;
    float Array[Count];

    kernel FirstKernel(FirstCLKernel, Context);
    kernel SecondKernel(SecondCLKernel, Context);
    int Value;
    {
      buffer<float, 1> FirstBuffer(Array, range<1>(1));
      buffer<int, 1> SecondBuffer(&Value, range<1>(1));
      Queue.submit([&](handler &CGH) {
        CGH.set_arg(0, FirstBuffer.get_access<access::mode::write>(CGH));
        CGH.set_arg(1, SecondBuffer.get_access<access::mode::write>(CGH));
        CGH.single_task(FirstKernel);
      });
    }
    Queue.wait_and_throw();

    assert(Array[0] == 42);
    assert(Value == 1);

    {
      buffer<float, 1> FirstBuffer(Array, range<1>(Count));
      Queue.submit([&](handler &CGH) {
        auto Acc = FirstBuffer.get_access<access::mode::read_write>(CGH);
        CGH.set_arg(0, FirstBuffer.get_access<access::mode::read_write>(CGH));
        CGH.parallel_for(range<1>{Count}, SecondKernel);
      });
    }
    Queue.wait_and_throw();

    for (size_t I = 0; I < Count; ++I) {
      assert(Array[I] == I);
    }

    clReleaseContext(ClContext);
    clReleaseKernel(FirstCLKernel);
    clReleaseKernel(SecondCLKernel);
    clReleaseProgram(ClProgram);
  }
  return 0;
}
