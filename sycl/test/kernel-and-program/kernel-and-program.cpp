// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//==--- kernel-and-program.cpp - SYCL kernel/program test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <utility>

int main() {

  // Single task invocation methods
  {
    cl::sycl::queue q;
    int data = 0;
    // Precompiled kernel invocation
    {
      cl::sycl::buffer<int, 1> buf(&data, cl::sycl::range<1>(1));
      cl::sycl::program prg(q.get_context());
      // Test program building
      assert(prg.get_state() == cl::sycl::program_state::none);
      prg.build_with_kernel_type<class SingleTask>();
      assert(prg.get_state() == cl::sycl::program_state::linked);
      assert(prg.has_kernel<class SingleTask>());
      cl::sycl::kernel krn = prg.get_kernel<class SingleTask>();
      assert(krn.get_context() == q.get_context());
      assert(krn.get_program() == prg);

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<class SingleTask>(krn, [=]() { acc[0] = acc[0] + 1; });
      });
      if (!q.is_host()) {
        const std::string integrationHeaderKernelName =
            cl::sycl::detail::KernelInfo<SingleTask>::getName();
        const std::string clKerneName =
            krn.get_info<cl::sycl::info::kernel::function_name>();
        assert(integrationHeaderKernelName == clKerneName);
      }
    }
    assert(data == 1);

    // OpenCL interoperability kernel invocation
    // TODO add set_args(cl::sycl::sampler) use case once it's supported
    if (!q.is_host()) {
      cl_int err;
      if (0) {
        cl::sycl::context ctx = q.get_context();
        cl_context clCtx = ctx.get();
        cl_command_queue clQ = q.get();
        cl_mem clBuffer =
            clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);
        err = clEnqueueWriteBuffer(clQ, clBuffer, CL_TRUE, 0, sizeof(int),
                                   &data, 0, NULL, NULL);
        // Kernel interoperability constructor
        assert(err == CL_SUCCESS);
        cl::sycl::program prog(ctx);
        prog.build_with_source(
            "kernel void SingleTask(global int* a) {*a+=1; }\n");
        q.submit([&](cl::sycl::handler &cgh) {
          cgh.set_args(clBuffer);
          cgh.single_task(prog.get_kernel("SingleTask"));
        });
        q.wait();
        err = clEnqueueReadBuffer(clQ, clBuffer, CL_TRUE, 0, sizeof(int), &data,
                                  0, NULL, NULL);
        clReleaseCommandQueue(clQ);
        clReleaseContext(clCtx);
        assert(err == CL_SUCCESS);
        assert(data == 2);
      }
      {
        cl::sycl::queue sycl_queue;
        cl::sycl::program prog(sycl_queue.get_context());
        prog.build_with_source("kernel void foo(global int* a, global int* b, "
                               "global int* c) {*a=*b+*c; }\n");
        int a = 13, b = 14, c = 15;
        {
          cl::sycl::buffer<int, 1> bufa(&a, cl::sycl::range<1>(1));
          cl::sycl::buffer<int, 1> bufb(&b, cl::sycl::range<1>(1));
          cl::sycl::buffer<int, 1> bufc(&c, cl::sycl::range<1>(1));
          sycl_queue.submit([&](cl::sycl::handler &cgh) {
            auto A = bufa.get_access<cl::sycl::access::mode::write>(cgh);
            auto B = bufb.get_access<cl::sycl::access::mode::read>(cgh);
            auto C = bufc.get_access<cl::sycl::access::mode::read>(cgh);
            cgh.set_args(A, B, C);
            cgh.single_task(prog.get_kernel("foo"));
          });
        }
        assert(a == b + c);
      }
    }
  }
  // Parallel for with range
  {
    cl::sycl::queue q;
    std::vector<int> dataVec(10);

    // Precompiled kernel invocation
    // TODO Disabled on GPU, revert once compile -> link issue is fixed there
    if (!q.get_device().is_gpu()) {
      std::iota(dataVec.begin(), dataVec.end(), 0);
      {
        cl::sycl::range<1> numOfItems(dataVec.size());
        cl::sycl::buffer<int, 1> buf(dataVec.data(), numOfItems);
        cl::sycl::program prg(q.get_context());
        assert(prg.get_state() == cl::sycl::program_state::none);
        // Test compiling -> linking
        prg.compile_with_kernel_type<class ParallelFor>();
        assert(prg.get_state() == cl::sycl::program_state::compiled);
        prg.link();
        assert(prg.get_state() == cl::sycl::program_state::linked);
        assert(prg.has_kernel<class ParallelFor>());
        cl::sycl::kernel krn = prg.get_kernel<class ParallelFor>();
        assert(krn.get_context() == q.get_context());
        assert(krn.get_program() == prg);

        q.submit([&](cl::sycl::handler &cgh) {
          auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<class ParallelFor>(
              numOfItems, krn,
              [=](cl::sycl::id<1> wiID) { acc[wiID] = acc[wiID] + 1; });
        });
      }
      for (size_t i = 0; i < dataVec.size(); ++i) {
        assert(dataVec[i] == i + 1);
      }
    }

    // OpenCL interoperability kernel invocation
    std::iota(dataVec.begin(), dataVec.end(), 0);
    if (!q.is_host()) {
      cl_int err;
      {
        cl::sycl::context ctx = q.get_context();
        cl_context clCtx = ctx.get();
        cl_command_queue clQ = q.get();
        cl_mem clBuffer = clCreateBuffer(
            clCtx, CL_MEM_WRITE_ONLY, sizeof(int) * dataVec.size(), NULL, NULL);
        err = clEnqueueWriteBuffer(clQ, clBuffer, CL_TRUE, 0,
                                   sizeof(int) * dataVec.size(), dataVec.data(),
                                   0, NULL, NULL);
        assert(err == CL_SUCCESS);

        cl::sycl::program prog(ctx);
        prog.build_with_source(
            "kernel void ParallelFor(__global int* a, int v, __local int *l) "
            "{ size_t index = get_global_id(0); l[index] = a[index];"
            " l[index] += v; a[index] = l[index]; }\n");

        q.submit([&](cl::sycl::handler &cgh) {
          const int value = 1;
          auto local_acc =
              cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local>(
                  cl::sycl::range<1>(10), cgh);
          cgh.set_args(clBuffer, value, local_acc);
          cgh.parallel_for(cl::sycl::range<1>(10),
                           prog.get_kernel("ParallelFor"));
        });

        q.wait();
        err = clEnqueueReadBuffer(clQ, clBuffer, CL_TRUE, 0,
                                  sizeof(int) * dataVec.size(), dataVec.data(),
                                  0, NULL, NULL);
        clReleaseCommandQueue(clQ);
        clReleaseContext(clCtx);
        assert(err == CL_SUCCESS);
        for (size_t i = 0; i < dataVec.size(); ++i) {
          assert(dataVec[i] == i + 1);
        }
      }
    }
  }

  // Parallel for with nd_range
  {
    cl::sycl::queue q;
    std::vector<int> dataVec(10);
    std::iota(dataVec.begin(), dataVec.end(), 0);

    // Precompiled kernel invocation
    // TODO run on host as well once local barrier is supported
    if (!q.is_host()) {
      {
        cl::sycl::range<1> numOfItems(dataVec.size());
        cl::sycl::range<1> localRange(2);
        cl::sycl::buffer<int, 1> buf(dataVec.data(), numOfItems);
        cl::sycl::program prg(q.get_context());
        assert(prg.get_state() == cl::sycl::program_state::none);
        prg.build_with_kernel_type<class ParallelForND>();
        assert(prg.get_state() == cl::sycl::program_state::linked);
        assert(prg.has_kernel<class ParallelForND>());
        cl::sycl::kernel krn = prg.get_kernel<class ParallelForND>();
        assert(krn.get_context() == q.get_context());
        assert(krn.get_program() == prg);

        q.submit([&](cl::sycl::handler &cgh) {
          auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
          cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local>
              localAcc(localRange, cgh);

          cgh.parallel_for<class ParallelForND>(
              cl::sycl::nd_range<1>(numOfItems, localRange), krn,
              [=](cl::sycl::nd_item<1> item) {
                size_t idx = item.get_global_linear_id();
                int pos = idx & 1;
                int opp = pos ^ 1;
                localAcc[pos] = acc[item.get_global_linear_id()];

                item.barrier(cl::sycl::access::fence_space::local_space);

                acc[idx] = localAcc[opp];
              });
        });
      }
      q.wait();
      for (size_t i = 0; i < dataVec.size(); ++i) {
        assert(dataVec[i] == (i ^ 1));
      }
    }

    // OpenCL interoperability kernel invocation
    if (!q.is_host()) {
      cl_int err;
      {
        cl::sycl::context ctx = q.get_context();
        cl_context clCtx = ctx.get();
        cl_command_queue clQ = q.get();
        cl_mem clBuffer = clCreateBuffer(
            clCtx, CL_MEM_WRITE_ONLY, sizeof(int) * dataVec.size(), NULL, NULL);
        err = clEnqueueWriteBuffer(clQ, clBuffer, CL_TRUE, 0,
                                   sizeof(int) * dataVec.size(), dataVec.data(),
                                   0, NULL, NULL);
        assert(err == CL_SUCCESS);

        cl::sycl::program prog(ctx);
        prog.build_with_source(
            "kernel void ParallelForND( local int* l,global int* a)"
            "{  size_t idx = get_global_id(0);"
            "  int pos = idx & 1;"
            "  int opp = pos ^ 1;"
            "  l[pos] = a[get_global_id(0)];"
            "  barrier(CLK_LOCAL_MEM_FENCE);"
            "  a[idx]=l[opp]; }");

        // TODO is there no way to set local memory size via interoperability?
        cl::sycl::kernel krn = prog.get_kernel("ParallelForND");
        clSetKernelArg(krn.get(), 0, sizeof(int) * 2, NULL);

        q.submit([&](cl::sycl::handler &cgh) {
          cgh.set_arg(1, clBuffer);
          cgh.parallel_for(cl::sycl::nd_range<1>(cl::sycl::range<1>(10),
                                                 cl::sycl::range<1>(2)),
                           krn);
        });

        q.wait();
        err = clEnqueueReadBuffer(clQ, clBuffer, CL_TRUE, 0,
                                  sizeof(int) * dataVec.size(), dataVec.data(),
                                  0, NULL, NULL);
        clReleaseCommandQueue(clQ);
        clReleaseContext(clCtx);
        assert(err == CL_SUCCESS);
      }
      for (size_t i = 0; i < dataVec.size(); ++i) {
        assert(dataVec[i] == i);
      }
    }
  }
}
