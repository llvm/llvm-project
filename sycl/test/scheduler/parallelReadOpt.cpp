// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out | FileCheck %s
//
// CHECK:Buffer A [[A:.*]]
// CHECK:Evolution of buffer [[A]]
// CHECK-NEXT:ID = [[ALLOCA_A:[0-9]+]] ; ALLOCA ON [[DEVICE_TYPE:.*]]
// CHECK-NEXT:  Buf : [[A]]  Access : read_write
// CHECK-NEXT:    Dependency:
// CHECK-NEXT:ID = [[INIT:[0-9]+]] ; RUN_KERNEL init_kernel ON [[DEVICE_TYPE]]
// CHECK-NEXT:    Dependency:
// CHECK-NEXT:        Dep on buf [[A]] write from Command ID = {{[0-9]+}}
// CHECK-NEXT:ID = [[READ1:[0-9]+]] ; RUN_KERNEL read1 ON [[DEVICE_TYPE]]
// CHECK-NEXT:    Dependency:
// CHECK-DAG:        Dep on buf [[B:.*]] write from Command ID = {{[0-9]+}}
// CHECK-DAG:        Dep on buf [[A]] read from Command ID = [[INIT]]
// CHECK-NEXT:ID = [[READ2:[0-9]+]] ; RUN_KERNEL read2 ON [[DEVICE_TYPE]]
// CHECK-NEXT:    Dependency:
// CHECK-NEXT:        Dep on buf [[C:.*]] write from Command ID = {{[0-9]+}}
// CHECK-NEXT:        Dep on buf [[A]] read from Command ID = [[INIT]]

//==---- parallelReadOpt.cpp - SYCL scheduler parallel read test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *

#include "CL/sycl.hpp"
#include "CL/sycl/detail/scheduler/scheduler.h"

using namespace cl::sycl;
static constexpr const detail::kernel_param_desc_t kernel_signatures[] = {
    //--- init_kernel
    {detail::kernel_param_kind_t::kind_accessor, 2014, 0},
    //--- read1 and read2
    {detail::kernel_param_kind_t::kind_accessor, 2014, 0},
    {detail::kernel_param_kind_t::kind_accessor, 2014, 192}};

int main() {
  auto M = detail::OSUtil::ExeModuleHandle;

  queue Queue;
  auto QueueImpl = detail::getSyclObjImpl(Queue);
  const size_t N = 10;

  buffer<float, 1> A(range<1>{N});
  buffer<float, 1> B(range<1>{N});
  buffer<float, 1> C(range<1>{N});

  { // Adding node that requires write access to A
    simple_scheduler::Node InitNode(QueueImpl);
    InitNode.template addBufRequirement<access::mode::write,
                                        access::target::global_buffer>(
        *detail::getSyclObjImpl(A));
    InitNode.addKernel(M, "init_kernel", 1, kernel_signatures, []() {});
    simple_scheduler::Scheduler::getInstance().addNode(std::move(InitNode));
  }

  { // Adding node that requires read access to A, write to B
    simple_scheduler::Node ReadNode1(QueueImpl);
    ReadNode1.template addBufRequirement<access::mode::read,
                                         access::target::global_buffer>(
        *detail::getSyclObjImpl(A));
    ReadNode1.template addBufRequirement<access::mode::write,
                                         access::target::global_buffer>(
        *detail::getSyclObjImpl(B));
    ReadNode1.addKernel(M, "read1", 2, kernel_signatures + 2, []() {});
    simple_scheduler::Scheduler::getInstance().addNode(std::move(ReadNode1));
  }

  { // Adding node that requires read access to A, write to C
    simple_scheduler::Node ReadNode2(QueueImpl);
    ReadNode2.template addBufRequirement<access::mode::read,
                                         access::target::global_buffer>(
        *detail::getSyclObjImpl(A));
    ReadNode2.template addBufRequirement<access::mode::write,
                                         access::target::global_buffer>(
        *detail::getSyclObjImpl(C));
    ReadNode2.addKernel(M, "read2", 2, kernel_signatures + 2, []() {});
    simple_scheduler::Scheduler::getInstance().addNode(std::move(ReadNode2));
  }

  std::cout << "Buffer A " << detail::getSyclObjImpl(A).get() << std::endl;

  // Expected that read2 kernel doesn't depend on read1.
  simple_scheduler::Scheduler::getInstance().parallelReadOpt();

  simple_scheduler::Scheduler::getInstance().dump();
}
