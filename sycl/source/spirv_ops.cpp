//===------------- spirv_ops.cpp - SPIRV operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/detail/platform_util.hpp>
#include <atomic>

// This operation is NOP on HOST as all operations there are blocking and
// by the moment this function was called, the operations generating
// the __ocl_event_t objects had already been finished.
void __spirv_GroupWaitEvents(__spv::Scope Execution, uint32_t NumEvents,
                              __ocl_event_t * WaitEvents) noexcept {
}

void __spirv_ControlBarrier(__spv::Scope Execution, __spv::Scope Memory,
                      uint32_t Semantics) noexcept {
  throw cl::sycl::runtime_error(
      "Barrier is not supported on the host device yet.");
}

void __spirv_MemoryBarrier(__spv::Scope Memory, uint32_t Semantics) noexcept {
  // 1. The 'Memory' parameter is ignored on HOST because there is no memory
  //    separation to global and local there.
  // 2. The 'Semantics' parameter is ignored because there is no need
  //    to distinguish the classes of memory (workgroup/cross-workgroup/etc).
  atomic_thread_fence(std::memory_order_seq_cst);
}

void __spirv_ocl_prefetch(const char *Ptr, size_t NumBytes) noexcept {
  cl::sycl::detail::PlatformUtil::prefetch(Ptr, NumBytes);
}
