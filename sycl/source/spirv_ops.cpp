//===------------- spirv_ops.cpp - SPIRV operations -----------------------===//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/exception.hpp>
#include <atomic>

namespace cl {
namespace __spirv {

// This operation is NOP on HOST as all operations there are blocking and
// by the moment this function was called, the operations generating
// the OpTypeEvent objects had already been finished.
void OpGroupWaitEvents(int32_t Scope, uint32_t NumEvents,
                              OpTypeEvent ** WaitEvents) noexcept {
}

void OpControlBarrier(Scope Execution, Scope Memory,
                      uint32_t Semantics) noexcept {
  throw cl::sycl::runtime_error(
      "Barrier is not supported on the host device yet.");
}

void OpMemoryBarrier(Scope Memory, uint32_t Semantics) noexcept {
  // 1. The 'Memory' parameter is ignored on HOST because there is no memory
  //    separation to global and local there.
  // 2. The 'Semantics' parameter is ignored because there is no need
  //    to distinguish the classes of memory (workgroup/cross-workgroup/etc).
  atomic_thread_fence(std::memory_order_seq_cst);
}

} // namespace __spirv
} // namespace cl
