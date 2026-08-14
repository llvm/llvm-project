//===--- MemoryAccessSPSCI.h - SPS CI for memory access ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the executor's memory-access
// operations. These wrappers perform the operation directly, so they take the
// operation's data arguments rather than a callee address.
//
// See CallSPSCI.h for a description of the descriptor scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_MEMORYACCESSSPSCI_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_MEMORYACCESSSPSCI_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc::rt::sps_ci {

struct MemWriteUInt8s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_write_uint8s";
  using SPSSig = void(shared::SPSSequence<shared::SPSMemoryAccessUInt8Write>);
};

struct MemWriteUInt16s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_write_uint16s";
  using SPSSig = void(shared::SPSSequence<shared::SPSMemoryAccessUInt16Write>);
};

struct MemWriteUInt32s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_write_uint32s";
  using SPSSig = void(shared::SPSSequence<shared::SPSMemoryAccessUInt32Write>);
};

struct MemWriteUInt64s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_write_uint64s";
  using SPSSig = void(shared::SPSSequence<shared::SPSMemoryAccessUInt64Write>);
};

struct MemWritePointers {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_write_pointers";
  using SPSSig = void(shared::SPSSequence<shared::SPSMemoryAccessPointerWrite>);
};

struct MemWriteBuffers {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_write_buffers";
  using SPSSig = void(shared::SPSSequence<shared::SPSMemoryAccessBufferWrite>);
};

struct MemReadUInt8s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_read_uint8s";
  using SPSSig = shared::SPSSequence<uint8_t>(
      shared::SPSSequence<shared::SPSExecutorAddr>);
};

struct MemReadUInt16s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_read_uint16s";
  using SPSSig = shared::SPSSequence<uint16_t>(
      shared::SPSSequence<shared::SPSExecutorAddr>);
};

struct MemReadUInt32s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_read_uint32s";
  using SPSSig = shared::SPSSequence<uint32_t>(
      shared::SPSSequence<shared::SPSExecutorAddr>);
};

struct MemReadUInt64s {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_read_uint64s";
  using SPSSig = shared::SPSSequence<uint64_t>(
      shared::SPSSequence<shared::SPSExecutorAddr>);
};

struct MemReadPointers {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_read_pointers";
  using SPSSig = shared::SPSSequence<shared::SPSExecutorAddr>(
      shared::SPSSequence<shared::SPSExecutorAddr>);
};

struct MemReadBuffers {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_read_buffers";
  using SPSSig = shared::SPSSequence<shared::SPSSequence<uint8_t>>(
      shared::SPSSequence<shared::SPSExecutorAddrRange>);
};

struct MemReadStrings {
  static constexpr char Name[] = "orc_rt_ci_sps_mem_read_strings";
  using SPSSig = shared::SPSSequence<shared::SPSString>(
      shared::SPSSequence<shared::SPSExecutorAddr>);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_MEMORYACCESSSPSCI_H
