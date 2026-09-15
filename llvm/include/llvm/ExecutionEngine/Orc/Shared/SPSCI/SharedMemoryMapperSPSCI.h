//===- SharedMemoryMapperSPSCI.h - SPS CI for shared-mem mapping *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the executor's shared-memory mapper
// service. See CallSPSCI.h for a description of the descriptor scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SHAREDMEMORYMAPPERSPSCI_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SHAREDMEMORYMAPPERSPSCI_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc::rt::sps_ci {

inline constexpr char SharedMemoryMapperInstanceName[] =
    "__llvm_orc_ExecutorSharedMemoryMapperService_Instance";

struct SharedMemoryMapperReserve {
  static constexpr char Name[] =
      "__llvm_orc_ExecutorSharedMemoryMapperService_Reserve";
  using SPSSig = shared::SPSExpected<
      shared::SPSTuple<shared::SPSExecutorAddr, shared::SPSString>>(
      shared::SPSExecutorAddr, uint64_t);
};

struct SharedMemoryMapperInitialize {
  static constexpr char Name[] =
      "__llvm_orc_ExecutorSharedMemoryMapperService_Initialize";
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
      shared::SPSExecutorAddr, shared::SPSExecutorAddr,
      shared::SPSSharedMemoryFinalizeRequest);
};

struct SharedMemoryMapperDeinitialize {
  static constexpr char Name[] =
      "__llvm_orc_ExecutorSharedMemoryMapperService_Deinitialize";
  using SPSSig = shared::SPSError(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>);
};

struct SharedMemoryMapperRelease {
  static constexpr char Name[] =
      "__llvm_orc_ExecutorSharedMemoryMapperService_Release";
  using SPSSig = shared::SPSError(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SHAREDMEMORYMAPPERSPSCI_H
