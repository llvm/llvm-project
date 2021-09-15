//===---- OrcRTBridge.h -- Utils for interacting with orc-rt ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares types and symbol names provided by the ORC runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_ORCRTBRIDGE_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_ORCRTBRIDGE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

namespace llvm {
namespace orc {
namespace rt {

extern const char *MemoryReserveWrapperName;
extern const char *MemoryFinalizeWrapperName;
extern const char *MemoryDeallocateWrapperName;
extern const char *MemoryWriteUInt8sWrapperName;
extern const char *MemoryWriteUInt16sWrapperName;
extern const char *MemoryWriteUInt32sWrapperName;
extern const char *MemoryWriteUInt64sWrapperName;
extern const char *MemoryWriteBuffersWrapperName;
extern const char *RunAsMainWrapperName;

using SPSMemoryReserveSignature =
    shared::SPSExpected<shared::SPSExecutorAddress>(uint64_t);
using SPSMemoryFinalizeSignature = shared::SPSError(shared::SPSFinalizeRequest);
using SPSMemoryDeallocateSignature =
    shared::SPSError(shared::SPSExecutorAddress, uint64_t);
using SPSRunAsMainSignature = int64_t(shared::SPSExecutorAddress,
                                      shared::SPSSequence<shared::SPSString>);

} // end namespace rt
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_ORCRTBRIDGE_H
