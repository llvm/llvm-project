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

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {
namespace rt {

LLVM_ABI extern const char *SimpleExecutorMemoryManagerInstanceName;
LLVM_ABI extern const char *SimpleExecutorMemoryManagerReserveWrapperName;
LLVM_ABI extern const char *SimpleExecutorMemoryManagerInitializeWrapperName;
LLVM_ABI extern const char *SimpleExecutorMemoryManagerDeinitializeWrapperName;
LLVM_ABI extern const char *SimpleExecutorMemoryManagerReleaseWrapperName;

LLVM_ABI extern const char *ExecutorSharedMemoryMapperServiceInstanceName;
LLVM_ABI extern const char *ExecutorSharedMemoryMapperServiceReserveWrapperName;
LLVM_ABI extern const char
    *ExecutorSharedMemoryMapperServiceInitializeWrapperName;
LLVM_ABI extern const char
    *ExecutorSharedMemoryMapperServiceDeinitializeWrapperName;
LLVM_ABI extern const char *ExecutorSharedMemoryMapperServiceReleaseWrapperName;

LLVM_ABI extern const char *RegisterEHFrameSectionAllocActionName;
LLVM_ABI extern const char *DeregisterEHFrameSectionAllocActionName;

LLVM_ABI extern const char *RegisterJITLoaderGDBAllocActionName;
LLVM_ABI extern const char *DeregisterJITLoaderGDBAllocActionName;

LLVM_ABI extern const char *const DispatchName;
LLVM_ABI extern const char *const DispatchCtxName;

/// Symbol names for the ORC runtime's StandaloneMachOUnwindInfoRegistrar
/// SPS interface.
struct MachOUnwindInfoRegistrarSymbolNames {
  StringRef RegisterSectionsName;
  StringRef DeregisterSectionsName;
};

/// Default symbol names for the ORC runtime's
/// StandaloneMachOUnwindInfoRegistrar SPS interface.
extern const LLVM_ABI MachOUnwindInfoRegistrarSymbolNames
    orc_rt_MachOUnwindInfoRegistrarSPSSymbols;

using SPSSimpleExecutorMemoryManagerReserveSignature =
    shared::SPSExpected<shared::SPSExecutorAddr>(shared::SPSExecutorAddr,
                                                 uint64_t);
using SPSSimpleExecutorMemoryManagerInitializeSignature =
    shared::SPSExpected<shared::SPSExecutorAddr>(shared::SPSExecutorAddr,
                                                 shared::SPSFinalizeRequest);
using SPSSimpleExecutorMemoryManagerDeinitializeSignature = shared::SPSError(
    shared::SPSExecutorAddr, shared::SPSSequence<shared::SPSExecutorAddr>);
using SPSSimpleExecutorMemoryManagerReleaseSignature = shared::SPSError(
    shared::SPSExecutorAddr, shared::SPSSequence<shared::SPSExecutorAddr>);

// ExecutorSharedMemoryMapperService
using SPSExecutorSharedMemoryMapperServiceReserveSignature =
    shared::SPSExpected<
        shared::SPSTuple<shared::SPSExecutorAddr, shared::SPSString>>(
        shared::SPSExecutorAddr, uint64_t);
using SPSExecutorSharedMemoryMapperServiceInitializeSignature =
    shared::SPSExpected<shared::SPSExecutorAddr>(
        shared::SPSExecutorAddr, shared::SPSExecutorAddr,
        shared::SPSSharedMemoryFinalizeRequest);
using SPSExecutorSharedMemoryMapperServiceDeinitializeSignature =
    shared::SPSError(shared::SPSExecutorAddr,
                     shared::SPSSequence<shared::SPSExecutorAddr>);
using SPSExecutorSharedMemoryMapperServiceReleaseSignature = shared::SPSError(
    shared::SPSExecutorAddr, shared::SPSSequence<shared::SPSExecutorAddr>);

} // end namespace rt

namespace rt_alt {
LLVM_ABI extern const char *UnwindInfoManagerRegisterActionName;
LLVM_ABI extern const char *UnwindInfoManagerDeregisterActionName;
} // end namespace rt_alt
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_ORCRTBRIDGE_H
