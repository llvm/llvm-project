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
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {
namespace rt {

LLVM_ABI extern const char *SimpleExecutorDylibManagerInstanceName;
LLVM_ABI extern const char *SimpleExecutorDylibManagerOpenWrapperName;
LLVM_ABI extern const char *SimpleExecutorDylibManagerResolveWrapperName;

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

LLVM_ABI extern const char *MemoryWriteUInt8sWrapperName;
LLVM_ABI extern const char *MemoryWriteUInt16sWrapperName;
LLVM_ABI extern const char *MemoryWriteUInt32sWrapperName;
LLVM_ABI extern const char *MemoryWriteUInt64sWrapperName;
LLVM_ABI extern const char *MemoryWritePointersWrapperName;
LLVM_ABI extern const char *MemoryWriteBuffersWrapperName;

LLVM_ABI extern const char *MemoryReadUInt8sWrapperName;
LLVM_ABI extern const char *MemoryReadUInt16sWrapperName;
LLVM_ABI extern const char *MemoryReadUInt32sWrapperName;
LLVM_ABI extern const char *MemoryReadUInt64sWrapperName;
LLVM_ABI extern const char *MemoryReadPointersWrapperName;
LLVM_ABI extern const char *MemoryReadBuffersWrapperName;
LLVM_ABI extern const char *MemoryReadStringsWrapperName;

LLVM_ABI extern const char *RegisterEHFrameSectionAllocActionName;
LLVM_ABI extern const char *DeregisterEHFrameSectionAllocActionName;

LLVM_ABI extern const char *RunAsMainWrapperName;
LLVM_ABI extern const char *RunAsVoidFunctionWrapperName;
LLVM_ABI extern const char *RunAsIntFunctionWrapperName;

using SPSSimpleExecutorDylibManagerOpenSignature =
    shared::SPSExpected<shared::SPSExecutorAddr>(shared::SPSExecutorAddr,
                                                 shared::SPSString, uint64_t);

using SPSSimpleExecutorDylibManagerResolveSignature = shared::SPSExpected<
    shared::SPSSequence<shared::SPSOptional<shared::SPSExecutorSymbolDef>>>(
    shared::SPSExecutorAddr, shared::SPSRemoteSymbolLookupSet);

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

// SimpleNativeMemoryMap APIs.
using SPSSimpleRemoteMemoryMapReserveSignature =
    shared::SPSExpected<shared::SPSExecutorAddr>(shared::SPSExecutorAddr,
                                                 uint64_t);
using SPSSimpleRemoteMemoryMapInitializeSignature =
    shared::SPSExpected<shared::SPSExecutorAddr>(shared::SPSExecutorAddr,
                                                 shared::SPSFinalizeRequest);
using SPSSimpleRemoteMemoryMapDeinitializeSignature = shared::SPSError(
    shared::SPSExecutorAddr, shared::SPSSequence<shared::SPSExecutorAddr>);
using SPSSimpleRemoteMemoryMapReleaseSignature = shared::SPSError(
    shared::SPSExecutorAddr, shared::SPSSequence<shared::SPSExecutorAddr>);

using SPSRunAsMainSignature = int64_t(shared::SPSExecutorAddr,
                                      shared::SPSSequence<shared::SPSString>);
using SPSRunAsVoidFunctionSignature = int32_t(shared::SPSExecutorAddr);
using SPSRunAsIntFunctionSignature = int32_t(shared::SPSExecutorAddr, int32_t);
} // end namespace rt

namespace rt_alt {
LLVM_ABI extern const char *UnwindInfoManagerRegisterActionName;
LLVM_ABI extern const char *UnwindInfoManagerDeregisterActionName;
} // end namespace rt_alt
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_ORCRTBRIDGE_H
