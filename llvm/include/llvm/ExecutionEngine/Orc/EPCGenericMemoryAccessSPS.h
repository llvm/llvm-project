//===- EPCGenericMemoryAccessSPS.h - SPS memory access ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binds EPCGenericMemoryAccess to the ORC runtime's SPS controller interface:
// a ProxySpec per operation, plus factories that resolve them and construct an
// instance.
//
// Each spec is built from the controller-interface descriptor for one of
// EPCGenericMemoryAccess's operations in Shared/SPSCI/MemoryAccessSPSCI.h,
// which supplies the wrapper name and wire signature. The specs are public so
// that clients can resolve the operations under non-default names, using
// recordProxy<Spec>(&P, Name) with lookupAndApply.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESSSPS_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESSSPS_H

#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/MemoryAccessSPSCI.h"
#include "llvm/Support/Compiler.h"

#include <memory>

namespace llvm::orc::sps {

using MemWriteUInt8sProxySpec = ProxySpec<rt::sps_ci::MemWriteUInt8s>;
using MemWriteUInt16sProxySpec = ProxySpec<rt::sps_ci::MemWriteUInt16s>;
using MemWriteUInt32sProxySpec = ProxySpec<rt::sps_ci::MemWriteUInt32s>;
using MemWriteUInt64sProxySpec = ProxySpec<rt::sps_ci::MemWriteUInt64s>;
using MemWritePointersProxySpec = ProxySpec<rt::sps_ci::MemWritePointers>;
using MemWriteBuffersProxySpec = ProxySpec<rt::sps_ci::MemWriteBuffers>;
using MemReadUInt8sProxySpec = ProxySpec<rt::sps_ci::MemReadUInt8s>;
using MemReadUInt16sProxySpec = ProxySpec<rt::sps_ci::MemReadUInt16s>;
using MemReadUInt32sProxySpec = ProxySpec<rt::sps_ci::MemReadUInt32s>;
using MemReadUInt64sProxySpec = ProxySpec<rt::sps_ci::MemReadUInt64s>;
using MemReadPointersProxySpec = ProxySpec<rt::sps_ci::MemReadPointers>;
using MemReadBuffersProxySpec = ProxySpec<rt::sps_ci::MemReadBuffers>;
using MemReadStringsProxySpec = ProxySpec<rt::sps_ci::MemReadStrings>;

/// Create an EPCGenericMemoryAccess that reaches the memory-access wrappers in
/// the given JITDylib via the runtime's SPS controller interface.
LLVM_ABI Expected<std::unique_ptr<EPCGenericMemoryAccess>>
createEPCGenericMemoryAccess(JITDylib &JD);

/// Create an EPCGenericMemoryAccess that reaches the memory-access wrappers in
/// ES's bootstrap JITDylib via the runtime's SPS controller interface.
LLVM_ABI Expected<std::unique_ptr<EPCGenericMemoryAccess>>
createEPCGenericMemoryAccess(ExecutionSession &ES);

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESSSPS_H
