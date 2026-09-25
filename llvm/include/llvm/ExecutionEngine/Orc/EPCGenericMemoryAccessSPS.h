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
// Each spec pairs one of EPCGenericMemoryAccess's proxies with its
// controller-interface descriptor in Shared/SPSCI/MemoryAccessSPSCI.h, which
// supplies the wrapper name and wire signature. The specs are public so that
// clients can resolve the operations under non-default names, using
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

using MemWriteUInt8sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::WriteUInt8sProxy,
              rt::sps_ci::MemWriteUInt8s>;
using MemWriteUInt16sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::WriteUInt16sProxy,
              rt::sps_ci::MemWriteUInt16s>;
using MemWriteUInt32sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::WriteUInt32sProxy,
              rt::sps_ci::MemWriteUInt32s>;
using MemWriteUInt64sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::WriteUInt64sProxy,
              rt::sps_ci::MemWriteUInt64s>;
using MemWritePointersProxySpec =
    ProxySpec<EPCGenericMemoryAccess::WritePointersProxy,
              rt::sps_ci::MemWritePointers>;
using MemWriteBuffersProxySpec =
    ProxySpec<EPCGenericMemoryAccess::WriteBuffersProxy,
              rt::sps_ci::MemWriteBuffers>;
using MemReadUInt8sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::ReadUInt8sProxy,
              rt::sps_ci::MemReadUInt8s>;
using MemReadUInt16sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::ReadUInt16sProxy,
              rt::sps_ci::MemReadUInt16s>;
using MemReadUInt32sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::ReadUInt32sProxy,
              rt::sps_ci::MemReadUInt32s>;
using MemReadUInt64sProxySpec =
    ProxySpec<EPCGenericMemoryAccess::ReadUInt64sProxy,
              rt::sps_ci::MemReadUInt64s>;
using MemReadPointersProxySpec =
    ProxySpec<EPCGenericMemoryAccess::ReadPointersProxy,
              rt::sps_ci::MemReadPointers>;
using MemReadBuffersProxySpec =
    ProxySpec<EPCGenericMemoryAccess::ReadBuffersProxy,
              rt::sps_ci::MemReadBuffers>;
using MemReadStringsProxySpec =
    ProxySpec<EPCGenericMemoryAccess::ReadStringsProxy,
              rt::sps_ci::MemReadStrings>;

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
