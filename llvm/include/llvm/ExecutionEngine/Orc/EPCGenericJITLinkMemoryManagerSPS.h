//===- EPCGenericJITLinkMemoryManagerSPS.h - SPS mem manager ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Factories that build an EPCGenericJITLinkMemoryManager over the ORC runtime's
// SPS controller interface.
//
// The bindings and their ProxySpecs are shared with the other drivers of the
// runtime's memory manager; see SimpleMemoryMapSPS.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGERSPS_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGERSPS_H

#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/SimpleMemoryMapSPS.h"
#include "llvm/Support/Compiler.h"

#include <memory>

namespace llvm::orc::sps {

/// Create an EPCGenericJITLinkMemoryManager for the ORC runtime's
/// SimpleNativeMemoryMap interface, resolving its symbols in the given
/// JITDylib.
LLVM_ABI Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(JITDylib &JD);

/// Create an EPCGenericJITLinkMemoryManager for the ORC runtime's
/// SimpleNativeMemoryMap interface, resolving its symbols in the given
/// ExecutionSession's bootstrap JITDylib.
LLVM_ABI Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(ExecutionSession &ES);

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGERSPS_H
