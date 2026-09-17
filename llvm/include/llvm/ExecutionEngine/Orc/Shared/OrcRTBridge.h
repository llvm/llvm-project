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

} // end namespace rt

namespace rt_alt {
LLVM_ABI extern const char *UnwindInfoManagerRegisterActionName;
LLVM_ABI extern const char *UnwindInfoManagerDeregisterActionName;
} // end namespace rt_alt
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_ORCRTBRIDGE_H
