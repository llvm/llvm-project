//===------ OrcRTBridge.cpp - Executor functions for bootstrap -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

namespace llvm {
namespace orc {
namespace rt {

const SymbolNameSpec RegisterEHFrameSectionAllocActionName =
    SymbolNameSpec::verbatim("llvm_orc_registerEHFrameAllocAction");
const SymbolNameSpec DeregisterEHFrameSectionAllocActionName =
    SymbolNameSpec::verbatim("llvm_orc_deregisterEHFrameAllocAction");

const SymbolNameSpec RegisterJITLoaderGDBAllocActionName =
    SymbolNameSpec::c("orc_rt_ci_aa_sps_GDBJITRegistrar_register");
const SymbolNameSpec DeregisterJITLoaderGDBAllocActionName =
    SymbolNameSpec::c("orc_rt_ci_aa_sps_GDBJITRegistrar_deregister");

const SymbolNameSpec DispatchName =
    SymbolNameSpec::verbatim("__orc_rt_jit_dispatch");
const SymbolNameSpec DispatchCtxName =
    SymbolNameSpec::verbatim("__orc_rt_jit_dispatch_ctx");

const MachOUnwindInfoRegistrarSymbolNames
    orc_rt_MachOUnwindInfoRegistrarSPSSymbols = {
        SymbolNameSpec::c(
            "orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_registerSections"),
        SymbolNameSpec::c(
            "orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_deregisterSections")};

} // end namespace rt
namespace rt_alt {
const SymbolNameSpec UnwindInfoManagerRegisterActionName =
    SymbolNameSpec::verbatim("orc_rt_alt_UnwindInfoManager_register");
const SymbolNameSpec UnwindInfoManagerDeregisterActionName =
    SymbolNameSpec::verbatim("orc_rt_alt_UnwindInfoManager_deregister");

} // end namespace rt_alt
} // end namespace orc
} // end namespace llvm
