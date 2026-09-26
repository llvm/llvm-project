//===- DWARFEHFrameRegistrarSPSCI.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for DWARFEHFrameRegistrar.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/DWARFEHFrameRegistrarSPSCI.h"
#include "orc-rt-internal/bedrock/sys/posix/DWARFEHFrameRegistrar.h"

namespace orc_rt::sps_ci {

ORC_RT_SPS_ALLOC_ACTION_IMPL(
    orc_rt_ci_aa_sps_DWARFEHFrameRegistrar_registerSection,
    (SPSExecutorAddrRange), &sys::posix::DWARFEHFrameRegistrar::registerSection)

ORC_RT_SPS_ALLOC_ACTION_IMPL(
    orc_rt_ci_aa_sps_DWARFEHFrameRegistrar_deregisterSection,
    (SPSExecutorAddrRange),
    &sys::posix::DWARFEHFrameRegistrar::deregisterSection)

static std::pair<SymbolNameSpec, const void *>
    orc_rt_ci_DWARFEHFrameRegistrar_sps_interface[] = {
        ORC_RT_SYMTAB_C_PAIR(
            orc_rt_ci_aa_sps_DWARFEHFrameRegistrar_registerSection),
        ORC_RT_SYMTAB_C_PAIR(
            orc_rt_ci_aa_sps_DWARFEHFrameRegistrar_deregisterSection)};

Error addDWARFEHFrameRegistrar(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_DWARFEHFrameRegistrar_sps_interface);
}

} // namespace orc_rt::sps_ci
