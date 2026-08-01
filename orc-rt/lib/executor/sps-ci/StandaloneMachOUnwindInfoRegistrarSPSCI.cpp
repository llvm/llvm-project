//===- StandaloneMachOUnwindInfoRegistrarSPSCI.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for
// StandaloneMachOUnwindInfoRegistrar.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/StandaloneMachOUnwindInfoRegistrarSPSCI.h"
#include "orc-rt/SPSAllocAction.h"
#include "orc-rt/StandaloneMachOUnwindInfoRegistrar.h"

namespace orc_rt::sps_ci {

ORC_RT_SPS_ALLOC_ACTION(
    orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_registerSections,
    (SPSSequence<SPSExecutorAddrRange>, SPSExecutorAddr, SPSExecutorAddrRange,
     SPSExecutorAddrRange),
    &StandaloneMachOUnwindInfoRegistrar::registerSections)

ORC_RT_SPS_ALLOC_ACTION(
    orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_deregisterSections,
    (SPSSequence<SPSExecutorAddrRange>),
    &StandaloneMachOUnwindInfoRegistrar::deregisterSections)

static std::pair<const char *, const void *>
    orc_rt_ci_StandaloneMachOUnwindInfoRegistrar_sps_interface[] = {
        ORC_RT_SYMTAB_PAIR(
            orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_registerSections),
        ORC_RT_SYMTAB_PAIR(
            orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_deregisterSections)};

Error addStandaloneMachOUnwindInfoRegistrar(SimpleSymbolTable &ST) {
  return ST.addUnique(
      orc_rt_ci_StandaloneMachOUnwindInfoRegistrar_sps_interface);
}

} // namespace orc_rt::sps_ci
