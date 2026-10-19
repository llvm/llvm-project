//===--------- StandaloneMachOUnwindInfoRegistrarSPSCI.h --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for
// StandaloneMachOUnwindInfoRegistrar.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_STANDALONEMACHOUNWINDINFOREGISTRARSPSCI_H
#define ORC_RT_BEDROCK_SPS_STANDALONEMACHOUNWINDINFOREGISTRARSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/sps/SPSAllocAction.h"

ORC_RT_SPS_ALLOC_ACTION_DECL(
    orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_registerSections)

ORC_RT_SPS_ALLOC_ACTION_DECL(
    orc_rt_ci_aa_sps_MachOUnwindInfoRegistrar_deregisterSections)

namespace orc_rt::sps_ci {

/// Add the StandaloneMachOUnwindInfoRegistrar SPS interface to the
/// controller interface.
Error addStandaloneMachOUnwindInfoRegistrar(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_BEDROCK_SPS_STANDALONEMACHOUNWINDINFOREGISTRARSPSCI_H
