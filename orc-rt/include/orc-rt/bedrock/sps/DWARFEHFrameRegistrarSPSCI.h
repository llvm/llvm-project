//===------------- DWARFEHFrameRegistrarSPSCI.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for DWARFEHFrameRegistrar.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_DWARFEHFRAMEREGISTRARSPSCI_H
#define ORC_RT_BEDROCK_SPS_DWARFEHFRAMEREGISTRARSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/sps/SPSAllocAction.h"

ORC_RT_SPS_ALLOC_ACTION_DECL(
    orc_rt_ci_aa_sps_DWARFEHFrameRegistrar_registerSection)
ORC_RT_SPS_ALLOC_ACTION_DECL(
    orc_rt_ci_aa_sps_DWARFEHFrameRegistrar_deregisterSection)

namespace orc_rt::sps_ci {

/// Add the DWARFEHFrameRegistrar SPS interface to the controller interface.
Error addDWARFEHFrameRegistrar(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_BEDROCK_SPS_DWARFEHFRAMEREGISTRARSPSCI_H
