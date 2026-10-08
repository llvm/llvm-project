//===-- NativeDylibManagerSPSCI.h -- NativeDylibManager SPS CI --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for NativeDylibManager.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_NATIVEDYLIBMANAGERSPSCI_H
#define ORC_RT_BEDROCK_SPS_NATIVEDYLIBMANAGERSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_NativeDylibManager_load)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_NativeDylibManager_lookup)

namespace orc_rt::sps_ci {

/// Add the NativeDylibManager SPS interface to the controller interface.
Error addNativeDylibManager(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_BEDROCK_SPS_NATIVEDYLIBMANAGERSPSCI_H
