//===------------ CallSPSCI.h - Function call SPS CI ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for function callers.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_CALLSPSCI_H
#define ORC_RT_BEDROCK_SPS_CALLSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_call_void_void)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_call_main)

namespace orc_rt::sps_ci {

/// Add the function callers SPS interface (orc_rt_ci_sps_call*) to the
/// controller interface.
Error addCall(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_BEDROCK_SPS_CALLSPSCI_H
