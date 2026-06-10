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

#ifndef ORC_RT_SPS_CI_CALLSPSCI_H
#define ORC_RT_SPS_CI_CALLSPSCI_H

#include "orc-rt/SimpleSymbolTable.h"

namespace orc_rt::sps_ci {

/// Add the function callers SPS interface (orc_rt_ci_sps_call*) to the
/// controller interface.
Error addCall(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_SPS_CI_CALLSPSCI_H
