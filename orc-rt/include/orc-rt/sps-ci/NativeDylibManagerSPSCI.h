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

#ifndef ORC_RT_SPS_CI_NATIVEDYLIBMANAGERSPSCI_H
#define ORC_RT_SPS_CI_NATIVEDYLIBMANAGERSPSCI_H

#include "orc-rt/SimpleSymbolTable.h"

namespace orc_rt::sps_ci {

/// Add the NativeDylibManager SPS interface to the controller interface.
Error addNativeDylibManager(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_SPS_CI_NATIVEDYLIBMANAGERSPSCI_H
