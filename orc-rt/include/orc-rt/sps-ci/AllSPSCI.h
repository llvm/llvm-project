//===- AllSPSCI.h -- All SPS Controller Interface registrations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convenience header that includes all SPS Controller Interface headers and
// declares addAll.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SPS_CI_ALLSPSCI_H
#define ORC_RT_SPS_CI_ALLSPSCI_H

#include "orc-rt/SimpleSymbolTable.h"

namespace orc_rt::sps_ci {

/// Add all SPS interfaces to the controller interface.
Error addAll(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_SPS_CI_ALLSPSCI_H
