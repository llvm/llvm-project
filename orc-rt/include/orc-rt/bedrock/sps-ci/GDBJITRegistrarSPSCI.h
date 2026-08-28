//===------------- GDBJITRegistrarSPSCI.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for GDBJITRegistrar.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SPS_CI_GDBJITREGISTRARSPSCI_H
#define ORC_RT_SPS_CI_GDBJITREGISTRARSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"

namespace orc_rt::sps_ci {

/// Add the GDBJITRegistrar SPS interface to the controller interface.
Error addGDBJITRegistrar(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_SPS_CI_GDBJITREGISTRARSPSCI_H
