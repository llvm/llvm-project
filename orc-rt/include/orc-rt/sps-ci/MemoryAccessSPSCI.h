//===-- MemoryAccessSPSCI.h -- MemoryAccess SPS CI --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for MemoryAccess.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SPS_CI_MEMORYACCESSSPSCI_H
#define ORC_RT_SPS_CI_MEMORYACCESSSPSCI_H

#include "orc-rt/SimpleSymbolTable.h"

namespace orc_rt::sps_ci {

/// Add the MemoryAccess SPS interface to the controller interface.
Error addMemoryAccess(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_SPS_CI_MEMORYACCESSSPSCI_H
