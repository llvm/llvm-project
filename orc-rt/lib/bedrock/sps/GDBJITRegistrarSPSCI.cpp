//===- GDBJITRegistrarSPSCI.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for GDBJITRegistrar.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/GDBJITRegistrarSPSCI.h"
#include "orc-rt/support/sps/SPSAllocAction.h"

#include "orc-rt-internal/bedrock/GDBJITRegistrar.h"

using namespace orc_rt;

namespace orc_rt::sps_ci {

ORC_RT_SPS_ALLOC_ACTION(orc_rt_ci_aa_sps_GDBJITRegistrar_register,
                        (SPSExecutorAddrRange), &gdb_jit::registerObject)

ORC_RT_SPS_ALLOC_ACTION(orc_rt_ci_aa_sps_GDBJITRegistrar_deregister,
                        (SPSExecutorAddrRange), &gdb_jit::deregisterObject)

static std::pair<const char *, const void *>
    orc_rt_ci_GDBJITRegistrar_sps_interface[] = {
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_aa_sps_GDBJITRegistrar_register),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_aa_sps_GDBJITRegistrar_deregister)};

Error addGDBJITRegistrar(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_GDBJITRegistrar_sps_interface);
}

} // namespace orc_rt::sps_ci
