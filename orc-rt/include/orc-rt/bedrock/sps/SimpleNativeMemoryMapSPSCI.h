//===------- SimpleNativeMemoryMapSPSCI.h -- SNMM SPS CI --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for SimpleNativeMemoryMap.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_SIMPLENATIVEMEMORYMAPSPSCI_H
#define ORC_RT_BEDROCK_SPS_SIMPLENATIVEMEMORYMAPSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_SimpleNativeMemoryMap_reserve)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_SimpleNativeMemoryMap_releaseMultiple)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_SimpleNativeMemoryMap_initialize)
ORC_RT_SPS_WRAPPER_DECL(
    orc_rt_ci_sps_SimpleNativeMemoryMap_deinitializeMultiple)

namespace orc_rt::sps_ci {

/// Add the SimpleNativeMemoryMap SPS interface to the controller interface.
Error addSimpleNativeMemoryMap(SimpleSymbolTable &ST);

} // namespace orc_rt::sps_ci

#endif // ORC_RT_BEDROCK_SPS_SIMPLENATIVEMEMORYMAPSPSCI_H
