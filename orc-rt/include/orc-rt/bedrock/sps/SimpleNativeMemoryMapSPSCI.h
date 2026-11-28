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

ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_SimpleNativeMemoryMap_create)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_SimpleNativeMemoryMap_reserve)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_SimpleNativeMemoryMap_releaseMultiple)
ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_SimpleNativeMemoryMap_initialize)
ORC_RT_SPS_WRAPPER_DECL(
    orc_rt_ci_sps_SimpleNativeMemoryMap_deinitializeMultiple)

namespace orc_rt::sps::ci {

/// Add the SimpleNativeMemoryMap SPS interface to the controller interface.
///
/// The interface consists of a create function, which constructs a new
/// SimpleNativeMemoryMap, adds it to the calling Session as a Service, and
/// returns its address, plus the memory operations, which take an instance
/// address as their first argument.
///
/// Adding this interface allows the controller to create memory managers
/// that map JIT'd code and data using the host's native virtual memory APIs.
Error addSimpleNativeMemoryMap(SimpleSymbolTable &ST);

} // namespace orc_rt::sps::ci

#endif // ORC_RT_BEDROCK_SPS_SIMPLENATIVEMEMORYMAPSPSCI_H
