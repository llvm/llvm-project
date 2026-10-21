//===-------------- SessionSPSCI.h - Session SPS CI -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for Session APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_SESSIONSPSCI_H
#define ORC_RT_BEDROCK_SPS_SESSIONSPSCI_H

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

ORC_RT_SPS_WRAPPER_DECL(orc_rt_ci_sps_Session_lookupInstanceSymbols)

namespace orc_rt::sps_ci {

/// Add the SPS controller interface to the Session APIs.
Error addSession(SimpleSymbolTable &ST) noexcept;

} // namespace orc_rt::sps_ci

#endif // ORC_RT_BEDROCK_SPS_SESSIONSPSCI_H
