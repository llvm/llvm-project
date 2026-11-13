//===- LookupInJITDylibSPS.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Resolve symbols via JITDylib lookup in the controller.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_LOOKUPINJITDYLIBSPS_H
#define ORC_RT_BEDROCK_SPS_LOOKUPINJITDYLIBSPS_H

#include "orc-rt/support/Error.h"
#include "orc-rt/support/JITDylibLookupFlags.h"
#include "orc-rt/support/SymbolLookupSet.h"
#include "orc-rt/support/move_only_function.h"

namespace orc_rt {

class Session;

namespace sps {

/// Lookup the given symbols in the JITDylib with the given base address.
///
/// JDLookupFlags determines whether the lookup matches all of the JITDylib's
/// symbols, or only its exported symbols.
///
/// On success the result has one entry per element of LS, in the same order.
/// A controller result of any other length is reported as an error, as are
/// failures of the lookup in the controller and failures to call the
/// controller.
void lookupInJITDylib(
    move_only_function<void(Expected<SymbolLookupResult>)> OnComplete,
    Session &S, const void *JITDylibKeyAddr, JITDylibLookupFlags JDLookupFlags,
    const SymbolLookupSet &LS) noexcept;

} // namespace sps
} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SPS_LOOKUPINJITDYLIBSPS_H
