//===-- SPSSymbolLookupResult.h -- SPS-serialize lookup results -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPSSerialization for SymbolLookupResult.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_SPS_SPSSYMBOLLOOKUPRESULT_H
#define ORC_RT_SUPPORT_SPS_SPSSYMBOLLOOKUPRESULT_H

#include "orc-rt/support/SymbolLookupResult.h"
#include "orc-rt/support/sps/SimplePackedSerialization.h"

namespace orc_rt {

using SPSSymbolLookupResult = SPSSequence<SPSOptional<SPSExecutorAddr>>;

/// Trivial SymbolLookupResult -> SPSSequence<SPSElementTagT> serialization.
template <typename SPSElementTagT>
class TrivialSPSSequenceSerialization<SPSElementTagT, SymbolLookupResult> {
public:
  static constexpr bool available = true;
};

/// Trivial SPSSequence<SPSElementTagT> -> SymbolLookupResult deserialization.
template <typename SPSElementTagT>
class TrivialSPSSequenceDeserialization<SPSElementTagT, SymbolLookupResult> {
public:
  static constexpr bool available = true;

  using element_type = SymbolLookupResult::value_type;

  static void reserve(SymbolLookupResult &R, uint64_t Size) { R.reserve(Size); }
  static bool append(SymbolLookupResult &R, element_type E) {
    R.push_back(std::move(E));
    return true;
  }
};

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_SPS_SPSSYMBOLLOOKUPRESULT_H
