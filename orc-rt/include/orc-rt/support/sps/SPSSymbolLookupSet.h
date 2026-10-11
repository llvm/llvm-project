//===-- SPSSymbolLookupSet.h -- SPS-serialization for lookups ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPSSerialization for relevant types in SymbolLookupFlags.h and
// SymbolLookupSet.h.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_SPS_SPSSYMBOLLOOKUPSET_H
#define ORC_RT_SUPPORT_SPS_SPSSYMBOLLOOKUPSET_H

#include "orc-rt/support/SymbolLookupSet.h"
#include "orc-rt/support/sps/SimplePackedSerialization.h"

namespace orc_rt {

using SPSSymbolLookupSet = SPSSequence<SPSTuple<SPSString, bool>>;

/// SPS serialization for SymbolLookupFlags as a bool.
///
/// RequiredSymbol serializes as true, WeaklyReferencedSymbol as false. This
/// matches the wire format of llvm::orc::RemoteSymbolLookupSetElement, which
/// uses a bool 'Required' field.
template <> class SPSSerializationTraits<bool, SymbolLookupFlags> {
public:
  static size_t size(SymbolLookupFlags) { return sizeof(bool); }

  static bool serialize(SPSOutputBuffer &OB, SymbolLookupFlags F) {
    return SPSSerializationTraits<bool, bool>::serialize(
        OB, F == SymbolLookupFlags::RequiredSymbol);
  }

  static bool deserialize(SPSInputBuffer &IB, SymbolLookupFlags &F) {
    bool Required;
    if (!SPSSerializationTraits<bool, bool>::deserialize(IB, Required))
      return false;
    F = Required ? SymbolLookupFlags::RequiredSymbol
                 : SymbolLookupFlags::WeaklyReferencedSymbol;
    return true;
  }
};

/// Trivial SymbolLookupSet -> SPSSequence<SPSElementTagT> serialization.
template <typename SPSElementTagT>
class TrivialSPSSequenceSerialization<SPSElementTagT, SymbolLookupSet> {
public:
  static constexpr bool available = true;
};

/// Trivial SPSSequence<SPSElementTagT> -> SymbolLookupSet deserialization.
template <typename SPSElementTagT>
class TrivialSPSSequenceDeserialization<SPSElementTagT, SymbolLookupSet> {
public:
  static constexpr bool available = true;

  using element_type = SymbolLookupSet::value_type;

  static void reserve(SymbolLookupSet &S, uint64_t Size) { S.reserve(Size); }
  static bool append(SymbolLookupSet &S, element_type E) {
    S.push_back(std::move(E));
    return true;
  }
};

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

#endif // ORC_RT_SUPPORT_SPS_SPSSYMBOLLOOKUPSET_H
