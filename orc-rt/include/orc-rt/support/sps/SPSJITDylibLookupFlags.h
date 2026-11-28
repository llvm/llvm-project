//===-- SPSJITDylibLookupFlags.h -- SPS-serialization for flags -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS serialization for JITDylibLookupFlags.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_SPS_SPSJITDYLIBLOOKUPFLAGS_H
#define ORC_RT_SUPPORT_SPS_SPSJITDYLIBLOOKUPFLAGS_H

#include "orc-rt/support/JITDylibLookupFlags.h"
#include "orc-rt/support/sps/SimplePackedSerialization.h"

namespace orc_rt {

/// SPS tag for JITDylibLookupFlags.
///
/// Serialized as a bool: MatchAllSymbols as true, MatchExportedSymbolsOnly as
/// false.
struct SPSJITDylibLookupFlags;

template <>
class SPSSerializationTraits<SPSJITDylibLookupFlags, JITDylibLookupFlags> {
public:
  static size_t size(JITDylibLookupFlags) { return sizeof(bool); }

  static bool serialize(SPSOutputBuffer &OB, JITDylibLookupFlags F) {
    return SPSSerializationTraits<bool, bool>::serialize(
        OB, F == JITDylibLookupFlags::MatchAllSymbols);
  }

  static bool deserialize(SPSInputBuffer &IB, JITDylibLookupFlags &F) {
    bool MatchAll;
    if (!SPSSerializationTraits<bool, bool>::deserialize(IB, MatchAll))
      return false;
    F = MatchAll ? JITDylibLookupFlags::MatchAllSymbols
                 : JITDylibLookupFlags::MatchExportedSymbolsOnly;
    return true;
  }
};

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_SPS_SPSJITDYLIBLOOKUPFLAGS_H
