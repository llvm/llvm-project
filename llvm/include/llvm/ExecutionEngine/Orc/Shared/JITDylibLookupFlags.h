//===- JITDylibLookupFlags.h - Flags for JITDylib lookups -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JITDylibLookupFlags and its SPS serialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_JITDYLIBLOOKUPFLAGS_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_JITDYLIBLOOKUPFLAGS_H

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

namespace llvm::orc {

/// Lookup flags that apply to each dylib in the search order for a lookup.
///
/// If MatchExportedSymbolsOnly is used (the default) for a given dylib, then
/// only symbols in that dylib's interface will be searched. If MatchAllSymbols
/// is used then symbols with hidden visibility will match as well.
enum class JITDylibLookupFlags { MatchExportedSymbolsOnly, MatchAllSymbols };

namespace shared {

/// SPS tag for JITDylibLookupFlags. Serialized as a bool: true for
/// MatchAllSymbols, false for MatchExportedSymbolsOnly.
struct SPSJITDylibLookupFlags;

template <>
class SPSSerializationTraits<SPSJITDylibLookupFlags, JITDylibLookupFlags> {
public:
  static size_t size(const JITDylibLookupFlags &F) {
    return SPSArgList<bool>::size(F == JITDylibLookupFlags::MatchAllSymbols);
  }

  static bool serialize(SPSOutputBuffer &OB, const JITDylibLookupFlags &F) {
    return SPSArgList<bool>::serialize(
        OB, F == JITDylibLookupFlags::MatchAllSymbols);
  }

  static bool deserialize(SPSInputBuffer &IB, JITDylibLookupFlags &F) {
    bool MatchAll;
    if (!SPSArgList<bool>::deserialize(IB, MatchAll))
      return false;
    F = MatchAll ? JITDylibLookupFlags::MatchAllSymbols
                 : JITDylibLookupFlags::MatchExportedSymbolsOnly;
    return true;
  }
};

} // namespace shared
} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_JITDYLIBLOOKUPFLAGS_H
