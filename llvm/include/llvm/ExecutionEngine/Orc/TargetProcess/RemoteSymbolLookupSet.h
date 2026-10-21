//===- RemoteSymbolLookupSet.h - Executor-side lookup sets ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The executor-side representation of a symbol lookup set and its result, and
// the lookup set's SPS serialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_REMOTESYMBOLLOOKUPSET_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_REMOTESYMBOLLOOKUPSET_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolLookupSetSPS.h"

#include <optional>
#include <string>
#include <vector>

namespace llvm::orc {

struct RemoteSymbolLookupSetElement {
  std::string Name;
  bool Required;
};

using RemoteSymbolLookupSet = std::vector<RemoteSymbolLookupSetElement>;

/// The result of looking up a RemoteSymbolLookupSet: one entry per element of
/// the set, in the same order.
///
/// An empty optional indicates that a required symbol was not found. A missing
/// weakly referenced symbol is reported as a present optional holding a null
/// address.
using RemoteSymbolLookupResult = std::vector<std::optional<ExecutorAddr>>;

namespace shared {

template <>
class SPSSerializationTraits<SPSSymbolLookupSetElement,
                             RemoteSymbolLookupSetElement> {
public:
  static size_t size(const RemoteSymbolLookupSetElement &V) {
    return SPSArgList<SPSString, bool>::size(V.Name, V.Required);
  }

  static size_t serialize(SPSOutputBuffer &OB,
                          const RemoteSymbolLookupSetElement &V) {
    return SPSArgList<SPSString, bool>::serialize(OB, V.Name, V.Required);
  }

  static size_t deserialize(SPSInputBuffer &IB,
                            RemoteSymbolLookupSetElement &V) {
    return SPSArgList<SPSString, bool>::deserialize(IB, V.Name, V.Required);
  }
};

} // namespace shared

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_REMOTESYMBOLLOOKUPSET_H
