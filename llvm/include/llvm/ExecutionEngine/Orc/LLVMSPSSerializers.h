//===-- LLVMSPSSerializers.h - SPS serialization for LLVM types -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Serialization for common LLVM types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LLVMSPSSERIALIZERS_H
#define LLVM_EXECUTIONENGINE_ORC_LLVMSPSSERIALIZERS_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

namespace llvm {
namespace orc {
namespace shared {

template <typename SPSValueT, typename ValueT>
class SPSSerializationTraits<SPSSequence<SPSTuple<SPSString, SPSValueT>>,
                             StringMap<ValueT>> {
public:
  static size_t size(const StringMap<ValueT> &M) {
    size_t Sz = SPSArgList<uint64_t>::size(static_cast<uint64_t>(M.size()));
    for (auto &E : M)
      Sz += SPSArgList<SPSString, SPSValueT>::size(E.first(), E.second);
    return Sz;
  }

  static bool serialize(SPSOutputBuffer &OB, const StringMap<ValueT> &M) {
    if (!SPSArgList<uint64_t>::serialize(OB, static_cast<uint64_t>(M.size())))
      return false;

    for (auto &E : M)
      if (!SPSArgList<SPSString, SPSValueT>::serialize(OB, E.first(), E.second))
        return false;

    return true;
  }

  static bool deserialize(SPSInputBuffer &IB, StringMap<ValueT> &M) {
    uint64_t Size;
    assert(M.empty() && "M already contains elements");

    if (!SPSArgList<uint64_t>::deserialize(IB, Size))
      return false;

    while (Size--) {
      StringRef S;
      ValueT V;
      if (!SPSArgList<SPSString, SPSValueT>::deserialize(IB, S, V))
        return false;
      if (!M.insert(std::make_pair(S, V)).second)
        return false;
    }

    return true;
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LLVMSPSSERIALIZERS_H
