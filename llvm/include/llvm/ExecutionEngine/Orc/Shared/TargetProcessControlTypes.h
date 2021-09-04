//===--- TargetProcessControlTypes.h -- Shared Core/TPC types ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TargetProcessControl types that are used by both the Orc and
// OrcTargetProcess libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_TARGETPROCESSCONTROLTYPES_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_TARGETPROCESSCONTROLTYPES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

#include <vector>

namespace llvm {
namespace orc {
namespace tpctypes {

template <typename T> struct UIntWrite {
  UIntWrite() = default;
  UIntWrite(JITTargetAddress Address, T Value)
      : Address(Address), Value(Value) {}

  JITTargetAddress Address = 0;
  T Value = 0;
};

/// Describes a write to a uint8_t.
using UInt8Write = UIntWrite<uint8_t>;

/// Describes a write to a uint16_t.
using UInt16Write = UIntWrite<uint16_t>;

/// Describes a write to a uint32_t.
using UInt32Write = UIntWrite<uint32_t>;

/// Describes a write to a uint64_t.
using UInt64Write = UIntWrite<uint64_t>;

/// Describes a write to a buffer.
/// For use with TargetProcessControl::MemoryAccess objects.
struct BufferWrite {
  BufferWrite() = default;
  BufferWrite(JITTargetAddress Address, StringRef Buffer)
      : Address(Address), Buffer(Buffer) {}

  JITTargetAddress Address = 0;
  StringRef Buffer;
};

/// A handle used to represent a loaded dylib in the target process.
using DylibHandle = JITTargetAddress;

using LookupResult = std::vector<JITTargetAddress>;

} // end namespace tpctypes

namespace shared {

template <typename T>
using SPSMemoryAccessUIntWrite = SPSTuple<SPSExecutorAddress, T>;

using SPSMemoryAccessUInt8Write = SPSMemoryAccessUIntWrite<uint8_t>;
using SPSMemoryAccessUInt16Write = SPSMemoryAccessUIntWrite<uint16_t>;
using SPSMemoryAccessUInt32Write = SPSMemoryAccessUIntWrite<uint32_t>;
using SPSMemoryAccessUInt64Write = SPSMemoryAccessUIntWrite<uint64_t>;

using SPSMemoryAccessBufferWrite =
    SPSTuple<SPSExecutorAddress, SPSSequence<char>>;

template <typename T>
class SPSSerializationTraits<SPSMemoryAccessUIntWrite<T>,
                             tpctypes::UIntWrite<T>> {
public:
  static size_t size(const tpctypes::UIntWrite<T> &W) {
    return SPSTuple<SPSExecutorAddress, T>::AsArgList::size(W.Address, W.Value);
  }

  static bool serialize(SPSOutputBuffer &OB, const tpctypes::UIntWrite<T> &W) {
    return SPSTuple<SPSExecutorAddress, T>::AsArgList::serialize(OB, W.Address,
                                                                 W.Value);
  }

  static bool deserialize(SPSInputBuffer &IB, tpctypes::UIntWrite<T> &W) {
    return SPSTuple<SPSExecutorAddress, T>::AsArgList::deserialize(
        IB, W.Address, W.Value);
  }
};

template <>
class SPSSerializationTraits<SPSMemoryAccessBufferWrite,
                             tpctypes::BufferWrite> {
public:
  static size_t size(const tpctypes::BufferWrite &W) {
    return SPSTuple<SPSExecutorAddress, SPSSequence<char>>::AsArgList::size(
        W.Address, W.Buffer);
  }

  static bool serialize(SPSOutputBuffer &OB, const tpctypes::BufferWrite &W) {
    return SPSTuple<SPSExecutorAddress,
                    SPSSequence<char>>::AsArgList ::serialize(OB, W.Address,
                                                              W.Buffer);
  }

  static bool deserialize(SPSInputBuffer &IB, tpctypes::BufferWrite &W) {
    return SPSTuple<SPSExecutorAddress,
                    SPSSequence<char>>::AsArgList ::deserialize(IB, W.Address,
                                                                W.Buffer);
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_TARGETPROCESSCONTROLTYPES_H
