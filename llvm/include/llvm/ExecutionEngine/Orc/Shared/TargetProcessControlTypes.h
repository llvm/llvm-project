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
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_TARGETPROCESSCONTROLTYPES_H
