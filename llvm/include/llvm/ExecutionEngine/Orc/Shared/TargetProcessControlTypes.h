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

/// Either a uint8_t array or a uint8_t*.
union CWrapperFunctionResultData {
  uint8_t Value[8];
  uint8_t *ValuePtr;
};

/// C ABI compatible wrapper function result.
///
/// This can be safely returned from extern "C" functions, but should be used
/// to construct a WrapperFunctionResult for safety.
struct CWrapperFunctionResult {
  uint64_t Size;
  CWrapperFunctionResultData Data;
  void (*Destroy)(CWrapperFunctionResultData Data, uint64_t Size);
};

/// C++ wrapper function result: Same as CWrapperFunctionResult but
/// auto-releases memory.
class WrapperFunctionResult {
public:
  /// Create a default WrapperFunctionResult.
  WrapperFunctionResult() { zeroInit(R); }

  /// Create a WrapperFunctionResult from a CWrapperFunctionResult. This
  /// instance takes ownership of the result object and will automatically
  /// call the Destroy member upon destruction.
  WrapperFunctionResult(CWrapperFunctionResult R) : R(R) {}

  WrapperFunctionResult(const WrapperFunctionResult &) = delete;
  WrapperFunctionResult &operator=(const WrapperFunctionResult &) = delete;

  WrapperFunctionResult(WrapperFunctionResult &&Other) {
    zeroInit(R);
    std::swap(R, Other.R);
  }

  WrapperFunctionResult &operator=(WrapperFunctionResult &&Other) {
    CWrapperFunctionResult Tmp;
    zeroInit(Tmp);
    std::swap(Tmp, Other.R);
    std::swap(R, Tmp);
    return *this;
  }

  ~WrapperFunctionResult() {
    if (R.Destroy)
      R.Destroy(R.Data, R.Size);
  }

  /// Relinquish ownership of and return the CWrapperFunctionResult.
  CWrapperFunctionResult release() {
    CWrapperFunctionResult Tmp;
    zeroInit(Tmp);
    std::swap(R, Tmp);
    return Tmp;
  }

  /// Get an ArrayRef covering the data in the result.
  ArrayRef<uint8_t> getData() const {
    if (R.Size <= 8)
      return ArrayRef<uint8_t>(R.Data.Value, R.Size);
    return ArrayRef<uint8_t>(R.Data.ValuePtr, R.Size);
  }

  /// Create a WrapperFunctionResult from the given integer, provided its
  /// size is no greater than 64 bits.
  template <typename T,
            typename _ = std::enable_if_t<std::is_integral<T>::value &&
                                          sizeof(T) <= sizeof(uint64_t)>>
  static WrapperFunctionResult from(T Value) {
    CWrapperFunctionResult R;
    R.Size = sizeof(T);
    memcpy(&R.Data.Value, Value, R.Size);
    R.Destroy = nullptr;
    return R;
  }

  /// Create a WrapperFunctionResult from the given string.
  static WrapperFunctionResult from(StringRef S);

  /// Always free Data.ValuePtr by calling free on it.
  static void destroyWithFree(CWrapperFunctionResultData Data, uint64_t Size);

  /// Always free Data.ValuePtr by calling delete[] on it.
  static void destroyWithDeleteArray(CWrapperFunctionResultData Data,
                                     uint64_t Size);

private:
  static void zeroInit(CWrapperFunctionResult &R) {
    R.Size = 0;
    R.Data.ValuePtr = nullptr;
    R.Destroy = nullptr;
  }

  CWrapperFunctionResult R;
};

} // end namespace tpctypes
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_TARGETPROCESSCONTROLTYPES_H
