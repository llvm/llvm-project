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
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/Support/Memory.h"

#include <vector>

namespace llvm {
namespace orc {
namespace tpctypes {

enum WireProtectionFlags : uint8_t {
  WPF_None = 0,
  WPF_Read = 1U << 0,
  WPF_Write = 1U << 1,
  WPF_Exec = 1U << 2,
  LLVM_MARK_AS_BITMASK_ENUM(WPF_Exec)
};

/// Convert from sys::Memory::ProtectionFlags
inline WireProtectionFlags
toWireProtectionFlags(sys::Memory::ProtectionFlags PF) {
  WireProtectionFlags WPF = WPF_None;
  if (PF & sys::Memory::MF_READ)
    WPF |= WPF_Read;
  if (PF & sys::Memory::MF_WRITE)
    WPF |= WPF_Write;
  if (PF & sys::Memory::MF_EXEC)
    WPF |= WPF_Exec;
  return WPF;
}

inline sys::Memory::ProtectionFlags
fromWireProtectionFlags(WireProtectionFlags WPF) {
  int PF = 0;
  if (WPF & WPF_Read)
    PF |= sys::Memory::MF_READ;
  if (WPF & WPF_Write)
    PF |= sys::Memory::MF_WRITE;
  if (WPF & WPF_Exec)
    PF |= sys::Memory::MF_EXEC;
  return static_cast<sys::Memory::ProtectionFlags>(PF);
}

inline std::string getWireProtectionFlagsStr(WireProtectionFlags WPF) {
  std::string Result;
  Result += (WPF & WPF_Read) ? 'R' : '-';
  Result += (WPF & WPF_Write) ? 'W' : '-';
  Result += (WPF & WPF_Exec) ? 'X' : '-';
  return Result;
}

struct SegFinalizeRequest {
  WireProtectionFlags Prot;
  ExecutorAddress Addr;
  uint64_t Size;
  ArrayRef<char> Content;
};

using FinalizeRequest = std::vector<SegFinalizeRequest>;

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

class SPSMemoryProtectionFlags {};

using SPSSegFinalizeRequest =
    SPSTuple<SPSMemoryProtectionFlags, SPSExecutorAddress, uint64_t,
             SPSSequence<char>>;

using SPSFinalizeRequest = SPSSequence<SPSSegFinalizeRequest>;

template <typename T>
using SPSMemoryAccessUIntWrite = SPSTuple<SPSExecutorAddress, T>;

using SPSMemoryAccessUInt8Write = SPSMemoryAccessUIntWrite<uint8_t>;
using SPSMemoryAccessUInt16Write = SPSMemoryAccessUIntWrite<uint16_t>;
using SPSMemoryAccessUInt32Write = SPSMemoryAccessUIntWrite<uint32_t>;
using SPSMemoryAccessUInt64Write = SPSMemoryAccessUIntWrite<uint64_t>;

using SPSMemoryAccessBufferWrite =
    SPSTuple<SPSExecutorAddress, SPSSequence<char>>;

template <>
class SPSSerializationTraits<SPSMemoryProtectionFlags,
                             tpctypes::WireProtectionFlags> {
public:
  static size_t size(const tpctypes::WireProtectionFlags &WPF) {
    return SPSArgList<uint8_t>::size(static_cast<uint8_t>(WPF));
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const tpctypes::WireProtectionFlags &WPF) {
    return SPSArgList<uint8_t>::serialize(OB, static_cast<uint8_t>(WPF));
  }

  static bool deserialize(SPSInputBuffer &IB,
                          tpctypes::WireProtectionFlags &WPF) {
    uint8_t Val;
    if (!SPSArgList<uint8_t>::deserialize(IB, Val))
      return false;
    WPF = static_cast<tpctypes::WireProtectionFlags>(Val);
    return true;
  }
};

template <>
class SPSSerializationTraits<SPSSegFinalizeRequest,
                             tpctypes::SegFinalizeRequest> {
  using SFRAL = SPSSegFinalizeRequest::AsArgList;

public:
  static size_t size(const tpctypes::SegFinalizeRequest &SFR) {
    return SFRAL::size(SFR.Prot, SFR.Addr, SFR.Size, SFR.Content);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const tpctypes::SegFinalizeRequest &SFR) {
    return SFRAL::serialize(OB, SFR.Prot, SFR.Addr, SFR.Size, SFR.Content);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          tpctypes::SegFinalizeRequest &SFR) {
    return SFRAL::deserialize(IB, SFR.Prot, SFR.Addr, SFR.Size, SFR.Content);
  }
};

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

using SPSOrcTargetProcessAllocate = SPSExpected<SPSExecutorAddress>(uint64_t);
using SPSOrcTargetProcessFinalize = SPSError(SPSFinalizeRequest);
using SPSOrcTargetProcessDeallocate = SPSError(SPSExecutorAddress, uint64_t);

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_TARGETPROCESSCONTROLTYPES_H
