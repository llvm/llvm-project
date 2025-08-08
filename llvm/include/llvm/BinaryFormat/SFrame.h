//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains data-structure definitions and constants to support
/// unwinding based on .sframe sections.  This only supports SFRAME_VERSION_2
/// as described at https://sourceware.org/binutils/docs/sframe-spec.html
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_SFRAME_H
#define LLVM_BINARYFORMAT_SFRAME_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Endian.h"

namespace llvm {

template <typename T> struct EnumEntry;

namespace sframe {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

constexpr uint16_t Magic = 0xdee2;

enum class Version : uint8_t {
#define HANDLE_SFRAME_VERSION(CODE, NAME) NAME = CODE,
#include "llvm/BinaryFormat/SFrameConstants.def"
};

enum class Flags : uint8_t {
#define HANDLE_SFRAME_FLAG(CODE, NAME) NAME = CODE,
#include "llvm/BinaryFormat/SFrameConstants.def"
  V2AllFlags = FDESorted | FramePointer | FDEFuncStartPCRel,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/0xff),
};

enum class ABI : uint8_t {
#define HANDLE_SFRAME_ABI(CODE, NAME) NAME = CODE,
#include "llvm/BinaryFormat/SFrameConstants.def"
};

/// SFrame FRE Types. Bits 0-3 of FuncDescEntry.Info.
enum class FREType : uint8_t {
#define HANDLE_SFRAME_FRE_TYPE(CODE, NAME) NAME = CODE,
#include "llvm/BinaryFormat/SFrameConstants.def"
};

/// SFrame FDE Types. Bit 4 of FuncDescEntry.Info.
enum class FDEType : uint8_t {
#define HANDLE_SFRAME_FDE_TYPE(CODE, NAME) NAME = CODE,
#include "llvm/BinaryFormat/SFrameConstants.def"
};

/// Speficies key used for signing return addresses. Bit 5 of
/// FuncDescEntry.Info.
enum class AArch64PAuthKey : uint8_t {
#define HANDLE_SFRAME_AARCH64_PAUTH_KEY(CODE, NAME) NAME = CODE,
#include "llvm/BinaryFormat/SFrameConstants.def"
};

/// Size of stack offsets. Bits 6-7 of FREInfo.Info.
enum class FREOffset : uint8_t {
#define HANDLE_SFRAME_FRE_OFFSET(CODE, NAME) NAME = CODE,
#include "llvm/BinaryFormat/SFrameConstants.def"
};

/// Stack frame base register. Bit 0 of FREInfo.Info.
enum class BaseReg : uint8_t {
  FP = 0,
  SP = 1,
};

namespace detail {
template <typename T, endianness E>
using packed =
    support::detail::packed_endian_specific_integral<T, E, support::unaligned>;
}

template <endianness E> struct Preamble {
  detail::packed<uint16_t, E> Magic;
  detail::packed<enum Version, E> Version;
  detail::packed<enum Flags, E> Flags;
};

template <endianness E> struct Header {
  struct Preamble<E> Preamble;
  detail::packed<ABI, E> ABIArch;
  detail::packed<int8_t, E> CFAFixedFPOffset;
  detail::packed<int8_t, E> CFAFixedRAOffset;
  detail::packed<uint8_t, E> AuxHdrLen;
  detail::packed<uint32_t, E> NumFDEs;
  detail::packed<uint32_t, E> NumFREs;
  detail::packed<uint32_t, E> FRELen;
  detail::packed<uint32_t, E> FDEOff;
  detail::packed<uint32_t, E> FREOff;
};

template <endianness E> struct FuncDescEntry {
  detail::packed<int32_t, E> StartAddress;
  detail::packed<uint32_t, E> Size;
  detail::packed<uint32_t, E> StartFREOff;
  detail::packed<uint32_t, E> NumFREs;
  detail::packed<uint8_t, E> Info;
  detail::packed<uint8_t, E> RepSize;
  detail::packed<uint16_t, E> Padding2;

  uint8_t getPAuthKey() const { return (Info >> 5) & 1; }
  FDEType getFDEType() const { return static_cast<FDEType>((Info >> 4) & 1); }
  FREType getFREType() const { return static_cast<FREType>(Info & 0xf); }
  void setPAuthKey(uint8_t P) { setFuncInfo(P, getFDEType(), getFREType()); }
  void setFDEType(FDEType D) { setFuncInfo(getPAuthKey(), D, getFREType()); }
  void setFREType(FREType R) { setFuncInfo(getPAuthKey(), getFDEType(), R); }
  void setFuncInfo(uint8_t PAuthKey, FDEType FDE, FREType FRE) {
    Info = ((PAuthKey & 1) << 5) | ((static_cast<uint8_t>(FDE) & 1) << 4) |
           (static_cast<uint8_t>(FRE) & 0xf);
  }
};

template <endianness E> struct FREInfo {
  detail::packed<uint8_t, E> Info;

  bool isReturnAddressSigned() const { return Info >> 7; }
  FREOffset getOffsetSize() const {
    return static_cast<FREOffset>((Info >> 5) & 3);
  }
  uint8_t getOffsetCount() const { return (Info >> 1) & 0xf; }
  BaseReg getBaseRegister() const { return static_cast<BaseReg>(Info & 1); }
  void setReturnAddressSigned(bool RA) {
    setFREInfo(RA, getOffsetSize(), getOffsetCount(), getBaseRegister());
  }
  void setOffsetSize(FREOffset Sz) {
    setFREInfo(isReturnAddressSigned(), Sz, getOffsetCount(),
               getBaseRegister());
  }
  void setOffsetCount(uint8_t N) {
    setFREInfo(isReturnAddressSigned(), getOffsetSize(), N, getBaseRegister());
  }
  void setBaseRegister(BaseReg Reg) {
    setFREInfo(isReturnAddressSigned(), getOffsetSize(), getOffsetCount(), Reg);
  }
  void setFREInfo(bool RA, FREOffset Sz, uint8_t N, BaseReg Reg) {
    Info = ((RA & 1) << 7) | ((static_cast<uint8_t>(Sz) & 3) << 5) |
           ((N & 0xf) << 1) | (static_cast<uint8_t>(Reg) & 1);
  }
};

template <typename T, endianness E> struct FrameRowEntry {
  detail::packed<T, E> StartAddress;
  FREInfo<E> Info;
};

template <endianness E> using FrameRowEntryAddr1 = FrameRowEntry<uint8_t, E>;
template <endianness E> using FrameRowEntryAddr2 = FrameRowEntry<uint16_t, E>;
template <endianness E> using FrameRowEntryAddr4 = FrameRowEntry<uint32_t, E>;

LLVM_ABI ArrayRef<EnumEntry<Version>> getVersions();
LLVM_ABI ArrayRef<EnumEntry<Flags>> getFlags();
LLVM_ABI ArrayRef<EnumEntry<ABI>> getABIs();
LLVM_ABI ArrayRef<EnumEntry<FREType>> getFRETypes();
LLVM_ABI ArrayRef<EnumEntry<FDEType>> getFDETypes();
LLVM_ABI ArrayRef<EnumEntry<AArch64PAuthKey>> getAArch64PAuthKeys();
LLVM_ABI ArrayRef<EnumEntry<FREOffset>> getFREOffsets();

} // namespace sframe
} // namespace llvm

#endif // LLVM_BINARYFORMAT_SFRAME_H
