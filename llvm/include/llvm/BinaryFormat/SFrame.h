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

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm::sframe {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

constexpr uint16_t MagicSignature = 0xdee2;

enum class Version : uint8_t {
  V1 = 1,
  V2 = 2,
};

enum class Flags : uint8_t {
  FDESorted = 0x01,
  FramePointer = 0x02,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/0xff),
};

enum class ABI : uint8_t {
  AArch64EndianBig = 1,
  AArch64EndianLittle = 2,
  AMD64EndianLittle = 3,
};

/// SFrame FRE Types. Bits 0-3 of FuncDescEntry.Info.
enum class FREType : uint8_t {
  Addr1 = 0,
  Addr2 = 1,
  Addr4 = 2,
};

/// SFrame FDE Types. Bit 4 of FuncDescEntry.Info.
enum class FDEType : uint8_t {
  PCInc = 0,
  PCMask = 1,
};

/// Speficies key used for signing return addresses. Bit 5 of
/// FuncDescEntry.Info.
enum class AArch64PAuthKey : uint8_t {
  A = 0,
  B = 1,
};

/// Size of stack offsets. Bits 5-6 of FREInfo.Info.
enum class FREOffset : uint8_t {
  B1 = 0,
  B2 = 1,
  B4 = 2,
};

/// Stack frame base register. Bit 0 of FREInfo.Info.
enum class BaseReg : uint8_t {
  FP = 0,
  SP = 1,
};

LLVM_PACKED_START

struct Preamble {
  uint16_t Magic;
  enum Version Version;
  enum Flags Flags;
};

struct Header {
  struct Preamble Preamble;
  ABI ABIArch;
  int8_t CFAFixedFPOffset;
  int8_t CFAFixedRAOffset;
  uint8_t AuxHdrLen;
  uint32_t NumFDEs;
  uint32_t NumFREs;
  uint32_t FRELen;
  uint32_t FDEOff;
  uint32_t FREOff;
};

struct FuncDescEntry {
  int32_t StartAddress;
  uint32_t Size;
  uint32_t StartFREOff;
  uint32_t NumFREs;
  uint8_t Info;
  uint8_t RepSize;
  uint16_t Padding2;

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

struct FREInfo {
  uint8_t Info;

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

struct FrameRowEntryAddr1 {
  uint8_t StartAddress;
  FREInfo Info;
};

struct FrameRowEntryAddr2 {
  uint16_t StartAddress;
  FREInfo Info;
};

struct FrameRowEntryAddr4 {
  uint32_t StartAddress;
  FREInfo Info;
};

LLVM_PACKED_END

} // namespace llvm::sframe

#endif // LLVM_BINARYFORMAT_SFRAME_H
