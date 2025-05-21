//===-- RISCVTargetParser - Parser for target features ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features
// for RISC-V CPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_RISCVTARGETPARSER_H
#define LLVM_TARGETPARSER_RISCVTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Triple;

namespace RISCV {

struct CPUModel {
  uint32_t MVendorID;
  uint64_t MArchID;
  uint64_t MImpID;
};

struct CPUInfo {
  StringLiteral Name;
  StringLiteral DefaultMarch;
  bool FastScalarUnalignedAccess;
  bool FastVectorUnalignedAccess;
  CPUModel Model;
  bool is64Bit() const { return DefaultMarch.starts_with("rv64"); }
};

// We use 64 bits as the known part in the scalable vector types.
static constexpr unsigned RVVBitsPerBlock = 64;
static constexpr unsigned RVVBytesPerBlock = RVVBitsPerBlock / 8;

void getFeaturesForCPU(StringRef CPU,
                       SmallVectorImpl<std::string> &EnabledFeatures,
                       bool NeedPlus = false);
bool parseCPU(StringRef CPU, bool IsRV64);
bool parseTuneCPU(StringRef CPU, bool IsRV64);
StringRef getMArchFromMcpu(StringRef CPU);
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64);
void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64);
bool hasFastScalarUnalignedAccess(StringRef CPU);
bool hasFastVectorUnalignedAccess(StringRef CPU);
bool hasValidCPUModel(StringRef CPU);
CPUModel getCPUModel(StringRef CPU);

} // namespace RISCV

namespace RISCVVType {
enum VLMUL : uint8_t {
  LMUL_1 = 0,
  LMUL_2,
  LMUL_4,
  LMUL_8,
  LMUL_RESERVED,
  LMUL_F8,
  LMUL_F4,
  LMUL_F2
};

enum {
  TAIL_UNDISTURBED_MASK_UNDISTURBED = 0,
  TAIL_AGNOSTIC = 1,
  MASK_AGNOSTIC = 2,
};

// Is this a SEW value that can be encoded into the VTYPE format.
inline static bool isValidSEW(unsigned SEW) {
  return isPowerOf2_32(SEW) && SEW >= 8 && SEW <= 64;
}

// Is this a LMUL value that can be encoded into the VTYPE format.
inline static bool isValidLMUL(unsigned LMUL, bool Fractional) {
  return isPowerOf2_32(LMUL) && LMUL <= 8 && (!Fractional || LMUL != 1);
}

unsigned encodeVTYPE(VLMUL VLMUL, unsigned SEW, bool TailAgnostic,
                     bool MaskAgnostic);

unsigned encodeXSfmmVType(unsigned SEW, unsigned Widen, bool AltFmt);

inline static VLMUL getVLMUL(unsigned VType) {
  unsigned VLMul = VType & 0x7;
  return static_cast<VLMUL>(VLMul);
}

// Decode VLMUL into 1,2,4,8 and fractional indicator.
std::pair<unsigned, bool> decodeVLMUL(VLMUL VLMul);

inline static VLMUL encodeLMUL(unsigned LMUL, bool Fractional) {
  assert(isValidLMUL(LMUL, Fractional) && "Unsupported LMUL");
  unsigned LmulLog2 = Log2_32(LMUL);
  return static_cast<VLMUL>(Fractional ? 8 - LmulLog2 : LmulLog2);
}

inline static unsigned decodeVSEW(unsigned VSEW) {
  assert(VSEW < 8 && "Unexpected VSEW value");
  return 1 << (VSEW + 3);
}

inline static unsigned encodeSEW(unsigned SEW) {
  assert(isValidSEW(SEW) && "Unexpected SEW value");
  return Log2_32(SEW) - 3;
}

inline static unsigned getSEW(unsigned VType) {
  unsigned VSEW = (VType >> 3) & 0x7;
  return decodeVSEW(VSEW);
}

inline static unsigned decodeTWiden(unsigned TWiden) {
  assert((TWiden == 1 || TWiden == 2 || TWiden == 3) &&
         "Unexpected TWiden value");
  return 1 << (TWiden - 1);
}

inline static bool hasXSfmmWiden(unsigned VType) {
  unsigned TWiden = (VType >> 9) & 0x3;
  return TWiden != 0;
}

inline static unsigned getXSfmmWiden(unsigned VType) {
  unsigned TWiden = (VType >> 9) & 0x3;
  assert(TWiden != 0 && "Invalid widen value");
  return 1 << (TWiden - 1);
}

static inline bool isValidXSfmmVType(unsigned VTypeI) {
  return (VTypeI & ~0x738) == 0 && RISCVVType::hasXSfmmWiden(VTypeI) &&
         RISCVVType::getSEW(VTypeI) * RISCVVType::getXSfmmWiden(VTypeI) <= 64;
}

inline static bool isTailAgnostic(unsigned VType) { return VType & 0x40; }

inline static bool isMaskAgnostic(unsigned VType) { return VType & 0x80; }

inline static bool isAltFmt(unsigned VType) { return VType & 0x100; }

void printVType(unsigned VType, raw_ostream &OS);

unsigned getSEWLMULRatio(unsigned SEW, VLMUL VLMul);

std::optional<VLMUL> getSameRatioLMUL(unsigned SEW, VLMUL VLMUL, unsigned EEW);
} // namespace RISCVVType

} // namespace llvm

#endif
