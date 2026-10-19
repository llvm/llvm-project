//===-- AGXBaseInfo.h - Top level definitions for AGX -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enumerations and helpers shared by the AGX MC layer, assembler,
// disassembler and code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_UTILS_AGXBASEINFO_H
#define LLVM_LIB_TARGET_AGX_UTILS_AGXBASEINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrDesc.h"
#include <cstdint>
#include <optional>

namespace llvm {

class MCRegisterInfo;
class MCRegister;

namespace AGX {

/// Operand types of AGX instructions.
enum OperandType : unsigned {
  /// ALU source: register, uniform register or 8-bit immediate.
  OPERAND_SRC = MCOI::OPERAND_FIRST_TARGET,
  /// ALU destination with cache hint.
  OPERAND_DST,
};

/// Flags of the modifier operand that accompanies every ALU source.
enum SrcMods : unsigned {
  SRC_MOD_ABS = 1 << 0,
  SRC_MOD_NEG = 1 << 1,
  SRC_MOD_SX = 1 << 2,
  /// Keep the value in the register cache.
  SRC_MOD_CACHE = 1 << 3,
  /// This is the last use of the register.
  SRC_MOD_DISCARD = 1 << 4,
};

/// Fields of the TSFlags of AGX instructions, see AGXInstrFormats.td.
namespace TSFlags {
inline unsigned getShortSize(uint64_t F) { return F & 0xf; }
inline unsigned getLongSize(uint64_t F) { return (F >> 4) & 0xf; }
inline unsigned getLengthBit(uint64_t F) { return (F >> 8) & 0x7f; }
inline bool isAsync(uint64_t F) { return F & (1ull << 15); }
inline bool isExecMaskOp(uint64_t F) { return F & (1ull << 16); }
inline bool isALU(uint64_t F) { return F & (1ull << 17); }
inline bool isMemory(uint64_t F) { return F & (1ull << 18); }
} // namespace TSFlags

/// Memory formats of load/store instructions.
enum MemFormat : unsigned {
  FMT_I8 = 0,
  FMT_I16 = 1,
  FMT_I32 = 2,
  FMT_F16 = 3,
  FMT_U8NORM = 4,
  FMT_S8NORM = 5,
  FMT_U16NORM = 6,
  FMT_S16NORM = 7,
  FMT_RGB10A2 = 8,
  FMT_SRGBA8 = 10,
  FMT_RG11B10F = 12,
  FMT_RGB9E5 = 13,
};

/// Operation numbers of SIMD-group reductions (llvm.agx.simd.reduce etc.).
enum SIMDOp : unsigned {
  SIMD_AND = 0,
  SIMD_FADD = 1,
  SIMD_OR = 2,
  SIMD_FMUL = 3,
  SIMD_XOR = 4,
  SIMD_FMIN = 5,
  SIMD_FMAX = 7,
  SIMD_IADD = 16,
  SIMD_SMIN = 20,
  SIMD_SMAX = 22,
  SIMD_UMIN = 28,
  SIMD_UMAX = 30,
};

/// Address spaces.
enum AddressSpace : unsigned {
  GENERIC_ADDRESS = 0,
  DEVICE_ADDRESS = 1,
  THREADGROUP_ADDRESS = 3,
  CONSTANT_ADDRESS = 4,
  PRIVATE_ADDRESS = 5,
};

/// Special register numbers read by get_sr.
enum SpecialReg : unsigned {
  SR_THREADGROUP_POSITION_IN_GRID_X = 0,
  SR_THREADS_PER_THREADGROUP_X = 4,
  SR_CORE_INDEX = 20,
  SR_THREAD_POSITION_IN_THREADGROUP_X = 48,
  SR_THREAD_INDEX_IN_THREADGROUP = 51,
  SR_THREAD_INDEX_IN_SIMDGROUP = 52,
  SR_SIMDGROUP_INDEX_IN_THREADGROUP = 53,
  SR_THREAD_POSITION_IN_GRID_X = 80,
};

/// Decode an 8-bit floating point ALU immediate: sign, 3-bit exponent and
/// 4-bit fraction. Exponent zero gives fraction / 64.
double decodeFPImm8(unsigned Imm);

/// Encode \p V as an 8-bit floating point ALU immediate if it is exactly
/// representable.
std::optional<unsigned> encodeFPImm8(double V);

/// Format an 8-bit floating point immediate the way the reference
/// disassembler does (shortest round-trip decimal, always with a '.').
std::string formatFPImm8(unsigned Imm);

/// Names of enumerated operands. Each returns an empty string for values
/// without a name.
StringRef getICondName(unsigned CC);
StringRef getFCondName(unsigned CC);
StringRef getMemFormatName(unsigned Fmt);
StringRef getAtomicOpName(unsigned Op);
StringRef getConvertModeName(unsigned Mode);
StringRef getRoundModeName(unsigned Mode);
StringRef getSpecialRegName(unsigned SR);
std::string getMemMaskName(unsigned Mask);

/// Reverse lookups, used by the assembler.
std::optional<unsigned> parseICond(StringRef Name);
std::optional<unsigned> parseFCond(StringRef Name);
std::optional<unsigned> parseMemFormat(StringRef Name);
std::optional<unsigned> parseAtomicOp(StringRef Name);
std::optional<unsigned> parseConvertMode(StringRef Name);
std::optional<unsigned> parseRoundMode(StringRef Name);
std::optional<unsigned> parseMemMask(StringRef Name);

/// Register kinds of the AGX register file.
enum class RegKind {
  Invalid,
  GPR16,
  GPR32,
  GPR64, ///< Register pair (any 32-bit tuple of two).
  GPR96,
  GPR128,
  UGPR16,
  UGPR32,
  UGPR64,
  Nest, ///< The nesting counter r0l used implicitly.
};

/// Kind and first half-register index of \p Reg.
struct RegInfo {
  RegKind Kind = RegKind::Invalid;
  unsigned HalfIndex = 0;

  bool isUniform() const {
    return Kind == RegKind::UGPR16 || Kind == RegKind::UGPR32 ||
           Kind == RegKind::UGPR64;
  }
  bool isGPR() const {
    return Kind == RegKind::GPR16 || Kind == RegKind::GPR32 ||
           Kind == RegKind::GPR64 || Kind == RegKind::GPR96 ||
           Kind == RegKind::GPR128;
  }
  unsigned getSizeInBits() const;
};

RegInfo getRegInfo(MCRegister Reg, const MCRegisterInfo &MRI);

/// The register of kind \p Kind starting at half register \p HalfIndex, or
/// no register if there is none.
MCRegister getRegister(RegKind Kind, unsigned HalfIndex,
                       const MCRegisterInfo &MRI);

} // namespace AGX
} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_UTILS_AGXBASEINFO_H
