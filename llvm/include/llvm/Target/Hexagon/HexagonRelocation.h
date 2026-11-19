//===- HexagonRelocation.h - Hexagon relocation application API -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API for applying relocations to Hexagon instruction encodings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_HEXAGON_HEXAGONRELOCATION_H
#define LLVM_TARGET_HEXAGON_HEXAGONRELOCATION_H

#include "llvm/Support/Compiler.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// ABI properties of a Hexagon ELF relocation type, as listed in the Hexagon
/// ABI document.
typedef struct LLVMHexagonRelocInfo {
  const char *Name;
  uint32_t Type;
  uint32_t EffectiveBits;
  uint32_t BitMask; ///< 0 if the mask depends on the instruction encoding.
  uint32_t Alignment;
  uint32_t Shift;
  bool VerifyRange;
  bool VerifyAlignment;
  bool IsSigned;
} LLVMHexagonRelocInfo;

/// Fill \p Info with the properties of \p RelocType. Returns false, leaving
/// \p Info untouched, if \p RelocType is not a known Hexagon relocation.
LLVM_ABI bool LLVMHexagonGetRelocationInfo(uint32_t RelocType,
                                           LLVMHexagonRelocInfo *Info);

/// Return true if \p Value fits the range the ABI specifies for \p RelocType.
/// \p Value must already be shifted right by LLVMHexagonRelocInfo::Shift.
/// Returns false for an unknown \p RelocType. The result is meaningful only
/// for relocations whose LLVMHexagonRelocInfo::VerifyRange is set.
LLVM_ABI bool LLVMHexagonVerifyRange(uint32_t RelocType, uint32_t Value);

/// Return true if \p Value satisfies the alignment required by \p RelocType.
/// \p Value is the raw, unshifted relocation value.
/// Returns false for an unknown \p RelocType. The result is meaningful only
/// for relocations whose LLVMHexagonRelocInfo::VerifyAlignment is set.
LLVM_ABI bool LLVMHexagonVerifyAlignment(uint32_t RelocType, uint32_t Value);

/// Return true if applying the unsigned relocation \p RelocType to \p Value
/// would lose set bits. \p Value must already be shifted right by
/// LLVMHexagonRelocInfo::Shift, as for LLVMHexagonVerifyRange. Returns false
/// for signed or unknown relocations.
LLVM_ABI bool LLVMHexagonIsTruncated(uint32_t RelocType, uint32_t Value);

/// Apply the ELF relocation \p RelocType with value \p Value to \p Instruction.
///
/// Hexagon instruction operands may be stored in non-contiguous bit positions.
/// This function locates the operand field for relocations whose mask depends
/// on the instruction encoding, then inserts the relocated value into that
/// field. \p Value is the raw, unshifted relocation value; the shift specified
/// by LLVMHexagonRelocInfo::Shift is applied internally. The caller is
/// responsible for validating the relocation value (see LLVMHexagonVerifyRange
/// and friends; note VerifyRange and IsTruncated expect a shifted value) and
/// for computing any PC-relative adjustment before calling this function.
/// \p Instruction is returned unchanged if \p RelocType is unknown.
LLVM_ABI uint32_t LLVMHexagonApplyRelocation(uint32_t RelocType,
                                             uint32_t Instruction,
                                             uint32_t Value);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LLVM_TARGET_HEXAGON_HEXAGONRELOCATION_H
