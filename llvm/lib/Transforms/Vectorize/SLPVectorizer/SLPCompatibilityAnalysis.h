//===- SLPCompatibilityAnalysis.h - SLP same-opcode helpers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It declares the same-opcode
// compatibility primitives that decide whether a group of values can be
// treated as sharing the same (or an interchangeable/alternate) opcode. These
// do not depend on BoUpSLP or any other SLP-private type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOMPATIBILITYANALYSIS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOMPATIBILITYANALYSIS_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"

#include <cstdint>
#include <utility>

namespace llvm {
class Constant;
class Value;
} // namespace llvm

namespace llvm::slpvectorizer {

/// \returns true if \p Opcode is allowed as part of the main/alternate
/// instruction for SLP vectorization.
///
/// Example of unsupported opcode is SDIV that can potentially cause UB if the
/// "shuffled out" lane would result in division by zero.
bool isValidForAlternation(unsigned Opcode);

/// Helper class that determines VL can use the same opcode.
/// Alternate instruction is supported. In addition, it supports interchangeable
/// instruction. An interchangeable instruction is an instruction that can be
/// converted to another instruction with same semantics. For example, x << 1 is
/// equal to x * 2. x * 1 is equal to x | 0.
class BinOpSameOpcodeHelper {
  using MaskType = std::uint_fast32_t;
  /// Sort SupportedOp because it is used by binary_search.
  constexpr static unsigned SupportedOp[] = {
      Instruction::Add, Instruction::FAdd, Instruction::Sub,  Instruction::FSub,
      Instruction::Mul, Instruction::Shl,  Instruction::AShr, Instruction::And,
      Instruction::Or,  Instruction::Xor};
  static_assert(llvm::is_sorted_constexpr(SupportedOp) &&
                "SupportedOp is not sorted.");
  enum : MaskType {
    ShlBIT = 1,
    AShrBIT = 1 << 1,
    MulBIT = 1 << 2,
    AddBIT = 1 << 3,
    SubBIT = 1 << 4,
    AndBIT = 1 << 5,
    OrBIT = 1 << 6,
    XorBIT = 1 << 7,
    FAddBIT = 1 << 8,
    FSubBIT = 1 << 9,
    MainOpBIT = 1 << 10,
    LLVM_MARK_AS_BITMASK_ENUM(MainOpBIT)
  };
  /// Return a non-nullptr if either operand of I is a ConstantInt (for the
  /// integer opcodes) or a ConstantFP (for FAdd/FSub).
  /// The second return value represents the operand position. We check the
  /// right-hand side first (1). If the right hand side is not a constant and
  /// the instruction is neither Sub, FSub, Shl, nor AShr, we then check the
  /// left hand side (0).
  static std::pair<Constant *, unsigned>
  isBinOpWithConstant(const Instruction *I);
  struct InterchangeableInfo {
    const Instruction *I = nullptr;
    /// The bit it sets represents whether MainOp can be converted to.
    MaskType Mask = MainOpBIT | XorBIT | OrBIT | AndBIT | SubBIT | AddBIT |
                    MulBIT | AShrBIT | ShlBIT | FSubBIT | FAddBIT;
    /// We cannot create an interchangeable instruction that does not exist in
    /// VL. For example, VL [x + 0, y * 1] can be converted to [x << 0, y << 0],
    /// but << does not exist in VL. In the end, we convert VL to [x * 1, y *
    /// 1]. SeenBefore is used to know what operations have been seen before.
    MaskType SeenBefore = 0;
    InterchangeableInfo(const Instruction *I) : I(I) {}
    /// Return false allows BinOpSameOpcodeHelper to find an alternate
    /// instruction. Directly setting the mask will destroy the mask state,
    /// preventing us from determining which instruction it should convert to.
    bool trySet(MaskType OpcodeInMaskForm, MaskType InterchangeableMask);
    bool equal(unsigned Opcode) {
      return Opcode == I->getOpcode() && trySet(MainOpBIT, MainOpBIT);
    }
    unsigned getOpcode() const;
    bool hasDefinedOpcode() const { return (Mask & SeenBefore) > 0; }
    /// Return true if the instruction can be converted to \p Opcode.
    bool hasCandidateOpcode(unsigned Opcode) const;
    SmallVector<Value *> getOperand(const Instruction *To) const;
  };
  InterchangeableInfo MainOp;
  InterchangeableInfo AltOp;
  bool isValidForAlternation(const Instruction *I) const;
  bool initializeAltOp(const Instruction *I);

public:
  BinOpSameOpcodeHelper(const Instruction *MainOp,
                        const Instruction *AltOp = nullptr)
      : MainOp(MainOp), AltOp(AltOp) {}
  bool add(const Instruction *I);
  unsigned getMainOpcode() const { return MainOp.getOpcode(); }
  bool hasDefinedMainOpcode() const { return MainOp.hasDefinedOpcode(); }
  /// Checks if the list of potential opcodes includes \p Opcode.
  bool hasCandidateOpcode(unsigned Opcode) const {
    return MainOp.hasCandidateOpcode(Opcode);
  }
  bool hasAltOp() const { return AltOp.I; }
  unsigned getAltOpcode() const {
    return hasAltOp() ? AltOp.getOpcode() : getMainOpcode();
  }
  bool hasDefinedAltOpcode() const {
    return !hasAltOp() || AltOp.hasDefinedOpcode();
  }
  SmallVector<Value *> getOperand(const Instruction *I) const {
    return MainOp.getOperand(I);
  }
};

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOMPATIBILITYANALYSIS_H
