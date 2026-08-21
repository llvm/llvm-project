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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/Instruction.h"

#include <cstdint>
#include <utility>

namespace llvm {
class Constant;
class TargetLibraryInfo;
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

/// Main data required for vectorization of instructions.
class InstructionsState {
  /// MainOp and AltOp are primarily determined by getSameOpcode. Currently,
  /// only BinaryOperator, CastInst, and CmpInst support alternate instructions
  /// (i.e., AltOp is not equal to MainOp; this can be checked using
  /// isAltShuffle).
  /// A rare exception is TrySplitNode, where the InstructionsState is derived
  /// from getMainAltOpsNoStateVL.
  /// For those InstructionsState that use alternate instructions, the resulting
  /// vectorized output ultimately comes from a shufflevector. For example,
  /// given a vector list (VL):
  /// VL[0] = add i32 a, e
  /// VL[1] = sub i32 b, f
  /// VL[2] = add i32 c, g
  /// VL[3] = sub i32 d, h
  /// The vectorized result would be:
  /// intermediated_0 = add <4 x i32> <a, b, c, d>, <e, f, g, h>
  /// intermediated_1 = sub <4 x i32> <a, b, c, d>, <e, f, g, h>
  /// result = shufflevector <4 x i32> intermediated_0,
  ///                        <4 x i32> intermediated_1,
  ///                        <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  /// Since shufflevector is used in the final result, when calculating the cost
  /// (getEntryCost), we must account for the usage of shufflevector in
  /// GetVectorCost.
  Instruction *MainOp = nullptr;
  Instruction *AltOp = nullptr;
  /// Whether the instruction state represents copyable instructions.
  bool HasCopyables = false;
  /// Index of the operand modeling the copyable values: the addend for
  /// fmuladd (retried with a multiplicand), the first operand otherwise.
  unsigned CopyableOpIdx = 0;
  /// Whether copyable single-use fmuls/fadds are modeled as
  /// fmuladd(a, b, -0.0)/fmuladd(1.0, a, b), absorbing the binop instead of
  /// computing and gathering its result.
  bool AbsorbCopyableFMulOrFAdd = false;

public:
  Instruction *getMainOp() const {
    assert(valid() && "InstructionsState is invalid.");
    return MainOp;
  }

  Instruction *getAltOp() const {
    assert(valid() && "InstructionsState is invalid.");
    return AltOp;
  }

  /// The main/alternate opcodes for the list of instructions.
  unsigned getOpcode() const { return getMainOp()->getOpcode(); }

  unsigned getAltOpcode() const { return getAltOp()->getOpcode(); }

  /// Some of the instructions in the list have alternate opcodes.
  bool isAltShuffle() const { return getMainOp() != getAltOp(); }

  /// Checks if \p I is the same operation as \p Op, distinguishing calls by
  /// intrinsic ID (all calls share the Call opcode, so e.g. umax != smax).
  static bool isSameOperation(const Instruction *I, const Instruction *Op);

  /// Checks if the instruction matches either the main or alternate opcode.
  /// \returns
  /// - MainOp if \param I matches MainOp's opcode directly or can be converted
  /// to it
  /// - AltOp if \param I matches AltOp's opcode directly or can be converted to
  /// it
  /// - nullptr if \param I cannot be matched or converted to either opcode
  Instruction *getMatchingMainOpOrAltOp(Instruction *I) const;

  /// Checks if main/alt instructions are shift operations.
  bool isShiftOp() const {
    return getMainOp()->isShift() && getAltOp()->isShift();
  }

  /// Checks if main/alt instructions are bitwise logic operations.
  bool isBitwiseLogicOp() const {
    return getMainOp()->isBitwiseLogicOp() && getAltOp()->isBitwiseLogicOp();
  }

  /// Checks if main/alt instructions are mul/div/rem/fmul/fdiv/frem operations.
  bool isMulDivLikeOp() const;

  /// Checks if main/alt instructions are add/sub/fadd/fsub operations.
  bool isAddSubLikeOp() const;

  /// Checks for an add/sub-like operation or an fneg, which models a
  /// flattened fadd reduction chain link: the link is dropped and its
  /// operand joins a sign-flipped combine, which may form an fma as well.
  bool isAddSubOrFNegLikeOp() const;

  /// Checks if main/alt instructions are cmp operations.
  bool isCmpOp() const {
    return (getOpcode() == Instruction::ICmp ||
            getOpcode() == Instruction::FCmp) &&
           getAltOpcode() == getOpcode();
  }

  /// Checks if the current state is valid, i.e. has non-null MainOp
  bool valid() const { return MainOp && AltOp; }

  explicit operator bool() const { return valid(); }

  InstructionsState() = delete;
  InstructionsState(Instruction *MainOp, Instruction *AltOp,
                    bool HasCopyables = false)
      : MainOp(MainOp), AltOp(AltOp), HasCopyables(HasCopyables),
        CopyableOpIdx(MainOp && RecurrenceDescriptor::isFMulAddIntrinsic(MainOp)
                          ? 2
                          : 0) {}
  static InstructionsState invalid() { return {nullptr, nullptr}; }

  /// Checks if the value is a copyable element.
  bool isCopyableElement(Value *V) const;

  /// Checks if the value \p V is a transformed instruction, compatible either
  /// with main or alternate ops.
  bool isExpandedBinOp(Value *V) const;

  /// Checks if the operand at index \p Idx of instruction \p I is an expanded
  /// operand.
  bool isExpandedOperand(Instruction *I, unsigned Idx) const;

  /// Checks if the value is non-schedulable.
  bool isNonSchedulable(Value *V) const;

  /// Checks if the state represents copyable instructions.
  bool areInstructionsWithCopyableElements() const {
    assert(valid() && "InstructionsState is invalid.");
    return HasCopyables;
  }

  /// Returns the index of the operand the copyable value is modeled in.
  unsigned getCopyableOpIdx() const {
    assert(valid() && "InstructionsState is invalid.");
    return CopyableOpIdx;
  }

  /// Sets the index of the operand the copyable value is modeled in.
  void setCopyableOpIdx(unsigned Idx) {
    assert((Idx == 0 || Idx == 2) && "Unexpected copyable operand index.");
    CopyableOpIdx = Idx;
  }

  /// Checks if copyable fmuls/fadds are absorbed as fmuladd(a, b, -0.0) or
  /// fmuladd(1.0, a, b).
  bool hasAbsorbedCopyableFMulOrFAdd() const {
    assert(valid() && "InstructionsState is invalid.");
    return AbsorbCopyableFMulOrFAdd;
  }

  /// Sets the absorbed-fmul/fadd modeling for copyable fmuls/fadds.
  void setAbsorbCopyableFMulOrFAdd(bool Absorb) {
    AbsorbCopyableFMulOrFAdd = Absorb;
  }
};

/// Checks if \p V is a single-use fmul/fadd with operands outside \p VL.
bool isAbsorbableFMulOrFAdd(ArrayRef<Value *> VL, Value *V);

/// Checks if \p V is a copyable single-use fmul/fadd, absorbable as
/// fmuladd(a, b, -0.0) or fmuladd(1.0, a, b).
bool isAbsorbableCopyableFMulOrFAdd(const InstructionsState &S, Value *V);

/// Checks if every copyable in \p VL is an absorbable fmul/fadd: the binops
/// die instead of being computed and gathered. Operand order is normalized
/// when the operands are built.
bool hasOnlyAbsorbableCopyableFMulOrFAdds(ArrayRef<Value *> VL);

/// \returns analysis of the Instructions in \p VL described in
/// InstructionsState, the Opcode that we suppose the whole list
/// could be vectorized even if its structure is diverse.
InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                const TargetLibraryInfo &TLI);

/// \returns the main or alternate operation from \p S matching \p I, together
/// with the operands of \p I adjusted to the selected operation.
std::pair<Instruction *, SmallVector<Value *>>
convertTo(Instruction *I, const InstructionsState &S);

/// Checks if the specified instruction \p I is an alternate operation for
/// the given \p MainOp and \p AltOp instructions.
bool isAlternateInstruction(Instruction *I, Instruction *MainOp,
                            Instruction *AltOp, const TargetLibraryInfo &TLI);

/// Peel the per-lane associative chains of an alternate node into operand
/// columns. Lanes peel in lockstep and only chain links with the lane's own
/// opcode, so every combine level keeps the root's main/alt opcode pattern
/// and a subtract lane never becomes an add of a negated leaf. Only the
/// leading (running) column peels: peeling a subtracted subtract would flip
/// signs. \p SubLanes records the subtract lanes for the realignment sign
/// query. Returns the flattened columns, empty when no level peels.
SmallVector<SmallVector<Value *>> scanAltAssociativeOperands(
    const InstructionsState &S, const TargetLibraryInfo &TLI,
    ArrayRef<Value *> VL, ArrayRef<Value *> Op0, ArrayRef<Value *> Op1,
    SmallVectorImpl<Value *> &ReassocScalars, SmallBitVector &SubLanes);
} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPCOMPATIBILITYANALYSIS_H
