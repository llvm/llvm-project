//===- PackedIntegerCombinePass.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the interface for LLVM's Packed Integer Combine pass.
/// This pass tries to treat integers as packed chunks of individual bytes,
/// and leverage this to coalesce needlessly fragmented
/// computations.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/PackedIntegerCombinePass.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include <variant>

using namespace llvm;

#define DEBUG_TYPE "packed-integer-combine"

namespace {

/// Reference to either a constant byte, or a byte extracted from an IR value.
class Byte {
  /// The base value from which the byte is obtained.
  Value *Base;

  /// If the base value is not null, then this holds the index of the byte
  /// being used, where 0 is the least significant byte.
  /// Otherwise, this is treated as a constant byte.
  unsigned Integer;

public:
  static constexpr unsigned BitWidth = 8;
  static constexpr unsigned AllOnes = 0xff;

  /// Construct a byte from a well-defined IR value.
  explicit Byte(Value &Base, unsigned Index) : Base(&Base), Integer(Index) {}

  /// Construct a constant byte.
  explicit Byte(unsigned Constant) : Base(nullptr), Integer(Constant) {
    assert(Constant <= AllOnes && "Constant is too large to fit in a byte.");
  }

  /// Construct a constant byte that is fully set.
  static Byte ones() { return Byte(Byte::AllOnes); }
  /// Construct the zero byte.
  static Byte zeroes() { return Byte(0); }

  /// Indicate whether the byte is a known integer constant.
  /// Note that poison or undef base values are not recognised as constant.
  bool isConstant() const { return !Base; }

  /// Get the constant byte value.
  unsigned getConstant() const {
    assert(isConstant() && "Expected a constant byte.");
    return Integer;
  }

  /// Get the base IR value from which this byte is obtained.
  Value *getBase() const {
    assert(!isConstant() && "Byte constants do not have a base value.");
    return Base;
  }

  /// Get the byte offset of the IR value referenced by the byte.
  unsigned getIndex() const {
    assert(!isConstant() && "Byte constants are not indexed.");
    return Integer;
  }

  bool operator==(const Byte &Other) const {
    return Base == Other.Base && Integer == Other.Integer;
  }

  void print(raw_ostream &ROS, bool NewLine = true) const {
    if (isConstant())
      ROS << "const";
    else
      Base->printAsOperand(ROS, false);

    ROS << '[' << Integer << ']';

    if (NewLine)
      ROS << '\n';
  }

  LLVM_DUMP_METHOD void dump() const { print(errs(), true); }
};

inline raw_ostream &operator<<(raw_ostream &ROS, const Byte &B) {
  B.print(ROS, false);
  return ROS;
}

/// Convenience data structure for describing the layout of bytes for vector and
/// integer types, treating integer types as singleton vectors.
struct ByteLayout {
  /// The number of bytes that fit in a single element.
  unsigned NumBytesPerElement;
  /// The number of vector elements (or 1, if the type is an integer type).
  unsigned NumVecElements;

  /// Get the total number of bytes held by the vector or integer type.
  unsigned getNumBytes() const { return NumBytesPerElement * NumVecElements; }
};

/// Interpret the given type as a number of packed bytes, if possible.
static std::optional<ByteLayout> getByteLayout(const Type *Ty) {
  unsigned IntBitWidth, NumElts;
  if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
    IntBitWidth = IntTy->getBitWidth();
    NumElts = 1;
  } else if (const auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    const auto *IntTy = dyn_cast<IntegerType>(VecTy->getElementType());
    if (!IntTy)
      return std::nullopt;
    IntBitWidth = IntTy->getBitWidth();
    NumElts = VecTy->getNumElements();
  } else
    return std::nullopt;

  if (IntBitWidth % Byte::BitWidth != 0)
    return std::nullopt;

  return ByteLayout{IntBitWidth / Byte::BitWidth, NumElts};
}

/// A convenience class for combining Byte instances obtained from the same base
/// value, and with a common relative offset, which can hence be obtained
/// simultaneously.
struct CoalescedBytes {
  /// The value from which the coalesced bytes are all derived. This pointer is
  /// never null.
  Value *Base;
  /// The number of bytes to shift right to align the coalesced bytes with the
  /// target value.
  ///
  /// For instance, if bytes 3, 4, 5 of some value %val are coalesced to provide
  /// bytes 0, 1, 2 of the target %tgt, then ShrByteOffset = 3.
  int SignedShrByteOffset;
  /// The bitmask identifying which bytes of the target value are covered by
  /// these coalesced bytes.
  ///
  /// For instance, if bytes 3, 4, 5 of some value %val are coalesced to provide
  /// bytes 0, 1, 2 of the target %tgt, then this mask's first three bits will
  /// be set, corresponding to the first three bits of %tgt.
  SmallBitVector Mask;

  explicit CoalescedBytes(Value &Base, int Offset, SmallBitVector Mask)
      : Base(&Base), SignedShrByteOffset(Offset), Mask(Mask) {}
  explicit CoalescedBytes(Value &Base, int Offset, unsigned NumBytes)
      : Base(&Base), SignedShrByteOffset(Offset), Mask(NumBytes) {}

  bool alignsWith(Value *V, int VOffset) const {
    return Base == V && SignedShrByteOffset == VOffset;
  }

  /// Get the number of bytes to shift the base value right to align with the
  /// target value.
  unsigned getShrBytes() const { return std::max(0, SignedShrByteOffset); }

  /// Get the number of bytes to shift the base value left to align with the
  /// target value.
  unsigned getShlBytes() const { return std::max(0, -SignedShrByteOffset); }

  /// Get the number of bits to shift the base value right to align with the
  /// target value.
  unsigned getShrBits() const { return getShrBytes() * Byte::BitWidth; }

  /// Get the number of bits to shift the base value left to align with the
  /// target value.
  unsigned getShlBits() const { return getShlBytes() * Byte::BitWidth; }

  void print(raw_ostream &ROS, bool NewLine = true) const {
    ROS << "{ ";
    for (unsigned Idx = 0; Idx < Mask.size(); ++Idx) {
      if (Mask.test(Idx)) {
        Base->printAsOperand(ROS, false);
        ROS << '[' << (static_cast<int>(Idx) + SignedShrByteOffset) << ']';
      } else
        ROS << 0;

      ROS << "; ";
    }
    ROS << '}';

    if (NewLine)
      ROS << '\n';
  }

  LLVM_DUMP_METHOD void dump() const { print(errs(), true); }
};

inline raw_ostream &operator<<(raw_ostream &ROS, const CoalescedBytes &CB) {
  CB.print(ROS, false);
  return ROS;
}

/// Association of a Byte (constant or byte extracted from an LLVM Value) to the
/// operand(s) responsible for producing it. A value of ByteUse::AllOperands
/// (-1) indicates that all operands are responsible for producing the given
/// byte.
class ByteUse {

  Byte B;
  int OpIdx;

public:
  /// Sentinel value representing that all operands are responsible for the
  /// given Byte.
  static constexpr int AllOperands = -1;

  ByteUse(Byte B, int OpIdx) : B(B), OpIdx(OpIdx) {}

  const Byte &getByte() const { return B; }
  int getOperandIndex() const { return OpIdx; }

  bool operator==(const ByteUse &BU) const {
    return BU.B == B && BU.OpIdx == OpIdx;
  }
};

using ByteVector = SmallVector<ByteUse, 8>;

/// The decomposition of an IR value into its individual bytes, tracking where
/// each byte is obtained.
struct ByteDefinition {
  std::variant<std::nullopt_t, ByteVector *, Value *> Ptr;
  ByteLayout Layout;

public:
  /// Indicate that a value cannot be decomposed into bytes in a known way.
  static ByteDefinition invalid() { return {std::nullopt, {0, 0}}; }
  /// Indicate that a value's bytes are known, and track their producers.
  static ByteDefinition vector(ByteVector &Ref, ByteLayout Layout) {
    return {&Ref, Layout};
  }
  /// Indicate that a value's bytes are opaque.
  static ByteDefinition value(Value &V) {
    return {&V, *getByteLayout(V.getType())};
  }

  bool isValid() const { return !std::holds_alternative<std::nullopt_t>(Ptr); }

  /// Return true iff the byte definition is valid.
  operator bool() const { return isValid(); }

  /// Get the definition of the byte at the specified byte offset, where 0 is
  /// the least significant byte.
  Byte getByte(unsigned Idx) const {
    struct Visitor {
      unsigned Idx;

      Byte operator()(std::nullopt_t) {
        llvm_unreachable("Invalid byte definition");
      }
      Byte operator()(ByteVector *BV) { return (*BV)[Idx].getByte(); }
      Byte operator()(Value *V) {
        if (auto *Int = dyn_cast<ConstantInt>(V))
          return Byte(Int->getValue().extractBitsAsZExtValue(
              Byte::BitWidth, Idx * Byte::BitWidth));

        if (V->getType()->isVectorTy()) {
          if (auto *Vec = dyn_cast<Constant>(V)) {
            const ByteLayout Layout = *getByteLayout(Vec->getType());
            const unsigned VecIdx = Idx / Layout.NumBytesPerElement;
            const unsigned EltIdx = Idx % Layout.NumBytesPerElement;

            if (Constant *Elt = Vec->getAggregateElement(VecIdx)) {
              if (const auto *Int = dyn_cast<ConstantInt>(Elt))
                return Byte(Int->getValue().extractBitsAsZExtValue(
                    Byte::BitWidth, EltIdx * Byte::BitWidth));

              return Byte(*Elt, EltIdx);
            }
          }
        }

        return Byte(*V, Idx);
      }
    };

    return std::visit(Visitor{Idx}, Ptr);
  }

  const ByteLayout &getLayout() const { return Layout; }

  void print(raw_ostream &ROS, bool NewLine = true) const {
    struct Visitor {
      raw_ostream &ROS;
      const ByteLayout &Layout;

      void operator()(std::nullopt_t) { ROS << "[INVALID]"; }
      void operator()(ByteVector *BV) {
        ROS << "{ ";
        for (unsigned ByteIdx = 0; ByteIdx < BV->size(); ++ByteIdx)
          ROS << ByteIdx << ": " << (*BV)[ByteIdx].getByte() << "; ";
        ROS << '}';
      }
      void operator()(Value *V) {
        ROS << '(';
        V->printAsOperand(ROS);
        ROS << ")[0:" << Layout.getNumBytes() << ']';
      }
    };

    std::visit(Visitor{ROS, Layout}, Ptr);
    if (NewLine)
      ROS << '\n';
  }

  LLVM_DUMP_METHOD void dump() const { print(errs(), true); }
};

inline raw_ostream &operator<<(raw_ostream &ROS, const ByteDefinition &Def) {
  Def.print(ROS, false);
  return ROS;
}

/// Tries to update byte definitions using the provided instruction.
///
/// In order to avoid eliminating values which are required for multiple packed
/// integers, the ByteExpander distinguishes two types of packed integer values:
/// - "Final" values, which are packed bytes which are either used by
///   instructions that cannot be classified as packed byte operations, or
///   values which are used by several other "final" values.
/// - "Intermediate" values, which are values whose sole raisons d'etre are to
///   produce bytes for a unique final value.
///
/// Effectively, intermediate values may be eliminated or replaced freely,
/// whereas final values must remain present in the IR after the pass completes.
/// Accordingly, byte defniitions of final values are expanded only up to other
/// final value producers.
class ByteExpander final : public InstVisitor<ByteExpander, ByteVector> {
  /// Resolution of values to their known definitions.
  DenseMap<Value *, ByteVector> Definitions;
  /// Map to all (eventual) non-intermediate users of a value.
  DenseMap<Value *, DenseSet<Value *>> FinalUsers;

  void updateFinalUsers(Value *V);
  bool checkIfIntermediate(Value *V, bool IsOperand);

public:
  // Visitation implementations return `true` iff a new byte definition was
  // successfully constructed.

  ByteVector visitAnd(BinaryOperator &I);
  ByteVector visitOr(BinaryOperator &I);
  ByteVector visitXor(BinaryOperator &I);
  ByteVector visitShl(BinaryOperator &I);
  ByteVector visitLShr(BinaryOperator &I);
  ByteVector visitTruncInst(TruncInst &I);
  ByteVector visitZExtInst(ZExtInst &I);
  ByteVector visitBitCastInst(BitCastInst &I);
  ByteVector visitExtractElementInst(ExtractElementInst &I);
  ByteVector visitInsertElementInst(InsertElementInst &I);
  ByteVector visitShuffleVectorInst(ShuffleVectorInst &I);
  // fallback for unhandled instructions
  ByteVector visitInstruction(Instruction &I) { return {}; }

  /// Return the final values producing each byte of a value, if known, or
  /// otherwise return a nullptr.
  ByteVector *expandByteDefinition(Value *V);

  /// Decompose a value into its bytes. If \p ExpandDef is true, expand each
  /// byte to the final values producing them if possible. The return value is
  /// guaranteed to be valid so long as the value passed can be viewed as packed
  /// bytes.
  ByteDefinition getByteDefinition(Value *V, bool ExpandDef = true);

  /// Same as above, but only expand bytes to their final value producers if the
  /// value \p V in question is an intermediate value. This is provided as a
  /// convenience for instruction visitation, as definitions should only expand
  /// until final value producers, even if the final value producers' bytes can
  /// be expanded further.
  ByteDefinition getByteDefinitionIfIntermediateOperand(Value *V);

  /// Get the set of all final values which use \p V.
  const DenseSet<Value *> &getFinalUsers(Value *V);

  /// Check if the provided value is known to be an intermediate value.
  bool checkIfIntermediate(Value *V) { return checkIfIntermediate(V, false); }

  /// Iterate over all instructions in a function over several passes to
  /// identify all final values and their byte definitions.
  std::vector<Instruction *>
  collectPIICandidates(Function &F, unsigned MaxCollectionIterations);
};

ByteVector ByteExpander::visitAnd(BinaryOperator &I) {
  const ByteDefinition RhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(1));
  if (!RhsDef)
    return {};
  const ByteDefinition LhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!LhsDef)
    return {};

  const ByteLayout &Layout = LhsDef.getLayout();
  const unsigned NumBytes = Layout.getNumBytes();

  ByteVector BV;
  BV.reserve(NumBytes);

  for (unsigned ByteIdx = 0; ByteIdx < NumBytes; ++ByteIdx) {
    const Byte Lhs = LhsDef.getByte(ByteIdx);
    const Byte Rhs = RhsDef.getByte(ByteIdx);

    if (Lhs == Rhs) {
      BV.emplace_back(Lhs, ByteUse::AllOperands);
      continue;
    }

    if (Lhs.isConstant()) {
      if (Lhs.getConstant() == 0) {
        BV.emplace_back(Byte::zeroes(), 0);
        continue;
      }
      if (Lhs.getConstant() == Byte::AllOnes) {
        BV.emplace_back(Rhs, 1);
        continue;
      }
    }
    if (Rhs.isConstant()) {
      if (Rhs.getConstant() == 0) {
        BV.emplace_back(Byte::zeroes(), 1);
        continue;
      }
      if (Rhs.getConstant() == Byte::AllOnes) {
        BV.emplace_back(Lhs, 0);
        continue;
      }
    }

    if (Lhs == Rhs) {
      BV.emplace_back(Lhs, ByteUse::AllOperands);
      continue;
    }
    return {};
  }

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitOr(BinaryOperator &I) {
  const ByteDefinition RhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(1));
  if (!RhsDef)
    return {};
  const ByteDefinition LhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!LhsDef)
    return {};

  const ByteLayout &Layout = LhsDef.getLayout();
  const unsigned NumBytes = Layout.getNumBytes();

  ByteVector BV;
  BV.reserve(NumBytes);

  for (unsigned ByteIdx = 0; ByteIdx < NumBytes; ++ByteIdx) {
    const Byte Lhs = LhsDef.getByte(ByteIdx);
    const Byte Rhs = RhsDef.getByte(ByteIdx);

    if (Lhs == Rhs) {
      BV.emplace_back(Lhs, ByteUse::AllOperands);
      continue;
    }

    if (Lhs.isConstant()) {
      if (Lhs.getConstant() == 0) {
        BV.emplace_back(Rhs, 1);
        continue;
      }
      if (Lhs.getConstant() == Byte::AllOnes) {
        BV.emplace_back(Byte::ones(), 0);
        continue;
      }
    }

    if (Rhs.isConstant()) {
      if (Rhs.getConstant() == 0) {
        BV.emplace_back(Lhs, 0);
        continue;
      }
      if (Rhs.getConstant() == Byte::AllOnes) {
        BV.emplace_back(Byte::ones(), 1);
        continue;
      }
    }

    return {};
  }

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitXor(BinaryOperator &I) {
  const ByteDefinition RhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(1));
  if (!RhsDef)
    return {};
  const ByteDefinition LhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!LhsDef)
    return {};

  const ByteLayout &Layout = LhsDef.getLayout();
  const unsigned NumBytes = Layout.getNumBytes();

  ByteVector BV;
  BV.reserve(NumBytes);

  for (unsigned ByteIdx = 0; ByteIdx < NumBytes; ++ByteIdx) {
    const Byte Lhs = LhsDef.getByte(ByteIdx);
    const Byte Rhs = RhsDef.getByte(ByteIdx);
    if (Lhs == Rhs)
      BV.emplace_back(Byte::zeroes(), ByteUse::AllOperands);
    else if (Lhs.isConstant() && Lhs.getConstant() == 0)
      BV.emplace_back(Rhs, 1);
    else if (Rhs.isConstant() && Rhs.getConstant() == 0)
      BV.emplace_back(Lhs, 0);
    else
      return {};
  }

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitShl(BinaryOperator &I) {
  const auto *Const = dyn_cast<Constant>(I.getOperand(1));
  if (!Const)
    return {};

  const ByteDefinition BaseDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!BaseDef)
    return {};

  const unsigned NumBytes = BaseDef.getLayout().getNumBytes();

  if (isa<ConstantInt>(Const)) {
    const unsigned ShAmt = Const->getUniqueInteger().getLimitedValue();
    if (ShAmt % Byte::BitWidth != 0)
      return {};
    const unsigned ByteShAmt = std::min(ShAmt / Byte::BitWidth, NumBytes);

    ByteVector BV;
    BV.reserve(NumBytes);
    BV.append(ByteShAmt, ByteUse(Byte::zeroes(), 0));
    for (unsigned ByteIdx = 0; ByteIdx + ByteShAmt < NumBytes; ++ByteIdx)
      BV.emplace_back(BaseDef.getByte(ByteIdx), 0);

    assert(BV.size() == NumBytes);
    return BV;
  }

  assert(Const->getType()->isVectorTy());

  ByteVector BV;
  BV.reserve(NumBytes);

  const unsigned NumBytesPerElt = BaseDef.getLayout().NumBytesPerElement;
  for (unsigned EltIdx = 0; EltIdx < BaseDef.getLayout().NumVecElements;
       ++EltIdx) {
    const auto *ConstInt =
        dyn_cast<ConstantInt>(Const->getAggregateElement(EltIdx));
    if (!ConstInt)
      return {};
    const unsigned ShAmt = ConstInt->getValue().getLimitedValue();
    if (ShAmt % Byte::BitWidth != 0)
      return {};

    const unsigned ByteShAmt = std::min(ShAmt / Byte::BitWidth, NumBytesPerElt);
    BV.append(ByteShAmt, ByteUse(Byte::zeroes(), 0));
    for (unsigned ByteIdx = 0; ByteIdx + ByteShAmt < NumBytesPerElt; ++ByteIdx)
      BV.emplace_back(BaseDef.getByte(EltIdx * NumBytesPerElt + ByteIdx), 0);
  }

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitLShr(BinaryOperator &I) {
  const auto *Const = dyn_cast<Constant>(I.getOperand(1));
  if (!Const)
    return {};

  const ByteDefinition BaseDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!BaseDef)
    return {};

  const unsigned NumBytes = BaseDef.getLayout().getNumBytes();

  if (isa<ConstantInt>(Const)) {
    const unsigned ShAmt = Const->getUniqueInteger().getLimitedValue();
    if (ShAmt % Byte::BitWidth != 0)
      return {};
    const unsigned ByteShAmt = std::min(ShAmt / Byte::BitWidth, NumBytes);

    ByteVector BV;
    BV.reserve(NumBytes);
    for (unsigned ByteIdx = ByteShAmt; ByteIdx < NumBytes; ++ByteIdx)
      BV.emplace_back(BaseDef.getByte(ByteIdx), 0);

    BV.append(ByteShAmt, ByteUse(Byte::zeroes(), 0));

    assert(BV.size() == NumBytes);
    return BV;
  }

  assert(Const->getType()->isVectorTy());

  ByteVector BV;
  BV.reserve(NumBytes);

  const unsigned NumBytesPerElt = BaseDef.getLayout().NumBytesPerElement;

  for (unsigned EltIdx = 0; EltIdx < BaseDef.getLayout().NumVecElements;
       ++EltIdx) {
    const auto *ConstInt =
        dyn_cast<ConstantInt>(Const->getAggregateElement(EltIdx));
    if (!ConstInt)
      return {};
    const unsigned ShAmt = ConstInt->getValue().getLimitedValue();
    if (ShAmt % Byte::BitWidth != 0)
      return {};
    const unsigned ByteShAmt = std::min(ShAmt / Byte::BitWidth, NumBytesPerElt);
    for (unsigned ByteIdx = ByteShAmt; ByteIdx < NumBytesPerElt; ++ByteIdx)
      BV.emplace_back(BaseDef.getByte(EltIdx * NumBytesPerElt + ByteIdx), 0);

    BV.append(ByteShAmt, ByteUse(Byte::zeroes(), 0));
  }

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitTruncInst(TruncInst &I) {
  const std::optional<ByteLayout> Layout = getByteLayout(I.getType());
  if (!Layout)
    return {};

  const std::optional<ByteLayout> SrcLayout =
      getByteLayout(I.getOperand(0)->getType());
  if (!SrcLayout)
    return {};

  const ByteDefinition SrcDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!SrcDef)
    return {};

  ByteVector BV;
  const unsigned NumBytes = Layout->getNumBytes();
  BV.reserve(NumBytes);

  const unsigned NumBytesPerElt = Layout->NumBytesPerElement;
  const unsigned NumSrcBytesPerElt = SrcLayout->NumBytesPerElement;
  for (unsigned EltIdx = 0; EltIdx < Layout->NumVecElements; ++EltIdx)
    for (unsigned ByteIdx = 0; ByteIdx < NumBytesPerElt; ++ByteIdx)
      BV.emplace_back(SrcDef.getByte(EltIdx * NumSrcBytesPerElt + ByteIdx), 0);

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitZExtInst(ZExtInst &I) {
  const std::optional<ByteLayout> Layout = getByteLayout(I.getType());
  if (!Layout)
    return {};

  const ByteDefinition SrcDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!SrcDef)
    return {};

  ByteVector BV;
  const unsigned NumBytes = Layout->getNumBytes();
  BV.reserve(NumBytes);

  const unsigned NumSrcBytesPerElt = SrcDef.getLayout().NumBytesPerElement;
  const unsigned NumZExtBytesPerElt =
      Layout->NumBytesPerElement - NumSrcBytesPerElt;

  unsigned SrcIdx = 0;
  for (unsigned EltIdx = 0; EltIdx < Layout->NumVecElements; ++EltIdx) {
    for (unsigned ByteIdx = 0; ByteIdx < NumSrcBytesPerElt; ++SrcIdx, ++ByteIdx)
      BV.emplace_back(SrcDef.getByte(SrcIdx), 0);

    BV.append(NumZExtBytesPerElt, ByteUse(Byte::zeroes(), 0));
  }

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitBitCastInst(BitCastInst &I) {
  const std::optional<ByteLayout> Layout = getByteLayout(I.getType());
  if (!Layout)
    return {};

  const ByteDefinition SrcDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!SrcDef)
    return {};

  const unsigned NumBytes = Layout->getNumBytes();
  ByteVector BV;
  BV.reserve(NumBytes);
  for (unsigned Idx = 0; Idx < NumBytes; ++Idx)
    BV.emplace_back(SrcDef.getByte(Idx), 0);

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitExtractElementInst(ExtractElementInst &I) {
  const ByteDefinition VecDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!VecDef)
    return {};

  const auto *VecIdx = dyn_cast<ConstantInt>(I.getOperand(1));
  if (!VecIdx)
    return {};

  const unsigned NumBytes = VecDef.getLayout().NumBytesPerElement;
  const unsigned ByteOffset = VecIdx->getLimitedValue() * NumBytes;
  ByteVector BV;
  BV.reserve(NumBytes);

  for (unsigned ByteIdx = 0; ByteIdx < NumBytes; ++ByteIdx)
    BV.emplace_back(VecDef.getByte(ByteIdx + ByteOffset), 0);

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitInsertElementInst(InsertElementInst &I) {
  const ByteDefinition VecDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  if (!VecDef)
    return {};

  const ByteDefinition EltDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(1));
  if (!EltDef)
    return {};

  const auto *VecIdx = dyn_cast<ConstantInt>(I.getOperand(2));
  if (!VecIdx)
    return {};

  const unsigned NumBytes = VecDef.getLayout().getNumBytes();
  const unsigned NumBytesPerElt = VecDef.getLayout().NumBytesPerElement;

  ByteVector BV;
  BV.reserve(NumBytes);
  for (unsigned EltIdx = 0; EltIdx < VecDef.getLayout().NumVecElements;
       ++EltIdx) {
    if (EltIdx == VecIdx->getLimitedValue()) {
      for (unsigned ByteIdx = 0; ByteIdx < NumBytesPerElt; ++ByteIdx)
        BV.emplace_back(EltDef.getByte(ByteIdx), 0);
    } else {
      for (unsigned ByteIdx = 0; ByteIdx < NumBytesPerElt; ++ByteIdx)
        BV.emplace_back(VecDef.getByte(EltIdx * NumBytesPerElt + ByteIdx), 1);
    }
  }

  assert(BV.size() == NumBytes);
  return BV;
}

ByteVector ByteExpander::visitShuffleVectorInst(ShuffleVectorInst &I) {
  const std::optional<ByteLayout> Layout = getByteLayout(I.getType());
  if (!Layout)
    return {};

  const ByteDefinition LhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(0));
  const ByteDefinition RhsDef =
      getByteDefinitionIfIntermediateOperand(I.getOperand(1));
  if (!LhsDef || !RhsDef)
    return {};

  const int LhsSize = LhsDef.getLayout().NumVecElements;
  const unsigned NumBytes = Layout->getNumBytes();
  const unsigned NumBytesPerElt = Layout->NumBytesPerElement;

  ByteVector BV;
  BV.reserve(NumBytes);

  for (unsigned EltIdx = 0; EltIdx < Layout->NumVecElements; ++EltIdx) {
    const int Idx = I.getMaskValue(EltIdx);
    if (Idx < 0) {
      auto *Poison = PoisonValue::get(I.getType()->getElementType());
      for (unsigned ByteIdx = 0; ByteIdx < NumBytesPerElt; ++ByteIdx)
        BV.emplace_back(Byte(*Poison, 0), ByteIdx);
    } else if (Idx < LhsSize) {
      for (unsigned ByteIdx = 0; ByteIdx < NumBytesPerElt; ++ByteIdx)
        BV.emplace_back(
            LhsDef.getByte(static_cast<unsigned>(Idx) * NumBytesPerElt +
                           ByteIdx),
            0);
    } else {
      for (unsigned ByteIdx = 0; ByteIdx < NumBytesPerElt; ++ByteIdx)
        BV.emplace_back(RhsDef.getByte(static_cast<unsigned>(Idx - LhsSize) *
                                           NumBytesPerElt +
                                       ByteIdx),
                        1);
    }
  }

  assert(BV.size() == NumBytes);
  return BV;
}

void ByteExpander::updateFinalUsers(Value *V) {
  assert(!isa<Constant>(V));

  // FIXME: Old users are copied because iterator is potentially invalidated by
  // intermediacy checks.
  DenseSet<Value *> OldFinalUsers = getFinalUsers(V);

  DenseSet<Value *> NewFinalUsers;
  for (Value *User : OldFinalUsers) {
    if (!Definitions.contains(User) || !checkIfIntermediate(User)) {
      NewFinalUsers.insert(User);
      continue;
    }
    const DenseSet<Value *> &NestedUses = getFinalUsers(User);
    NewFinalUsers.insert_range(NestedUses);
  }

  FinalUsers[V] = std::move(NewFinalUsers);
}

const DenseSet<Value *> &ByteExpander::getFinalUsers(Value *V) {
  assert(!isa<Constant>(V));

  auto It = FinalUsers.find(V);

  if (It != FinalUsers.end())
    return It->getSecond();

  DenseSet<Value *> &Uses = FinalUsers[V];
  for (Use &U : V->uses())
    Uses.insert(U.getUser());

  return Uses;
}

ByteVector *ByteExpander::expandByteDefinition(Value *V) {
  auto It = Definitions.find(V);
  if (It == Definitions.end())
    return nullptr;
  return &It->getSecond();
}

ByteDefinition ByteExpander::getByteDefinition(Value *V, bool ExpandDef) {
  const std::optional<ByteLayout> Layout = getByteLayout(V->getType());
  if (!Layout)
    return ByteDefinition::invalid();

  if (ExpandDef)
    if (ByteVector *BV = expandByteDefinition(V))
      return ByteDefinition::vector(*BV, *Layout);

  return ByteDefinition::value(*V);
}

ByteDefinition ByteExpander::getByteDefinitionIfIntermediateOperand(Value *V) {
  return getByteDefinition(V, checkIfIntermediate(V, true));
}

bool ByteExpander::checkIfIntermediate(Value *V, bool IsOperand) {
  if (isa<Constant>(V))
    return true;

  // Short-circuit check.
  if (IsOperand && V->hasOneUse())
    return true;

  const DenseSet<Value *> &FU = getFinalUsers(V);
  if (FU.size() != 1)
    return false;

  return Definitions.contains(*FU.begin());
}

std::vector<Instruction *>
ByteExpander::collectPIICandidates(Function &F,
                                   unsigned MaxCollectionIterations) {
  std::vector<Instruction *> PackedIntInsts;

  unsigned NumIterations = 1;
  for (;;) {
    LLVM_DEBUG(dbgs() << "PICP: Iteration " << NumIterations << '\n');
    bool Converged = true;

    std::vector<Instruction *> CollectedInsts;
    SetVector<Value *> WorkList;

    for (BasicBlock *BB : ReversePostOrderTraversal<Function *>(&F)) {
      for (Instruction &I : *BB) {
        ByteVector BV = visit(I);
        if (BV.empty())
          continue;

        CollectedInsts.push_back(&I);

        ByteVector &Def = Definitions[&I];
        if (Def == BV)
          continue;

        Converged = false;
        Def = std::move(BV);
        for (ByteUse &BU : Def) {
          const Byte &B = BU.getByte();
          if (!B.isConstant() && !isa<Constant>(B.getBase()))
            WorkList.insert(B.getBase());
        }

        WorkList.insert(&I);

        LLVM_DEBUG({
          dbgs() << "PICP: Updating definition: ";
          I.printAsOperand(dbgs());
          dbgs() << " = " << getByteDefinition(&I) << '\n';
        });
      }
    }

    PackedIntInsts.swap(CollectedInsts);

    if (Converged) {
      LLVM_DEBUG(dbgs() << "PICP: Reached fixpoint\n");
      break;
    }
    if (NumIterations == MaxCollectionIterations) {
      LLVM_DEBUG(dbgs() << "PICP: Reached maximum iteration limit\n");
      break;
    }

    // Update final uses of values before their operands.
    for (auto RI = WorkList.rbegin(); RI != WorkList.rend(); ++RI)
      updateFinalUsers(*RI);
    ++NumIterations;
  }

  LLVM_DEBUG(dbgs() << "PICP: Total iterations: " << NumIterations << '\n');
  return PackedIntInsts;
}

/// Return the value of all bits in a range, or std::nullopt if the bits vary.
static std::optional<bool> checkAllBits(const SmallBitVector &Mask, unsigned Lo,
                                        unsigned NumBits) {
  bool Bit = Mask[Lo];
  for (unsigned Idx = 1; Idx < NumBits; ++Idx)
    if (Mask[Lo + Idx] != Bit)
      return std::nullopt;
  return Bit;
}

/// Structure for tracking the set of bytes of a final value which are produced
/// by a given byte pack.
struct PartialBytePack {

  /// The value which produces the subset of bytes of a final value.
  /// The byte pack is invalid if this pointer is null.
  Value *BytePack;
  /// A mask which identifies which bytes of a final value are provided by the
  /// given byte pack. If a mask bit is not set, then the corresponding byte of
  /// the byte pack must be zero.
  SmallBitVector SetBytes;

  PartialBytePack(Value *BytePack, SmallBitVector SetBytes)
      : BytePack(BytePack), SetBytes(SetBytes) {}
  PartialBytePack(Value *BytePack, unsigned NumBytes)
      : BytePack(BytePack), SetBytes(NumBytes) {}

  static PartialBytePack invalid() { return {nullptr, {}}; }

  bool isValid() const { return BytePack != nullptr; }
};

/// Construct an integer whose bytes are set depending on the value of the
/// corresponding \p Mask bit. A bit of \p Mask corresponds to an entire byte of
/// the resulting APInt.
static APInt createMaskConstant(unsigned BitWidth, const SmallBitVector &Mask) {
  APInt BitMaskInt(BitWidth, 0);
  for (unsigned ByteIdx : Mask.set_bits()) {
    const unsigned BitIdx = ByteIdx * Byte::BitWidth;
    if (BitIdx >= BitWidth)
      break;
    BitMaskInt.setBits(BitIdx, BitIdx + Byte::BitWidth);
  }
  return BitMaskInt;
}

/// Construct a mask whose bits correspond to vector elements identified by the
/// \p ByteMask, or an empty vector if the \p Bytemask does not identify whole
/// vector elements.
static SmallBitVector getVectorElementMask(SmallBitVector &ByteMask,
                                           unsigned NumBytesPerElement) {
  if (ByteMask.size() % NumBytesPerElement != 0)
    return {};
  const unsigned NumElts = ByteMask.size() / NumBytesPerElement;

  SmallBitVector EltMask;
  EltMask.reserve(NumElts);
  for (unsigned EltIdx = 0; EltIdx < NumElts; ++EltIdx) {
    const std::optional<bool> Bits =
        checkAllBits(ByteMask, EltIdx * NumBytesPerElement, NumBytesPerElement);
    if (!Bits)
      return {};
    EltMask.push_back(*Bits);
  }

  assert(EltMask.size() == NumElts);
  return EltMask;
}

/// A key for the CastCache of the BytePackFolder.
struct CastEntry {
  /// The value being casted.
  Value *Base;
  /// The type being casted into.
  Type *CastTy;
  /// The opcode of the cast instruction.
  Instruction::CastOps OpCode;

  struct MapInfo {
    static CastEntry getEmptyKey() {
      return {DenseMapInfo<Value *>::getEmptyKey(),
              DenseMapInfo<Type *>::getEmptyKey(),
              DenseMapInfo<Instruction::CastOps>::getEmptyKey()};
    }
    static CastEntry getTombstoneKey() {
      return {DenseMapInfo<Value *>::getTombstoneKey(),
              DenseMapInfo<Type *>::getTombstoneKey(),
              DenseMapInfo<Instruction::CastOps>::getTombstoneKey()};
    }
    static unsigned getHashValue(const CastEntry &E) {
      return hash_combine(
          DenseMapInfo<Value *>::getHashValue(E.Base),
          DenseMapInfo<Type *>::getHashValue(E.CastTy),
          DenseMapInfo<Instruction::CastOps>::getHashValue(E.OpCode));
    }
    static bool isEqual(const CastEntry &Lhs, const CastEntry &Rhs) {
      return Lhs.Base == Rhs.Base && Lhs.CastTy == Rhs.CastTy &&
             Lhs.OpCode == Rhs.OpCode;
    }
  };
};

/// The class responsible for taking coalesced bytes and folding them together
/// to produce the desired final value.
///
/// When coalesced bytes are pushed, they are promoted to the target type, and
/// shifted to align the bytes to their corresponding offsets in the target
/// value.
class BytePackFolder {
  /// The target final value to produce.
  Instruction *TargetInst;
  /// The layout of the target value.
  ByteLayout Layout;
  /// The collection of intermediate partial byte packs generated while folding
  /// coalesced bytes.
  std::vector<PartialBytePack> WorkList;
  /// The list of non-cast instructions generated while folding coalesced bytes.
  SmallVector<Instruction *> Insts;
  /// A dedicated partial byte pack for collecting vector-aligned coalesced
  /// bytes, if the target value is a vector type.
  PartialBytePack VectorAlignedPack;
  /// A cache holding all value casts needed, to avoid generating duplicate
  /// casts.
  MapVector<CastEntry, Instruction *,
            DenseMap<CastEntry, unsigned, CastEntry::MapInfo>>
      CastCache;

  /// Create or reuse a cast of a given value.
  Value *pushCast(Instruction::CastOps OpCode, Value *V, Type *DstTy) {
    if (V->getType() == DstTy)
      return V;

    CastEntry E{V, DstTy, OpCode};
    auto *It = CastCache.find(E);
    if (It != CastCache.end())
      return It->second;

    auto *CI = CastInst::Create(OpCode, V, DstTy, V->getName() + ".cast");
    CastCache[E] = CI;

    LLVM_DEBUG({
      dbgs() << "PICP [";
      TargetInst->printAsOperand(dbgs());
      dbgs() << "]: Queuing cast " << *CI << '\n';
    });
    return CI;
  }

  Instruction *pushInst(Instruction *I) {
    // Cast instructions should be handled with pushCast.
    assert(!isa<CastInst>(I));
    Insts.push_back(I);

    LLVM_DEBUG({
      dbgs() << "PICP [";
      TargetInst->printAsOperand(dbgs());
      dbgs() << "]: Queuing inst " << *I << '\n';
    });
    return I;
  }

  /// Common functionality for promoting coalesced bytes to a vector.
  bool pushToVectorImpl(Value *V, SmallBitVector &ByteMask, unsigned NumSrcElts,
                        int ShrEltOffset, const Twine &Name) {
    auto *TargetVecTy = cast<FixedVectorType>(TargetInst->getType());
    auto *I32Ty = IntegerType::getInt32Ty(V->getContext());

    // Try to push bytes to the vector-aligned builder.
    SmallBitVector VecMask =
        getVectorElementMask(ByteMask, Layout.NumBytesPerElement);
    if (!VecMask.empty()) {
      if (!VectorAlignedPack.isValid())
        VectorAlignedPack = PartialBytePack(
            ConstantVector::getNullValue(TargetVecTy), Layout.getNumBytes());

      if (NumSrcElts == 1) {
        // Insert a single element
        assert(ShrEltOffset <= 0);
        VectorAlignedPack.BytePack = pushInst(InsertElementInst::Create(
            VectorAlignedPack.BytePack, V,
            ConstantInt::get(I32Ty, -ShrEltOffset), Name + ".insert"));
        VectorAlignedPack.SetBytes |= ByteMask;
        return true;
      }

      assert(isa<FixedVectorType>(V->getType()));

      if (NumSrcElts != Layout.NumVecElements) {
        // We need to construct a vector of the same size as the vector-aligned
        // byte pack before shuffling it in.
        SmallVector<int> ExtractMask;
        ExtractMask.reserve(Layout.NumVecElements);
        for (unsigned EltIdx = 0; EltIdx < Layout.NumVecElements; ++EltIdx) {
          if (VecMask.test(EltIdx)) {
            const int SrcIdx = static_cast<int>(EltIdx) + ShrEltOffset;
            assert(SrcIdx >= 0);
            ExtractMask.push_back(SrcIdx);
          } else
            ExtractMask.push_back(PoisonMaskElem);
        }
        assert(ExtractMask.size() == Layout.NumVecElements);

        V = pushInst(new ShuffleVectorInst(V, ExtractMask, Name + ".extract"));
        // We have accounted for the shift already, so no need to account for it
        // when shuffling into the vector-aligned byte pack.
        ShrEltOffset = 0;
      }

      assert(V->getType() == TargetVecTy);

      if (VecMask.all()) {
        VectorAlignedPack.BytePack = V;
        VectorAlignedPack.SetBytes.set();
        return true;
      }

      SmallVector<int> ShuffleMask;
      ShuffleMask.reserve(Layout.NumVecElements);
      for (unsigned EltIdx = 0; EltIdx < Layout.NumVecElements; ++EltIdx) {
        if (VecMask.test(EltIdx)) {
          const int SrcIdx = static_cast<int>(EltIdx) + ShrEltOffset;
          assert(SrcIdx >= 0);
          ShuffleMask.push_back(SrcIdx);
        } else
          ShuffleMask.push_back(EltIdx + Layout.NumVecElements);
      }
      assert(ShuffleMask.size() == Layout.NumVecElements);

      // We can shuffle directly into the vector-aligned byte pack.
      VectorAlignedPack.BytePack = pushInst(new ShuffleVectorInst(
          V, VectorAlignedPack.BytePack, ShuffleMask, Name + ".shuffle"));
      VectorAlignedPack.SetBytes |= ByteMask;
      return true;
    }

    // Otherwise, just extract and mask the relevant elements, and append to the
    // worklist.

    if (NumSrcElts == 1) {
      assert(ShrEltOffset <= 0);
      V = pushInst(InsertElementInst::Create(
          ConstantVector::getNullValue(TargetVecTy), V,
          ConstantInt::get(I32Ty, -ShrEltOffset), Name + ".insert"));
    } else if (NumSrcElts != Layout.NumVecElements) {
      SmallVector<int> ShuffleMask;
      ShuffleMask.reserve(Layout.NumVecElements);
      ShuffleMask.append(std::max(0, -ShrEltOffset), Layout.NumVecElements);
      for (unsigned SrcIdx = std::max(0, ShrEltOffset);
           SrcIdx < Layout.NumVecElements; ++SrcIdx)
        ShuffleMask.push_back(SrcIdx);
      ShuffleMask.append(std::max(0, ShrEltOffset), Layout.NumVecElements);
      assert(ShuffleMask.size() == Layout.NumVecElements);

      V = pushInst(
          new ShuffleVectorInst(V, ConstantVector::getNullValue(V->getType()),
                                ShuffleMask, Name + ".shuffle"));
    }

    assert(V->getType() == TargetVecTy);

    const unsigned TargetBitWidth = Layout.getNumBytes() * Byte::BitWidth;
    const unsigned TargetEltBitWidth =
        Layout.NumBytesPerElement * Byte::BitWidth;
    Type *TargetEltTy = TargetVecTy->getElementType();

    APInt MaskBits = createMaskConstant(TargetBitWidth, ByteMask);
    SmallVector<Constant *> EltwiseMask;
    EltwiseMask.reserve(Layout.NumVecElements);
    for (unsigned EltIdx = 0; EltIdx < Layout.NumVecElements; ++EltIdx)
      EltwiseMask.push_back(ConstantInt::get(
          TargetEltTy,
          MaskBits.extractBits(TargetEltBitWidth, EltIdx * TargetEltBitWidth)));

    V = pushInst(BinaryOperator::CreateAnd(V, ConstantVector::get(EltwiseMask),
                                           Name + ".mask"));

    WorkList.emplace_back(V, ByteMask);
    return true;
  }

  bool pushIntegerToInteger(CoalescedBytes CB) {
    assert(isa<IntegerType>(CB.Base->getType()));
    auto *TargetIntTy = cast<IntegerType>(TargetInst->getType());

    const unsigned NumTargetBytes = Layout.getNumBytes();
    Value *V = CB.Base;
    const unsigned NumSrcBytes = getByteLayout(V->getType())->getNumBytes();
    const StringRef &Name = V->getName();

    // Transformation: shr -> trunc -> mask -> zext -> shl
    if (const unsigned ShrAmt = CB.getShrBits())
      V = pushInst(BinaryOperator::CreateLShr(
          V, ConstantInt::get(V->getType(), ShrAmt), Name + ".shift"));

    if (NumSrcBytes > NumTargetBytes)
      V = pushCast(Instruction::Trunc, V, TargetIntTy);

    const unsigned ShlByteOffset = CB.getShlBytes();
    const unsigned NumBytesToCheck = std::min(
        ShlByteOffset < NumTargetBytes ? NumTargetBytes - ShlByteOffset : 0,
        CB.getShrBytes() < NumSrcBytes ? NumSrcBytes - CB.getShrBytes() : 0);
    if (!checkAllBits(CB.Mask, ShlByteOffset, NumBytesToCheck)) {
      SmallBitVector RelMask = CB.Mask;
      RelMask >>= ShlByteOffset;
      Constant *Mask = ConstantInt::get(
          V->getType(),
          createMaskConstant(V->getType()->getIntegerBitWidth(), RelMask));
      V = pushInst(BinaryOperator::CreateAnd(V, Mask, Name + ".mask"));
    }

    if (NumSrcBytes < NumTargetBytes)
      V = pushCast(Instruction::ZExt, V, TargetIntTy);

    if (const unsigned ShlAmt = CB.getShlBits())
      V = pushInst(BinaryOperator::CreateShl(
          V, ConstantInt::get(V->getType(), ShlAmt), Name + ".shift"));

    WorkList.emplace_back(V, CB.Mask);
    return true;
  }

  bool pushIntegerToVector(CoalescedBytes CB) {
    assert(isa<IntegerType>(CB.Base->getType()));
    auto *TargetVecTy = cast<FixedVectorType>(TargetInst->getType());
    Type *TargetEltTy = TargetVecTy->getElementType();

    Value *V = CB.Base;
    const unsigned NumSrcBytes =
        V->getType()->getIntegerBitWidth() / Byte::BitWidth;
    const StringRef &Name = V->getName();

    // Give up if bytes are obtained from a strange offset.
    if (CB.SignedShrByteOffset % Layout.NumBytesPerElement != 0)
      return {};
    const int ShrEltOffset =
        CB.SignedShrByteOffset / static_cast<int>(Layout.NumBytesPerElement);

    // Give up if the source integer does not decompose naturally into vector
    // elements.
    if (NumSrcBytes % Layout.NumBytesPerElement != 0)
      return {};
    const unsigned NumSrcElts = NumSrcBytes / Layout.NumBytesPerElement;

    if (NumSrcElts > 1) {
      auto *CastTy = FixedVectorType::get(TargetEltTy, NumSrcElts);
      V = pushCast(Instruction::BitCast, V, CastTy);
    }

    return pushToVectorImpl(V, CB.Mask, NumSrcElts, ShrEltOffset, Name);
  }

  bool pushVectorToInteger(CoalescedBytes CB) {
    assert(isa<FixedVectorType>(CB.Base->getType()));
    auto *TargetIntTy = cast<IntegerType>(TargetInst->getType());

    const unsigned NumTargetBytes = Layout.getNumBytes();
    Value *V = CB.Base;
    const StringRef &Name = V->getName();
    ByteLayout VecLayout = *getByteLayout(V->getType());

    // For sub-element accesses, try to subdivide the vector into smaller
    // elements.
    if (VecLayout.NumBytesPerElement > NumTargetBytes) {
      if (VecLayout.NumBytesPerElement % NumTargetBytes != 0)
        return {};

      const unsigned SplitFactor =
          VecLayout.NumBytesPerElement / NumTargetBytes;
      auto *NewTy = FixedVectorType::get(TargetIntTy, VecLayout.NumVecElements *
                                                          SplitFactor);
      V = pushCast(Instruction::BitCast, V, NewTy);
      VecLayout = *getByteLayout(V->getType());
    }

    // Give up if bytes are obtained from a strange offset.
    if (CB.SignedShrByteOffset % VecLayout.NumBytesPerElement != 0)
      return {};

    int ShrEltOffset =
        CB.SignedShrByteOffset / static_cast<int>(VecLayout.NumBytesPerElement);

    // Give up if the target integer does not decompose naturally into vector
    // elements.
    if (NumTargetBytes % VecLayout.NumBytesPerElement != 0)
      return {};
    const unsigned NumTargetElts =
        NumTargetBytes / VecLayout.NumBytesPerElement;

    auto *I32Ty = IntegerType::getInt32Ty(V->getContext());

    // Coarsely isolate elements of interest, and use a bitmask to clean up the
    // rest.
    const bool NeedsBitMask = [&] {
      if (NumTargetElts == 1) {
        // Extract the unique relevant element
        const int ExtractIdx = ShrEltOffset;
        assert(ExtractIdx >= 0);
        V = pushInst(ExtractElementInst::Create(
            V, ConstantInt::get(I32Ty, ExtractIdx), Name + ".extract"));
        ShrEltOffset = 0;
        return !CB.Mask.all();
      }

      if (NumTargetElts != VecLayout.NumVecElements) {
        bool IsVectorAligned = true;

        // Extract all relevant elements into a shufflevector
        SmallVector<int> ShuffleMask;
        ShuffleMask.reserve(NumTargetElts);

        for (unsigned EltIdx = 0; EltIdx < NumTargetElts; ++EltIdx) {
          const std::optional<bool> EltMask =
              checkAllBits(CB.Mask, EltIdx * VecLayout.NumBytesPerElement,
                           VecLayout.NumBytesPerElement);

          IsVectorAligned &= EltMask.has_value();
          if (!EltMask || *EltMask) {
            const int ExtractIdx = static_cast<int>(EltIdx) + ShrEltOffset;
            assert(ExtractIdx >= 0);
            ShuffleMask.push_back(ExtractIdx);
          } else {
            ShuffleMask.push_back(VecLayout.NumVecElements);
          }
        }

        V = pushInst(
            new ShuffleVectorInst(V, ConstantVector::getNullValue(V->getType()),
                                  ShuffleMask, Name + ".shuffle"));
        V = pushCast(Instruction::BitCast, V, TargetIntTy);

        ShrEltOffset = 0;
        return !IsVectorAligned;
      }

      V = pushCast(Instruction::BitCast, V, TargetIntTy);
      return !CB.Mask.all();
    }();

    assert(V->getType() == TargetIntTy);

    const int ShrBitOffset = ShrEltOffset *
                             static_cast<int>(VecLayout.NumBytesPerElement) *
                             static_cast<int>(Byte::BitWidth);
    if (ShrBitOffset > 0)
      V = pushInst(BinaryOperator::CreateLShr(
          V, ConstantInt::get(V->getType(), ShrBitOffset), Name + ".shift"));
    else if (ShrBitOffset < 0)
      V = pushInst(BinaryOperator::CreateShl(
          V, ConstantInt::get(V->getType(), -ShrBitOffset), Name + ".shift"));

    if (NeedsBitMask) {
      // Mask out unwanted bytes.
      Constant *Mask = ConstantInt::get(
          TargetIntTy, createMaskConstant(TargetIntTy->getBitWidth(), CB.Mask));
      V = pushInst(BinaryOperator::CreateAnd(V, Mask, Name + ".mask"));
    }

    WorkList.emplace_back(V, CB.Mask);
    return true;
  }

  bool pushVectorToVector(CoalescedBytes CB) {
    assert(isa<FixedVectorType>(CB.Base->getType()));
    auto *TargetVecTy = cast<FixedVectorType>(TargetInst->getType());
    Type *TargetEltTy = TargetVecTy->getElementType();

    const ByteLayout SrcLayout = *getByteLayout(CB.Base->getType());
    Value *V = CB.Base;
    const StringRef &Name = V->getName();

    // Give up if the source vector cannot be converted to match the elements of
    // the target vector.
    if (SrcLayout.getNumBytes() % Layout.NumBytesPerElement != 0)
      return {};
    const unsigned NumSrcElts =
        SrcLayout.getNumBytes() / Layout.NumBytesPerElement;

    // Give up if the shift amount is not aligned to the target vector.
    if (CB.SignedShrByteOffset % Layout.NumBytesPerElement != 0)
      return {};

    const int ShrEltOffset =
        CB.SignedShrByteOffset / static_cast<int>(Layout.NumBytesPerElement);

    Type *SrcTy;
    if (NumSrcElts > 1)
      SrcTy = FixedVectorType::get(TargetEltTy, NumSrcElts);
    else
      SrcTy = TargetEltTy;

    V = pushCast(Instruction::BitCast, V, SrcTy);

    return pushToVectorImpl(V, CB.Mask, NumSrcElts, ShrEltOffset, Name);
  }

  PartialBytePack mergeIntegerPacks(PartialBytePack &Lhs,
                                    PartialBytePack &Rhs) {
    assert(isa<IntegerType>(Lhs.BytePack->getType()) &&
           isa<IntegerType>(Rhs.BytePack->getType()));
    Value *Merge = pushInst(BinaryOperator::CreateDisjointOr(
        Lhs.BytePack, Rhs.BytePack, TargetInst->getName() + ".merge", nullptr));
    return {Merge, Lhs.SetBytes | Rhs.SetBytes};
  }

  PartialBytePack mergeVectorPacks(PartialBytePack &Lhs, PartialBytePack &Rhs) {
    assert(isa<FixedVectorType>(Lhs.BytePack->getType()) &&
           isa<FixedVectorType>(Rhs.BytePack->getType()));
    SmallVector<int> ShuffleMask;
    ShuffleMask.reserve(Layout.NumVecElements);
    for (unsigned EltIdx = 0; EltIdx < Layout.NumVecElements; ++EltIdx) {
      const unsigned Lo = EltIdx * Layout.NumBytesPerElement;
      const std::optional<bool> LhsBits =
          checkAllBits(Lhs.SetBytes, Lo, Layout.NumBytesPerElement);
      if (!LhsBits) {
        const std::optional<bool> RhsBits =
            checkAllBits(Rhs.SetBytes, Lo, Layout.NumBytesPerElement);
        if (!RhsBits) {
          ShuffleMask.clear();
          break;
        }
        ShuffleMask.push_back(*RhsBits ? EltIdx + Layout.NumVecElements
                                       : EltIdx);
        continue;
      }

      ShuffleMask.push_back(*LhsBits ? EltIdx : EltIdx + Layout.NumVecElements);
    }

    const Twine &Name = TargetInst->getName() + ".merge";
    Value *Merge;
    if (ShuffleMask.empty())
      Merge = pushInst(BinaryOperator::CreateDisjointOr(
          Lhs.BytePack, Rhs.BytePack, Name, nullptr));
    else
      Merge = pushInst(
          new ShuffleVectorInst(Lhs.BytePack, Rhs.BytePack, ShuffleMask, Name));

    return {Merge, Lhs.SetBytes | Rhs.SetBytes};
  }

public:
  BytePackFolder(Instruction *TargetV)
      : TargetInst(TargetV), Layout(*getByteLayout(TargetV->getType())),
        VectorAlignedPack(PartialBytePack::invalid()) {}

  ~BytePackFolder() {
    /// If instructions are not committed, they need to be cleaned up.

    for (auto &[_, I] : CastCache) {
      LLVM_DEBUG({
        dbgs() << "PICP [";
        TargetInst->printAsOperand(dbgs());
        dbgs() << "]: Dequeuing cast " << *I << '\n';
      });
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->deleteValue();
    }

    while (!Insts.empty()) {
      LLVM_DEBUG({
        dbgs() << "PICP [";
        TargetInst->printAsOperand(dbgs());
        dbgs() << "]: Dequeuing inst " << *Insts.back() << '\n';
      });
      Insts.back()->deleteValue();
      Insts.pop_back();
    }
  }

  /// Try to generate instructions for coalescing the given bytes and aligning
  /// them to the target value. Returns true iff this is successful.
  bool pushCoalescedBytes(CoalescedBytes CB) {
    if (CB.SignedShrByteOffset == 0)
      if (auto *Const = dyn_cast<Constant>(CB.Base)) {
        WorkList.emplace_back(
            ConstantExpr::getBitCast(Const, TargetInst->getType()), CB.Mask);
        return true;
      }

    LLVM_DEBUG({
      dbgs() << "PICP [";
      TargetInst->printAsOperand(dbgs());
      dbgs() << "]: Preparing bytes " << CB << '\n';
    });
    if (isa<FixedVectorType>(TargetInst->getType())) {
      if (isa<FixedVectorType>(CB.Base->getType()))
        return pushVectorToVector(CB);

      return pushIntegerToVector(CB);
    }

    if (isa<FixedVectorType>(CB.Base->getType()))
      return pushVectorToInteger(CB);

    return pushIntegerToInteger(CB);
  }

  /// After coalescing all byte packs individually, this folds the coalesced
  /// byte packs together to (re)produce the final value and return it.
  Value *foldBytePacks(IRBuilder<> &IRB) {
    Type *TargetTy = TargetInst->getType();

    if (VectorAlignedPack.isValid()) {
      WorkList.push_back(VectorAlignedPack);
      VectorAlignedPack = PartialBytePack::invalid();
    }

    while (WorkList.size() > 1) {
      std::vector<PartialBytePack> NewWorkList;
      NewWorkList.reserve((WorkList.size() + 1) / 2);

      for (unsigned Item = 0; Item + 1 < WorkList.size(); Item += 2) {
        PartialBytePack &Lhs = WorkList[Item];
        PartialBytePack &Rhs = WorkList[Item + 1];
        NewWorkList.push_back(isa<FixedVectorType>(TargetTy)
                                  ? mergeVectorPacks(Lhs, Rhs)
                                  : mergeIntegerPacks(Lhs, Rhs));
      }
      if (WorkList.size() % 2 == 1)
        NewWorkList.push_back(WorkList.back());

      WorkList.swap(NewWorkList);
    }

    IRB.SetInsertPoint(TargetInst);
    for (Value *I : Insts)
      IRB.Insert(I, I->getName());

    Insts.clear();

    for (auto &[E, I] : CastCache) {
      if (auto *BaseI = dyn_cast<Instruction>(E.Base))
        I->insertInto(BaseI->getParent(), *BaseI->getInsertionPointAfterDef());
      else {
        BasicBlock &BB = TargetInst->getFunction()->getEntryBlock();
        I->insertInto(&BB, BB.getFirstInsertionPt());
      }
    }

    CastCache.clear();

    // Note: WorkList may be empty if the value is known to be zero.
    return WorkList.empty() ? Constant::getNullValue(TargetTy)
                            : WorkList.back().BytePack;
  }
};

/// A final value (or an operand thereof, if the rewriter is not aggressive)
/// queued up to be reconstructed.
struct PackedIntInstruction {
  /// The target value to reconstruct.
  Instruction *TargetInst;
  /// The chosen partitioning of its bytes.
  SmallVector<CoalescedBytes, 8> CBV;

  PackedIntInstruction(Instruction &I, SmallVector<CoalescedBytes, 8> &&CBV)
      : TargetInst(&I), CBV(CBV) {}

  /// Try to reconstruct a value given its coalesced byte partitioning,
  /// returning the reconstructed value on success, or a nullptr on failure.
  Value *rewrite(IRBuilder<> &IRB) const {
    BytePackFolder BPF(TargetInst);
    for (const CoalescedBytes &CB : CBV) {
      if (!BPF.pushCoalescedBytes(CB)) {
        LLVM_DEBUG(dbgs() << "PICP: Coalescing rejected!\n");
        return nullptr;
      }
    }

    return BPF.foldBytePacks(IRB);
  }
};

/// Coalesce the bytes in a definition into a partition for rewriting.
/// If the rewriter is non-aggressive, return nullopt if the rewriting is
/// determined to be unnecessary.
static std::optional<SmallVector<CoalescedBytes, 8>>
getCoalescingOpportunity(Type *Ty, const ByteVector &BV,
                         bool AggressiveRewriting) {
  const ByteLayout Layout = *getByteLayout(Ty);
  assert(Layout.getNumBytes() == BV.size() &&
         "Byte definition has unexpected width.");

  SmallVector<CoalescedBytes, 8> CBV;
  SmallVector<int, 8> CBVOperands;
  const unsigned BitWidth = Layout.getNumBytes() * Byte::BitWidth;
  APInt ConstBits(BitWidth, 0);
  SmallBitVector ConstBytes(BV.size());

  bool OperandsAlreadyCoalesced = true;
  bool UsesSingleSource = true;
  for (unsigned ByteIdx = 0; ByteIdx < BV.size(); ++ByteIdx) {
    const ByteUse &BU = BV[ByteIdx];
    const Byte &B = BU.getByte();
    if (B.isConstant()) {
      const unsigned Const = B.getConstant();
      if (!Const)
        continue;

      ConstBits.insertBits(Const, ByteIdx * Byte::BitWidth, Byte::BitWidth);
      ConstBytes.set(ByteIdx);
    } else {
      CoalescedBytes *CB = nullptr;
      Value *Base = B.getBase();
      const int Offset =
          static_cast<int>(B.getIndex()) - static_cast<int>(ByteIdx);
      for (unsigned CBIdx = 0; CBIdx < CBV.size(); ++CBIdx) {
        if (CBV[CBIdx].alignsWith(Base, Offset)) {
          CB = &CBV[CBIdx];
          int &OpIdx = CBVOperands[CBIdx];
          if (OpIdx < 0)
            OpIdx = BU.getOperandIndex();
          else if (BU.getOperandIndex() >= 0 && OpIdx != BU.getOperandIndex()) {
            LLVM_DEBUG(dbgs()
                       << "PICP: Bytes " << *CB << " from operand " << OpIdx
                       << " can be coalesced with byte " << B
                       << " from operand " << BU.getOperandIndex() << '\n');
            OperandsAlreadyCoalesced = false;
          }
        }
      }

      if (!CB) {
        CB = &CBV.emplace_back(*Base, Offset, BV.size());
        CBVOperands.push_back(BU.getOperandIndex());
      }

      UsesSingleSource &= CB->Base == CBV.front().Base;
      CB->Mask.set(ByteIdx);
    }
  }

  if (!AggressiveRewriting) {
    if (OperandsAlreadyCoalesced && !CBV.empty()) {
      // If packed bytes from the same source and offset are not split between
      // operands, then this instruction does not need to be rewritten.
      LLVM_DEBUG(dbgs() << "PICP: Operands are already coalesced.\n");
      return std::nullopt;
    }
    if (UsesSingleSource && CBV.size() > 1) {
      // If packed bytes come from the same source, but cannot be coalesced
      // (e.g., bytes from one operand are shuffled), then rewriting this
      // instruction may lead to strange IR.
      LLVM_DEBUG(
          dbgs()
          << "PICP: Instruction rearranges bytes from a single source.\n");
      return std::nullopt;
    }
  }

  // The CBV will be used for rewriting; append the constant value that was also
  // accumulated, if nonzero.
  if (ConstBytes.any()) {
    // Create initial constant as desired type.
    if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
      SmallVector<Constant *> EltwiseMask;
      const unsigned NumBitsPerElt = Layout.NumBytesPerElement * Byte::BitWidth;
      EltwiseMask.reserve(Layout.NumVecElements);
      for (unsigned EltIdx = 0; EltIdx < Layout.NumVecElements; ++EltIdx)
        EltwiseMask.push_back(ConstantInt::get(
            VecTy->getElementType(),
            ConstBits.extractBits(NumBitsPerElt, EltIdx * NumBitsPerElt)));

      CBV.emplace_back(*ConstantVector::get(EltwiseMask), 0, ConstBytes);
    } else
      CBV.emplace_back(*ConstantInt::get(Ty, ConstBits), 0, ConstBytes);
  }

  return CBV;
}

/// Queue into \p PIIV the set of final values (or operands thereof, if the
/// rewriter is non-aggressive) which are deemed beneficial to rewrite.
static void queueRewriting(std::vector<PackedIntInstruction> &PIIV,
                           Instruction &FinalInst, ByteExpander &BE,
                           bool AggressiveRewriting) {
  SmallVector<Instruction *, 8> WorkList{&FinalInst};
  SmallPtrSet<Instruction *, 8> Seen{&FinalInst};

  do {
    Instruction *I = WorkList.back();
    WorkList.pop_back();

    const ByteVector *BV = BE.expandByteDefinition(I);
    if (!BV)
      // This instruction is beyond the analysis scope of PICP.
      continue;

    LLVM_DEBUG(dbgs() << "PICP rewrite candidate: " << *I << '\n'
                      << "             byte pack: " << BE.getByteDefinition(I)
                      << '\n');
    auto CBV = [&]() -> std::optional<SmallVector<CoalescedBytes, 8>> {
      // Short-circuit check for casts.
      if (!AggressiveRewriting && I->getNumOperands() == 1)
        return std::nullopt;

      return getCoalescingOpportunity(I->getType(), *BV, AggressiveRewriting);
    }();

    if (!CBV) {
      // Narrow rewriting to the operands of this instruction instead.
      for (Use &U : I->operands())
        if (auto *Op = dyn_cast<Instruction>(U.get()))
          if (Seen.insert(Op).second)
            WorkList.push_back(Op);
      continue;
    }

    PIIV.emplace_back(*I, std::move(*CBV));
  } while (!WorkList.empty());
}

static bool runImpl(Function &F, PackedIntegerCombineOptions Options) {
  ByteExpander BE;

  std::vector<Instruction *> PIICandidates =
      BE.collectPIICandidates(F, Options.MaxCollectionIterations);
  std::vector<PackedIntInstruction> PIIV;

  for (Instruction *I : PIICandidates) {
    if (!BE.checkIfIntermediate(I))
      queueRewriting(PIIV, *I, BE, Options.AggressiveRewriting);
    else
      LLVM_DEBUG(dbgs() << "PICP intermediate inst: " << *I << '\n'
                        << "            final user: "
                        << **BE.getFinalUsers(I).begin() << '\n');
  }

  DenseMap<Instruction *, Value *> InstSubs;
  IRBuilder<> IRB(F.getContext());
  for (const PackedIntInstruction &PII : PIIV)
    if (Value *V = PII.rewrite(IRB)) {
      LLVM_DEBUG(dbgs() << "PICP rewrite successful for " << *PII.TargetInst
                        << '\n');
      InstSubs[PII.TargetInst] = V;
    }

  if (InstSubs.empty())
    return false;

  for (auto &[OldI, NewV] : InstSubs)
    OldI->replaceAllUsesWith(NewV);

  for (auto RIt = PIICandidates.rbegin(); RIt != PIICandidates.rend(); ++RIt) {
    Instruction *I = *RIt;
    if (I->getNumUses() == 0)
      I->eraseFromParent();
  }
  return true;
}

class PackedIntegerCombineLegacyPass : public FunctionPass {
  PackedIntegerCombineOptions Options;

public:
  static char ID;

  PackedIntegerCombineLegacyPass(PackedIntegerCombineOptions Options)
      : FunctionPass(ID), Options(Options) {}

  bool runOnFunction(Function &F) override { return runImpl(F, Options); }
};
char PackedIntegerCombineLegacyPass::ID = 0;

} // namespace

PreservedAnalyses PackedIntegerCombinePass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  if (!runImpl(F, Options))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

void PackedIntegerCombinePass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<PackedIntegerCombinePass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << '<';
  if (Options.AggressiveRewriting)
    OS << "aggressive;";
  OS << "max-iterations=" << Options.MaxCollectionIterations << '>';
}

INITIALIZE_PASS(PackedIntegerCombineLegacyPass, DEBUG_TYPE,
                "Packed Integer Combine", false, false)

FunctionPass *
llvm::createPackedIntegerCombinePass(unsigned MaxCollectionIterations,
                                     bool AggressiveRewriting) {
  return new PackedIntegerCombineLegacyPass(
      {MaxCollectionIterations, AggressiveRewriting});
}
