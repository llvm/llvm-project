//===- VecUtils.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collector for SandboxVectorizer related convenience functions that don't
// belong in other classes.

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/SandboxIR/Type.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
/// Traits for DenseMap.
template <> struct DenseMapInfo<SmallVector<sandboxir::Value *>> {
  static inline SmallVector<sandboxir::Value *> getEmptyKey() {
    return SmallVector<sandboxir::Value *>({(sandboxir::Value *)-1});
  }
  static inline SmallVector<sandboxir::Value *> getTombstoneKey() {
    return SmallVector<sandboxir::Value *>({(sandboxir::Value *)-2});
  }
  static unsigned getHashValue(const SmallVector<sandboxir::Value *> &Vec) {
    return hash_combine_range(Vec);
  }
  static bool isEqual(const SmallVector<sandboxir::Value *> &Vec1,
                      const SmallVector<sandboxir::Value *> &Vec2) {
    return Vec1 == Vec2;
  }
};

namespace sandboxir {

class VecUtils {
public:
  /// \Returns the number of elements in \p Ty. That is the number of lanes if a
  /// fixed vector or 1 if scalar. ScalableVectors have unknown size and
  /// therefore are unsupported.
  static int getNumElements(Type *Ty) {
    assert(!isa<ScalableVectorType>(Ty));
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getNumElements() : 1;
  }
  /// Returns \p Ty if scalar or its element type if vector.
  static Type *getElementType(Type *Ty) {
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getElementType() : Ty;
  }

  /// \Returns true if \p I1 and \p I2 are load/stores accessing consecutive
  /// memory addresses.
  template <typename LoadOrStoreT>
  static bool areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                             ScalarEvolution &SE, const DataLayout &DL) {
    static_assert(std::is_same<LoadOrStoreT, LoadInst>::value ||
                      std::is_same<LoadOrStoreT, StoreInst>::value,
                  "Expected Load or Store!");
    auto Diff = Utils::getPointerDiffInBytes(I1, I2, SE);
    if (!Diff)
      return false;
    int ElmBytes = Utils::getNumBits(I1) / 8;
    return *Diff == ElmBytes;
  }

  template <typename LoadOrStoreT>
  static bool areConsecutive(ArrayRef<Value *> &Bndl, ScalarEvolution &SE,
                             const DataLayout &DL) {
    static_assert(std::is_same<LoadOrStoreT, LoadInst>::value ||
                      std::is_same<LoadOrStoreT, StoreInst>::value,
                  "Expected Load or Store!");
    assert(isa<LoadOrStoreT>(Bndl[0]) && "Expected Load or Store!");
    auto *LastLS = cast<LoadOrStoreT>(Bndl[0]);
    for (Value *V : drop_begin(Bndl)) {
      assert(isa<LoadOrStoreT>(V) &&
             "Unimplemented: we only support StoreInst!");
      auto *LS = cast<LoadOrStoreT>(V);
      if (!VecUtils::areConsecutive(LastLS, LS, SE, DL))
        return false;
      LastLS = LS;
    }
    return true;
  }

  /// \Returns the number of vector lanes of \p Ty or 1 if not a vector.
  /// NOTE: It asserts that \p Ty is a fixed vector type.
  static unsigned getNumLanes(Type *Ty) {
    assert(!isa<ScalableVectorType>(Ty) && "Expect scalar or fixed vector");
    if (auto *FixedVecTy = dyn_cast<FixedVectorType>(Ty))
      return FixedVecTy->getNumElements();
    return 1u;
  }

  /// \Returns the expected vector lanes of \p V or 1 if not a vector.
  /// NOTE: It asserts that \p V is a fixed vector.
  static unsigned getNumLanes(Value *V) {
    return VecUtils::getNumLanes(Utils::getExpectedType(V));
  }

  /// \Returns the total number of lanes across all values in \p Bndl.
  static unsigned getNumLanes(ArrayRef<Value *> Bndl) {
    unsigned Lanes = 0;
    for (Value *V : Bndl)
      Lanes += getNumLanes(V);
    return Lanes;
  }

  /// \Returns <NumElts x ElemTy>.
  /// It works for both scalar and vector \p ElemTy.
  static Type *getWideType(Type *ElemTy, unsigned NumElts) {
    if (ElemTy->isVectorTy()) {
      auto *VecTy = cast<FixedVectorType>(ElemTy);
      ElemTy = VecTy->getElementType();
      NumElts = VecTy->getNumElements() * NumElts;
    }
    return FixedVectorType::get(ElemTy, NumElts);
  }
  /// \Returns the instruction in \p Instrs that is lowest in the BB. Expects
  /// that all instructions are in the same BB.
  static Instruction *getLowest(ArrayRef<Instruction *> Instrs) {
    Instruction *LowestI = Instrs.front();
    for (auto *I : drop_begin(Instrs)) {
      if (LowestI->comesBefore(I))
        LowestI = I;
    }
    return LowestI;
  }
  /// \Returns the lowest instruction in \p Vals, or nullptr if no instructions
  /// are found. Skips instructions not in \p BB.
  static Instruction *getLowest(ArrayRef<Value *> Vals, BasicBlock *BB) {
    // Find the first Instruction in Vals that is also in `BB`.
    auto It = find_if(Vals, [BB](Value *V) {
      return isa<Instruction>(V) && cast<Instruction>(V)->getParent() == BB;
    });
    // If we couldn't find an instruction return nullptr.
    if (It == Vals.end())
      return nullptr;
    Instruction *FirstI = cast<Instruction>(*It);
    // Now look for the lowest instruction in Vals starting from one position
    // after FirstI.
    Instruction *LowestI = FirstI;
    for (auto *V : make_range(std::next(It), Vals.end())) {
      auto *I = dyn_cast<Instruction>(V);
      // Skip non-instructions.
      if (I == nullptr)
        continue;
      // Skips instructions not in \p BB.
      if (I->getParent() != BB)
        continue;
      // If `LowestI` comes before `I` then `I` is the new lowest.
      if (LowestI->comesBefore(I))
        LowestI = I;
    }
    return LowestI;
  }

  /// If \p I is not a PHI it returns it. Else it walks down the instruction
  /// chain looking for the last PHI and returns it. \Returns nullptr if \p I is
  /// nullptr.
  static Instruction *getLastPHIOrSelf(Instruction *I) {
    Instruction *LastI = I;
    while (I != nullptr && isa<PHINode>(I)) {
      LastI = I;
      I = I->getNextNode();
    }
    return LastI;
  }

  /// If all values in \p Bndl are of the same scalar type then return it,
  /// otherwise return nullptr.
  static Type *tryGetCommonScalarType(ArrayRef<Value *> Bndl) {
    Value *V0 = Bndl[0];
    Type *Ty0 = Utils::getExpectedType(V0);
    Type *ScalarTy = VecUtils::getElementType(Ty0);
    for (auto *V : drop_begin(Bndl)) {
      Type *NTy = Utils::getExpectedType(V);
      Type *NScalarTy = VecUtils::getElementType(NTy);
      if (NScalarTy != ScalarTy)
        return nullptr;
    }
    return ScalarTy;
  }

  /// Similar to tryGetCommonScalarType() but will assert that there is a common
  /// type. So this is faster in release builds as it won't iterate through the
  /// values.
  static Type *getCommonScalarType(ArrayRef<Value *> Bndl) {
    Value *V0 = Bndl[0];
    Type *Ty0 = Utils::getExpectedType(V0);
    Type *ScalarTy = VecUtils::getElementType(Ty0);
    assert(tryGetCommonScalarType(Bndl) && "Expected common scalar type!");
    return ScalarTy;
  }
  /// \Returns the first integer power of 2 that is <= Num.
  LLVM_ABI static unsigned getFloorPowerOf2(unsigned Num);

  /// Helper struct for `matchPack()`. Describes the instructions and operands
  /// of a pack pattern.
  struct PackPattern {
    /// The insertelement instructions that form the pack pattern in bottom-up
    /// order, i.e., the first instruction in `Instrs` is the bottom-most
    /// InsertElement instruction of the pack pattern.
    /// For example in this simple pack pattern:
    ///  %Pack0 = insertelement <2 x i8> poison, i8 %v0, i64 0
    ///  %Pack1 = insertelement <2 x i8> %Pack0, i8 %v1, i64 1
    /// this is [ %Pack1, %Pack0 ].
    SmallVector<Instruction *> Instrs;
    /// The "external" operands of the pack pattern, i.e., the values that get
    /// packed into a vector, skipping the ones in `Instrs`. The operands are in
    /// bottom-up order, starting from the operands of the bottom-most insert.
    /// So in our example this would be [ %v1, %v0 ].
    SmallVector<Value *> Operands;
  };

  /// If \p I is the last instruction of a pack pattern (i.e., an InsertElement
  /// into a vector), then this function returns the instructions in the pack
  /// and the operands in the pack, else returns nullopt.
  /// Here is an example of a matched pattern:
  ///  %PackA0 = insertelement <2 x i8> poison, i8 %v0, i64 0
  ///  %PackA1 = insertelement <2 x i8> %PackA0, i8 %v1, i64 1
  /// TODO: this currently detects only simple canonicalized patterns.
  static std::optional<PackPattern> matchPack(Instruction *I) {
    // TODO: Support vector pack patterns.
    // TODO: Support out-of-order inserts.

    // Early return if `I` is not an Insert.
    if (!isa<InsertElementInst>(I))
      return std::nullopt;
    auto *BB0 = I->getParent();
    // The pack contains as many instrs as the lanes of the bottom-most Insert
    unsigned ExpectedNumInserts = VecUtils::getNumLanes(I);
    assert(ExpectedNumInserts >= 2 && "Expected at least 2 inserts!");
    PackPattern Pack;
    Pack.Operands.resize(ExpectedNumInserts);
    // Collect the inserts by walking up the use-def chain.
    Instruction *InsertI = I;
    for (auto ExpectedLane : reverse(seq<unsigned>(ExpectedNumInserts))) {
      if (InsertI == nullptr)
        return std::nullopt;
      if (InsertI->getParent() != BB0)
        return std::nullopt;
      // Check the lane.
      auto *LaneC = dyn_cast<ConstantInt>(InsertI->getOperand(2));
      if (LaneC == nullptr || LaneC->getSExtValue() != ExpectedLane)
        return std::nullopt;
      Pack.Instrs.push_back(InsertI);
      Pack.Operands[ExpectedLane] = InsertI->getOperand(1);

      Value *Op = InsertI->getOperand(0);
      if (ExpectedLane == 0) {
        // Check the topmost insert. The operand should be a Poison.
        if (!isa<PoisonValue>(Op))
          return std::nullopt;
      } else {
        InsertI = dyn_cast<InsertElementInst>(Op);
      }
    }
    return Pack;
  }

#ifndef NDEBUG
  /// Helper dump function for debugging.
  LLVM_DUMP_METHOD static void dump(ArrayRef<Value *> Bndl);
  LLVM_DUMP_METHOD static void dump(ArrayRef<Instruction *> Bndl);
#endif // NDEBUG
};

} // namespace sandboxir

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H
