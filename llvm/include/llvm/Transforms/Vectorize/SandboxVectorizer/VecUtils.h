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

namespace llvm::sandboxir {

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
  static unsigned getFloorPowerOf2(unsigned Num);

#ifndef NDEBUG
  /// Helper dump function for debugging.
  LLVM_DUMP_METHOD static void dump(ArrayRef<Value *> Bndl);
  LLVM_DUMP_METHOD static void dump(ArrayRef<Instruction *> Bndl);
#endif // NDEBUG
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H
