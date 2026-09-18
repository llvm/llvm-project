//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collector for SandboxIR related convenience functions that don't belong in
// other classes.

#ifndef LLVM_SANDBOXIR_UTILS_H
#define LLVM_SANDBOXIR_UTILS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Verifier.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include <optional>

namespace llvm::sandboxir {

class Utils {
public:
  /// \Returns the expected type of \p Value V. For most Values this is
  /// equivalent to getType, but for stores returns the stored type, rather
  /// than void, and for ReturnInsts returns the returned type.
  static Type *getExpectedType(const Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      // A Return's value operand can be null if it returns void.
      if (auto *RI = dyn_cast<ReturnInst>(I)) {
        if (RI->getReturnValue() == nullptr)
          return RI->getType();
      }
      return getExpectedValue(I)->getType();
    }
    return V->getType();
  }

  /// \Returns the expected Value for this instruction. For most instructions,
  /// this is the instruction itself, but for stores returns the stored
  /// operand, and for ReturnInstructions returns the returned value.
  static Value *getExpectedValue(const Instruction *I) {
    if (auto *SI = dyn_cast<StoreInst>(I))
      return SI->getValueOperand();
    if (auto *RI = dyn_cast<ReturnInst>(I))
      return RI->getReturnValue();
    return const_cast<Instruction *>(I);
  }

  /// \Returns the base Value for load or store instruction \p LSI.
  template <typename LoadOrStoreT>
  static Value *getMemInstructionBase(const LoadOrStoreT *LSI) {
    static_assert(std::is_same_v<LoadOrStoreT, LoadInst> ||
                      std::is_same_v<LoadOrStoreT, StoreInst>,
                  "Expected sandboxir::Load or sandboxir::Store!");
    return LSI->Ctx.getOrCreateValue(
        getUnderlyingObject(LSI->getPointerOperand()->Val));
  }

  /// \Returns the number of bits of \p Ty.
  static unsigned getNumBits(Type *Ty, const DataLayout &DL) {
    return DL.getTypeSizeInBits(Ty->LLVMTy);
  }

  /// \Returns the number of bits required to represent the operands or return
  /// value of \p V in \p DL.
  static unsigned getNumBits(Value *V, const DataLayout &DL) {
    Type *Ty = getExpectedType(V);
    return getNumBits(Ty, DL);
  }

  /// \Returns the number of bits required to represent the operands or
  /// return value of \p I.
  static unsigned getNumBits(Instruction *I) {
    return I->getDataLayout().getTypeSizeInBits(getExpectedType(I)->LLVMTy);
  }

  /// Equivalent to MemoryLocation::getOrNone(I).
  static std::optional<llvm::MemoryLocation>
  memoryLocationGetOrNone(const Instruction *I) {
    return llvm::MemoryLocation::getOrNone(cast<llvm::Instruction>(I->Val));
  }

  static std::optional<int64_t>
  getPointersDiff(Type *ElemTyA, Value *PtrA, Type *ElemTyB, Value *PtrB,
                  const DataLayout &DL, ScalarEvolution &SE, bool StrictCheck,
                  bool CheckType, bool ExpensivePtrCheck = false) {
    assert(PtrA && PtrB && "Expected non-nullptr pointers.");

    // Make sure that A and B are different pointers.
    if (PtrA == PtrB)
      return 0;

    // Make sure that the element types are the same if required.
    if (CheckType && ElemTyA != ElemTyB)
      return std::nullopt;

    unsigned ASA = PtrA->getType()->getPointerAddressSpace();
    unsigned ASB = PtrB->getType()->getPointerAddressSpace();

    // Check that the address spaces match.
    if (ASA != ASB)
      return std::nullopt;
    unsigned IdxWidth = DL.getIndexSizeInBits(ASA);

    APInt OffsetA(IdxWidth, 0), OffsetB(IdxWidth, 0);
    const Value *PtrA1 = PtrA->stripAndAccumulateConstantOffsets(
        DL, OffsetA, /*AllowNonInbounds=*/true);
    const Value *PtrB1 = PtrB->stripAndAccumulateConstantOffsets(
        DL, OffsetB, /*AllowNonInbounds=*/true);

    std::optional<int64_t> Val;
    if (PtrA1 == PtrB1) {
      // Retrieve the address space again as pointer stripping now tracks
      // through `addrspacecast`.
      ASA = cast<PointerType>(PtrA1->getType())->getAddressSpace();
      ASB = cast<PointerType>(PtrB1->getType())->getAddressSpace();
      // Check that the address spaces match and that the pointers are valid.
      if (ASA != ASB)
        return std::nullopt;

      IdxWidth = DL.getIndexSizeInBits(ASA);
      OffsetA = OffsetA.sextOrTrunc(IdxWidth);
      OffsetB = OffsetB.sextOrTrunc(IdxWidth);

      OffsetB -= OffsetA;
      Val = OffsetB.trySExtValue();
    } else {
      // Otherwise compute the distance with SCEV between the base pointers.
      const SCEV *PtrSCEVA = SE.getSCEV(PtrA->Val);
      const SCEV *PtrSCEVB = SE.getSCEV(PtrB->Val);
      if (ExpensivePtrCheck) {
        const SCEV *DistScev =
            SE.getMinusSCEV(SE.getSCEV(PtrB->Val), SE.getSCEV(PtrA->Val));
        if (DistScev == SE.getCouldNotCompute())
          return std::nullopt;
        ConstantRange DistRange = SE.getSignedRange(DistScev);
        if (!DistRange.isSingleElement())
          return std::nullopt;
        // Handle index width (the width of Dist) != pointer width (the width of
        // the Offset*s at this point).
        APInt Dist = DistRange.getSingleElement()->sextOrTrunc(64);
        return (OffsetB - OffsetA + Dist).sextOrTrunc(64).trySExtValue();
      }

      std::optional<APInt> Diff =
          SE.computeConstantDifference(PtrSCEVB, PtrSCEVA);
      if (!Diff)
        return std::nullopt;
      Val = Diff->trySExtValue();
    }

    if (!Val)
      return std::nullopt;

    int64_t Size = DL.getTypeStoreSize(ElemTyA->LLVMTy);
    int64_t Dist = *Val / Size;

    // Ensure that the calculated distance matches the type-based one after all
    // the bitcasts removal in the provided pointers.
    if (!StrictCheck || Dist * Size == Val)
      return Dist;
    return std::nullopt;
  }

  /// \Returns the gap between the memory locations accessed by \p I0 and
  /// \p I1 in bytes. Returns nullopt if the gap can't be determined.
  template <typename LoadOrStoreT>
  static std::optional<int>
  getPointerDiffInBytes(LoadOrStoreT *I0, LoadOrStoreT *I1, ScalarEvolution &SE,
                        bool ExpensivePtrCheck) {
    static_assert(std::is_same_v<LoadOrStoreT, LoadInst> ||
                      std::is_same_v<LoadOrStoreT, StoreInst>,
                  "Expected sandboxir::Load or sandboxir::Store!");
    Value *Opnd0 = I0->getPointerOperand();
    Value *Opnd1 = I1->getPointerOperand();
    llvm::Value *LLVMOpnd0 = Opnd0->Val;
    llvm::Value *LLVMOpnd1 = Opnd1->Val;
    llvm::Value *LLVMPtr0 = getUnderlyingObject(LLVMOpnd0);
    llvm::Value *LLVMPtr1 = getUnderlyingObject(LLVMOpnd1);
    if (LLVMPtr0 != LLVMPtr1)
      return std::nullopt;
    Type *ElemTy = Type::getInt8Ty(I0->getContext());
    return getPointersDiff(ElemTy, Opnd0, ElemTy, Opnd1, I0->getDataLayout(),
                           SE, /*StrictCheck=*/false, /*CheckType=*/false,
                           ExpensivePtrCheck);
  }

  /// \Returns true if \p I0 accesses a memory location lower than \p I1.
  /// Returns false if the memory locations are equal, or if I1 accesses a
  /// memory location greater than I0. Returns nullopt if the difference cannot
  /// be determined.
  template <typename LoadOrStoreT>
  static std::optional<bool> atLowerAddress(LoadOrStoreT *I0, LoadOrStoreT *I1,
                                            ScalarEvolution &SE,
                                            bool ExpensivePtrChecks) {
    auto Diff = getPointerDiffInBytes(I0, I1, SE, ExpensivePtrChecks);
    if (!Diff)
      return std::nullopt;
    return *Diff > 0;
  }

  /// Equivalent to BatchAA::getModRefInfo().
  static ModRefInfo
  aliasAnalysisGetModRefInfo(BatchAAResults &BatchAA, const Instruction *I,
                             const std::optional<MemoryLocation> &OptLoc) {
    return BatchAA.getModRefInfo(cast<llvm::Instruction>(I->Val), OptLoc);
  }

  /// Equivalent to llvm::verifyFunction().
  /// \Returns true if the IR is broken.
  static bool verifyFunction(const Function *F, raw_ostream &OS) {
    const auto &LLVMF = *cast<llvm::Function>(F->Val);
    return llvm::verifyFunction(LLVMF, &OS);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_UTILS_H
