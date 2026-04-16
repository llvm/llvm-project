//===- ScalableToFixedVectors.cpp - Convert scalable to fixed vectors -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts IR operations on scalable vector types to fixed-length
// vectors when the effective length is known and is less than the minimum
// possible scaled vector length. For a scalable vector type with
// element count VF (known min elements), if minvscale * VF > VL, the we can
// convert to a fixed length vector of length VL.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ScalableToFixedVectors.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "scalable-to-fixed-vectors"

STATISTIC(NumScalableInstructionsConvertedToFixed,
          "Number of scalable instructions with constant EVL converted to "
          "fixed vectors");
STATISTIC(NumFunctionsWithScalableConvertedToFixed,
          "Number of functions with scalable vector converted to fixed width");
STATISTIC(NumViableSeeds,
          "Number of scalable stores that are viable to start chains");

/// For a given Instruction \p I, find the vector length needed required by
/// the vector instruction
/// If a vector intrinsic, must have:
/// - A constant VL
/// - All `true` mask
/// For other instructions, return 0 since length depends on demanded Elt's
unsigned
ScalableToFixedVectorsPass::getMinimumVLOfInst(const Instruction *I) const {
  unsigned MinVL = 0;
  if (const auto *VPI = dyn_cast<VPIntrinsic>(I)) {
    auto getMinVLVPIntrinsic = [&](const VPIntrinsic *VPI) -> unsigned {
      Value *Mask = VPI->getMaskParam();
      if (!Mask)
        return MaxVL;
      Value *SV = getSplatValue(Mask);
      if (!SV)
        return MaxVL;
      if (auto *CI = dyn_cast<ConstantInt>(SV); !CI || !CI->isOne())
        return MaxVL;
      Value *EVL = VPI->getVectorLengthParam();
      const auto *CInt = dyn_cast_or_null<ConstantInt>(EVL);
      if (!CInt)
        return MaxVL;
      uint64_t EVLVal64 =
          std::min(CInt->getLimitedValue(MaxVL),
                   static_cast<uint64_t>(std::numeric_limits<unsigned>::max()));
      return static_cast<unsigned>(EVLVal64);
    };
    VectorType *VTy = nullptr;
    switch (VPI->getIntrinsicID()) {
    case Intrinsic::vp_store: {
      VTy = dyn_cast<VectorType>(VPI->getMemoryDataParam()->getType());
      break;
    }
    case Intrinsic::vp_load:
      VTy = dyn_cast<VectorType>(I->getType());
      break;
    }
    assert(VTy && "VP load/store must carry a scalable vector type");
    ElementCount EC = VTy->getElementCount();
    assert(EC.isScalable() && "Unexpected fixed type vector");
    uint64_t MinElems = EC.getKnownMinValue();
    uint64_t MinWidth = static_cast<uint64_t>(MinVScale) * MinElems;
    unsigned EVLVal = getMinVLVPIntrinsic(VPI);
    if (EVLVal <= MinWidth)
      MinVL = EVLVal;
    else
      MinVL = MaxVL;
  }
  LLVM_DEBUG(dbgs() << "Found MinVL="; if (MinVL == MaxVL) dbgs() << "MaxVL";
             else dbgs() << MinVL; dbgs() << " for " << *I << "\n");
  return MinVL;
}

/// Is the given instruction a scalable vp_store
/// Only consider vp_store since it can be converted to store
static bool isScalableStoreVPIntrinsic(const Instruction *I) {
  const auto *VPI = dyn_cast<VPIntrinsic>(I);
  if (!VPI)
    return false;
  if (VPI->getIntrinsicID() != Intrinsic::vp_store)
    return false;
  return VPI->getMemoryDataParam()->getType()->isScalableTy();
}

/// Identify seed instructions to start conversion (vp_store only).
bool ScalableToFixedVectorsPass::isSeedCandidate(const Instruction *I) const {
  return isScalableStoreVPIntrinsic(I);
}

/// Check whether all scalable vector operands of I have been converted.
bool ScalableToFixedVectorsPass::allVectorOperandsConverted(
    Instruction *I) const {
  for (Value *Op : vector_operands(I)) {
    if (auto *OpI = dyn_cast<Instruction>(Op)) {
      if (!ScaledToFixed.count(OpI))
        return false;
    }
  }
  return true;
}

/// Clear persistent data structures
void ScalableToFixedVectorsPass::reset() {
  DemandedVLs.clear();
  ScaledToFixed.clear();
}

PreservedAnalyses
ScalableToFixedVectorsPass::run(Function &F, FunctionAnalysisManager &FAM) {
  LLVM_DEBUG(dbgs() << "Running ScalableToFixedVectorsPass on " << F.getName()
                    << "\n");

  assert(DemandedVLs.empty() &&
         "DemandedVLs ought to be reset for each call to run()");
  assert(Worklist.empty() &&
         "Worklist ought to be reset for each call to run()");
  assert(ScaledToFixed.empty() &&
         "ScaledToFixed ought to be reset for each call to run()");

  // If vscale is unknown, conservatively assume 1
  Attribute VSR = F.getFnAttribute(Attribute::VScaleRange);
  MinVScale = VSR.isValid() ? VSR.getVScaleRangeMin() : 1;

  // For each instruction that defines a vector, propagate the VL it
  // uses to its inputs.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isSeedCandidate(&I)) {
        unsigned MinVL = getMinimumVLOfInst(&I);
        if (MinVL != MaxVL) {
          DemandedVLs[&I] = MinVL;
          Worklist.insert(&I);
          ++NumViableSeeds;
        }
      }
    }
  }

  while (!Worklist.empty()) {
    Instruction *I = Worklist.front();
    Worklist.remove(I);
    transfer(I);
  }

  // Find instructions that don't have any vector operands to start vectorizing
  // from Need to go top down since we can't temporarily use the old
  // instructions due to type mismatch
  for (auto const &[I, VL] : DemandedVLs) {
    if (VL == MaxVL)
      continue;
    if (llvm::none_of(vector_operands(I),
                      [&](const Value *V) { return isa<Instruction>(V); }))
      Worklist.insert(I);
  }

  if (Worklist.empty()) {
    reset();
    return PreservedAnalyses::all();
  }

  IRBuilder<> Builder(F.getContext());

  while (!Worklist.empty()) {
    Instruction *I = Worklist.front();
    Worklist.remove(I);
    unsigned VL = DemandedVLs[I];
    assert(VL != MaxVL && "Should reduce VL");
    convertToFixed(Builder, I, VL);
    for (auto *U : I->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI)
        continue;
      if (allVectorOperandsConverted(UI))
        Worklist.insert(UI);
    }
  }

  assert(all_of(DemandedVLs,
                [&](const auto &Pair) {
                  return Pair.second == MaxVL ||
                         ScaledToFixed.count(Pair.first);
                }) &&
         "Instructions not converted despite valid VL.");

  // Clean up old instructions
  // Work bottom up, only deleting once instruction has no users
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (isSeedCandidate(&I) && DemandedVLs[&I] != MaxVL)
        Worklist.insert(&I);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.front();
    Worklist.remove(I);
    for (Value *V : I->operands())
      if (Instruction *OpI = dyn_cast<Instruction>(V))
        if (OpI->hasOneUser())
          Worklist.insert(OpI);
    I->eraseFromParent();
  }

  reset();
  ++NumFunctionsWithScalableConvertedToFixed;
  return PreservedAnalyses::none();
}

/// Look through \p I's operands and propagate what it demands to its uses.
/// Also capture its operations demands and transfer to itself
/// Is bi-directional since we want all instructions in the graph
/// to have the same type/length
/// TODO: Can we shorten non-masked loads?
void ScalableToFixedVectorsPass::transfer(Instruction *I) {
  for (Value *Op : llvm::concat<Value *>(vector_operands(I), vector_users(I))) {
    if (isSupported(Op)) {
      if (Instruction *OpI = dyn_cast<Instruction>(Op)) {
        unsigned Prev = DemandedVLs[OpI];
        if (Prev == 0)
          DemandedVLs[OpI] = std::max(DemandedVLs[I], getMinimumVLOfInst(OpI));
        else
          DemandedVLs[OpI] = std::max(DemandedVLs[I], DemandedVLs[OpI]);
        // TODO: Support changing the length of VP intrinsics
        if (isa<VPIntrinsic>(OpI) && getMinimumVLOfInst(OpI) != DemandedVLs[I])
          DemandedVLs[OpI] = MaxVL;
        if (DemandedVLs[OpI] != Prev) {
          LLVM_DEBUG(dbgs() << "Updated " << *OpI << " with VL=";
                     if (DemandedVLs[OpI] == MaxVL) dbgs() << "MaxVL";
                     else dbgs() << DemandedVLs[OpI]; dbgs() << "\n");
          Worklist.insert(OpI);
        }
      }
    } else {
      // Can't shorten non-intructions/unsupported instructions
      LLVM_DEBUG(dbgs() << "Unsupported value " << *Op << "\n");
      if (DemandedVLs[I] != MaxVL) {
        DemandedVLs[I] = MaxVL;
        Worklist.insert(I);
      }
    }
  }
}

/// Return true if we can reason about demanded VLs elementwise for \p I.
/// Right now just handle just a few easy to reason about operations
/// TODO: Support other arithmetic instructions
/// TODO: Support insert/extract subvectors
/// TODO: Support PHIs
/// TODO: Support arguments
bool ScalableToFixedVectorsPass::isSupported(const Value *V) const {
  if (auto *VPI = dyn_cast<VPIntrinsic>(V)) {
    unsigned VPID = VPI->getIntrinsicID();
    if (VPID != Intrinsic::vp_store && VPID != Intrinsic::vp_load)
      return false;
    return VPI->getType()->isScalableTy() || isScalableStoreVPIntrinsic(VPI);
  } else if (auto *SV = getSplatValue(V)) {
    return isa<Constant>(SV);
  } else if (auto *I = dyn_cast<Instruction>(V)) {
    return I->getType()->isScalableTy() && (I->isBinaryOp() || I->isCast());
  }
  return false;
}

/// Convert a single scalable vector instruction into a fixed width vector
/// instruction Will convert select VP intrinsics into base vector instructions
/// (e.g. vp_store -> store)
void ScalableToFixedVectorsPass::convertToFixed(IRBuilder<> &Builder,
                                                Instruction *I, unsigned VL) {
  assert(isSupported(I) && "Cannot convert unsupported instruction");
  Builder.SetInsertPoint(I);

  // Return the transformed operation
  auto TransformOp = [&](Value *Op) -> Value * {
    if (auto *OpI = dyn_cast<Instruction>(Op)) {
      if (ScaledToFixed.count(OpI)) {
        return ScaledToFixed[OpI];
      } else {
        assert(
            !DemandedVLs.count(OpI) &&
            "Expected to find Fixed version of instruction with demanded VL");
        return Op;
      }
    } else if (auto *SV = getSplatValue(Op); SV && isa<Constant>(SV)) {
      return ConstantVector::getSplat(ElementCount::getFixed(VL),
                                      cast<Constant>(SV));
    }
    return Op;
  };

  auto GetFixedType = [&]() -> Type * {
    return llvm::FixedVectorType::get(I->getType()->getScalarType(), VL);
  };

  Value *Fixed = nullptr;
  if (const auto *BOp = dyn_cast<BinaryOperator>(I)) {
    Fixed = Builder.CreateBinOp(BOp->getOpcode(), TransformOp(I->getOperand(0)),
                                TransformOp(I->getOperand(1)));
  } else if (const auto *COp = dyn_cast<CastInst>(I)) {
    Fixed = Builder.CreateCast(COp->getOpcode(), TransformOp(I->getOperand(0)),
                               GetFixedType());
  } else if (const auto *VPI = dyn_cast<VPIntrinsic>(I)) {
    assert(VL == getMinimumVLOfInst(I) &&
           "Can't reduce VP intrinsics with changed VL");
    switch (VPI->getIntrinsicID()) {
    case Intrinsic::vp_store:
      Fixed = Builder.CreateAlignedStore(TransformOp(I->getOperand(0)),
                                         TransformOp(I->getOperand(1)),
                                         VPI->getParamAlign(1));
      break;
    case Intrinsic::vp_load:
      Fixed = Builder.CreateAlignedLoad(
          GetFixedType(), TransformOp(I->getOperand(0)), VPI->getParamAlign(0));
      break;
    }
  }
  assert(Fixed && "Failed to create FixedOp");

  if (Instruction *FI = dyn_cast<Instruction>(Fixed)) {
    FI->copyMetadata(*I);
    FI->copyIRFlags(I);
    FI->setDebugLoc(I->getDebugLoc());
    FI->takeName(I);
  }
  ScaledToFixed[I] = Fixed;
  LLVM_DEBUG(dbgs() << "Converted scalable:\n"
                    << *I << "\nto fixed:\n"
                    << *Fixed << "\n");
  ++NumScalableInstructionsConvertedToFixed;
}
