//===- ExpandReductions.cpp - Expand reduction intrinsics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR expansion for reduction intrinsics, allowing targets
// to enable the intrinsics until just before codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <optional>

using namespace llvm;

namespace {

void updateDomTreeForScalableExpansion(DominatorTree *DT, BasicBlock *Preheader,
                                       BasicBlock *Loop, BasicBlock *Exit) {
  DT->addNewBlock(Loop, Preheader);
  DT->changeImmediateDominator(Exit, Loop);
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
}

/// Expand a reduction on a scalable vector into a loop
/// that iterates over one element after the other.
Value *expandScalableReduction(IRBuilderBase &Builder, IntrinsicInst *II,
                               Value *Acc, Value *Vec,
                               Instruction::BinaryOps BinOp,
                               DominatorTree *DT) {
  ScalableVectorType *VecTy = cast<ScalableVectorType>(Vec->getType());

  // Split the original BB in two and create a new BB between them,
  // which will be a loop.
  BasicBlock *BeforeBB = II->getParent();
  BasicBlock *AfterBB = SplitBlock(BeforeBB, II, DT);
  BasicBlock *LoopBB = BasicBlock::Create(Builder.getContext(), "rdx.loop",
                                          BeforeBB->getParent(), AfterBB);
  BeforeBB->getTerminator()->setSuccessor(0, LoopBB);

  // Calculate the number of elements in the vector:
  Builder.SetInsertPoint(BeforeBB->getTerminator());
  Value *NumElts =
      Builder.CreateVScale(Builder.getInt64(VecTy->getMinNumElements()));

  // Create two PHIs, one for the index of the current lane and one for
  // the reduction.
  Builder.SetInsertPoint(LoopBB);
  PHINode *IV = Builder.CreatePHI(Builder.getInt64Ty(), 2, "index");
  IV->addIncoming(Builder.getInt64(0), BeforeBB);
  PHINode *RdxPhi = Builder.CreatePHI(VecTy->getScalarType(), 2, "rdx.phi");
  RdxPhi->addIncoming(Acc, BeforeBB);

  Value *IVInc =
      Builder.CreateAdd(IV, Builder.getInt64(1), "index.next", true, true);
  IV->addIncoming(IVInc, LoopBB);

  // Extract the value at the current lane from the vector and perform
  // the scalar reduction binop:
  Value *Lane = Builder.CreateExtractElement(Vec, IV, "elm");
  Value *Rdx = Builder.CreateBinOp(BinOp, RdxPhi, Lane, "rdx");
  RdxPhi->addIncoming(Rdx, LoopBB);

  // Exit when all lanes have been treated (assuming there will be at least
  // one element in the vector):
  Value *Done = Builder.CreateCmp(CmpInst::ICMP_EQ, IVInc, NumElts, "exitcond");
  Builder.CreateCondBr(Done, AfterBB, LoopBB);

  if (DT)
    updateDomTreeForScalableExpansion(DT, BeforeBB, LoopBB, AfterBB);

  return Rdx;
}

/// Expand a reduction on a scalable vector in a parallel-tree like
/// manner, meaning halving the number of elements to treat in every
/// iteration.
Value *expandScalableTreeReduction(
    IRBuilderBase &Builder, IntrinsicInst *II, std::optional<Value *> Acc,
    Value *Vec, Instruction::BinaryOps BinOp,
    function_ref<bool(Constant *)> IsNeutralElement, DominatorTree *DT,
    std::optional<unsigned> FixedVScale) {
  ScalableVectorType *VecTy = cast<ScalableVectorType>(Vec->getType());
  ScalableVectorType *VecTyX2 = ScalableVectorType::get(
      VecTy->getScalarType(), VecTy->getMinNumElements() * 2);

  // If the VScale is fixed, do not generate a loop, and instead to
  // something similar to llvm::getShuffleReduction(). That function
  // cannot be used directly because it uses shuffle masks, which
  // are not avaiable for scalable vectors (even if vscale is fixed).
  // The approach is effectively the same.
  if (FixedVScale.has_value()) {
    unsigned VF = VecTy->getMinNumElements() * FixedVScale.value();
    assert(isPowerOf2_64(VF));
    for (unsigned I = VF; I != 1; I >>= 1) {
      Value *Extended = Builder.CreateInsertVector(
          VecTyX2, PoisonValue::get(VecTyX2), Vec, Builder.getInt64(0));
      Value *Pair = Builder.CreateIntrinsic(Intrinsic::vector_deinterleave2,
                                            {VecTyX2}, {Extended});

      Value *Vec1 = Builder.CreateExtractValue(Pair, {0});
      Value *Vec2 = Builder.CreateExtractValue(Pair, {1});
      Vec = Builder.CreateBinOp(BinOp, Vec1, Vec2, "rdx");
    }
    Value *FinalVal = Builder.CreateExtractElement(Vec, uint64_t(0));
    if (Acc)
      if (auto *C = dyn_cast<Constant>(*Acc); !C || !IsNeutralElement(C))
        FinalVal = Builder.CreateBinOp(BinOp, *Acc, FinalVal, "rdx.final");
    return FinalVal;
  }

  // Split the original BB in two and create a new BB between them,
  // which will be a loop.
  BasicBlock *BeforeBB = II->getParent();
  BasicBlock *AfterBB = SplitBlock(BeforeBB, II, DT);
  BasicBlock *LoopBB = BasicBlock::Create(Builder.getContext(), "rdx.loop",
                                          BeforeBB->getParent(), AfterBB);
  BeforeBB->getTerminator()->setSuccessor(0, LoopBB);

  // This tree reduction only needs to do log2(N) iterations.
  // Note: Calculating log2(N) using count-trailing-zeros (cttz) only works if
  // `vscale` the vector size is a power of two.
  Builder.SetInsertPoint(BeforeBB->getTerminator());
  Value *NumElts =
      Builder.CreateVScale(Builder.getInt64(VecTy->getMinNumElements()));
  Value *NumIters = Builder.CreateIntrinsic(NumElts->getType(), Intrinsic::cttz,
                                            {NumElts, Builder.getTrue()});

  // Create two PHIs, one for the IV and one for the reduction.
  Builder.SetInsertPoint(LoopBB);
  PHINode *IV = Builder.CreatePHI(Builder.getInt64Ty(), 2, "iter");
  IV->addIncoming(Builder.getInt64(0), BeforeBB);
  PHINode *VecPhi = Builder.CreatePHI(VecTy, 2, "rdx.phi");
  VecPhi->addIncoming(Vec, BeforeBB);

  Value *IVInc =
      Builder.CreateAdd(IV, Builder.getInt64(1), "iter.next", true, true);
  IV->addIncoming(IVInc, LoopBB);

  // The deinterleave intrinsic takes a vector of, for example, type
  // <vscale x 8 x float> and produces a pair of vectors with half the size,
  // so 2 x <vscale x 4 x float>. An insert vector operation is used to
  // create a double-sized vector where the upper half is poison, because
  // we never care about that upper half anyways!
  Value *Extended = Builder.CreateInsertVector(
      VecTyX2, PoisonValue::get(VecTyX2), VecPhi, Builder.getInt64(0));
  Value *Pair = Builder.CreateIntrinsic(Intrinsic::vector_deinterleave2,
                                        {VecTyX2}, {Extended});
  Value *Vec1 = Builder.CreateExtractValue(Pair, {0});
  Value *Vec2 = Builder.CreateExtractValue(Pair, {1});
  Value *Rdx = Builder.CreateBinOp(BinOp, Vec1, Vec2, "rdx");
  VecPhi->addIncoming(Rdx, LoopBB);

  // Reduction-loop exit condition:
  Value *Done =
      Builder.CreateCmp(CmpInst::ICMP_EQ, IVInc, NumIters, "exitcond");
  Builder.CreateCondBr(Done, AfterBB, LoopBB);
  Builder.SetInsertPoint(AfterBB, AfterBB->getFirstInsertionPt());
  Value *FinalVal = Builder.CreateExtractElement(Rdx, uint64_t(0));

  // If the Acc value is not the neutral element of the reduction operation,
  // then we need to do the binop one last time with the end result of the
  // tree reduction.
  if (Acc)
    if (auto *C = dyn_cast<Constant>(*Acc); !C || !IsNeutralElement(C))
      FinalVal = Builder.CreateBinOp(BinOp, *Acc, FinalVal, "rdx.final");

  if (DT)
    updateDomTreeForScalableExpansion(DT, BeforeBB, LoopBB, AfterBB);

  return FinalVal;
}

std::pair<bool, bool> expandReductions(Function &F,
                                       const TargetTransformInfo *TTI,
                                       DominatorTree *DT) {
  bool Changed = false, CFGChanged = false;
  SmallVector<IntrinsicInst *, 4> Worklist;
  for (auto &I : instructions(F)) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::vector_reduce_fadd:
      case Intrinsic::vector_reduce_fmul:
      case Intrinsic::vector_reduce_add:
      case Intrinsic::vector_reduce_mul:
      case Intrinsic::vector_reduce_and:
      case Intrinsic::vector_reduce_or:
      case Intrinsic::vector_reduce_xor:
      case Intrinsic::vector_reduce_smax:
      case Intrinsic::vector_reduce_smin:
      case Intrinsic::vector_reduce_umax:
      case Intrinsic::vector_reduce_umin:
      case Intrinsic::vector_reduce_fmax:
      case Intrinsic::vector_reduce_fmin:
        if (TTI->shouldExpandReduction(II))
          Worklist.push_back(II);

        break;
      }
    }
  }

  const auto &Attrs = F.getAttributes().getFnAttrs();
  unsigned MinVScale = Attrs.getVScaleRangeMin();
  std::optional<unsigned> FixedVScale = Attrs.getVScaleRangeMax();
  if (FixedVScale != MinVScale)
    FixedVScale = std::nullopt;

  for (auto *II : Worklist) {
    FastMathFlags FMF =
        isa<FPMathOperator>(II) ? II->getFastMathFlags() : FastMathFlags{};
    Intrinsic::ID ID = II->getIntrinsicID();
    RecurKind RK = getMinMaxReductionRecurKind(ID);
    TargetTransformInfo::ReductionShuffle RS =
        TTI->getPreferredExpandedReductionShuffle(II);

    Value *Rdx = nullptr;
    IRBuilder<> Builder(II);
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.setFastMathFlags(FMF);
    switch (ID) {
    default: llvm_unreachable("Unexpected intrinsic!");
    case Intrinsic::vector_reduce_fadd:
    case Intrinsic::vector_reduce_fmul: {
      // FMFs must be attached to the call, otherwise it's an ordered reduction
      // and it can't be handled by generating a shuffle sequence.
      Value *Acc = II->getArgOperand(0);
      Value *Vec = II->getArgOperand(1);
      auto RdxOpcode =
          Instruction::BinaryOps(getArithmeticReductionInstruction(ID));

      bool ScalableTy = Vec->getType()->isScalableTy();
      if (ScalableTy && (!FixedVScale || FMF.allowReassoc())) {
        CFGChanged |= !FixedVScale;
        assert(TTI->isVScaleKnownToBeAPowerOfTwo() &&
               "Scalable tree reduction unimplemented for targets with a "
               "VScale not known to be a power of 2.");
        if (FMF.allowReassoc())
          Rdx = expandScalableTreeReduction(
              Builder, II, Acc, Vec, RdxOpcode,
              [&](Constant *C) {
                switch (ID) {
                case Intrinsic::vector_reduce_fadd:
                  return C->isZeroValue();
                case Intrinsic::vector_reduce_fmul:
                  return C->isOneValue();
                default:
                  llvm_unreachable("Binop not handled");
                }
              },
              DT, FixedVScale);
        else
          Rdx = expandScalableReduction(Builder, II, Acc, Vec, RdxOpcode, DT);
        break;
      }

      if (!FMF.allowReassoc())
        Rdx = getOrderedReduction(Builder, Acc, Vec, RdxOpcode, RK);
      else {
        if (!isPowerOf2_32(
                cast<FixedVectorType>(Vec->getType())->getNumElements()))
          continue;
        Rdx = getShuffleReduction(Builder, Vec, RdxOpcode, RS, RK);
        Rdx = Builder.CreateBinOp((Instruction::BinaryOps)RdxOpcode, Acc, Rdx,
                                  "bin.rdx");
      }
      break;
    }
    case Intrinsic::vector_reduce_and:
    case Intrinsic::vector_reduce_or: {
      // Canonicalize logical or/and reductions:
      // Or reduction for i1 is represented as:
      // %val = bitcast <ReduxWidth x i1> to iReduxWidth
      // %res = cmp ne iReduxWidth %val, 0
      // And reduction for i1 is represented as:
      // %val = bitcast <ReduxWidth x i1> to iReduxWidth
      // %res = cmp eq iReduxWidth %val, 11111
      Value *Vec = II->getArgOperand(0);
      auto *FTy = cast<FixedVectorType>(Vec->getType());
      unsigned NumElts = FTy->getNumElements();
      if (!isPowerOf2_32(NumElts))
        continue;

      if (FTy->getElementType() == Builder.getInt1Ty()) {
        Rdx = Builder.CreateBitCast(Vec, Builder.getIntNTy(NumElts));
        if (ID == Intrinsic::vector_reduce_and) {
          Rdx = Builder.CreateICmpEQ(
              Rdx, ConstantInt::getAllOnesValue(Rdx->getType()));
        } else {
          assert(ID == Intrinsic::vector_reduce_or && "Expected or reduction.");
          Rdx = Builder.CreateIsNotNull(Rdx);
        }
        break;
      }
      unsigned RdxOpcode = getArithmeticReductionInstruction(ID);
      Rdx = getShuffleReduction(Builder, Vec, RdxOpcode, RS, RK);
      break;
    }
    case Intrinsic::vector_reduce_add:
    case Intrinsic::vector_reduce_mul:
    case Intrinsic::vector_reduce_xor:
    case Intrinsic::vector_reduce_smax:
    case Intrinsic::vector_reduce_smin:
    case Intrinsic::vector_reduce_umax:
    case Intrinsic::vector_reduce_umin: {
      Value *Vec = II->getArgOperand(0);
      unsigned RdxOpcode = getArithmeticReductionInstruction(ID);
      if (Vec->getType()->isScalableTy()) {
        CFGChanged |= !FixedVScale;
        assert(TTI->isVScaleKnownToBeAPowerOfTwo() &&
               "Scalable tree reduction unimplemented for targets with a "
               "VScale not known to be a power of 2.");
        Rdx = expandScalableTreeReduction(
            Builder, II, std::nullopt, Vec, Instruction::BinaryOps(RdxOpcode),
            [](Constant *C) -> bool { llvm_unreachable("No accumulator!"); },
            DT, FixedVScale);
        break;
      }

      if (!isPowerOf2_32(
              cast<FixedVectorType>(Vec->getType())->getNumElements()))
        continue;
      Rdx = getShuffleReduction(Builder, Vec, RdxOpcode, RS, RK);
      break;
    }
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_fmin: {
      // We require "nnan" to use a shuffle reduction; "nsz" is implied by the
      // semantics of the reduction.
      Value *Vec = II->getArgOperand(0);
      if (!isPowerOf2_32(
              cast<FixedVectorType>(Vec->getType())->getNumElements()) ||
          !FMF.noNaNs())
        continue;
      unsigned RdxOpcode = getArithmeticReductionInstruction(ID);
      Rdx = getShuffleReduction(Builder, Vec, RdxOpcode, RS, RK);
      break;
    }
    }
    II->replaceAllUsesWith(Rdx);
    II->eraseFromParent();
    Changed = true;
  }
  return {CFGChanged, Changed};
}

class ExpandReductions : public FunctionPass {
public:
  static char ID;
  ExpandReductions() : FunctionPass(ID) {
    initializeExpandReductionsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    const auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *DTA = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    return expandReductions(F, TTI, DTA ? &DTA->getDomTree() : nullptr).second;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addUsedIfAvailable<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};
}

char ExpandReductions::ID;
INITIALIZE_PASS_BEGIN(ExpandReductions, "expand-reductions",
                      "Expand reduction intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(ExpandReductions, "expand-reductions",
                    "Expand reduction intrinsics", false, false)

FunctionPass *llvm::createExpandReductionsPass() {
  return new ExpandReductions();
}

PreservedAnalyses ExpandReductionsPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  const auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto *DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  auto [CFGChanged, Changed] = expandReductions(F, &TTI, DT);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  if (!CFGChanged)
    PA.preserveSet<CFGAnalyses>();
  else
    PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
