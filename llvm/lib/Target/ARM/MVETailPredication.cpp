//===- MVETailPredication.cpp - MVE Tail Predication ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Armv8.1m introduced MVE, M-Profile Vector Extension, and low-overhead
/// branches to help accelerate DSP applications. These two extensions can be
/// combined to provide implicit vector predication within a low-overhead loop.
/// The HardwareLoops pass inserts intrinsics identifying loops that the
/// backend will attempt to convert into a low-overhead loop. The vectorizer is
/// responsible for generating a vectorized loop in which the lanes are
/// predicated upon the iteration counter. This pass looks at these predicated
/// vector loops, that are targets for low-overhead loops, and prepares it for
/// code generation. Once the vectorizer has produced a masked loop, there's a
/// couple of final forms:
/// - A tail-predicated loop, with implicit predication.
/// - A loop containing multiple VCPT instructions, predicating multiple VPT
///   blocks of instructions operating on different vector types.
///
/// This pass inserts the inserts the VCTP intrinsic to represent the effect of
/// tail predication. This will be picked up by the ARM Low-overhead loop pass,
/// which performs the final transformation to a DLSTP or WLSTP tail-predicated
/// loop.

#include "ARM.h"
#include "ARMSubtarget.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "mve-tail-predication"
#define DESC "Transform predicated vector loops to use MVE tail predication"

cl::opt<bool>
DisableTailPredication("disable-mve-tail-predication", cl::Hidden,
                       cl::init(true),
                       cl::desc("Disable MVE Tail Predication"));
namespace {

class MVETailPredication : public LoopPass {
  SmallVector<IntrinsicInst*, 4> MaskedInsts;
  Loop *L = nullptr;
  ScalarEvolution *SE = nullptr;
  TargetTransformInfo *TTI = nullptr;

public:
  static char ID;

  MVETailPredication() : LoopPass(ID) { }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
  }

  bool runOnLoop(Loop *L, LPPassManager&) override;

private:

  /// Perform the relevant checks on the loop and convert if possible.
  bool TryConvert(Value *TripCount);

  /// Return whether this is a vectorized loop, that contains masked
  /// load/stores.
  bool IsPredicatedVectorLoop();

  /// Compute a value for the total number of elements that the predicated
  /// loop will process.
  Value *ComputeElements(Value *TripCount, VectorType *VecTy);

  /// Is the icmp that generates an i1 vector, based upon a loop counter
  /// and a limit that is defined outside the loop.
  bool isTailPredicate(Instruction *Predicate, Value *NumElements);

  /// Insert the intrinsic to represent the effect of tail predication.
  void InsertVCTPIntrinsic(Instruction *Predicate,
                           DenseMap<Instruction*, Instruction*> &NewPredicates,
                           VectorType *VecTy,
                           Value *NumElements);
};

} // end namespace

static bool IsDecrement(Instruction &I) {
  auto *Call = dyn_cast<IntrinsicInst>(&I);
  if (!Call)
    return false;

  Intrinsic::ID ID = Call->getIntrinsicID();
  return ID == Intrinsic::loop_decrement_reg;
}

static bool IsMasked(Instruction *I) {
  auto *Call = dyn_cast<IntrinsicInst>(I);
  if (!Call)
    return false;

  Intrinsic::ID ID = Call->getIntrinsicID();
  // TODO: Support gather/scatter expand/compress operations.
  return ID == Intrinsic::masked_store || ID == Intrinsic::masked_load;
}

bool MVETailPredication::runOnLoop(Loop *L, LPPassManager&) {
  if (skipLoop(L) || DisableTailPredication)
    return false;

  Function &F = *L->getHeader()->getParent();
  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<TargetMachine>();
  auto *ST = &TM.getSubtarget<ARMSubtarget>(F);
  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  this->L = L;

  // The MVE and LOB extensions are combined to enable tail-predication, but
  // there's nothing preventing us from generating VCTP instructions for v8.1m.
  if (!ST->hasMVEIntegerOps() || !ST->hasV8_1MMainlineOps()) {
    LLVM_DEBUG(dbgs() << "ARM TP: Not a v8.1m.main+mve target.\n");
    return false;
  }

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader)
    return false;

  auto FindLoopIterations = [](BasicBlock *BB) -> IntrinsicInst* {
    for (auto &I : *BB) {
      auto *Call = dyn_cast<IntrinsicInst>(&I);
      if (!Call)
        continue;

      Intrinsic::ID ID = Call->getIntrinsicID();
      if (ID == Intrinsic::set_loop_iterations ||
          ID == Intrinsic::test_set_loop_iterations)
        return cast<IntrinsicInst>(&I);
    }
    return nullptr;
  };

  // Look for the hardware loop intrinsic that sets the iteration count.
  IntrinsicInst *Setup = FindLoopIterations(Preheader);

  // The test.set iteration could live in the pre-preheader.
  if (!Setup) {
    if (!Preheader->getSinglePredecessor())
      return false;
    Setup = FindLoopIterations(Preheader->getSinglePredecessor());
    if (!Setup)
      return false;
  }

  // Search for the hardware loop intrinic that decrements the loop counter.
  IntrinsicInst *Decrement = nullptr;
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (IsDecrement(I)) {
        Decrement = cast<IntrinsicInst>(&I);
        break;
      }
    }
  }

  if (!Decrement)
    return false;

  LLVM_DEBUG(dbgs() << "ARM TP: Running on Loop: " << *L << *Setup << "\n"
             << *Decrement << "\n");
  return TryConvert(Setup->getArgOperand(0));
}

bool MVETailPredication::isTailPredicate(Instruction *I, Value *NumElements) {
  // Look for the following:

  // %trip.count.minus.1 = add i32 %N, -1
  // %broadcast.splatinsert10 = insertelement <4 x i32> undef,
  //                                          i32 %trip.count.minus.1, i32 0
  // %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10,
  //                                    <4 x i32> undef,
  //                                    <4 x i32> zeroinitializer
  // ...
  // ...
  // %index = phi i32
  // %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  // %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert,
  //                                  <4 x i32> undef,
  //                                  <4 x i32> zeroinitializer
  // %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  // %pred = icmp ule <4 x i32> %induction, %broadcast.splat11

  // And return whether V == %pred.

  using namespace PatternMatch;

  CmpInst::Predicate Pred;
  Instruction *Shuffle = nullptr;
  Instruction *Induction = nullptr;

  // The vector icmp
  if (!match(I, m_ICmp(Pred, m_Instruction(Induction),
                       m_Instruction(Shuffle))) ||
      Pred != ICmpInst::ICMP_ULE)
    return false;

  // First find the stuff outside the loop which is setting up the limit
  // vector....
  // The invariant shuffle that broadcast the limit into a vector.
  Instruction *Insert = nullptr;
  if (!match(Shuffle, m_ShuffleVector(m_Instruction(Insert), m_Undef(),
                                      m_Zero())))
    return false;

  // Insert the limit into a vector.
  Instruction *BECount = nullptr;
  if (!match(Insert, m_InsertElement(m_Undef(), m_Instruction(BECount),
                                     m_Zero())))
    return false;

  // The limit calculation, backedge count.
  Value *TripCount = nullptr;
  if (!match(BECount, m_Add(m_Value(TripCount), m_AllOnes())))
    return false;

  if (TripCount != NumElements || !L->isLoopInvariant(BECount))
    return false;

  // Now back to searching inside the loop body...
  // Find the add with takes the index iv and adds a constant vector to it.
  Instruction *BroadcastSplat = nullptr;
  Constant *Const = nullptr;
  if (!match(Induction, m_Add(m_Instruction(BroadcastSplat),
                              m_Constant(Const))))
   return false;

  // Check that we're adding <0, 1, 2, 3...
  if (auto *CDS = dyn_cast<ConstantDataSequential>(Const)) {
    for (unsigned i = 0; i < CDS->getNumElements(); ++i) {
      if (CDS->getElementAsInteger(i) != i)
        return false;
    }
  } else
    return false;

  // The shuffle which broadcasts the index iv into a vector.
  if (!match(BroadcastSplat, m_ShuffleVector(m_Instruction(Insert), m_Undef(),
                                             m_Zero())))
    return false;

  // The insert element which initialises a vector with the index iv.
  Instruction *IV = nullptr;
  if (!match(Insert, m_InsertElement(m_Undef(), m_Instruction(IV), m_Zero())))
    return false;

  // The index iv.
  auto *Phi = dyn_cast<PHINode>(IV);
  if (!Phi)
    return false;

  // TODO: Don't think we need to check the entry value.
  Value *OnEntry = Phi->getIncomingValueForBlock(L->getLoopPreheader());
  if (!match(OnEntry, m_Zero()))
    return false;

  Value *InLoop = Phi->getIncomingValueForBlock(L->getLoopLatch());
  unsigned Lanes = cast<VectorType>(Insert->getType())->getNumElements();

  Instruction *LHS = nullptr;
  if (!match(InLoop, m_Add(m_Instruction(LHS), m_SpecificInt(Lanes))))
    return false;

  return LHS == Phi;
}

static VectorType* getVectorType(IntrinsicInst *I) {
  unsigned TypeOp = I->getIntrinsicID() == Intrinsic::masked_load ? 0 : 1;
  auto *PtrTy = cast<PointerType>(I->getOperand(TypeOp)->getType());
  return cast<VectorType>(PtrTy->getElementType());
}

bool MVETailPredication::IsPredicatedVectorLoop() {
  // Check that the loop contains at least one masked load/store intrinsic.
  // We only support 'normal' vector instructions - other than masked
  // load/stores.
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (IsMasked(&I)) {
        VectorType *VecTy = getVectorType(cast<IntrinsicInst>(&I));
        unsigned Lanes = VecTy->getNumElements();
        unsigned ElementWidth = VecTy->getScalarSizeInBits();
        // MVE vectors are 128-bit, but don't support 128 x i1.
        // TODO: Can we support vectors larger than 128-bits?
        unsigned MaxWidth = TTI->getRegisterBitWidth(true);
        if (Lanes * ElementWidth > MaxWidth || Lanes == MaxWidth)
          return false;
        MaskedInsts.push_back(cast<IntrinsicInst>(&I));
      } else if (auto *Int = dyn_cast<IntrinsicInst>(&I)) {
        for (auto &U : Int->args()) {
          if (isa<VectorType>(U->getType()))
            return false;
        }
      }
    }
  }

  return !MaskedInsts.empty();
}

Value* MVETailPredication::ComputeElements(Value *TripCount,
                                           VectorType *VecTy) {
  const SCEV *TripCountSE = SE->getSCEV(TripCount);
  ConstantInt *VF = ConstantInt::get(cast<IntegerType>(TripCount->getType()),
                                     VecTy->getNumElements());

  if (VF->equalsInt(1))
    return nullptr;

  // TODO: Support constant trip counts.
  auto VisitAdd = [&](const SCEVAddExpr *S) -> const SCEVMulExpr* {
    if (auto *Const = dyn_cast<SCEVConstant>(S->getOperand(0))) {
      if (Const->getAPInt() != -VF->getValue())
        return nullptr;
    } else
      return nullptr;
    return dyn_cast<SCEVMulExpr>(S->getOperand(1));
  };

  auto VisitMul = [&](const SCEVMulExpr *S) -> const SCEVUDivExpr* {
    if (auto *Const = dyn_cast<SCEVConstant>(S->getOperand(0))) {
      if (Const->getValue() != VF)
        return nullptr;
    } else
      return nullptr;
    return dyn_cast<SCEVUDivExpr>(S->getOperand(1));
  };

  auto VisitDiv = [&](const SCEVUDivExpr *S) -> const SCEV* {
    if (auto *Const = dyn_cast<SCEVConstant>(S->getRHS())) {
      if (Const->getValue() != VF)
        return nullptr;
    } else
      return nullptr;

    if (auto *RoundUp = dyn_cast<SCEVAddExpr>(S->getLHS())) {
      if (auto *Const = dyn_cast<SCEVConstant>(RoundUp->getOperand(0))) {
        if (Const->getAPInt() != (VF->getValue() - 1))
          return nullptr;
      } else
        return nullptr;

      return RoundUp->getOperand(1);
    }
    return nullptr;
  };

  // TODO: Can we use SCEV helpers, such as findArrayDimensions, and friends to
  // determine the numbers of elements instead? Looks like this is what is used
  // for delinearization, but I'm not sure if it can be applied to the
  // vectorized form - at least not without a bit more work than I feel
  // comfortable with.

  // Search for Elems in the following SCEV:
  // (1 + ((-VF + (VF * (((VF - 1) + %Elems) /u VF))<nuw>) /u VF))<nuw><nsw>
  const SCEV *Elems = nullptr;
  if (auto *TC = dyn_cast<SCEVAddExpr>(TripCountSE))
    if (auto *Div = dyn_cast<SCEVUDivExpr>(TC->getOperand(1)))
      if (auto *Add = dyn_cast<SCEVAddExpr>(Div->getLHS()))
        if (auto *Mul = VisitAdd(Add))
          if (auto *Div = VisitMul(Mul))
            if (auto *Res = VisitDiv(Div))
              Elems = Res;

  if (!Elems)
    return nullptr;

  Instruction *InsertPt = L->getLoopPreheader()->getTerminator();
  if (!isSafeToExpandAt(Elems, InsertPt, *SE))
    return nullptr;

  auto DL = L->getHeader()->getModule()->getDataLayout();
  SCEVExpander Expander(*SE, DL, "elements");
  return Expander.expandCodeFor(Elems, Elems->getType(), InsertPt);
}

// Look through the exit block to see whether there's a duplicate predicate
// instruction. This can happen when we need to perform a select on values
// from the last and previous iteration. Instead of doing a straight
// replacement of that predicate with the vctp, clone the vctp and place it
// in the block. This means that the VPR doesn't have to be live into the
// exit block which should make it easier to convert this loop into a proper
// tail predicated loop.
static void Cleanup(DenseMap<Instruction*, Instruction*> &NewPredicates,
                    SetVector<Instruction*> &MaybeDead, Loop *L) {
  BasicBlock *Exit = L->getUniqueExitBlock();
  if (!Exit) {
    LLVM_DEBUG(dbgs() << "ARM TP: can't find loop exit block\n");
    return;
  }

  for (auto &Pair : NewPredicates) {
    Instruction *OldPred = Pair.first;
    Instruction *NewPred = Pair.second;

    for (auto &I : *Exit) {
      if (I.isSameOperationAs(OldPred)) {
        Instruction *PredClone = NewPred->clone();
        PredClone->insertBefore(&I);
        I.replaceAllUsesWith(PredClone);
        MaybeDead.insert(&I);
        LLVM_DEBUG(dbgs() << "ARM TP: replacing: "; I.dump();
                   dbgs() << "ARM TP: with:      "; PredClone->dump());
        break;
      }
    }
  }

  // Drop references and add operands to check for dead.
  SmallPtrSet<Instruction*, 4> Dead;
  while (!MaybeDead.empty()) {
    auto *I = MaybeDead.front();
    MaybeDead.remove(I);
    if (I->hasNUsesOrMore(1))
      continue;

    for (auto &U : I->operands()) {
      if (auto *OpI = dyn_cast<Instruction>(U))
        MaybeDead.insert(OpI);
    }
    I->dropAllReferences();
    Dead.insert(I);
  }

  for (auto *I : Dead) {
    LLVM_DEBUG(dbgs() << "ARM TP: removing dead insn: "; I->dump());
    I->eraseFromParent();
  }

  for (auto I : L->blocks())
    DeleteDeadPHIs(I);
}

void MVETailPredication::InsertVCTPIntrinsic(Instruction *Predicate,
    DenseMap<Instruction*, Instruction*> &NewPredicates,
    VectorType *VecTy, Value *NumElements) {
  IRBuilder<> Builder(L->getHeader()->getFirstNonPHI());
  Module *M = L->getHeader()->getModule();
  Type *Ty = IntegerType::get(M->getContext(), 32);

  // Insert a phi to count the number of elements processed by the loop.
  PHINode *Processed = Builder.CreatePHI(Ty, 2);
  Processed->addIncoming(NumElements, L->getLoopPreheader());

  // Insert the intrinsic to represent the effect of tail predication.
  Builder.SetInsertPoint(cast<Instruction>(Predicate));
  ConstantInt *Factor =
    ConstantInt::get(cast<IntegerType>(Ty), VecTy->getNumElements());

  Intrinsic::ID VCTPID;
  switch (VecTy->getNumElements()) {
  default:
    llvm_unreachable("unexpected number of lanes");
  case 4:  VCTPID = Intrinsic::arm_mve_vctp32; break;
  case 8:  VCTPID = Intrinsic::arm_mve_vctp16; break;
  case 16: VCTPID = Intrinsic::arm_mve_vctp8; break;

    // FIXME: vctp64 currently not supported because the predicate
    // vector wants to be <2 x i1>, but v2i1 is not a legal MVE
    // type, so problems happen at isel time.
    // Intrinsic::arm_mve_vctp64 exists for ACLE intrinsics
    // purposes, but takes a v4i1 instead of a v2i1.
  }
  Function *VCTP = Intrinsic::getDeclaration(M, VCTPID);
  Value *TailPredicate = Builder.CreateCall(VCTP, Processed);
  Predicate->replaceAllUsesWith(TailPredicate);
  NewPredicates[Predicate] = cast<Instruction>(TailPredicate);

  // Add the incoming value to the new phi.
  // TODO: This add likely already exists in the loop.
  Value *Remaining = Builder.CreateSub(Processed, Factor);
  Processed->addIncoming(Remaining, L->getLoopLatch());
  LLVM_DEBUG(dbgs() << "ARM TP: Insert processed elements phi: "
             << *Processed << "\n"
             << "ARM TP: Inserted VCTP: " << *TailPredicate << "\n");
}

bool MVETailPredication::TryConvert(Value *TripCount) {
  if (!IsPredicatedVectorLoop()) {
    LLVM_DEBUG(dbgs() << "ARM TP: no masked instructions in loop");
    return false;
  }

  LLVM_DEBUG(dbgs() << "ARM TP: Found predicated vector loop.\n");

  // Walk through the masked intrinsics and try to find whether the predicate
  // operand is generated from an induction variable.
  SetVector<Instruction*> Predicates;
  DenseMap<Instruction*, Instruction*> NewPredicates;

  for (auto *I : MaskedInsts) {
    Intrinsic::ID ID = I->getIntrinsicID();
    unsigned PredOp = ID == Intrinsic::masked_load ? 2 : 3;
    auto *Predicate = dyn_cast<Instruction>(I->getArgOperand(PredOp));
    if (!Predicate || Predicates.count(Predicate))
      continue;

    VectorType *VecTy = getVectorType(I);
    Value *NumElements = ComputeElements(TripCount, VecTy);
    if (!NumElements)
      continue;

    if (!isTailPredicate(Predicate, NumElements)) {
      LLVM_DEBUG(dbgs() << "ARM TP: Not tail predicate: " << *Predicate << "\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "ARM TP: Found tail predicate: " << *Predicate << "\n");
    Predicates.insert(Predicate);

    InsertVCTPIntrinsic(Predicate, NewPredicates, VecTy, NumElements);
  }

  // Now clean up.
  Cleanup(NewPredicates, Predicates, L);
  return true;
}

Pass *llvm::createMVETailPredicationPass() {
  return new MVETailPredication();
}

char MVETailPredication::ID = 0;

INITIALIZE_PASS_BEGIN(MVETailPredication, DEBUG_TYPE, DESC, false, false)
INITIALIZE_PASS_END(MVETailPredication, DEBUG_TYPE, DESC, false, false)
