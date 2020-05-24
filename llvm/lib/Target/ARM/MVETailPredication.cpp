//===- MVETailPredication.cpp - MVE Tail Predication ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Armv8.1m introduced MVE, M-Profile Vector Extension, and low-overhead
/// branches to help accelerate DSP applications. These two extensions,
/// combined with a new form of predication called tail-predication, can be used
/// to provide implicit vector predication within a low-overhead loop.
/// This is implicit because the predicate of active/inactive lanes is
/// calculated by hardware, and thus does not need to be explicitly passed
/// to vector instructions. The instructions responsible for this are the
/// DLSTP and WLSTP instructions, which setup a tail-predicated loop and the
/// the total number of data elements processed by the loop. The loop-end
/// LETP instruction is responsible for decrementing and setting the remaining
/// elements to be processed and generating the mask of active lanes.
///
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
/// This pass:
/// 1) Pattern matches the scalar iteration count produced by the vectoriser.
///    The scalar loop iteration count represents the number of elements to be
///    processed.
///    TODO: this could be emitted using an intrinsic, similar to the hardware
///    loop intrinsics, so that we don't need to pattern match this here.
/// 2) Inserts the VCTP intrinsic to represent the effect of
///    tail predication. This will be picked up by the ARM Low-overhead loop
///    pass, which performs the final transformation to a DLSTP or WLSTP
///    tail-predicated loop.

#include "ARM.h"
#include "ARMSubtarget.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

#define DEBUG_TYPE "mve-tail-predication"
#define DESC "Transform predicated vector loops to use MVE tail predication"

cl::opt<bool>
DisableTailPredication("disable-mve-tail-predication", cl::Hidden,
                       cl::init(true),
                       cl::desc("Disable MVE Tail Predication"));
namespace {

// Bookkeeping for pattern matching the loop trip count and the number of
// elements processed by the loop.
struct TripCountPattern {
  // An icmp instruction that calculates a predicate of active/inactive lanes
  // used by the masked loads/stores.
  Instruction *Predicate = nullptr;

  // The add instruction that increments the IV.
  Value *TripCount = nullptr;

  // The number of elements processed by the vector loop.
  Value *NumElements = nullptr;

  // Other instructions in the icmp chain that calculate the predicate.
  FixedVectorType *VecTy = nullptr;
  Instruction *Shuffle = nullptr;
  Instruction *Induction = nullptr;

  TripCountPattern(Instruction *P, Value *TC, FixedVectorType *VT)
      : Predicate(P), TripCount(TC), VecTy(VT){};
};

class MVETailPredication : public LoopPass {
  SmallVector<IntrinsicInst*, 4> MaskedInsts;
  Loop *L = nullptr;
  LoopInfo *LI = nullptr;
  const DataLayout *DL;
  DominatorTree *DT = nullptr;
  ScalarEvolution *SE = nullptr;
  TargetTransformInfo *TTI = nullptr;
  TargetLibraryInfo *TLI = nullptr;
  bool ClonedVCTPInExitBlock = false;

public:
  static char ID;

  MVETailPredication() : LoopPass(ID) { }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
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
  /// loop will process if it is a runtime value.
  bool ComputeRuntimeElements(TripCountPattern &TCP);

  /// Return whether this is the icmp that generates an i1 vector, based
  /// upon a loop counter and a limit that is defined outside the loop,
  /// that generates the active/inactive lanes required for tail-predication.
  bool isTailPredicate(TripCountPattern &TCP);

  /// Insert the intrinsic to represent the effect of tail predication.
  void InsertVCTPIntrinsic(TripCountPattern &TCP,
                           DenseMap<Instruction *, Instruction *> &NewPredicates);

  /// Rematerialize the iteration count in exit blocks, which enables
  /// ARMLowOverheadLoops to better optimise away loop update statements inside
  /// hardware-loops.
  void RematerializeIterCount();
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

void MVETailPredication::RematerializeIterCount() {
  SmallVector<WeakTrackingVH, 16> DeadInsts;
  SCEVExpander Rewriter(*SE, *DL, "mvetp");
  ReplaceExitVal ReplaceExitValue = AlwaysRepl;

  formLCSSARecursively(*L, *DT, LI, SE);
  rewriteLoopExitValues(L, LI, TLI, SE, TTI, Rewriter, DT, ReplaceExitValue,
                        DeadInsts);
}

bool MVETailPredication::runOnLoop(Loop *L, LPPassManager&) {
  if (skipLoop(L) || DisableTailPredication)
    return false;

  MaskedInsts.clear();
  Function &F = *L->getHeader()->getParent();
  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<TargetMachine>();
  auto *ST = &TM.getSubtarget<ARMSubtarget>(F);
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  TLI = TLIP ? &TLIP->getTLI(*L->getHeader()->getParent()) : nullptr;
  DL = &L->getHeader()->getModule()->getDataLayout();
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

  ClonedVCTPInExitBlock = false;
  LLVM_DEBUG(dbgs() << "ARM TP: Running on Loop: " << *L << *Setup << "\n"
             << *Decrement << "\n");

  if (TryConvert(Setup->getArgOperand(0))) {
    if (ClonedVCTPInExitBlock)
      RematerializeIterCount();
    return true;
  }

  LLVM_DEBUG(dbgs() << "ARM TP: Can't tail-predicate this loop.\n");
  return false;
}

// Pattern match predicates/masks and determine if they use the loop induction
// variable to control the number of elements processed by the loop. If so,
// the loop is a candidate for tail-predication.
bool MVETailPredication::isTailPredicate(TripCountPattern &TCP) {
  using namespace PatternMatch;

  // Pattern match the loop body and find the add with takes the index iv
  // and adds a constant vector to it:
  //
  // vector.body:
  // ..
  // %index = phi i32
  // %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  // %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert,
  //                                  <4 x i32> undef,
  //                                  <4 x i32> zeroinitializer
  // %induction = [add|or] <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  // %pred = icmp ule <4 x i32> %induction, %broadcast.splat11
  //
  // Please note that the 'or' is equivalent to the 'and' here, this relies on
  // BroadcastSplat being the IV which we know is a phi with 0 start and Lanes
  // increment, which is all being checked below.
  Instruction *BroadcastSplat = nullptr;
  Constant *Const = nullptr;
  if (!match(TCP.Induction,
             m_Add(m_Instruction(BroadcastSplat), m_Constant(Const))) &&
      !match(TCP.Induction,
             m_Or(m_Instruction(BroadcastSplat), m_Constant(Const))))
    return false;

  // Check that we're adding <0, 1, 2, 3...
  if (auto *CDS = dyn_cast<ConstantDataSequential>(Const)) {
    for (unsigned i = 0; i < CDS->getNumElements(); ++i) {
      if (CDS->getElementAsInteger(i) != i)
        return false;
    }
  } else
    return false;

  Instruction *Insert = nullptr;
  // The shuffle which broadcasts the index iv into a vector.
  if (!match(BroadcastSplat,
             m_Shuffle(m_Instruction(Insert), m_Undef(), m_ZeroMask())))
    return false;

  // The insert element which initialises a vector with the index iv.
  Instruction *IV = nullptr;
  if (!match(Insert, m_InsertElt(m_Undef(), m_Instruction(IV), m_Zero())))
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
  unsigned Lanes = cast<FixedVectorType>(Insert->getType())->getNumElements();

  Instruction *LHS = nullptr;
  if (!match(InLoop, m_Add(m_Instruction(LHS), m_SpecificInt(Lanes))))
    return false;

  return LHS == Phi;
}

static FixedVectorType *getVectorType(IntrinsicInst *I) {
  unsigned TypeOp = I->getIntrinsicID() == Intrinsic::masked_load ? 0 : 1;
  auto *PtrTy = cast<PointerType>(I->getOperand(TypeOp)->getType());
  return cast<FixedVectorType>(PtrTy->getElementType());
}

bool MVETailPredication::IsPredicatedVectorLoop() {
  // Check that the loop contains at least one masked load/store intrinsic.
  // We only support 'normal' vector instructions - other than masked
  // load/stores.
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (IsMasked(&I)) {
        FixedVectorType *VecTy = getVectorType(cast<IntrinsicInst>(&I));
        unsigned Lanes = VecTy->getNumElements();
        unsigned ElementWidth = VecTy->getScalarSizeInBits();
        // MVE vectors are 128-bit, but don't support 128 x i1.
        // TODO: Can we support vectors larger than 128-bits?
        unsigned MaxWidth = TTI->getRegisterBitWidth(true);
        if (Lanes * ElementWidth > MaxWidth || Lanes == MaxWidth)
          return false;
        MaskedInsts.push_back(cast<IntrinsicInst>(&I));
      } else if (auto *Int = dyn_cast<IntrinsicInst>(&I)) {
        if (Int->getIntrinsicID() == Intrinsic::fma)
          continue;
        for (auto &U : Int->args()) {
          if (isa<VectorType>(U->getType()))
            return false;
        }
      }
    }
  }

  return !MaskedInsts.empty();
}

// Pattern match the predicate, which is an icmp with a constant vector of this
// form:
//
//   icmp ult <4 x i32> %induction, <i32 32002, i32 32002, i32 32002, i32 32002>
//
// and return the constant, i.e. 32002 in this example. This is assumed to be
// the scalar loop iteration count: the number of loop elements by the
// the vector loop. Further checks are performed in function isTailPredicate(),
// to verify 'induction' behaves as an induction variable.
//
static bool ComputeConstElements(TripCountPattern &TCP) {
  if (!dyn_cast<ConstantInt>(TCP.TripCount))
    return false;

  ConstantInt *VF = ConstantInt::get(
      cast<IntegerType>(TCP.TripCount->getType()), TCP.VecTy->getNumElements());
  using namespace PatternMatch;
  CmpInst::Predicate CC;

  if (!match(TCP.Predicate, m_ICmp(CC, m_Instruction(TCP.Induction),
                                   m_AnyIntegralConstant())) ||
      CC != ICmpInst::ICMP_ULT)
    return false;

  LLVM_DEBUG(dbgs() << "ARM TP: icmp with constants: "; TCP.Predicate->dump(););
  Value *ConstVec = TCP.Predicate->getOperand(1);

  auto *CDS = dyn_cast<ConstantDataSequential>(ConstVec);
  if (!CDS || CDS->getNumElements() != VF->getSExtValue())
    return false;

  if ((TCP.NumElements = CDS->getSplatValue())) {
    assert(dyn_cast<ConstantInt>(TCP.NumElements)->getSExtValue() %
                   VF->getSExtValue() !=
               0 &&
           "tail-predication: trip count should not be a multiple of the VF");
    LLVM_DEBUG(dbgs() << "ARM TP: Found const elem count: " << *TCP.NumElements
                      << "\n");
    return true;
  }
  return false;
}

// Pattern match the loop iteration count setup:
//
// %trip.count.minus.1 = add i32 %N, -1
// %broadcast.splatinsert10 = insertelement <4 x i32> undef,
//                                          i32 %trip.count.minus.1, i32 0
// %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10,
//                                    <4 x i32> undef,
//                                    <4 x i32> zeroinitializer
// ..
// vector.body:
// ..
//
static bool MatchElemCountLoopSetup(Loop *L, Instruction *Shuffle,
                                    Value *NumElements) {
  using namespace PatternMatch;
  Instruction *Insert = nullptr;

  if (!match(Shuffle,
             m_Shuffle(m_Instruction(Insert), m_Undef(), m_ZeroMask())))
    return false;

  // Insert the limit into a vector.
  Instruction *BECount = nullptr;
  if (!match(Insert,
             m_InsertElt(m_Undef(), m_Instruction(BECount), m_Zero())))
    return false;

  // The limit calculation, backedge count.
  Value *TripCount = nullptr;
  if (!match(BECount, m_Add(m_Value(TripCount), m_AllOnes())))
    return false;

  if (TripCount != NumElements || !L->isLoopInvariant(BECount))
    return false;

  return true;
}

bool MVETailPredication::ComputeRuntimeElements(TripCountPattern &TCP) {
  using namespace PatternMatch;
  const SCEV *TripCountSE = SE->getSCEV(TCP.TripCount);
  ConstantInt *VF = ConstantInt::get(
      cast<IntegerType>(TCP.TripCount->getType()), TCP.VecTy->getNumElements());

  if (VF->equalsInt(1))
    return false;

  CmpInst::Predicate Pred;
  if (!match(TCP.Predicate, m_ICmp(Pred, m_Instruction(TCP.Induction),
                                   m_Instruction(TCP.Shuffle))) ||
      Pred != ICmpInst::ICMP_ULE)
    return false;

  LLVM_DEBUG(dbgs() << "Computing number of elements for vector trip count: ";
             TCP.TripCount->dump());

  // Otherwise, continue and try to pattern match the vector iteration
  // count expression
  auto VisitAdd = [&](const SCEVAddExpr *S) -> const SCEVMulExpr * {
    if (auto *Const = dyn_cast<SCEVConstant>(S->getOperand(0))) {
      if (Const->getAPInt() != -VF->getValue())
        return nullptr;
    } else
      return nullptr;
    return dyn_cast<SCEVMulExpr>(S->getOperand(1));
  };

  auto VisitMul = [&](const SCEVMulExpr *S) -> const SCEVUDivExpr * {
    if (auto *Const = dyn_cast<SCEVConstant>(S->getOperand(0))) {
      if (Const->getValue() != VF)
        return nullptr;
    } else
      return nullptr;
    return dyn_cast<SCEVUDivExpr>(S->getOperand(1));
  };

  auto VisitDiv = [&](const SCEVUDivExpr *S) -> const SCEV * {
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
    return false;

  Instruction *InsertPt = L->getLoopPreheader()->getTerminator();
  if (!isSafeToExpandAt(Elems, InsertPt, *SE))
    return false;

  auto DL = L->getHeader()->getModule()->getDataLayout();
  SCEVExpander Expander(*SE, DL, "elements");
  TCP.NumElements = Expander.expandCodeFor(Elems, Elems->getType(), InsertPt);

  if (!MatchElemCountLoopSetup(L, TCP.Shuffle, TCP.NumElements))
    return false;

  return true;
}

// Look through the exit block to see whether there's a duplicate predicate
// instruction. This can happen when we need to perform a select on values
// from the last and previous iteration. Instead of doing a straight
// replacement of that predicate with the vctp, clone the vctp and place it
// in the block. This means that the VPR doesn't have to be live into the
// exit block which should make it easier to convert this loop into a proper
// tail predicated loop.
static bool Cleanup(DenseMap<Instruction*, Instruction*> &NewPredicates,
                    SetVector<Instruction*> &MaybeDead, Loop *L) {
  BasicBlock *Exit = L->getUniqueExitBlock();
  if (!Exit) {
    LLVM_DEBUG(dbgs() << "ARM TP: can't find loop exit block\n");
    return false;
  }

  bool ClonedVCTPInExitBlock = false;

  for (auto &Pair : NewPredicates) {
    Instruction *OldPred = Pair.first;
    Instruction *NewPred = Pair.second;

    for (auto &I : *Exit) {
      if (I.isSameOperationAs(OldPred)) {
        Instruction *PredClone = NewPred->clone();
        PredClone->insertBefore(&I);
        I.replaceAllUsesWith(PredClone);
        MaybeDead.insert(&I);
        ClonedVCTPInExitBlock = true;
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

    for (auto &U : I->operands())
      if (auto *OpI = dyn_cast<Instruction>(U))
        MaybeDead.insert(OpI);

    I->dropAllReferences();
    Dead.insert(I);
  }

  for (auto *I : Dead) {
    LLVM_DEBUG(dbgs() << "ARM TP: removing dead insn: "; I->dump());
    I->eraseFromParent();
  }

  for (auto I : L->blocks())
    DeleteDeadPHIs(I);

  return ClonedVCTPInExitBlock;
}

void MVETailPredication::InsertVCTPIntrinsic(TripCountPattern &TCP,
    DenseMap<Instruction*, Instruction*> &NewPredicates) {
  IRBuilder<> Builder(L->getHeader()->getFirstNonPHI());
  Module *M = L->getHeader()->getModule();
  Type *Ty = IntegerType::get(M->getContext(), 32);

  // Insert a phi to count the number of elements processed by the loop.
  PHINode *Processed = Builder.CreatePHI(Ty, 2);
  Processed->addIncoming(TCP.NumElements, L->getLoopPreheader());

  // Insert the intrinsic to represent the effect of tail predication.
  Builder.SetInsertPoint(cast<Instruction>(TCP.Predicate));
  ConstantInt *Factor =
    ConstantInt::get(cast<IntegerType>(Ty), TCP.VecTy->getNumElements());

  Intrinsic::ID VCTPID;
  switch (TCP.VecTy->getNumElements()) {
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
  TCP.Predicate->replaceAllUsesWith(TailPredicate);
  NewPredicates[TCP.Predicate] = cast<Instruction>(TailPredicate);

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
    LLVM_DEBUG(dbgs() << "ARM TP: no masked instructions in loop.\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "ARM TP: Found predicated vector loop.\n");

  // Walk through the masked intrinsics and try to find whether the predicate
  // operand is generated from an induction variable.
  SetVector<Instruction*> Predicates;
  DenseMap<Instruction*, Instruction*> NewPredicates;

#ifndef NDEBUG
  // For debugging purposes, use this to indicate we have been able to
  // pattern match the scalar loop trip count.
  bool FoundScalarTC = false;
#endif

  for (auto *I : MaskedInsts) {
    Intrinsic::ID ID = I->getIntrinsicID();
    // First, find the icmp used by this masked load/store.
    unsigned PredOp = ID == Intrinsic::masked_load ? 2 : 3;
    auto *Predicate = dyn_cast<Instruction>(I->getArgOperand(PredOp));
    if (!Predicate || Predicates.count(Predicate))
      continue;

    // Step 1: using this icmp, now calculate the number of elements
    // processed by this loop.
    TripCountPattern TCP(Predicate, TripCount, getVectorType(I));
    if (!(ComputeConstElements(TCP) || ComputeRuntimeElements(TCP)))
      continue;

    LLVM_DEBUG(FoundScalarTC = true);

    if (!isTailPredicate(TCP)) {
      LLVM_DEBUG(dbgs() << "ARM TP: Not an icmp that generates tail predicate: "
                        << *Predicate << "\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "ARM TP: Found icmp generating tail predicate: "
                      << *Predicate << "\n");
    Predicates.insert(Predicate);

    // Step 2: emit the VCTP intrinsic representing the effect of TP.
    InsertVCTPIntrinsic(TCP, NewPredicates);
  }

  if (!NewPredicates.size()) {
      LLVM_DEBUG(if (!FoundScalarTC)
                   dbgs() << "ARM TP: Can't determine loop itertion count\n");
    return false;
  }

  // Now clean up.
  ClonedVCTPInExitBlock = Cleanup(NewPredicates, Predicates, L);
  return true;
}

Pass *llvm::createMVETailPredicationPass() {
  return new MVETailPredication();
}

char MVETailPredication::ID = 0;

INITIALIZE_PASS_BEGIN(MVETailPredication, DEBUG_TYPE, DESC, false, false)
INITIALIZE_PASS_END(MVETailPredication, DEBUG_TYPE, DESC, false, false)
