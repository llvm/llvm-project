//===-- VPlanConstruction.cpp - Transforms for initial VPlan construction -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transforms for initial VPlan construction.
///
//===----------------------------------------------------------------------===//

#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"

#define DEBUG_TYPE "vplan"

using namespace llvm;

namespace {
// Class that is used to build the plain CFG for the incoming IR.
class PlainCFGBuilder {
  // The outermost loop of the input loop nest considered for vectorization.
  Loop *TheLoop;

  // Loop Info analysis.
  LoopInfo *LI;

  // Vectorization plan that we are working on.
  std::unique_ptr<VPlan> Plan;

  // Builder of the VPlan instruction-level representation.
  VPBuilder VPIRBuilder;

  // NOTE: The following maps are intentionally destroyed after the plain CFG
  // construction because subsequent VPlan-to-VPlan transformation may
  // invalidate them.
  // Map incoming BasicBlocks to their newly-created VPBasicBlocks.
  DenseMap<BasicBlock *, VPBasicBlock *> BB2VPBB;
  // Map incoming Value definitions to their newly-created VPValues.
  DenseMap<Value *, VPValue *> IRDef2VPValue;

  // Hold phi node's that need to be fixed once the plain CFG has been built.
  SmallVector<PHINode *, 8> PhisToFix;

  // Utility functions.
  void setVPBBPredsFromBB(VPBasicBlock *VPBB, BasicBlock *BB);
  void fixHeaderPhis();
  VPBasicBlock *getOrCreateVPBB(BasicBlock *BB);
#ifndef NDEBUG
  bool isExternalDef(Value *Val);
#endif
  VPValue *getOrCreateVPOperand(Value *IRVal);
  void createVPInstructionsForVPBB(VPBasicBlock *VPBB, BasicBlock *BB);

public:
  PlainCFGBuilder(Loop *Lp, LoopInfo *LI)
      : TheLoop(Lp), LI(LI), Plan(std::make_unique<VPlan>(Lp)) {}

  /// Build plain CFG for TheLoop  and connects it to Plan's entry.
  std::unique_ptr<VPlan>
  buildPlainCFG(DenseMap<const VPBlockBase *, BasicBlock *> &VPB2IRBB);
};
} // anonymous namespace

// Set predecessors of \p VPBB in the same order as they are in \p BB. \p VPBB
// must have no predecessors.
void PlainCFGBuilder::setVPBBPredsFromBB(VPBasicBlock *VPBB, BasicBlock *BB) {
  // Collect VPBB predecessors.
  SmallVector<VPBlockBase *, 2> VPBBPreds;
  for (BasicBlock *Pred : predecessors(BB))
    VPBBPreds.push_back(getOrCreateVPBB(Pred));
  VPBB->setPredecessors(VPBBPreds);
}

static bool isHeaderBB(BasicBlock *BB, Loop *L) {
  return L && BB == L->getHeader();
}

// Add operands to VPInstructions representing phi nodes from the input IR.
void PlainCFGBuilder::fixHeaderPhis() {
  for (auto *Phi : PhisToFix) {
    assert(IRDef2VPValue.count(Phi) && "Missing VPInstruction for PHINode.");
    VPValue *VPVal = IRDef2VPValue[Phi];
    assert(isa<VPWidenPHIRecipe>(VPVal) &&
           "Expected WidenPHIRecipe for phi node.");
    auto *VPPhi = cast<VPWidenPHIRecipe>(VPVal);
    assert(VPPhi->getNumOperands() == 0 &&
           "Expected VPInstruction with no operands.");
    assert(isHeaderBB(Phi->getParent(), LI->getLoopFor(Phi->getParent())) &&
           "Expected Phi in header block.");
    assert(Phi->getNumOperands() == 2 &&
           "header phi must have exactly 2 operands");
    for (BasicBlock *Pred : predecessors(Phi->getParent()))
      VPPhi->addOperand(
          getOrCreateVPOperand(Phi->getIncomingValueForBlock(Pred)));
  }
}

// Create a new empty VPBasicBlock for an incoming BasicBlock or retrieve an
// existing one if it was already created.
VPBasicBlock *PlainCFGBuilder::getOrCreateVPBB(BasicBlock *BB) {
  if (auto *VPBB = BB2VPBB.lookup(BB)) {
    // Retrieve existing VPBB.
    return VPBB;
  }

  // Create new VPBB.
  StringRef Name = BB->getName();
  LLVM_DEBUG(dbgs() << "Creating VPBasicBlock for " << Name << "\n");
  VPBasicBlock *VPBB = Plan->createVPBasicBlock(Name);
  BB2VPBB[BB] = VPBB;
  return VPBB;
}

#ifndef NDEBUG
// Return true if \p Val is considered an external definition. An external
// definition is either:
// 1. A Value that is not an Instruction. This will be refined in the future.
// 2. An Instruction that is outside of the IR region represented in VPlan,
// i.e., is not part of the loop nest.
bool PlainCFGBuilder::isExternalDef(Value *Val) {
  // All the Values that are not Instructions are considered external
  // definitions for now.
  Instruction *Inst = dyn_cast<Instruction>(Val);
  if (!Inst)
    return true;

  // Check whether Instruction definition is in loop body.
  return !TheLoop->contains(Inst);
}
#endif

// Create a new VPValue or retrieve an existing one for the Instruction's
// operand \p IRVal. This function must only be used to create/retrieve VPValues
// for *Instruction's operands* and not to create regular VPInstruction's. For
// the latter, please, look at 'createVPInstructionsForVPBB'.
VPValue *PlainCFGBuilder::getOrCreateVPOperand(Value *IRVal) {
  auto VPValIt = IRDef2VPValue.find(IRVal);
  if (VPValIt != IRDef2VPValue.end())
    // Operand has an associated VPInstruction or VPValue that was previously
    // created.
    return VPValIt->second;

  // Operand doesn't have a previously created VPInstruction/VPValue. This
  // means that operand is:
  //   A) a definition external to VPlan,
  //   B) any other Value without specific representation in VPlan.
  // For now, we use VPValue to represent A and B and classify both as external
  // definitions. We may introduce specific VPValue subclasses for them in the
  // future.
  assert(isExternalDef(IRVal) && "Expected external definition as operand.");

  // A and B: Create VPValue and add it to the pool of external definitions and
  // to the Value->VPValue map.
  VPValue *NewVPVal = Plan->getOrAddLiveIn(IRVal);
  IRDef2VPValue[IRVal] = NewVPVal;
  return NewVPVal;
}

// Create new VPInstructions in a VPBasicBlock, given its BasicBlock
// counterpart. This function must be invoked in RPO so that the operands of a
// VPInstruction in \p BB have been visited before (except for Phi nodes).
void PlainCFGBuilder::createVPInstructionsForVPBB(VPBasicBlock *VPBB,
                                                  BasicBlock *BB) {
  VPIRBuilder.setInsertPoint(VPBB);
  // TODO: Model and preserve debug intrinsics in VPlan.
  for (Instruction &InstRef : BB->instructionsWithoutDebug(false)) {
    Instruction *Inst = &InstRef;

    // There shouldn't be any VPValue for Inst at this point. Otherwise, we
    // visited Inst when we shouldn't, breaking the RPO traversal order.
    assert(!IRDef2VPValue.count(Inst) &&
           "Instruction shouldn't have been visited.");

    if (auto *Br = dyn_cast<BranchInst>(Inst)) {
      // Conditional branch instruction are represented using BranchOnCond
      // recipes.
      if (Br->isConditional()) {
        VPValue *Cond = getOrCreateVPOperand(Br->getCondition());
        VPIRBuilder.createNaryOp(VPInstruction::BranchOnCond, {Cond}, Inst);
      }

      // Skip the rest of the Instruction processing for Branch instructions.
      continue;
    }

    if (auto *SI = dyn_cast<SwitchInst>(Inst)) {
      SmallVector<VPValue *> Ops = {getOrCreateVPOperand(SI->getCondition())};
      for (auto Case : SI->cases())
        Ops.push_back(getOrCreateVPOperand(Case.getCaseValue()));
      VPIRBuilder.createNaryOp(Instruction::Switch, Ops, Inst);
      continue;
    }

    VPSingleDefRecipe *NewR;
    if (auto *Phi = dyn_cast<PHINode>(Inst)) {
      // Phi node's operands may have not been visited at this point. We create
      // an empty VPInstruction that we will fix once the whole plain CFG has
      // been built.
      NewR = new VPWidenPHIRecipe(Phi, nullptr, Phi->getDebugLoc(), "vec.phi");
      VPBB->appendRecipe(NewR);
      if (isHeaderBB(Phi->getParent(), LI->getLoopFor(Phi->getParent()))) {
        // Header phis need to be fixed after the VPBB for the latch has been
        // created.
        PhisToFix.push_back(Phi);
      } else {
        // Add operands for VPPhi in the order matching its predecessors in
        // VPlan.
        DenseMap<const VPBasicBlock *, VPValue *> VPPredToIncomingValue;
        for (unsigned I = 0; I != Phi->getNumOperands(); ++I) {
          VPPredToIncomingValue[BB2VPBB[Phi->getIncomingBlock(I)]] =
              getOrCreateVPOperand(Phi->getIncomingValue(I));
        }
        for (VPBlockBase *Pred : VPBB->getPredecessors())
          NewR->addOperand(
              VPPredToIncomingValue.lookup(Pred->getExitingBasicBlock()));
      }
    } else {
      // Translate LLVM-IR operands into VPValue operands and set them in the
      // new VPInstruction.
      SmallVector<VPValue *, 4> VPOperands;
      for (Value *Op : Inst->operands())
        VPOperands.push_back(getOrCreateVPOperand(Op));

      // Build VPInstruction for any arbitrary Instruction without specific
      // representation in VPlan.
      NewR = cast<VPInstruction>(
          VPIRBuilder.createNaryOp(Inst->getOpcode(), VPOperands, Inst));
    }

    IRDef2VPValue[Inst] = NewR;
  }
}

// Main interface to build the plain CFG.
std::unique_ptr<VPlan> PlainCFGBuilder::buildPlainCFG(
    DenseMap<const VPBlockBase *, BasicBlock *> &VPB2IRBB) {
  VPIRBasicBlock *Entry = cast<VPIRBasicBlock>(Plan->getEntry());
  BB2VPBB[Entry->getIRBasicBlock()] = Entry;
  for (VPIRBasicBlock *ExitVPBB : Plan->getExitBlocks())
    BB2VPBB[ExitVPBB->getIRBasicBlock()] = ExitVPBB;

  // 1. Scan the body of the loop in a topological order to visit each basic
  // block after having visited its predecessor basic blocks. Create a VPBB for
  // each BB and link it to its successor and predecessor VPBBs. Note that
  // predecessors must be set in the same order as they are in the incomming IR.
  // Otherwise, there might be problems with existing phi nodes and algorithm
  // based on predecessors traversal.

  // Loop PH needs to be explicitly visited since it's not taken into account by
  // LoopBlocksDFS.
  BasicBlock *ThePreheaderBB = TheLoop->getLoopPreheader();
  assert((ThePreheaderBB->getTerminator()->getNumSuccessors() == 1) &&
         "Unexpected loop preheader");
  for (auto &I : *ThePreheaderBB) {
    if (I.getType()->isVoidTy())
      continue;
    IRDef2VPValue[&I] = Plan->getOrAddLiveIn(&I);
  }

  LoopBlocksRPO RPO(TheLoop);
  RPO.perform(LI);

  for (BasicBlock *BB : RPO) {
    // Create or retrieve the VPBasicBlock for this BB.
    VPBasicBlock *VPBB = getOrCreateVPBB(BB);
    // Set VPBB predecessors in the same order as they are in the incoming BB.
    setVPBBPredsFromBB(VPBB, BB);

    // Create VPInstructions for BB.
    createVPInstructionsForVPBB(VPBB, BB);

    // Set VPBB successors. We create empty VPBBs for successors if they don't
    // exist already. Recipes will be created when the successor is visited
    // during the RPO traversal.
    if (auto *SI = dyn_cast<SwitchInst>(BB->getTerminator())) {
      SmallVector<VPBlockBase *> Succs = {
          getOrCreateVPBB(SI->getDefaultDest())};
      for (auto Case : SI->cases())
        Succs.push_back(getOrCreateVPBB(Case.getCaseSuccessor()));
      VPBB->setSuccessors(Succs);
      continue;
    }
    auto *BI = cast<BranchInst>(BB->getTerminator());
    unsigned NumSuccs = succ_size(BB);
    if (NumSuccs == 1) {
      VPBB->setOneSuccessor(getOrCreateVPBB(BB->getSingleSuccessor()));
      continue;
    }
    assert(BI->isConditional() && NumSuccs == 2 && BI->isConditional() &&
           "block must have conditional branch with 2 successors");

    BasicBlock *IRSucc0 = BI->getSuccessor(0);
    BasicBlock *IRSucc1 = BI->getSuccessor(1);
    VPBasicBlock *Successor0 = getOrCreateVPBB(IRSucc0);
    VPBasicBlock *Successor1 = getOrCreateVPBB(IRSucc1);
    VPBB->setTwoSuccessors(Successor0, Successor1);
  }

  for (auto *EB : Plan->getExitBlocks())
    setVPBBPredsFromBB(EB, EB->getIRBasicBlock());

  // 2. The whole CFG has been built at this point so all the input Values must
  // have a VPlan counterpart. Fix VPlan header phi by adding their
  // corresponding VPlan operands.
  fixHeaderPhis();

  Plan->getEntry()->setOneSuccessor(getOrCreateVPBB(TheLoop->getHeader()));
  Plan->getEntry()->setPlan(&*Plan);

  // Fix VPlan loop-closed-ssa exit phi's by adding incoming operands to the
  // VPIRInstructions wrapping them.
  // // Note that the operand order corresponds to IR predecessor order, and may
  // need adjusting when VPlan predecessors are added, if an exit block has
  // multiple predecessor.
  for (auto *EB : Plan->getExitBlocks()) {
    for (VPRecipeBase &R : EB->phis()) {
      auto *PhiR = cast<VPIRPhi>(&R);
      PHINode &Phi = PhiR->getIRPhi();
      assert(PhiR->getNumOperands() == 0 &&
             "no phi operands should be added yet");
      for (BasicBlock *Pred : predecessors(EB->getIRBasicBlock()))
        PhiR->addOperand(
            getOrCreateVPOperand(Phi.getIncomingValueForBlock(Pred)));
    }
  }

  for (const auto &[IRBB, VPB] : BB2VPBB)
    VPB2IRBB[VPB] = IRBB;

  LLVM_DEBUG(Plan->setName("Plain CFG\n"); dbgs() << *Plan);
  return std::move(Plan);
}

std::unique_ptr<VPlan> VPlanTransforms::buildPlainCFG(
    Loop *TheLoop, LoopInfo &LI,
    DenseMap<const VPBlockBase *, BasicBlock *> &VPB2IRBB) {
  PlainCFGBuilder Builder(TheLoop, &LI);
  return Builder.buildPlainCFG(VPB2IRBB);
}

/// Checks if \p HeaderVPB is a loop header block in the plain CFG; that is, it
/// has exactly 2 predecessors (preheader and latch), where the block
/// dominates the latch and the preheader dominates the block. If it is a
/// header block return true and canonicalize the predecessors of the header
/// (making sure the preheader appears first and the latch second) and the
/// successors of the latch (making sure the loop exit comes first). Otherwise
/// return false.
static bool canonicalHeaderAndLatch(VPBlockBase *HeaderVPB,
                                    const VPDominatorTree &VPDT) {
  ArrayRef<VPBlockBase *> Preds = HeaderVPB->getPredecessors();
  if (Preds.size() != 2)
    return false;

  auto *PreheaderVPBB = Preds[0];
  auto *LatchVPBB = Preds[1];
  if (!VPDT.dominates(PreheaderVPBB, HeaderVPB) ||
      !VPDT.dominates(HeaderVPB, LatchVPBB)) {
    std::swap(PreheaderVPBB, LatchVPBB);

    if (!VPDT.dominates(PreheaderVPBB, HeaderVPB) ||
        !VPDT.dominates(HeaderVPB, LatchVPBB))
      return false;

    // Canonicalize predecessors of header so that preheader is first and
    // latch second.
    HeaderVPB->swapPredecessors();
    for (VPRecipeBase &R : cast<VPBasicBlock>(HeaderVPB)->phis())
      R.swapOperands();
  }

  // The two successors of conditional branch match the condition, with the
  // first successor corresponding to true and the second to false. We
  // canonicalize the successors of the latch when introducing the region, such
  // that the latch exits the region when its condition is true; invert the
  // original condition if the original CFG branches to the header on true.
  // Note that the exit edge is not yet connected for top-level loops.
  if (LatchVPBB->getSingleSuccessor() ||
      LatchVPBB->getSuccessors()[0] != HeaderVPB)
    return true;

  assert(LatchVPBB->getNumSuccessors() == 2 && "Must have 2 successors");
  auto *Term = cast<VPBasicBlock>(LatchVPBB)->getTerminator();
  assert(cast<VPInstruction>(Term)->getOpcode() ==
             VPInstruction::BranchOnCond &&
         "terminator must be a BranchOnCond");
  auto *Not = new VPInstruction(VPInstruction::Not, {Term->getOperand(0)});
  Not->insertBefore(Term);
  Term->setOperand(0, Not);
  LatchVPBB->swapSuccessors();

  return true;
}

/// Create a new VPRegionBlock for the loop starting at \p HeaderVPB.
static void createLoopRegion(VPlan &Plan, VPBlockBase *HeaderVPB) {
  auto *PreheaderVPBB = HeaderVPB->getPredecessors()[0];
  auto *LatchVPBB = HeaderVPB->getPredecessors()[1];

  VPBlockUtils::disconnectBlocks(PreheaderVPBB, HeaderVPB);
  VPBlockUtils::disconnectBlocks(LatchVPBB, HeaderVPB);
  VPBlockBase *LatchExitVPB = LatchVPBB->getSingleSuccessor();
  assert(LatchExitVPB && "Latch expected to be left with a single successor");

  // Create an empty region first and insert it between PreheaderVPBB and
  // LatchExitVPB, taking care to preserve the original predecessor & successor
  // order of blocks. Set region entry and exiting after both HeaderVPB and
  // LatchVPBB have been disconnected from their predecessors/successors.
  auto *R = Plan.createVPRegionBlock("", false /*isReplicator*/);
  VPBlockUtils::insertOnEdge(LatchVPBB, LatchExitVPB, R);
  VPBlockUtils::disconnectBlocks(LatchVPBB, R);
  VPBlockUtils::connectBlocks(PreheaderVPBB, R);
  R->setEntry(HeaderVPB);
  R->setExiting(LatchVPBB);

  // All VPBB's reachable shallowly from HeaderVPB belong to the current region.
  for (VPBlockBase *VPBB : vp_depth_first_shallow(HeaderVPB))
    VPBB->setParent(R);
}

// Add the necessary canonical IV and branch recipes required to control the
// loop.
static void addCanonicalIVRecipes(VPlan &Plan, VPBasicBlock *HeaderVPBB,
                                  VPBasicBlock *LatchVPBB, Type *IdxTy,
                                  DebugLoc DL) {
  using namespace VPlanPatternMatch;
  Value *StartIdx = ConstantInt::get(IdxTy, 0);
  auto *StartV = Plan.getOrAddLiveIn(StartIdx);

  // Add a VPCanonicalIVPHIRecipe starting at 0 to the header.
  auto *CanonicalIVPHI = new VPCanonicalIVPHIRecipe(StartV, DL);
  HeaderVPBB->insert(CanonicalIVPHI, HeaderVPBB->begin());

  // We are about to replace the branch to exit the region. Remove the original
  // BranchOnCond, if there is any.
  if (!LatchVPBB->empty() &&
      match(&LatchVPBB->back(), m_BranchOnCond(m_VPValue())))
    LatchVPBB->getTerminator()->eraseFromParent();

  VPBuilder Builder(LatchVPBB);
  // Add a VPInstruction to increment the scalar canonical IV by VF * UF.
  // Initially the induction increment is guaranteed to not wrap, but that may
  // change later, e.g. when tail-folding, when the flags need to be dropped.
  auto *CanonicalIVIncrement = Builder.createOverflowingOp(
      Instruction::Add, {CanonicalIVPHI, &Plan.getVFxUF()}, {true, false}, DL,
      "index.next");
  CanonicalIVPHI->addOperand(CanonicalIVIncrement);

  // Add the BranchOnCount VPInstruction to the latch.
  Builder.createNaryOp(VPInstruction::BranchOnCount,
                       {CanonicalIVIncrement, &Plan.getVectorTripCount()}, DL);
}

void VPlanTransforms::prepareForVectorization(
    VPlan &Plan, Type *InductionTy, PredicatedScalarEvolution &PSE,
    bool RequiresScalarEpilogueCheck, bool TailFolded, Loop *TheLoop,
    DebugLoc IVDL, bool HasUncountableEarlyExit, VFRange &Range) {
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);

  VPBlockBase *HeaderVPB = Plan.getEntry()->getSingleSuccessor();
  canonicalHeaderAndLatch(HeaderVPB, VPDT);
  VPBlockBase *LatchVPB = HeaderVPB->getPredecessors()[1];

  VPBasicBlock *VecPreheader = Plan.createVPBasicBlock("vector.ph");
  VPBlockUtils::insertBlockAfter(VecPreheader, Plan.getEntry());

  VPBasicBlock *MiddleVPBB = Plan.createVPBasicBlock("middle.block");
  // The canonical LatchVPB has the header block as last successor. If it has
  // another successor, this successor is an exit block - insert middle block on
  // its edge. Otherwise, add middle block as another successor retaining header
  // as last.
  if (LatchVPB->getNumSuccessors() == 2) {
    VPBlockBase *LatchExitVPB = LatchVPB->getSuccessors()[0];
    VPBlockUtils::insertOnEdge(LatchVPB, LatchExitVPB, MiddleVPBB);
  } else {
    VPBlockUtils::connectBlocks(LatchVPB, MiddleVPBB);
    LatchVPB->swapSuccessors();
  }

  addCanonicalIVRecipes(Plan, cast<VPBasicBlock>(HeaderVPB),
                        cast<VPBasicBlock>(LatchVPB), InductionTy, IVDL);

  [[maybe_unused]] bool HandledUncountableEarlyExit = false;
  // Disconnect all early exits from the loop leaving it with a single exit from
  // the latch. Early exits that are countable are left for a scalar epilog. The
  // condition of uncountable early exits (currently at most one is supported)
  // is fused into the latch exit, and used to branch from middle block to the
  // early exit destination.
  for (VPIRBasicBlock *EB : Plan.getExitBlocks()) {
    for (VPBlockBase *Pred : to_vector(EB->getPredecessors())) {
      if (Pred == MiddleVPBB)
        continue;
      if (HasUncountableEarlyExit) {
        assert(!HandledUncountableEarlyExit &&
               "can handle exactly one uncountable early exit");
        handleUncountableEarlyExit(cast<VPBasicBlock>(Pred), EB, Plan,
                                   cast<VPBasicBlock>(HeaderVPB),
                                   cast<VPBasicBlock>(LatchVPB), Range);
        HandledUncountableEarlyExit = true;
      }

      cast<VPBasicBlock>(Pred)->getTerminator()->eraseFromParent();
      VPBlockUtils::disconnectBlocks(Pred, EB);
    }
  }

  assert((!HasUncountableEarlyExit || HandledUncountableEarlyExit) &&
         "missed an uncountable exit that must be handled");

  // Create SCEV and VPValue for the trip count.
  // We use the symbolic max backedge-taken-count, which works also when
  // vectorizing loops with uncountable early exits.
  const SCEV *BackedgeTakenCountSCEV = PSE.getSymbolicMaxBackedgeTakenCount();
  assert(!isa<SCEVCouldNotCompute>(BackedgeTakenCountSCEV) &&
         "Invalid loop count");
  ScalarEvolution &SE = *PSE.getSE();
  const SCEV *TripCount = SE.getTripCountFromExitCount(BackedgeTakenCountSCEV,
                                                       InductionTy, TheLoop);
  Plan.setTripCount(
      vputils::getOrCreateVPValueForSCEVExpr(Plan, TripCount, SE));

  VPBasicBlock *ScalarPH = Plan.createVPBasicBlock("scalar.ph");
  VPBlockUtils::connectBlocks(ScalarPH, Plan.getScalarHeader());

  // If needed, add a check in the middle block to see if we have completed
  // all of the iterations in the first vector loop.  Three cases:
  // 1) If we require a scalar epilogue, there is no conditional branch as
  //    we unconditionally branch to the scalar preheader.  Remove the recipes
  //    from the exit blocks.
  // 2) If (N - N%VF) == N, then we *don't* need to run the remainder.
  //    Thus if tail is to be folded, we know we don't need to run the
  //    remainder and we can set the condition to true.
  // 3) Otherwise, construct a runtime check.

  if (!RequiresScalarEpilogueCheck) {
    if (auto *LatchExitVPB = MiddleVPBB->getSingleSuccessor())
      VPBlockUtils::disconnectBlocks(MiddleVPBB, LatchExitVPB);
    VPBlockUtils::connectBlocks(MiddleVPBB, ScalarPH);
    // The exit blocks are unreachable, remove their recipes to make sure no
    // users remain that may pessimize transforms.
    for (auto *EB : Plan.getExitBlocks()) {
      for (VPRecipeBase &R : make_early_inc_range(*EB))
        R.eraseFromParent();
    }
    return;
  }

  // The connection order corresponds to the operands of the conditional branch,
  // with the middle block already connected to the exit block.
  VPBlockUtils::connectBlocks(MiddleVPBB, ScalarPH);

  auto *ScalarLatchTerm = TheLoop->getLoopLatch()->getTerminator();
  // Here we use the same DebugLoc as the scalar loop latch terminator instead
  // of the corresponding compare because they may have ended up with
  // different line numbers and we want to avoid awkward line stepping while
  // debugging. Eg. if the compare has got a line number inside the loop.
  VPBuilder Builder(MiddleVPBB);
  VPValue *Cmp =
      TailFolded
          ? Plan.getOrAddLiveIn(ConstantInt::getTrue(
                IntegerType::getInt1Ty(TripCount->getType()->getContext())))
          : Builder.createICmp(CmpInst::ICMP_EQ, Plan.getTripCount(),
                               &Plan.getVectorTripCount(),
                               ScalarLatchTerm->getDebugLoc(), "cmp.n");
  Builder.createNaryOp(VPInstruction::BranchOnCond, {Cmp},
                       ScalarLatchTerm->getDebugLoc());
}

void VPlanTransforms::createLoopRegions(VPlan &Plan) {
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);
  for (VPBlockBase *HeaderVPB : vp_post_order_shallow(Plan.getEntry()))
    if (canonicalHeaderAndLatch(HeaderVPB, VPDT))
      createLoopRegion(Plan, HeaderVPB);

  VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  TopRegion->setName("vector loop");
  TopRegion->getEntryBasicBlock()->setName("vector.body");
}
