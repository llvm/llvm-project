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
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"

#define DEBUG_TYPE "vplan"

using namespace llvm;
using namespace VPlanPatternMatch;

namespace {
// Class that is used to build the plain CFG for the incoming IR.
class PlainCFGBuilder {
  // The outermost loop of the input loop nest considered for vectorization.
  Loop *TheLoop;

  // Loop Info analysis.
  LoopInfo *LI;

  // Loop versioning for alias metadata.
  LoopVersioning *LVer;

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
  PlainCFGBuilder(Loop *Lp, LoopInfo *LI, LoopVersioning *LVer)
      : TheLoop(Lp), LI(LI), LVer(LVer), Plan(std::make_unique<VPlan>(Lp)) {}

  /// Build plain CFG for TheLoop and connect it to Plan's entry.
  std::unique_ptr<VPlan> buildPlainCFG();
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
    assert(isa<VPPhi>(VPVal) && "Expected VPPhi for phi node.");
    auto *PhiR = cast<VPPhi>(VPVal);
    assert(PhiR->getNumOperands() == 0 && "Expected VPPhi with no operands.");
    assert(isHeaderBB(Phi->getParent(), LI->getLoopFor(Phi->getParent())) &&
           "Expected Phi in header block.");
    assert(Phi->getNumOperands() == 2 &&
           "header phi must have exactly 2 operands");
    for (BasicBlock *Pred : predecessors(Phi->getParent()))
      PhiR->addOperand(
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
        VPIRBuilder.createNaryOp(VPInstruction::BranchOnCond, {Cond}, Inst, {},
                                 VPIRMetadata(*Inst), Inst->getDebugLoc());
      }

      // Skip the rest of the Instruction processing for Branch instructions.
      continue;
    }

    if (auto *SI = dyn_cast<SwitchInst>(Inst)) {
      // Don't emit recipes for unconditional switch instructions.
      if (SI->getNumCases() == 0)
        continue;
      SmallVector<VPValue *> Ops = {getOrCreateVPOperand(SI->getCondition())};
      for (auto Case : SI->cases())
        Ops.push_back(getOrCreateVPOperand(Case.getCaseValue()));
      VPIRBuilder.createNaryOp(Instruction::Switch, Ops, Inst, {},
                               VPIRMetadata(*Inst), Inst->getDebugLoc());
      continue;
    }

    VPSingleDefRecipe *NewR;
    if (auto *Phi = dyn_cast<PHINode>(Inst)) {
      // Phi node's operands may not have been visited at this point. We create
      // an empty VPInstruction that we will fix once the whole plain CFG has
      // been built.
      NewR = VPIRBuilder.createScalarPhi({}, Phi->getDebugLoc(), "vec.phi");
      NewR->setUnderlyingValue(Phi);
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
      // Build VPIRMetadata from the instruction and add loop versioning
      // metadata for loads and stores.
      VPIRMetadata MD(*Inst);
      if (isa<LoadInst, StoreInst>(Inst) && LVer) {
        const auto &[AliasScopeMD, NoAliasMD] =
            LVer->getNoAliasMetadataFor(Inst);
        if (AliasScopeMD)
          MD.setMetadata(LLVMContext::MD_alias_scope, AliasScopeMD);
        if (NoAliasMD)
          MD.setMetadata(LLVMContext::MD_noalias, NoAliasMD);
      }

      // Translate LLVM-IR operands into VPValue operands and set them in the
      // new VPInstruction.
      SmallVector<VPValue *, 4> VPOperands;
      for (Value *Op : Inst->operands())
        VPOperands.push_back(getOrCreateVPOperand(Op));

      if (auto *CI = dyn_cast<CastInst>(Inst)) {
        NewR = VPIRBuilder.createScalarCast(CI->getOpcode(), VPOperands[0],
                                            CI->getType(), CI->getDebugLoc(),
                                            VPIRFlags(*CI), MD);
        NewR->setUnderlyingValue(CI);
      } else {
        // Build VPInstruction for any arbitrary Instruction without specific
        // representation in VPlan.
        NewR =
            VPIRBuilder.createNaryOp(Inst->getOpcode(), VPOperands, Inst,
                                     VPIRFlags(*Inst), MD, Inst->getDebugLoc());
      }
    }

    IRDef2VPValue[Inst] = NewR;
  }
}

// Main interface to build the plain CFG.
std::unique_ptr<VPlan> PlainCFGBuilder::buildPlainCFG() {
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

  LLVM_DEBUG(Plan->setName("Plain CFG\n"); dbgs() << *Plan);
  return std::move(Plan);
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

  // Create an empty region first and insert it between PreheaderVPBB and
  // the exit blocks, taking care to preserve the original predecessor &
  // successor order of blocks. Set region entry and exiting after both
  // HeaderVPB and LatchVPBB have been disconnected from their
  // predecessors/successors.
  auto *R = Plan.createLoopRegion();

  // Transfer latch's successors to the region.
  VPBlockUtils::transferSuccessors(LatchVPBB, R);

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
  Value *StartIdx = ConstantInt::get(IdxTy, 0);
  auto *StartV = Plan.getOrAddLiveIn(StartIdx);

  // Add a VPCanonicalIVPHIRecipe starting at 0 to the header.
  auto *CanonicalIVPHI = new VPCanonicalIVPHIRecipe(StartV, DL);
  HeaderVPBB->insert(CanonicalIVPHI, HeaderVPBB->begin());

  // We are about to replace the branch to exit the region. Remove the original
  // BranchOnCond, if there is any.
  DebugLoc LatchDL = DL;
  if (!LatchVPBB->empty() && match(&LatchVPBB->back(), m_BranchOnCond())) {
    LatchDL = LatchVPBB->getTerminator()->getDebugLoc();
    LatchVPBB->getTerminator()->eraseFromParent();
  }

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
                       {CanonicalIVIncrement, &Plan.getVectorTripCount()},
                       LatchDL);
}

/// Creates extracts for values in \p Plan defined in a loop region and used
/// outside a loop region.
static void createExtractsForLiveOuts(VPlan &Plan, VPBasicBlock *MiddleVPBB) {
  VPBuilder B(MiddleVPBB, MiddleVPBB->getFirstNonPhi());
  for (VPBasicBlock *EB : Plan.getExitBlocks()) {
    if (EB->getSinglePredecessor() != MiddleVPBB)
      continue;

    for (VPRecipeBase &R : EB->phis()) {
      auto *ExitIRI = cast<VPIRPhi>(&R);
      for (unsigned Idx = 0; Idx != ExitIRI->getNumIncoming(); ++Idx) {
        VPRecipeBase *Inc = ExitIRI->getIncomingValue(Idx)->getDefiningRecipe();
        if (!Inc)
          continue;
        assert(ExitIRI->getNumOperands() == 1 &&
               ExitIRI->getParent()->getSinglePredecessor() == MiddleVPBB &&
               "exit values from early exits must be fixed when branch to "
               "early-exit is added");
        ExitIRI->extractLastLaneOfLastPartOfFirstOperand(B);
      }
    }
  }
}

static void addInitialSkeleton(VPlan &Plan, Type *InductionTy, DebugLoc IVDL,
                               PredicatedScalarEvolution &PSE, Loop *TheLoop) {
  VPDominatorTree VPDT(Plan);

  auto *HeaderVPBB = cast<VPBasicBlock>(Plan.getEntry()->getSingleSuccessor());
  canonicalHeaderAndLatch(HeaderVPBB, VPDT);
  auto *LatchVPBB = cast<VPBasicBlock>(HeaderVPBB->getPredecessors()[1]);

  VPBasicBlock *VecPreheader = Plan.createVPBasicBlock("vector.ph");
  VPBlockUtils::insertBlockAfter(VecPreheader, Plan.getEntry());

  VPBasicBlock *MiddleVPBB = Plan.createVPBasicBlock("middle.block");
  // The canonical LatchVPBB has the header block as last successor. If it has
  // another successor, this successor is an exit block - insert middle block on
  // its edge. Otherwise, add middle block as another successor retaining header
  // as last.
  if (LatchVPBB->getNumSuccessors() == 2) {
    VPBlockBase *LatchExitVPB = LatchVPBB->getSuccessors()[0];
    VPBlockUtils::insertOnEdge(LatchVPBB, LatchExitVPB, MiddleVPBB);
  } else {
    VPBlockUtils::connectBlocks(LatchVPBB, MiddleVPBB);
    LatchVPBB->swapSuccessors();
  }

  addCanonicalIVRecipes(Plan, HeaderVPBB, LatchVPBB, InductionTy, IVDL);

  // Create SCEV and VPValue for the trip count.
  // We use the symbolic max backedge-taken-count, which works also when
  // vectorizing loops with uncountable early exits.
  const SCEV *BackedgeTakenCountSCEV = PSE.getSymbolicMaxBackedgeTakenCount();
  assert(!isa<SCEVCouldNotCompute>(BackedgeTakenCountSCEV) &&
         "Invalid backedge-taken count");
  ScalarEvolution &SE = *PSE.getSE();
  const SCEV *TripCount = SE.getTripCountFromExitCount(BackedgeTakenCountSCEV,
                                                       InductionTy, TheLoop);
  Plan.setTripCount(vputils::getOrCreateVPValueForSCEVExpr(Plan, TripCount));

  VPBasicBlock *ScalarPH = Plan.createVPBasicBlock("scalar.ph");
  VPBlockUtils::connectBlocks(ScalarPH, Plan.getScalarHeader());

  // The connection order corresponds to the operands of the conditional branch,
  // with the middle block already connected to the exit block.
  VPBlockUtils::connectBlocks(MiddleVPBB, ScalarPH);
  // Also connect the entry block to the scalar preheader.
  // TODO: Also introduce a branch recipe together with the minimum trip count
  // check.
  VPBlockUtils::connectBlocks(Plan.getEntry(), ScalarPH);
  Plan.getEntry()->swapSuccessors();

  createExtractsForLiveOuts(Plan, MiddleVPBB);

  VPBuilder ScalarPHBuilder(ScalarPH);
  for (const auto &[PhiR, ScalarPhiR] : zip_equal(
           drop_begin(HeaderVPBB->phis()), Plan.getScalarHeader()->phis())) {
    auto *VectorPhiR = cast<VPPhi>(&PhiR);
    auto *ResumePhiR = ScalarPHBuilder.createScalarPhi(
        {VectorPhiR, VectorPhiR->getOperand(0)}, VectorPhiR->getDebugLoc());
    cast<VPIRPhi>(&ScalarPhiR)->addOperand(ResumePhiR);
  }
}

/// Check \p Plan's live-in and replace them with constants, if they can be
/// simplified via SCEV.
static void simplifyLiveInsWithSCEV(VPlan &Plan,
                                    PredicatedScalarEvolution &PSE) {
  auto GetSimplifiedLiveInViaSCEV = [&](VPValue *VPV) -> VPValue * {
    const SCEV *Expr = vputils::getSCEVExprForVPValue(VPV, PSE);
    if (auto *C = dyn_cast<SCEVConstant>(Expr))
      return Plan.getOrAddLiveIn(C->getValue());
    return nullptr;
  };

  for (VPValue *LiveIn : to_vector(Plan.getLiveIns())) {
    if (VPValue *SimplifiedLiveIn = GetSimplifiedLiveInViaSCEV(LiveIn))
      LiveIn->replaceAllUsesWith(SimplifiedLiveIn);
  }
}

std::unique_ptr<VPlan>
VPlanTransforms::buildVPlan0(Loop *TheLoop, LoopInfo &LI, Type *InductionTy,
                             DebugLoc IVDL, PredicatedScalarEvolution &PSE,
                             LoopVersioning *LVer) {
  PlainCFGBuilder Builder(TheLoop, &LI, LVer);
  std::unique_ptr<VPlan> VPlan0 = Builder.buildPlainCFG();
  addInitialSkeleton(*VPlan0, InductionTy, IVDL, PSE, TheLoop);
  simplifyLiveInsWithSCEV(*VPlan0, PSE);
  return VPlan0;
}

/// Creates a VPWidenIntOrFpInductionRecipe or VPWidenPointerInductionRecipe
/// for \p Phi based on \p IndDesc.
static VPHeaderPHIRecipe *
createWidenInductionRecipe(PHINode *Phi, VPPhi *PhiR, VPIRValue *Start,
                           const InductionDescriptor &IndDesc, VPlan &Plan,
                           PredicatedScalarEvolution &PSE, Loop &OrigLoop,
                           DebugLoc DL) {
  [[maybe_unused]] ScalarEvolution &SE = *PSE.getSE();
  assert(SE.isLoopInvariant(IndDesc.getStep(), &OrigLoop) &&
         "step must be loop invariant");
  assert((Plan.getLiveIn(IndDesc.getStartValue()) == Start ||
          (SE.isSCEVable(IndDesc.getStartValue()->getType()) &&
           SE.getSCEV(IndDesc.getStartValue()) ==
               vputils::getSCEVExprForVPValue(Start, PSE))) &&
         "Start VPValue must match IndDesc's start value");

  VPValue *Step =
      vputils::getOrCreateVPValueForSCEVExpr(Plan, IndDesc.getStep());

  if (IndDesc.getKind() == InductionDescriptor::IK_PtrInduction)
    return new VPWidenPointerInductionRecipe(Phi, Start, Step, &Plan.getVFxUF(),
                                             IndDesc, DL);

  assert((IndDesc.getKind() == InductionDescriptor::IK_IntInduction ||
          IndDesc.getKind() == InductionDescriptor::IK_FpInduction) &&
         "must have an integer or float induction at this point");

  // Update wide induction increments to use the same step as the corresponding
  // wide induction. This enables detecting induction increments directly in
  // VPlan and removes redundant splats.
  using namespace llvm::VPlanPatternMatch;
  if (match(PhiR->getOperand(1), m_Add(m_Specific(PhiR), m_VPValue())))
    PhiR->getOperand(1)->getDefiningRecipe()->setOperand(1, Step);

  // It is always safe to copy over the NoWrap and FastMath flags. In
  // particular, when folding tail by masking, the masked-off lanes are never
  // used, so it is safe.
  VPIRFlags Flags = vputils::getFlagsFromIndDesc(IndDesc);

  return new VPWidenIntOrFpInductionRecipe(Phi, Start, Step, &Plan.getVF(),
                                           IndDesc, Flags, DL);
}

void VPlanTransforms::createHeaderPhiRecipes(
    VPlan &Plan, PredicatedScalarEvolution &PSE, Loop &OrigLoop,
    const MapVector<PHINode *, InductionDescriptor> &Inductions,
    const MapVector<PHINode *, RecurrenceDescriptor> &Reductions,
    const SmallPtrSetImpl<const PHINode *> &FixedOrderRecurrences,
    const SmallPtrSetImpl<PHINode *> &InLoopReductions, bool AllowReordering) {
  // Retrieve the header manually from the intial plain-CFG VPlan.
  VPBasicBlock *HeaderVPBB = cast<VPBasicBlock>(
      Plan.getEntry()->getSuccessors()[1]->getSingleSuccessor());
  assert(VPDominatorTree(Plan).dominates(HeaderVPBB,
                                         HeaderVPBB->getPredecessors()[1]) &&
         "header must dominate its latch");

  auto CreateHeaderPhiRecipe = [&](VPPhi *PhiR) -> VPHeaderPHIRecipe * {
    // TODO: Gradually replace uses of underlying instruction by analyses on
    // VPlan.
    auto *Phi = cast<PHINode>(PhiR->getUnderlyingInstr());
    assert(PhiR->getNumOperands() == 2 &&
           "Must have 2 operands for header phis");

    // Extract common values once.
    VPIRValue *Start = cast<VPIRValue>(PhiR->getOperand(0));
    VPValue *BackedgeValue = PhiR->getOperand(1);

    if (FixedOrderRecurrences.contains(Phi)) {
      // TODO: Currently fixed-order recurrences are modeled as chains of
      // first-order recurrences. If there are no users of the intermediate
      // recurrences in the chain, the fixed order recurrence should be
      // modeled directly, enabling more efficient codegen.
      return new VPFirstOrderRecurrencePHIRecipe(Phi, *Start, *BackedgeValue);
    }

    auto InductionIt = Inductions.find(Phi);
    if (InductionIt != Inductions.end())
      return createWidenInductionRecipe(Phi, PhiR, Start, InductionIt->second,
                                        Plan, PSE, OrigLoop,
                                        PhiR->getDebugLoc());

    assert(Reductions.contains(Phi) && "only reductions are expected now");
    const RecurrenceDescriptor &RdxDesc = Reductions.lookup(Phi);
    assert(RdxDesc.getRecurrenceStartValue() ==
               Phi->getIncomingValueForBlock(OrigLoop.getLoopPreheader()) &&
           "incoming value must match start value");
    // Will be updated later to >1 if reduction is partial.
    unsigned ScaleFactor = 1;
    bool UseOrderedReductions = !AllowReordering && RdxDesc.isOrdered();
    return new VPReductionPHIRecipe(
        Phi, RdxDesc.getRecurrenceKind(), *Start, *BackedgeValue,
        getReductionStyle(InLoopReductions.contains(Phi), UseOrderedReductions,
                          ScaleFactor),
        RdxDesc.hasUsesOutsideReductionChain());
  };

  for (VPRecipeBase &R : make_early_inc_range(HeaderVPBB->phis())) {
    if (isa<VPCanonicalIVPHIRecipe>(&R))
      continue;
    auto *PhiR = cast<VPPhi>(&R);
    VPHeaderPHIRecipe *HeaderPhiR = CreateHeaderPhiRecipe(PhiR);
    HeaderPhiR->insertBefore(PhiR);
    PhiR->replaceAllUsesWith(HeaderPhiR);
    PhiR->eraseFromParent();
  }
}

void VPlanTransforms::createInLoopReductionRecipes(
    VPlan &Plan, const DenseMap<VPBasicBlock *, VPValue *> &BlockMaskCache,
    const DenseSet<BasicBlock *> &BlocksNeedingPredication,
    ElementCount MinVF) {
  VPTypeAnalysis TypeInfo(Plan);
  VPBasicBlock *Header = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  SmallVector<VPRecipeBase *> ToDelete;

  for (VPRecipeBase &R : Header->phis()) {
    auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!PhiR || !PhiR->isInLoop() || (MinVF.isScalar() && !PhiR->isOrdered()))
      continue;

    RecurKind Kind = PhiR->getRecurrenceKind();
    assert(!RecurrenceDescriptor::isFindLastRecurrenceKind(Kind) &&
           !RecurrenceDescriptor::isAnyOfRecurrenceKind(Kind) &&
           !RecurrenceDescriptor::isFindIVRecurrenceKind(Kind) &&
           "AnyOf and Find reductions are not allowed for in-loop reductions");

    bool IsFPRecurrence =
        RecurrenceDescriptor::isFloatingPointRecurrenceKind(Kind);
    FastMathFlags FMFs =
        IsFPRecurrence ? FastMathFlags::getFast() : FastMathFlags();

    // Collect the chain of "link" recipes for the reduction starting at PhiR.
    SetVector<VPSingleDefRecipe *> Worklist;
    Worklist.insert(PhiR);
    for (unsigned I = 0; I != Worklist.size(); ++I) {
      VPSingleDefRecipe *Cur = Worklist[I];
      for (VPUser *U : Cur->users()) {
        auto *UserRecipe = cast<VPSingleDefRecipe>(U);
        if (!UserRecipe->getParent()->getEnclosingLoopRegion()) {
          assert((UserRecipe->getParent() == Plan.getMiddleBlock() ||
                  UserRecipe->getParent() == Plan.getScalarPreheader()) &&
                 "U must be either in the loop region, the middle block or the "
                 "scalar preheader.");
          continue;
        }

        // Stores using instructions will be sunk later.
        if (match(UserRecipe, m_VPInstruction<Instruction::Store>()))
          continue;
        Worklist.insert(UserRecipe);
      }
    }

    // Visit operation "Links" along the reduction chain top-down starting from
    // the phi until LoopExitValue. We keep track of the previous item
    // (PreviousLink) to tell which of the two operands of a Link will remain
    // scalar and which will be reduced. For minmax by select(cmp), Link will be
    // the select instructions. Blend recipes of in-loop reduction phi's will
    // get folded to their non-phi operand, as the reduction recipe handles the
    // condition directly.
    VPSingleDefRecipe *PreviousLink = PhiR; // Aka Worklist[0].
    for (VPSingleDefRecipe *CurrentLink : drop_begin(Worklist)) {
      if (auto *Blend = dyn_cast<VPBlendRecipe>(CurrentLink)) {
        assert(Blend->getNumIncomingValues() == 2 &&
               "Blend must have 2 incoming values");
        unsigned PhiRIdx = Blend->getIncomingValue(0) == PhiR ? 0 : 1;
        assert(Blend->getIncomingValue(PhiRIdx) == PhiR &&
               "PhiR must be an operand of the blend");
        Blend->replaceAllUsesWith(Blend->getIncomingValue(1 - PhiRIdx));
        continue;
      }

      if (IsFPRecurrence) {
        FastMathFlags CurFMF =
            cast<VPRecipeWithIRFlags>(CurrentLink)->getFastMathFlags();
        if (match(CurrentLink, m_Select(m_VPValue(), m_VPValue(), m_VPValue())))
          CurFMF |= cast<VPRecipeWithIRFlags>(CurrentLink->getOperand(0))
                        ->getFastMathFlags();
        FMFs &= CurFMF;
      }

      Instruction *CurrentLinkI = CurrentLink->getUnderlyingInstr();

      // Recognize a call to the llvm.fmuladd intrinsic.
      bool IsFMulAdd = Kind == RecurKind::FMulAdd;
      VPValue *VecOp;
      VPBasicBlock *LinkVPBB = CurrentLink->getParent();
      if (IsFMulAdd) {
        assert(RecurrenceDescriptor::isFMulAddIntrinsic(CurrentLinkI) &&
               "Expected current VPInstruction to be a call to the "
               "llvm.fmuladd intrinsic");
        assert(CurrentLink->getOperand(2) == PreviousLink &&
               "expected a call where the previous link is the added operand");

        // If the instruction is a call to the llvm.fmuladd intrinsic then we
        // need to create an fmul recipe (multiplying the first two operands of
        // the fmuladd together) to use as the vector operand for the fadd
        // reduction.
        auto *FMulRecipe = new VPInstruction(
            Instruction::FMul,
            {CurrentLink->getOperand(0), CurrentLink->getOperand(1)},
            CurrentLinkI->getFastMathFlags());
        LinkVPBB->insert(FMulRecipe, CurrentLink->getIterator());
        VecOp = FMulRecipe;
      } else if (Kind == RecurKind::AddChainWithSubs &&
                 match(CurrentLink, m_Sub(m_VPValue(), m_VPValue()))) {
        Type *PhiTy = TypeInfo.inferScalarType(PhiR);
        auto *Zero = Plan.getConstantInt(PhiTy, 0);
        auto *Sub = new VPInstruction(Instruction::Sub,
                                      {Zero, CurrentLink->getOperand(1)}, {},
                                      {}, CurrentLinkI->getDebugLoc());
        Sub->setUnderlyingValue(CurrentLinkI);
        LinkVPBB->insert(Sub, CurrentLink->getIterator());
        VecOp = Sub;
      } else {
        // Index of the first operand which holds a non-mask vector operand.
        unsigned IndexOfFirstOperand = 0;
        if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind)) {
          if (match(CurrentLink, m_Cmp(m_VPValue(), m_VPValue())))
            continue;
          assert(match(CurrentLink,
                       m_Select(m_VPValue(), m_VPValue(), m_VPValue())) &&
                 "must be a select recipe");
          IndexOfFirstOperand = 1;
        }
        // Note that for non-commutable operands (cmp-selects), the semantics of
        // the cmp-select are captured in the recurrence kind.
        unsigned VecOpId =
            CurrentLink->getOperand(IndexOfFirstOperand) == PreviousLink
                ? IndexOfFirstOperand + 1
                : IndexOfFirstOperand;
        VecOp = CurrentLink->getOperand(VecOpId);
        assert(VecOp != PreviousLink &&
               CurrentLink->getOperand(CurrentLink->getNumOperands() - 1 -
                                       (VecOpId - IndexOfFirstOperand)) ==
                   PreviousLink &&
               "PreviousLink must be the operand other than VecOp");
      }

      // Get block mask from BlockMaskCache if the block needs predication.
      VPValue *CondOp = nullptr;
      if (BlocksNeedingPredication.contains(CurrentLinkI->getParent()))
        CondOp = BlockMaskCache.lookup(LinkVPBB);

      assert(PhiR->getVFScaleFactor() == 1 &&
             "inloop reductions must be unscaled");
      auto *RedRecipe = new VPReductionRecipe(
          Kind, FMFs, CurrentLinkI, PreviousLink, VecOp, CondOp,
          getReductionStyle(/*IsInLoop=*/true, PhiR->isOrdered(), 1),
          CurrentLinkI->getDebugLoc());
      // Append the recipe to the end of the VPBasicBlock because we need to
      // ensure that it comes after all of it's inputs, including CondOp.
      // Delete CurrentLink as it will be invalid if its operand is replaced
      // with a reduction defined at the bottom of the block in the next link.
      if (LinkVPBB->getNumSuccessors() == 0)
        RedRecipe->insertBefore(&*std::prev(std::prev(LinkVPBB->end())));
      else
        LinkVPBB->appendRecipe(RedRecipe);

      CurrentLink->replaceAllUsesWith(RedRecipe);
      ToDelete.push_back(CurrentLink);
      PreviousLink = RedRecipe;
    }
  }

  for (VPRecipeBase *R : ToDelete)
    R->eraseFromParent();
}

void VPlanTransforms::handleEarlyExits(VPlan &Plan,
                                       bool HasUncountableEarlyExit) {
  auto *MiddleVPBB = cast<VPBasicBlock>(
      Plan.getScalarHeader()->getSinglePredecessor()->getPredecessors()[0]);
  auto *LatchVPBB = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());
  VPBlockBase *HeaderVPB = cast<VPBasicBlock>(LatchVPBB->getSuccessors()[1]);

  // Disconnect all early exits from the loop leaving it with a single exit from
  // the latch. Early exits that are countable are left for a scalar epilog. The
  // condition of uncountable early exits (currently at most one is supported)
  // is fused into the latch exit, and used to branch from middle block to the
  // early exit destination.
  [[maybe_unused]] bool HandledUncountableEarlyExit = false;
  for (VPIRBasicBlock *EB : Plan.getExitBlocks()) {
    for (VPBlockBase *Pred : to_vector(EB->getPredecessors())) {
      if (Pred == MiddleVPBB)
        continue;
      if (HasUncountableEarlyExit) {
        assert(!HandledUncountableEarlyExit &&
               "can handle exactly one uncountable early exit");
        handleUncountableEarlyExit(cast<VPBasicBlock>(Pred), EB, Plan,
                                   cast<VPBasicBlock>(HeaderVPB), LatchVPBB);
        HandledUncountableEarlyExit = true;
      } else {
        for (VPRecipeBase &R : EB->phis())
          cast<VPIRPhi>(&R)->removeIncomingValueFor(Pred);
      }
      cast<VPBasicBlock>(Pred)->getTerminator()->eraseFromParent();
      VPBlockUtils::disconnectBlocks(Pred, EB);
    }
  }

  assert((!HasUncountableEarlyExit || HandledUncountableEarlyExit) &&
         "missed an uncountable exit that must be handled");
}

void VPlanTransforms::addMiddleCheck(VPlan &Plan,
                                     bool RequiresScalarEpilogueCheck,
                                     bool TailFolded) {
  auto *MiddleVPBB = cast<VPBasicBlock>(
      Plan.getScalarHeader()->getSinglePredecessor()->getPredecessors()[0]);
  // If MiddleVPBB has a single successor then the original loop does not exit
  // via the latch and the single successor must be the scalar preheader.
  // There's no need to add a runtime check to MiddleVPBB.
  if (MiddleVPBB->getNumSuccessors() == 1) {
    assert(MiddleVPBB->getSingleSuccessor() == Plan.getScalarPreheader() &&
           "must have ScalarPH as single successor");
    return;
  }

  assert(MiddleVPBB->getNumSuccessors() == 2 && "must have 2 successors");

  // Add a check in the middle block to see if we have completed all of the
  // iterations in the first vector loop.
  //
  // Three cases:
  // 1) If we require a scalar epilogue, the scalar ph must execute. Set the
  //    condition to false.
  // 2) If (N - N%VF) == N, then we *don't* need to run the
  //    remainder. Thus if tail is to be folded, we know we don't need to run
  //    the remainder and we can set the condition to true.
  // 3) Otherwise, construct a runtime check.

  // We use the same DebugLoc as the scalar loop latch terminator instead of
  // the corresponding compare because they may have ended up with different
  // line numbers and we want to avoid awkward line stepping while debugging.
  // E.g., if the compare has got a line number inside the loop.
  auto *LatchVPBB = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());
  DebugLoc LatchDL = LatchVPBB->getTerminator()->getDebugLoc();
  VPBuilder Builder(MiddleVPBB);
  VPValue *Cmp;
  if (!RequiresScalarEpilogueCheck)
    Cmp = Plan.getFalse();
  else if (TailFolded)
    Cmp = Plan.getTrue();
  else
    Cmp = Builder.createICmp(CmpInst::ICMP_EQ, Plan.getTripCount(),
                             &Plan.getVectorTripCount(), LatchDL, "cmp.n");
  Builder.createNaryOp(VPInstruction::BranchOnCond, {Cmp}, LatchDL);
}

void VPlanTransforms::createLoopRegions(VPlan &Plan) {
  VPDominatorTree VPDT(Plan);
  for (VPBlockBase *HeaderVPB : vp_post_order_shallow(Plan.getEntry()))
    if (canonicalHeaderAndLatch(HeaderVPB, VPDT))
      createLoopRegion(Plan, HeaderVPB);

  VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  TopRegion->setName("vector loop");
  TopRegion->getEntryBasicBlock()->setName("vector.body");
}

// Likelyhood of bypassing the vectorized loop due to a runtime check block,
// including memory overlap checks block and wrapping/unit-stride checks block.
static constexpr uint32_t CheckBypassWeights[] = {1, 127};

void VPlanTransforms::attachCheckBlock(VPlan &Plan, Value *Cond,
                                       BasicBlock *CheckBlock,
                                       bool AddBranchWeights) {
  VPValue *CondVPV = Plan.getOrAddLiveIn(Cond);
  VPBasicBlock *CheckBlockVPBB = Plan.createVPIRBasicBlock(CheckBlock);
  VPBlockBase *VectorPH = Plan.getVectorPreheader();
  VPBlockBase *ScalarPH = Plan.getScalarPreheader();
  VPBlockBase *PreVectorPH = VectorPH->getSinglePredecessor();
  VPBlockUtils::insertOnEdge(PreVectorPH, VectorPH, CheckBlockVPBB);
  VPBlockUtils::connectBlocks(CheckBlockVPBB, ScalarPH);
  CheckBlockVPBB->swapSuccessors();

  // We just connected a new block to the scalar preheader. Update all
  // VPPhis by adding an incoming value for it, replicating the last value.
  unsigned NumPredecessors = ScalarPH->getNumPredecessors();
  for (VPRecipeBase &R : cast<VPBasicBlock>(ScalarPH)->phis()) {
    assert(isa<VPPhi>(&R) && "Phi expected to be VPPhi");
    assert(cast<VPPhi>(&R)->getNumIncoming() == NumPredecessors - 1 &&
           "must have incoming values for all operands");
    R.addOperand(R.getOperand(NumPredecessors - 2));
  }

  VPIRMetadata VPBranchWeights;
  auto *Term =
      VPBuilder(CheckBlockVPBB)
          .createNaryOp(
              VPInstruction::BranchOnCond, {CondVPV},
              Plan.getVectorLoopRegion()->getCanonicalIV()->getDebugLoc());
  if (AddBranchWeights) {
    MDBuilder MDB(Plan.getContext());
    MDNode *BranchWeights =
        MDB.createBranchWeights(CheckBypassWeights, /*IsExpected=*/false);
    Term->setMetadata(LLVMContext::MD_prof, BranchWeights);
  }
}

void VPlanTransforms::addMinimumIterationCheck(
    VPlan &Plan, ElementCount VF, unsigned UF,
    ElementCount MinProfitableTripCount, bool RequiresScalarEpilogue,
    bool TailFolded, bool CheckNeededWithTailFolding, Loop *OrigLoop,
    const uint32_t *MinItersBypassWeights, DebugLoc DL,
    PredicatedScalarEvolution &PSE) {
  // Generate code to check if the loop's trip count is less than VF * UF, or
  // equal to it in case a scalar epilogue is required; this implies that the
  // vector trip count is zero. This check also covers the case where adding one
  // to the backedge-taken count overflowed leading to an incorrect trip count
  // of zero. In this case we will also jump to the scalar loop.
  CmpInst::Predicate CmpPred =
      RequiresScalarEpilogue ? ICmpInst::ICMP_ULE : ICmpInst::ICMP_ULT;
  // If tail is to be folded, vector loop takes care of all iterations.
  VPValue *TripCountVPV = Plan.getTripCount();
  const SCEV *TripCount = vputils::getSCEVExprForVPValue(TripCountVPV, PSE);
  Type *TripCountTy = TripCount->getType();
  ScalarEvolution &SE = *PSE.getSE();
  auto GetMinTripCount = [&]() -> const SCEV * {
    // Compute max(MinProfitableTripCount, UF * VF) and return it.
    const SCEV *VFxUF =
        SE.getElementCount(TripCountTy, (VF * UF), SCEV::FlagNUW);
    if (UF * VF.getKnownMinValue() >=
        MinProfitableTripCount.getKnownMinValue()) {
      // TODO: SCEV should be able to simplify test.
      return VFxUF;
    }
    const SCEV *MinProfitableTripCountSCEV =
        SE.getElementCount(TripCountTy, MinProfitableTripCount, SCEV::FlagNUW);
    return SE.getUMaxExpr(MinProfitableTripCountSCEV, VFxUF);
  };

  VPBasicBlock *EntryVPBB = Plan.getEntry();
  VPBuilder Builder(EntryVPBB);
  VPValue *TripCountCheck = Plan.getFalse();
  const SCEV *Step = GetMinTripCount();
  if (TailFolded) {
    if (CheckNeededWithTailFolding) {
      // vscale is not necessarily a power-of-2, which means we cannot guarantee
      // an overflow to zero when updating induction variables and so an
      // additional overflow check is required before entering the vector loop.

      // Get the maximum unsigned value for the type.
      VPValue *MaxUIntTripCount =
          Plan.getConstantInt(cast<IntegerType>(TripCountTy)->getMask());
      VPValue *DistanceToMax = Builder.createNaryOp(
          Instruction::Sub, {MaxUIntTripCount, TripCountVPV},
          DebugLoc::getUnknown());

      // Don't execute the vector loop if (UMax - n) < (VF * UF).
      // FIXME: Should only check VF * UF, but currently checks Step=max(VF*UF,
      // minProfitableTripCount).
      TripCountCheck = Builder.createICmp(ICmpInst::ICMP_ULT, DistanceToMax,
                                          Builder.createExpandSCEV(Step), DL);
    } else {
      // TripCountCheck = false, folding tail implies positive vector trip
      // count.
    }
  } else {
    // TODO: Emit unconditional branch to vector preheader instead of
    // conditional branch with known condition.
    TripCount = SE.applyLoopGuards(TripCount, OrigLoop);
    // Check if the trip count is < the step.
    if (SE.isKnownPredicate(CmpPred, TripCount, Step)) {
      // TODO: Ensure step is at most the trip count when determining max VF and
      // UF, w/o tail folding.
      TripCountCheck = Plan.getTrue();
    } else if (!SE.isKnownPredicate(CmpInst::getInversePredicate(CmpPred),
                                    TripCount, Step)) {
      // Generate the minimum iteration check only if we cannot prove the
      // check is known to be true, or known to be false.
      VPValue *MinTripCountVPV = Builder.createExpandSCEV(Step);
      TripCountCheck = Builder.createICmp(
          CmpPred, TripCountVPV, MinTripCountVPV, DL, "min.iters.check");
    } // else step known to be < trip count, use TripCountCheck preset to false.
  }
  VPInstruction *Term =
      Builder.createNaryOp(VPInstruction::BranchOnCond, {TripCountCheck}, DL);
  if (MinItersBypassWeights) {
    MDBuilder MDB(Plan.getContext());
    MDNode *BranchWeights = MDB.createBranchWeights(
        ArrayRef(MinItersBypassWeights, 2), /*IsExpected=*/false);
    Term->setMetadata(LLVMContext::MD_prof, BranchWeights);
  }
}

void VPlanTransforms::addMinimumVectorEpilogueIterationCheck(
    VPlan &Plan, Value *TripCount, Value *VectorTripCount,
    bool RequiresScalarEpilogue, ElementCount EpilogueVF, unsigned EpilogueUF,
    unsigned MainLoopStep, unsigned EpilogueLoopStep, ScalarEvolution &SE) {
  // Add the minimum iteration check for the epilogue vector loop.
  VPValue *TC = Plan.getOrAddLiveIn(TripCount);
  VPBuilder Builder(cast<VPBasicBlock>(Plan.getEntry()));
  VPValue *VFxUF = Builder.createExpandSCEV(SE.getElementCount(
      TripCount->getType(), (EpilogueVF * EpilogueUF), SCEV::FlagNUW));
  VPValue *Count = Builder.createNaryOp(
      Instruction::Sub, {TC, Plan.getOrAddLiveIn(VectorTripCount)},
      DebugLoc::getUnknown(), "n.vec.remaining");

  // Generate code to check if the loop's trip count is less than VF * UF of
  // the vector epilogue loop.
  auto P = RequiresScalarEpilogue ? ICmpInst::ICMP_ULE : ICmpInst::ICMP_ULT;
  auto *CheckMinIters = Builder.createICmp(
      P, Count, VFxUF, DebugLoc::getUnknown(), "min.epilog.iters.check");
  VPInstruction *Branch =
      Builder.createNaryOp(VPInstruction::BranchOnCond, CheckMinIters);

  // We assume the remaining `Count` is equally distributed in
  // [0, MainLoopStep)
  // So the probability for `Count < EpilogueLoopStep` should be
  // min(MainLoopStep, EpilogueLoopStep) / MainLoopStep
  // TODO: Improve the estimate by taking the estimated trip count into
  // consideration.
  unsigned EstimatedSkipCount = std::min(MainLoopStep, EpilogueLoopStep);
  const uint32_t Weights[] = {EstimatedSkipCount,
                              MainLoopStep - EstimatedSkipCount};
  MDBuilder MDB(Plan.getContext());
  MDNode *BranchWeights =
      MDB.createBranchWeights(Weights, /*IsExpected=*/false);
  Branch->setMetadata(LLVMContext::MD_prof, BranchWeights);
}

/// If \p V is used by a recipe matching pattern \p P, return it. Otherwise
/// return nullptr;
template <typename MatchT>
static VPRecipeBase *findUserOf(VPValue *V, const MatchT &P) {
  auto It = find_if(V->users(), match_fn(P));
  return It == V->user_end() ? nullptr : cast<VPRecipeBase>(*It);
}

/// If \p V is used by a VPInstruction with \p Opcode, return it. Otherwise
/// return nullptr.
template <unsigned Opcode> static VPInstruction *findUserOf(VPValue *V) {
  return cast_or_null<VPInstruction>(findUserOf(V, m_VPInstruction<Opcode>()));
}

/// Find the ComputeReductionResult recipe for \p PhiR, looking through selects
/// inserted for predicated reductions or tail folding.
static VPInstruction *findComputeReductionResult(VPReductionPHIRecipe *PhiR) {
  VPValue *BackedgeVal = PhiR->getBackedgeValue();
  if (auto *Res =
          findUserOf<VPInstruction::ComputeReductionResult>(BackedgeVal))
    return Res;

  // Look through selects inserted for tail folding or predicated reductions.
  VPRecipeBase *SelR =
      findUserOf(BackedgeVal, m_Select(m_VPValue(), m_VPValue(), m_VPValue()));
  if (!SelR)
    return nullptr;
  return findUserOf<VPInstruction::ComputeReductionResult>(
      cast<VPSingleDefRecipe>(SelR));
}

bool VPlanTransforms::handleMaxMinNumReductions(VPlan &Plan) {
  auto GetMinOrMaxCompareValue =
      [](VPReductionPHIRecipe *RedPhiR) -> VPValue * {
    auto *MinOrMaxR =
        dyn_cast_or_null<VPRecipeWithIRFlags>(RedPhiR->getBackedgeValue());
    if (!MinOrMaxR)
      return nullptr;

    // Check that MinOrMaxR is a VPWidenIntrinsicRecipe or VPReplicateRecipe
    // with an intrinsic that matches the reduction kind.
    Intrinsic::ID ExpectedIntrinsicID =
        getMinMaxReductionIntrinsicOp(RedPhiR->getRecurrenceKind());
    if (!match(MinOrMaxR, m_Intrinsic(ExpectedIntrinsicID)))
      return nullptr;

    if (MinOrMaxR->getOperand(0) == RedPhiR)
      return MinOrMaxR->getOperand(1);

    assert(MinOrMaxR->getOperand(1) == RedPhiR &&
           "Reduction phi operand expected");
    return MinOrMaxR->getOperand(0);
  };

  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  SmallVector<std::pair<VPReductionPHIRecipe *, VPValue *>>
      MinOrMaxNumReductionsToHandle;
  bool HasUnsupportedPhi = false;
  for (auto &R : LoopRegion->getEntryBasicBlock()->phis()) {
    if (isa<VPCanonicalIVPHIRecipe, VPWidenIntOrFpInductionRecipe>(&R))
      continue;
    auto *Cur = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!Cur) {
      // TODO: Also support fixed-order recurrence phis.
      HasUnsupportedPhi = true;
      continue;
    }
    if (!RecurrenceDescriptor::isFPMinMaxNumRecurrenceKind(
            Cur->getRecurrenceKind())) {
      HasUnsupportedPhi = true;
      continue;
    }

    VPValue *MinOrMaxOp = GetMinOrMaxCompareValue(Cur);
    if (!MinOrMaxOp)
      return false;

    MinOrMaxNumReductionsToHandle.emplace_back(Cur, MinOrMaxOp);
  }

  if (MinOrMaxNumReductionsToHandle.empty())
    return true;

  // We won't be able to resume execution in the scalar tail, if there are
  // unsupported header phis or there is no scalar tail at all, due to
  // tail-folding.
  if (HasUnsupportedPhi || !Plan.hasScalarTail())
    return false;

  /// Check if the vector loop of \p Plan can early exit and restart
  /// execution of last vector iteration in the scalar loop. This requires all
  /// recipes up to early exit point be side-effect free as they are
  /// re-executed. Currently we check that the loop is free of any recipe that
  /// may write to memory. Expected to operate on an early VPlan w/o nested
  /// regions.
  for (VPBlockBase *VPB : vp_depth_first_shallow(
           Plan.getVectorLoopRegion()->getEntryBasicBlock())) {
    auto *VPBB = cast<VPBasicBlock>(VPB);
    for (auto &R : *VPBB) {
      if (R.mayWriteToMemory() && !match(&R, m_BranchOnCount()))
        return false;
    }
  }

  VPBasicBlock *LatchVPBB = LoopRegion->getExitingBasicBlock();
  VPBuilder LatchBuilder(LatchVPBB->getTerminator());
  VPValue *AllNaNLanes = nullptr;
  SmallPtrSet<VPValue *, 2> RdxResults;
  for (const auto &[_, MinOrMaxOp] : MinOrMaxNumReductionsToHandle) {
    VPValue *RedNaNLanes =
        LatchBuilder.createFCmp(CmpInst::FCMP_UNO, MinOrMaxOp, MinOrMaxOp);
    AllNaNLanes = AllNaNLanes ? LatchBuilder.createOr(AllNaNLanes, RedNaNLanes)
                              : RedNaNLanes;
  }

  VPValue *AnyNaNLane =
      LatchBuilder.createNaryOp(VPInstruction::AnyOf, {AllNaNLanes});
  VPBasicBlock *MiddleVPBB = Plan.getMiddleBlock();
  VPBuilder MiddleBuilder(MiddleVPBB, MiddleVPBB->begin());
  for (const auto &[RedPhiR, _] : MinOrMaxNumReductionsToHandle) {
    assert(RecurrenceDescriptor::isFPMinMaxNumRecurrenceKind(
               RedPhiR->getRecurrenceKind()) &&
           "unsupported reduction");

    // If we exit early due to NaNs, compute the final reduction result based on
    // the reduction phi at the beginning of the last vector iteration.
    auto *RdxResult = findComputeReductionResult(RedPhiR);
    assert(RdxResult && "must find a ComputeReductionResult");

    auto *NewSel = MiddleBuilder.createSelect(AnyNaNLane, RedPhiR,
                                              RdxResult->getOperand(0));
    RdxResult->setOperand(0, NewSel);
    assert(!RdxResults.contains(RdxResult) && "RdxResult already used");
    RdxResults.insert(RdxResult);
  }

  auto *LatchExitingBranch = LatchVPBB->getTerminator();
  assert(match(LatchExitingBranch, m_BranchOnCount(m_VPValue(), m_VPValue())) &&
         "Unexpected terminator");
  auto *IsLatchExitTaken = LatchBuilder.createICmp(
      CmpInst::ICMP_EQ, LatchExitingBranch->getOperand(0),
      LatchExitingBranch->getOperand(1));
  auto *AnyExitTaken = LatchBuilder.createNaryOp(
      Instruction::Or, {AnyNaNLane, IsLatchExitTaken});
  LatchBuilder.createNaryOp(VPInstruction::BranchOnCond, AnyExitTaken);
  LatchExitingBranch->eraseFromParent();

  // Update resume phis for inductions in the scalar preheader. If AnyNaNLane is
  // true, the resume from the start of the last vector iteration via the
  // canonical IV, otherwise from the original value.
  for (auto &R : Plan.getScalarPreheader()->phis()) {
    auto *ResumeR = cast<VPPhi>(&R);
    VPValue *VecV = ResumeR->getOperand(0);
    if (RdxResults.contains(VecV))
      continue;
    if (auto *DerivedIV = dyn_cast<VPDerivedIVRecipe>(VecV)) {
      if (DerivedIV->getNumUsers() == 1 &&
          DerivedIV->getOperand(1) == &Plan.getVectorTripCount()) {
        auto *NewSel =
            MiddleBuilder.createSelect(AnyNaNLane, LoopRegion->getCanonicalIV(),
                                       &Plan.getVectorTripCount());
        DerivedIV->moveAfter(&*MiddleBuilder.getInsertPoint());
        DerivedIV->setOperand(1, NewSel);
        continue;
      }
    }
    // Bail out and abandon the current, partially modified, VPlan if we
    // encounter resume phi that cannot be updated yet.
    if (VecV != &Plan.getVectorTripCount()) {
      LLVM_DEBUG(dbgs() << "Found resume phi we cannot update for VPlan with "
                           "FMaxNum/FMinNum reduction.\n");
      return false;
    }
    auto *NewSel = MiddleBuilder.createSelect(
        AnyNaNLane, LoopRegion->getCanonicalIV(), VecV);
    ResumeR->setOperand(0, NewSel);
  }

  auto *MiddleTerm = MiddleVPBB->getTerminator();
  MiddleBuilder.setInsertPoint(MiddleTerm);
  VPValue *MiddleCond = MiddleTerm->getOperand(0);
  VPValue *NewCond =
      MiddleBuilder.createAnd(MiddleCond, MiddleBuilder.createNot(AnyNaNLane));
  MiddleTerm->setOperand(0, NewCond);
  return true;
}

bool VPlanTransforms::handleFindLastReductions(VPlan &Plan) {
  if (Plan.hasScalarVFOnly())
    return false;

  // We want to create the following nodes:
  // vector.body:
  //   ...new WidenPHI recipe introduced to keep the mask value for the latest
  //      iteration where any lane was active.
  //   mask.phi = phi [ ir<false>, vector.ph ], [ vp<new.mask>, vector.body ]
  //   ...data.phi (a VPReductionPHIRecipe for a FindLast reduction) already
  //      exists, but needs updating to use 'new.data' for the backedge value.
  //   data.phi = phi ir<default.val>, vp<new.data>
  //
  //   ...'data' and 'compare' created by existing nodes...
  //
  //   ...new recipes introduced to determine whether to update the reduction
  //      values or keep the current one.
  //   any.active = i1 any-of ir<compare>
  //   new.mask = select vp<any.active>, ir<compare>, vp<mask.phi>
  //   new.data = select vp<any.active>, ir<data>, ir<data.phi>
  //
  // middle.block:
  //   ...extract-last-active replaces compute-reduction-result.
  //   result = extract-last-active vp<new.data>, vp<new.mask>, ir<default.val>

  for (auto &Phi : Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&Phi);
    if (!PhiR || !RecurrenceDescriptor::isFindLastRecurrenceKind(
                     PhiR->getRecurrenceKind()))
      continue;

    // Find the condition for the select.
    auto *SelectR = cast<VPSingleDefRecipe>(&PhiR->getBackedgeRecipe());
    VPValue *Cond = nullptr, *Op1 = nullptr, *Op2 = nullptr;
    if (!match(SelectR,
               m_Select(m_VPValue(Cond), m_VPValue(Op1), m_VPValue(Op2))))
      return false;

    // Add mask phi.
    VPBuilder Builder = VPBuilder::getToInsertAfter(PhiR);
    auto *MaskPHI = new VPWidenPHIRecipe(nullptr, /*Start=*/Plan.getFalse());
    Builder.insert(MaskPHI);

    // Add select for mask.
    Builder.setInsertPoint(SelectR);

    if (Op1 == PhiR) {
      // Normalize to selecting the data operand when the condition is true by
      // swapping operands and negating the condition.
      std::swap(Op1, Op2);
      Cond = Builder.createNot(Cond);
    }
    assert(Op2 == PhiR && "data value must be selected if Cond is true");

    VPValue *AnyOf = Builder.createNaryOp(VPInstruction::AnyOf, {Cond});
    VPValue *MaskSelect = Builder.createSelect(AnyOf, Cond, MaskPHI);
    MaskPHI->addOperand(MaskSelect);

    // Replace select for data.
    VPValue *DataSelect =
        Builder.createSelect(AnyOf, Op1, Op2, SelectR->getDebugLoc());
    SelectR->replaceAllUsesWith(DataSelect);
    SelectR->eraseFromParent();

    // Find final reduction computation and replace it with an
    // extract.last.active intrinsic.
    auto *RdxResult =
        findUserOf<VPInstruction::ComputeReductionResult>(DataSelect);
    // TODO: Handle tail-folding.
    if (!RdxResult)
      return false;
    Builder.setInsertPoint(RdxResult);
    auto *ExtractLastActive =
        Builder.createNaryOp(VPInstruction::ExtractLastActive,
                             {DataSelect, MaskSelect, PhiR->getStartValue()},
                             RdxResult->getDebugLoc());
    RdxResult->replaceAllUsesWith(ExtractLastActive);
    RdxResult->eraseFromParent();
  }

  return true;
}

bool VPlanTransforms::handleMultiUseReductions(VPlan &Plan) {
  for (auto &PhiR : make_early_inc_range(
           Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis())) {
    auto *MinOrMaxPhiR = dyn_cast<VPReductionPHIRecipe>(&PhiR);
    // TODO: check for multi-uses in VPlan directly.
    if (!MinOrMaxPhiR || !MinOrMaxPhiR->hasUsesOutsideReductionChain())
      continue;

    // MinOrMaxPhiR has users outside the reduction cycle in the loop. Check if
    // the only other user is a FindLastIV reduction. MinOrMaxPhiR must have
    // exactly 2 users:
    // 1) the min/max operation of the reduction cycle, and
    // 2) the compare of a FindLastIV reduction cycle. This compare must match
    // the min/max operation - comparing MinOrMaxPhiR with the operand of the
    // min/max operation, and be used only by the select of the FindLastIV
    // reduction cycle.
    RecurKind RdxKind = MinOrMaxPhiR->getRecurrenceKind();
    assert(
        RecurrenceDescriptor::isIntMinMaxRecurrenceKind(RdxKind) &&
        "only min/max recurrences support users outside the reduction chain");

    auto *MinOrMaxOp =
        dyn_cast<VPRecipeWithIRFlags>(MinOrMaxPhiR->getBackedgeValue());
    if (!MinOrMaxOp)
      return false;

    // Check that MinOrMaxOp is a VPWidenIntrinsicRecipe or VPReplicateRecipe
    // with an intrinsic that matches the reduction kind.
    Intrinsic::ID ExpectedIntrinsicID = getMinMaxReductionIntrinsicOp(RdxKind);
    if (!match(MinOrMaxOp, m_Intrinsic(ExpectedIntrinsicID)))
      return false;

    // MinOrMaxOp must have 2 users: 1) MinOrMaxPhiR and 2)
    // ComputeReductionResult.
    assert(MinOrMaxOp->getNumUsers() == 2 &&
           "MinOrMaxOp must have exactly 2 users");
    VPValue *MinOrMaxOpValue = MinOrMaxOp->getOperand(0);
    if (MinOrMaxOpValue == MinOrMaxPhiR)
      MinOrMaxOpValue = MinOrMaxOp->getOperand(1);

    VPValue *CmpOpA;
    VPValue *CmpOpB;
    CmpPredicate Pred;
    auto *Cmp = dyn_cast_or_null<VPRecipeWithIRFlags>(findUserOf(
        MinOrMaxPhiR, m_Cmp(Pred, m_VPValue(CmpOpA), m_VPValue(CmpOpB))));
    if (!Cmp || Cmp->getNumUsers() != 1 ||
        (CmpOpA != MinOrMaxOpValue && CmpOpB != MinOrMaxOpValue))
      return false;

    if (MinOrMaxOpValue != CmpOpB)
      Pred = CmpInst::getSwappedPredicate(Pred);

    // MinOrMaxPhiR must have exactly 2 users:
    // * MinOrMaxOp,
    // * Cmp (that's part of a FindLastIV chain).
    if (MinOrMaxPhiR->getNumUsers() != 2)
      return false;

    VPInstruction *MinOrMaxResult =
        findUserOf<VPInstruction::ComputeReductionResult>(MinOrMaxOp);
    assert(is_contained(MinOrMaxPhiR->users(), MinOrMaxOp) &&
           "one user must be MinOrMaxOp");
    assert(MinOrMaxResult &&
           "MinOrMaxOp must have a ComputeReductionResult user");

    // Cmp must be used by the select of a FindLastIV chain.
    VPValue *Sel = dyn_cast<VPSingleDefRecipe>(Cmp->getSingleUser());
    VPValue *IVOp, *FindIV;
    if (!Sel || Sel->getNumUsers() != 2 ||
        !match(Sel,
               m_Select(m_Specific(Cmp), m_VPValue(IVOp), m_VPValue(FindIV))))
      return false;

    if (!isa<VPReductionPHIRecipe>(FindIV)) {
      std::swap(FindIV, IVOp);
      Pred = CmpInst::getInversePredicate(Pred);
    }

    auto *FindIVPhiR = dyn_cast<VPReductionPHIRecipe>(FindIV);
    if (!FindIVPhiR || !RecurrenceDescriptor::isFindLastIVRecurrenceKind(
                           FindIVPhiR->getRecurrenceKind()))
      return false;

    // TODO: Support cases where IVOp is the IV increment.
    if (!match(IVOp, m_TruncOrSelf(m_VPValue(IVOp))) ||
        !isa<VPWidenIntOrFpInductionRecipe>(IVOp))
      return false;

    CmpInst::Predicate RdxPredicate = [RdxKind]() {
      switch (RdxKind) {
      case RecurKind::UMin:
        return CmpInst::ICMP_UGE;
      case RecurKind::UMax:
        return CmpInst::ICMP_ULE;
      case RecurKind::SMax:
        return CmpInst::ICMP_SLE;
      case RecurKind::SMin:
        return CmpInst::ICMP_SGE;
      default:
        llvm_unreachable("unhandled recurrence kind");
      }
    }();

    // TODO: Strict predicates need to find the first IV value for which the
    // predicate holds, not the last.
    if (Pred != RdxPredicate)
      return false;

    assert(!FindIVPhiR->isInLoop() && !FindIVPhiR->isOrdered() &&
           "cannot handle inloop/ordered reductions yet");

    // The reduction using MinOrMaxPhiR needs adjusting to compute the correct
    // result:
    //  1. We need to find the last IV for which the condition based on the
    //     min/max recurrence is true,
    //  2. Compare the partial min/max reduction result to its final value and,
    //  3. Select the lanes of the partial FindLastIV reductions which
    //     correspond to the lanes matching the min/max reduction result.
    //
    // For example, this transforms
    // vp<%min.result> = compute-reduction-result ir<%min.val.next>
    // vp<%find.iv.result = compute-find-iv-result ir<0>, SENTINEL,
    //                                             vp<%min.idx.next>
    //
    // into:
    //
    // vp<min.result> = compute-reduction-result ir<%min.val.next>
    // vp<%final.min.cmp> = icmp eq ir<%min.val.next>, vp<min.result>
    // vp<%final.iv> = select vp<%final.min.cmp>, ir<%min.idx.next>, SENTINEL
    // vp<%find.iv.result> = compute-find-iv-result ir<0>, SENTINEL,
    //                                              vp<%final.iv>
    VPInstruction *FindIVResult =
        findUserOf<VPInstruction::ComputeFindIVResult>(
            FindIVPhiR->getBackedgeValue());
    assert(FindIVResult && "Backedge value feeding FindIVPhiR expected to also "
                           "feed a ComputeFindIVResult");
    assert(FindIVResult->getParent() == MinOrMaxResult->getParent() &&
           "both results must be computed in the same block");
    MinOrMaxResult->moveBefore(*FindIVResult->getParent(),
                               FindIVResult->getIterator());

    VPBuilder B(FindIVResult);
    VPValue *MinOrMaxExiting = MinOrMaxResult->getOperand(0);
    auto *FinalMinOrMaxCmp =
        B.createICmp(CmpInst::ICMP_EQ, MinOrMaxExiting, MinOrMaxResult);
    VPValue *Sentinel = FindIVResult->getOperand(1);
    VPValue *LastIVExiting = FindIVResult->getOperand(2);
    auto *FinalIVSelect =
        B.createSelect(FinalMinOrMaxCmp, LastIVExiting, Sentinel);
    FindIVResult->setOperand(2, FinalIVSelect);
  }
  return true;
}
