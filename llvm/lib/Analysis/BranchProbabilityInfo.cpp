//===-- BranchProbabilityInfo.cpp - Branch Probability Analysis -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "branch-prob"

INITIALIZE_PASS_BEGIN(BranchProbabilityInfoWrapperPass, "branch-prob",
                      "Branch Probability Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(BranchProbabilityInfoWrapperPass, "branch-prob",
                    "Branch Probability Analysis", false, true)

char BranchProbabilityInfoWrapperPass::ID = 0;

// Weights are for internal use only. They are used by heuristics to help to
// estimate edges' probability. Example:
//
// Using "Loop Branch Heuristics" we predict weights of edges for the
// block BB2.
//         ...
//          |
//          V
//         BB1<-+
//          |   |
//          |   | (Weight = 124)
//          V   |
//         BB2--+
//          |
//          | (Weight = 4)
//          V
//         BB3
//
// Probability of the edge BB2->BB1 = 124 / (124 + 4) = 0.96875
// Probability of the edge BB2->BB3 = 4 / (124 + 4) = 0.03125
static const uint32_t LBH_TAKEN_WEIGHT = 124;
static const uint32_t LBH_NONTAKEN_WEIGHT = 4;

/// \brief Unreachable-terminating branch taken weight.
///
/// This is the weight for a branch being taken to a block that terminates
/// (eventually) in unreachable. These are predicted as unlikely as possible.
static const uint32_t UR_TAKEN_WEIGHT = 1;

/// \brief Unreachable-terminating branch not-taken weight.
///
/// This is the weight for a branch not being taken toward a block that
/// terminates (eventually) in unreachable. Such a branch is essentially never
/// taken. Set the weight to an absurdly high value so that nested loops don't
/// easily subsume it.
static const uint32_t UR_NONTAKEN_WEIGHT = 1024*1024 - 1;

/// \brief Weight for a branch taken going into a cold block.
///
/// This is the weight for a branch taken toward a block marked
/// cold.  A block is marked cold if it's postdominated by a
/// block containing a call to a cold function.  Cold functions
/// are those marked with attribute 'cold'.
static const uint32_t CC_TAKEN_WEIGHT = 4;

/// \brief Weight for a branch not-taken into a cold block.
///
/// This is the weight for a branch not taken toward a block marked
/// cold.
static const uint32_t CC_NONTAKEN_WEIGHT = 64;

static const uint32_t PH_TAKEN_WEIGHT = 20;
static const uint32_t PH_NONTAKEN_WEIGHT = 12;

static const uint32_t ZH_TAKEN_WEIGHT = 20;
static const uint32_t ZH_NONTAKEN_WEIGHT = 12;

static const uint32_t FPH_TAKEN_WEIGHT = 20;
static const uint32_t FPH_NONTAKEN_WEIGHT = 12;

/// \brief Invoke-terminating normal branch taken weight
///
/// This is the weight for branching to the normal destination of an invoke
/// instruction. We expect this to happen most of the time. Set the weight to an
/// absurdly high value so that nested loops subsume it.
static const uint32_t IH_TAKEN_WEIGHT = 1024 * 1024 - 1;

/// \brief Invoke-terminating normal branch not-taken weight.
///
/// This is the weight for branching to the unwind destination of an invoke
/// instruction. This is essentially never taken.
static const uint32_t IH_NONTAKEN_WEIGHT = 1;

/// \brief Calculate edge weights for successors lead to unreachable.
///
/// Predict that a successor which leads necessarily to an
/// unreachable-terminated block as extremely unlikely.
bool BranchProbabilityInfo::calcUnreachableHeuristics(BasicBlock *BB) {
  TerminatorInst *TI = BB->getTerminator();
  if (TI->getNumSuccessors() == 0) {
    if (isa<UnreachableInst>(TI))
      PostDominatedByUnreachable.insert(BB);
    return false;
  }

  SmallVector<unsigned, 4> UnreachableEdges;
  SmallVector<unsigned, 4> ReachableEdges;

  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    if (PostDominatedByUnreachable.count(*I))
      UnreachableEdges.push_back(I.getSuccessorIndex());
    else
      ReachableEdges.push_back(I.getSuccessorIndex());
  }

  // If all successors are in the set of blocks post-dominated by unreachable,
  // this block is too.
  if (UnreachableEdges.size() == TI->getNumSuccessors())
    PostDominatedByUnreachable.insert(BB);

  // Skip probabilities if this block has a single successor or if all were
  // reachable.
  if (TI->getNumSuccessors() == 1 || UnreachableEdges.empty())
    return false;

  // If the terminator is an InvokeInst, check only the normal destination block
  // as the unwind edge of InvokeInst is also very unlikely taken.
  if (auto *II = dyn_cast<InvokeInst>(TI))
    if (PostDominatedByUnreachable.count(II->getNormalDest())) {
      PostDominatedByUnreachable.insert(BB);
      // Return false here so that edge weights for InvokeInst could be decided
      // in calcInvokeHeuristics().
      return false;
    }

  if (ReachableEdges.empty()) {
    BranchProbability Prob(1, UnreachableEdges.size());
    for (unsigned SuccIdx : UnreachableEdges)
      setEdgeProbability(BB, SuccIdx, Prob);
    return true;
  }

  BranchProbability UnreachableProb(UR_TAKEN_WEIGHT,
                                    (UR_TAKEN_WEIGHT + UR_NONTAKEN_WEIGHT) *
                                        UnreachableEdges.size());
  BranchProbability ReachableProb(UR_NONTAKEN_WEIGHT,
                                  (UR_TAKEN_WEIGHT + UR_NONTAKEN_WEIGHT) *
                                      ReachableEdges.size());

  for (unsigned SuccIdx : UnreachableEdges)
    setEdgeProbability(BB, SuccIdx, UnreachableProb);
  for (unsigned SuccIdx : ReachableEdges)
    setEdgeProbability(BB, SuccIdx, ReachableProb);

  return true;
}

// Propagate existing explicit probabilities from either profile data or
// 'expect' intrinsic processing.
bool BranchProbabilityInfo::calcMetadataWeights(BasicBlock *BB) {
  TerminatorInst *TI = BB->getTerminator();
  if (TI->getNumSuccessors() == 1)
    return false;
  if (!isa<BranchInst>(TI) && !isa<SwitchInst>(TI))
    return false;

  MDNode *WeightsNode = TI->getMetadata(LLVMContext::MD_prof);
  if (!WeightsNode)
    return false;

  // Check that the number of successors is manageable.
  assert(TI->getNumSuccessors() < UINT32_MAX && "Too many successors");

  // Ensure there are weights for all of the successors. Note that the first
  // operand to the metadata node is a name, not a weight.
  if (WeightsNode->getNumOperands() != TI->getNumSuccessors() + 1)
    return false;

  // Build up the final weights that will be used in a temporary buffer.
  // Compute the sum of all weights to later decide whether they need to
  // be scaled to fit in 32 bits.
  uint64_t WeightSum = 0;
  SmallVector<uint32_t, 2> Weights;
  Weights.reserve(TI->getNumSuccessors());
  for (unsigned i = 1, e = WeightsNode->getNumOperands(); i != e; ++i) {
    ConstantInt *Weight =
        mdconst::dyn_extract<ConstantInt>(WeightsNode->getOperand(i));
    if (!Weight)
      return false;
    assert(Weight->getValue().getActiveBits() <= 32 &&
           "Too many bits for uint32_t");
    Weights.push_back(Weight->getZExtValue());
    WeightSum += Weights.back();
  }
  assert(Weights.size() == TI->getNumSuccessors() && "Checked above");

  // If the sum of weights does not fit in 32 bits, scale every weight down
  // accordingly.
  uint64_t ScalingFactor =
      (WeightSum > UINT32_MAX) ? WeightSum / UINT32_MAX + 1 : 1;

  WeightSum = 0;
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
    Weights[i] /= ScalingFactor;
    WeightSum += Weights[i];
  }

  if (WeightSum == 0) {
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      setEdgeProbability(BB, i, {1, e});
  } else {
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      setEdgeProbability(BB, i, {Weights[i], static_cast<uint32_t>(WeightSum)});
  }

  assert(WeightSum <= UINT32_MAX &&
         "Expected weights to scale down to 32 bits");

  return true;
}

/// \brief Calculate edge weights for edges leading to cold blocks.
///
/// A cold block is one post-dominated by  a block with a call to a
/// cold function.  Those edges are unlikely to be taken, so we give
/// them relatively low weight.
///
/// Return true if we could compute the weights for cold edges.
/// Return false, otherwise.
bool BranchProbabilityInfo::calcColdCallHeuristics(BasicBlock *BB) {
  TerminatorInst *TI = BB->getTerminator();
  if (TI->getNumSuccessors() == 0)
    return false;

  // Determine which successors are post-dominated by a cold block.
  SmallVector<unsigned, 4> ColdEdges;
  SmallVector<unsigned, 4> NormalEdges;
  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I)
    if (PostDominatedByColdCall.count(*I))
      ColdEdges.push_back(I.getSuccessorIndex());
    else
      NormalEdges.push_back(I.getSuccessorIndex());

  // If all successors are in the set of blocks post-dominated by cold calls,
  // this block is in the set post-dominated by cold calls.
  if (ColdEdges.size() == TI->getNumSuccessors())
    PostDominatedByColdCall.insert(BB);
  else {
    // Otherwise, if the block itself contains a cold function, add it to the
    // set of blocks postdominated by a cold call.
    assert(!PostDominatedByColdCall.count(BB));
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (CallInst *CI = dyn_cast<CallInst>(I))
        if (CI->hasFnAttr(Attribute::Cold)) {
          PostDominatedByColdCall.insert(BB);
          break;
        }
  }

  // Skip probabilities if this block has a single successor.
  if (TI->getNumSuccessors() == 1 || ColdEdges.empty())
    return false;

  if (NormalEdges.empty()) {
    BranchProbability Prob(1, ColdEdges.size());
    for (unsigned SuccIdx : ColdEdges)
      setEdgeProbability(BB, SuccIdx, Prob);
    return true;
  }

  BranchProbability ColdProb(CC_TAKEN_WEIGHT,
                             (CC_TAKEN_WEIGHT + CC_NONTAKEN_WEIGHT) *
                                 ColdEdges.size());
  BranchProbability NormalProb(CC_NONTAKEN_WEIGHT,
                               (CC_TAKEN_WEIGHT + CC_NONTAKEN_WEIGHT) *
                                   NormalEdges.size());

  for (unsigned SuccIdx : ColdEdges)
    setEdgeProbability(BB, SuccIdx, ColdProb);
  for (unsigned SuccIdx : NormalEdges)
    setEdgeProbability(BB, SuccIdx, NormalProb);

  return true;
}

// Calculate Edge Weights using "Pointer Heuristics". Predict a comparsion
// between two pointer or pointer and NULL will fail.
bool BranchProbabilityInfo::calcPointerHeuristics(BasicBlock *BB) {
  BranchInst * BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isConditional())
    return false;

  Value *Cond = BI->getCondition();
  ICmpInst *CI = dyn_cast<ICmpInst>(Cond);
  if (!CI || !CI->isEquality())
    return false;

  Value *LHS = CI->getOperand(0);

  if (!LHS->getType()->isPointerTy())
    return false;

  assert(CI->getOperand(1)->getType()->isPointerTy());

  // p != 0   ->   isProb = true
  // p == 0   ->   isProb = false
  // p != q   ->   isProb = true
  // p == q   ->   isProb = false;
  unsigned TakenIdx = 0, NonTakenIdx = 1;
  bool isProb = CI->getPredicate() == ICmpInst::ICMP_NE;
  if (!isProb)
    std::swap(TakenIdx, NonTakenIdx);

  BranchProbability TakenProb(PH_TAKEN_WEIGHT,
                              PH_TAKEN_WEIGHT + PH_NONTAKEN_WEIGHT);
  setEdgeProbability(BB, TakenIdx, TakenProb);
  setEdgeProbability(BB, NonTakenIdx, TakenProb.getCompl());
  return true;
}

// Calculate Edge Weights using "Loop Branch Heuristics". Predict backedges
// as taken, exiting edges as not-taken.
bool BranchProbabilityInfo::calcLoopBranchHeuristics(BasicBlock *BB,
                                                     const LoopInfo &LI) {
  Loop *L = LI.getLoopFor(BB);
  if (!L)
    return false;

  SmallVector<unsigned, 8> BackEdges;
  SmallVector<unsigned, 8> ExitingEdges;
  SmallVector<unsigned, 8> InEdges; // Edges from header to the loop.

  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    if (!L->contains(*I))
      ExitingEdges.push_back(I.getSuccessorIndex());
    else if (L->getHeader() == *I)
      BackEdges.push_back(I.getSuccessorIndex());
    else
      InEdges.push_back(I.getSuccessorIndex());
  }

  if (BackEdges.empty() && ExitingEdges.empty())
    return false;

  // Collect the sum of probabilities of back-edges/in-edges/exiting-edges, and
  // normalize them so that they sum up to one.
  SmallVector<BranchProbability, 4> Probs(3, BranchProbability::getZero());
  unsigned Denom = (BackEdges.empty() ? 0 : LBH_TAKEN_WEIGHT) +
                   (InEdges.empty() ? 0 : LBH_TAKEN_WEIGHT) +
                   (ExitingEdges.empty() ? 0 : LBH_NONTAKEN_WEIGHT);
  if (!BackEdges.empty())
    Probs[0] = BranchProbability(LBH_TAKEN_WEIGHT, Denom);
  if (!InEdges.empty())
    Probs[1] = BranchProbability(LBH_TAKEN_WEIGHT, Denom);
  if (!ExitingEdges.empty())
    Probs[2] = BranchProbability(LBH_NONTAKEN_WEIGHT, Denom);

  if (uint32_t numBackEdges = BackEdges.size()) {
    auto Prob = Probs[0] / numBackEdges;
    for (unsigned SuccIdx : BackEdges)
      setEdgeProbability(BB, SuccIdx, Prob);
  }

  if (uint32_t numInEdges = InEdges.size()) {
    auto Prob = Probs[1] / numInEdges;
    for (unsigned SuccIdx : InEdges)
      setEdgeProbability(BB, SuccIdx, Prob);
  }

  if (uint32_t numExitingEdges = ExitingEdges.size()) {
    auto Prob = Probs[2] / numExitingEdges;
    for (unsigned SuccIdx : ExitingEdges)
      setEdgeProbability(BB, SuccIdx, Prob);
  }

  return true;
}

bool BranchProbabilityInfo::calcZeroHeuristics(BasicBlock *BB) {
  BranchInst * BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isConditional())
    return false;

  Value *Cond = BI->getCondition();
  ICmpInst *CI = dyn_cast<ICmpInst>(Cond);
  if (!CI)
    return false;

  Value *RHS = CI->getOperand(1);
  ConstantInt *CV = dyn_cast<ConstantInt>(RHS);
  if (!CV)
    return false;

  // If the LHS is the result of AND'ing a value with a single bit bitmask,
  // we don't have information about probabilities.
  if (Instruction *LHS = dyn_cast<Instruction>(CI->getOperand(0)))
    if (LHS->getOpcode() == Instruction::And)
      if (ConstantInt *AndRHS = dyn_cast<ConstantInt>(LHS->getOperand(1)))
        if (AndRHS->getUniqueInteger().isPowerOf2())
          return false;

  bool isProb;
  if (CV->isZero()) {
    switch (CI->getPredicate()) {
    case CmpInst::ICMP_EQ:
      // X == 0   ->  Unlikely
      isProb = false;
      break;
    case CmpInst::ICMP_NE:
      // X != 0   ->  Likely
      isProb = true;
      break;
    case CmpInst::ICMP_SLT:
      // X < 0   ->  Unlikely
      isProb = false;
      break;
    case CmpInst::ICMP_SGT:
      // X > 0   ->  Likely
      isProb = true;
      break;
    default:
      return false;
    }
  } else if (CV->isOne() && CI->getPredicate() == CmpInst::ICMP_SLT) {
    // InstCombine canonicalizes X <= 0 into X < 1.
    // X <= 0   ->  Unlikely
    isProb = false;
  } else if (CV->isAllOnesValue()) {
    switch (CI->getPredicate()) {
    case CmpInst::ICMP_EQ:
      // X == -1  ->  Unlikely
      isProb = false;
      break;
    case CmpInst::ICMP_NE:
      // X != -1  ->  Likely
      isProb = true;
      break;
    case CmpInst::ICMP_SGT:
      // InstCombine canonicalizes X >= 0 into X > -1.
      // X >= 0   ->  Likely
      isProb = true;
      break;
    default:
      return false;
    }
  } else {
    return false;
  }

  unsigned TakenIdx = 0, NonTakenIdx = 1;

  if (!isProb)
    std::swap(TakenIdx, NonTakenIdx);

  BranchProbability TakenProb(ZH_TAKEN_WEIGHT,
                              ZH_TAKEN_WEIGHT + ZH_NONTAKEN_WEIGHT);
  setEdgeProbability(BB, TakenIdx, TakenProb);
  setEdgeProbability(BB, NonTakenIdx, TakenProb.getCompl());
  return true;
}

bool BranchProbabilityInfo::calcFloatingPointHeuristics(BasicBlock *BB) {
  BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isConditional())
    return false;

  Value *Cond = BI->getCondition();
  FCmpInst *FCmp = dyn_cast<FCmpInst>(Cond);
  if (!FCmp)
    return false;

  bool isProb;
  if (FCmp->isEquality()) {
    // f1 == f2 -> Unlikely
    // f1 != f2 -> Likely
    isProb = !FCmp->isTrueWhenEqual();
  } else if (FCmp->getPredicate() == FCmpInst::FCMP_ORD) {
    // !isnan -> Likely
    isProb = true;
  } else if (FCmp->getPredicate() == FCmpInst::FCMP_UNO) {
    // isnan -> Unlikely
    isProb = false;
  } else {
    return false;
  }

  unsigned TakenIdx = 0, NonTakenIdx = 1;

  if (!isProb)
    std::swap(TakenIdx, NonTakenIdx);

  BranchProbability TakenProb(FPH_TAKEN_WEIGHT,
                              FPH_TAKEN_WEIGHT + FPH_NONTAKEN_WEIGHT);
  setEdgeProbability(BB, TakenIdx, TakenProb);
  setEdgeProbability(BB, NonTakenIdx, TakenProb.getCompl());
  return true;
}

bool BranchProbabilityInfo::calcInvokeHeuristics(BasicBlock *BB) {
  InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator());
  if (!II)
    return false;

  BranchProbability TakenProb(IH_TAKEN_WEIGHT,
                              IH_TAKEN_WEIGHT + IH_NONTAKEN_WEIGHT);
  setEdgeProbability(BB, 0 /*Index for Normal*/, TakenProb);
  setEdgeProbability(BB, 1 /*Index for Unwind*/, TakenProb.getCompl());
  return true;
}

void BranchProbabilityInfo::releaseMemory() {
  Probs.clear();
}

void BranchProbabilityInfo::print(raw_ostream &OS) const {
  OS << "---- Branch Probabilities ----\n";
  // We print the probabilities from the last function the analysis ran over,
  // or the function it is currently running over.
  assert(LastF && "Cannot print prior to running over a function");
  for (const auto &BI : *LastF) {
    for (succ_const_iterator SI = succ_begin(&BI), SE = succ_end(&BI); SI != SE;
         ++SI) {
      printEdgeProbability(OS << "  ", &BI, *SI);
    }
  }
}

bool BranchProbabilityInfo::
isEdgeHot(const BasicBlock *Src, const BasicBlock *Dst) const {
  // Hot probability is at least 4/5 = 80%
  // FIXME: Compare against a static "hot" BranchProbability.
  return getEdgeProbability(Src, Dst) > BranchProbability(4, 5);
}

BasicBlock *BranchProbabilityInfo::getHotSucc(BasicBlock *BB) const {
  auto MaxProb = BranchProbability::getZero();
  BasicBlock *MaxSucc = nullptr;

  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    BasicBlock *Succ = *I;
    auto Prob = getEdgeProbability(BB, Succ);
    if (Prob > MaxProb) {
      MaxProb = Prob;
      MaxSucc = Succ;
    }
  }

  // Hot probability is at least 4/5 = 80%
  if (MaxProb > BranchProbability(4, 5))
    return MaxSucc;

  return nullptr;
}

/// Get the raw edge probability for the edge. If can't find it, return a
/// default probability 1/N where N is the number of successors. Here an edge is
/// specified using PredBlock and an
/// index to the successors.
BranchProbability
BranchProbabilityInfo::getEdgeProbability(const BasicBlock *Src,
                                          unsigned IndexInSuccessors) const {
  auto I = Probs.find(std::make_pair(Src, IndexInSuccessors));

  if (I != Probs.end())
    return I->second;

  return {1,
          static_cast<uint32_t>(std::distance(succ_begin(Src), succ_end(Src)))};
}

BranchProbability
BranchProbabilityInfo::getEdgeProbability(const BasicBlock *Src,
                                          succ_const_iterator Dst) const {
  return getEdgeProbability(Src, Dst.getSuccessorIndex());
}

/// Get the raw edge probability calculated for the block pair. This returns the
/// sum of all raw edge probabilities from Src to Dst.
BranchProbability
BranchProbabilityInfo::getEdgeProbability(const BasicBlock *Src,
                                          const BasicBlock *Dst) const {
  auto Prob = BranchProbability::getZero();
  bool FoundProb = false;
  for (succ_const_iterator I = succ_begin(Src), E = succ_end(Src); I != E; ++I)
    if (*I == Dst) {
      auto MapI = Probs.find(std::make_pair(Src, I.getSuccessorIndex()));
      if (MapI != Probs.end()) {
        FoundProb = true;
        Prob += MapI->second;
      }
    }
  uint32_t succ_num = std::distance(succ_begin(Src), succ_end(Src));
  return FoundProb ? Prob : BranchProbability(1, succ_num);
}

/// Set the edge probability for a given edge specified by PredBlock and an
/// index to the successors.
void BranchProbabilityInfo::setEdgeProbability(const BasicBlock *Src,
                                               unsigned IndexInSuccessors,
                                               BranchProbability Prob) {
  Probs[std::make_pair(Src, IndexInSuccessors)] = Prob;
  DEBUG(dbgs() << "set edge " << Src->getName() << " -> " << IndexInSuccessors
               << " successor probability to " << Prob << "\n");
}

raw_ostream &
BranchProbabilityInfo::printEdgeProbability(raw_ostream &OS,
                                            const BasicBlock *Src,
                                            const BasicBlock *Dst) const {

  const BranchProbability Prob = getEdgeProbability(Src, Dst);
  OS << "edge " << Src->getName() << " -> " << Dst->getName()
     << " probability is " << Prob
     << (isEdgeHot(Src, Dst) ? " [HOT edge]\n" : "\n");

  return OS;
}

void BranchProbabilityInfo::calculate(Function &F, const LoopInfo& LI) {
  DEBUG(dbgs() << "---- Branch Probability Info : " << F.getName()
               << " ----\n\n");
  LastF = &F; // Store the last function we ran on for printing.
  assert(PostDominatedByUnreachable.empty());
  assert(PostDominatedByColdCall.empty());

  // Walk the basic blocks in post-order so that we can build up state about
  // the successors of a block iteratively.
  for (auto BB : post_order(&F.getEntryBlock())) {
    DEBUG(dbgs() << "Computing probabilities for " << BB->getName() << "\n");
    if (calcUnreachableHeuristics(BB))
      continue;
    if (calcMetadataWeights(BB))
      continue;
    if (calcColdCallHeuristics(BB))
      continue;
    if (calcLoopBranchHeuristics(BB, LI))
      continue;
    if (calcPointerHeuristics(BB))
      continue;
    if (calcZeroHeuristics(BB))
      continue;
    if (calcFloatingPointHeuristics(BB))
      continue;
    calcInvokeHeuristics(BB);
  }

  PostDominatedByUnreachable.clear();
  PostDominatedByColdCall.clear();
}

void BranchProbabilityInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.setPreservesAll();
}

bool BranchProbabilityInfoWrapperPass::runOnFunction(Function &F) {
  const LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  BPI.calculate(F, LI);
  return false;
}

void BranchProbabilityInfoWrapperPass::releaseMemory() { BPI.releaseMemory(); }

void BranchProbabilityInfoWrapperPass::print(raw_ostream &OS,
                                             const Module *) const {
  BPI.print(OS);
}
