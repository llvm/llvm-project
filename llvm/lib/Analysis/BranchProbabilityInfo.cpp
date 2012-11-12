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

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(BranchProbabilityInfo, "branch-prob",
                      "Branch Probability Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(BranchProbabilityInfo, "branch-prob",
                    "Branch Probability Analysis", false, true)

char BranchProbabilityInfo::ID = 0;

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

// Standard weight value. Used when none of the heuristics set weight for
// the edge.
static const uint32_t NORMAL_WEIGHT = 16;

// Minimum weight of an edge. Please note, that weight is NEVER 0.
static const uint32_t MIN_WEIGHT = 1;

static uint32_t getMaxWeightFor(BasicBlock *BB) {
  return UINT32_MAX / BB->getTerminator()->getNumSuccessors();
}


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

  uint32_t UnreachableWeight =
    std::max(UR_TAKEN_WEIGHT / (unsigned)UnreachableEdges.size(), MIN_WEIGHT);
  for (SmallVector<unsigned, 4>::iterator I = UnreachableEdges.begin(),
                                          E = UnreachableEdges.end();
       I != E; ++I)
    setEdgeWeight(BB, *I, UnreachableWeight);

  if (ReachableEdges.empty())
    return true;
  uint32_t ReachableWeight =
    std::max(UR_NONTAKEN_WEIGHT / (unsigned)ReachableEdges.size(),
             NORMAL_WEIGHT);
  for (SmallVector<unsigned, 4>::iterator I = ReachableEdges.begin(),
                                          E = ReachableEdges.end();
       I != E; ++I)
    setEdgeWeight(BB, *I, ReachableWeight);

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

  // Ensure there are weights for all of the successors. Note that the first
  // operand to the metadata node is a name, not a weight.
  if (WeightsNode->getNumOperands() != TI->getNumSuccessors() + 1)
    return false;

  // Build up the final weights that will be used in a temporary buffer, but
  // don't add them until all weihts are present. Each weight value is clamped
  // to [1, getMaxWeightFor(BB)].
  uint32_t WeightLimit = getMaxWeightFor(BB);
  SmallVector<uint32_t, 2> Weights;
  Weights.reserve(TI->getNumSuccessors());
  for (unsigned i = 1, e = WeightsNode->getNumOperands(); i != e; ++i) {
    ConstantInt *Weight = dyn_cast<ConstantInt>(WeightsNode->getOperand(i));
    if (!Weight)
      return false;
    Weights.push_back(
      std::max<uint32_t>(1, Weight->getLimitedValue(WeightLimit)));
  }
  assert(Weights.size() == TI->getNumSuccessors() && "Checked above");
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    setEdgeWeight(BB, i, Weights[i]);

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

  setEdgeWeight(BB, TakenIdx, PH_TAKEN_WEIGHT);
  setEdgeWeight(BB, NonTakenIdx, PH_NONTAKEN_WEIGHT);
  return true;
}

// Calculate Edge Weights using "Loop Branch Heuristics". Predict backedges
// as taken, exiting edges as not-taken.
bool BranchProbabilityInfo::calcLoopBranchHeuristics(BasicBlock *BB) {
  Loop *L = LI->getLoopFor(BB);
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

  if (uint32_t numBackEdges = BackEdges.size()) {
    uint32_t backWeight = LBH_TAKEN_WEIGHT / numBackEdges;
    if (backWeight < NORMAL_WEIGHT)
      backWeight = NORMAL_WEIGHT;

    for (SmallVector<unsigned, 8>::iterator EI = BackEdges.begin(),
         EE = BackEdges.end(); EI != EE; ++EI) {
      setEdgeWeight(BB, *EI, backWeight);
    }
  }

  if (uint32_t numInEdges = InEdges.size()) {
    uint32_t inWeight = LBH_TAKEN_WEIGHT / numInEdges;
    if (inWeight < NORMAL_WEIGHT)
      inWeight = NORMAL_WEIGHT;

    for (SmallVector<unsigned, 8>::iterator EI = InEdges.begin(),
         EE = InEdges.end(); EI != EE; ++EI) {
      setEdgeWeight(BB, *EI, inWeight);
    }
  }

  if (uint32_t numExitingEdges = ExitingEdges.size()) {
    uint32_t exitWeight = LBH_NONTAKEN_WEIGHT / numExitingEdges;
    if (exitWeight < MIN_WEIGHT)
      exitWeight = MIN_WEIGHT;

    for (SmallVector<unsigned, 8>::iterator EI = ExitingEdges.begin(),
         EE = ExitingEdges.end(); EI != EE; ++EI) {
      setEdgeWeight(BB, *EI, exitWeight);
    }
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
  } else if (CV->isAllOnesValue() && CI->getPredicate() == CmpInst::ICMP_SGT) {
    // InstCombine canonicalizes X >= 0 into X > -1.
    // X >= 0   ->  Likely
    isProb = true;
  } else {
    return false;
  }

  unsigned TakenIdx = 0, NonTakenIdx = 1;

  if (!isProb)
    std::swap(TakenIdx, NonTakenIdx);

  setEdgeWeight(BB, TakenIdx, ZH_TAKEN_WEIGHT);
  setEdgeWeight(BB, NonTakenIdx, ZH_NONTAKEN_WEIGHT);

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

  setEdgeWeight(BB, TakenIdx, FPH_TAKEN_WEIGHT);
  setEdgeWeight(BB, NonTakenIdx, FPH_NONTAKEN_WEIGHT);

  return true;
}

bool BranchProbabilityInfo::calcInvokeHeuristics(BasicBlock *BB) {
  InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator());
  if (!II)
    return false;

  setEdgeWeight(BB, 0/*Index for Normal*/, IH_TAKEN_WEIGHT);
  setEdgeWeight(BB, 1/*Index for Unwind*/, IH_NONTAKEN_WEIGHT);
  return true;
}

void BranchProbabilityInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();
  AU.setPreservesAll();
}

bool BranchProbabilityInfo::runOnFunction(Function &F) {
  LastF = &F; // Store the last function we ran on for printing.
  LI = &getAnalysis<LoopInfo>();
  assert(PostDominatedByUnreachable.empty());

  // Walk the basic blocks in post-order so that we can build up state about
  // the successors of a block iteratively.
  for (po_iterator<BasicBlock *> I = po_begin(&F.getEntryBlock()),
                                 E = po_end(&F.getEntryBlock());
       I != E; ++I) {
    DEBUG(dbgs() << "Computing probabilities for " << I->getName() << "\n");
    if (calcUnreachableHeuristics(*I))
      continue;
    if (calcMetadataWeights(*I))
      continue;
    if (calcLoopBranchHeuristics(*I))
      continue;
    if (calcPointerHeuristics(*I))
      continue;
    if (calcZeroHeuristics(*I))
      continue;
    if (calcFloatingPointHeuristics(*I))
      continue;
    calcInvokeHeuristics(*I);
  }

  PostDominatedByUnreachable.clear();
  return false;
}

void BranchProbabilityInfo::print(raw_ostream &OS, const Module *) const {
  OS << "---- Branch Probabilities ----\n";
  // We print the probabilities from the last function the analysis ran over,
  // or the function it is currently running over.
  assert(LastF && "Cannot print prior to running over a function");
  for (Function::const_iterator BI = LastF->begin(), BE = LastF->end();
       BI != BE; ++BI) {
    for (succ_const_iterator SI = succ_begin(BI), SE = succ_end(BI);
         SI != SE; ++SI) {
      printEdgeProbability(OS << "  ", BI, *SI);
    }
  }
}

uint32_t BranchProbabilityInfo::getSumForBlock(const BasicBlock *BB) const {
  uint32_t Sum = 0;

  for (succ_const_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    uint32_t Weight = getEdgeWeight(BB, I.getSuccessorIndex());
    uint32_t PrevSum = Sum;

    Sum += Weight;
    assert(Sum > PrevSum); (void) PrevSum;
  }

  return Sum;
}

bool BranchProbabilityInfo::
isEdgeHot(const BasicBlock *Src, const BasicBlock *Dst) const {
  // Hot probability is at least 4/5 = 80%
  // FIXME: Compare against a static "hot" BranchProbability.
  return getEdgeProbability(Src, Dst) > BranchProbability(4, 5);
}

BasicBlock *BranchProbabilityInfo::getHotSucc(BasicBlock *BB) const {
  uint32_t Sum = 0;
  uint32_t MaxWeight = 0;
  BasicBlock *MaxSucc = 0;

  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    BasicBlock *Succ = *I;
    uint32_t Weight = getEdgeWeight(BB, Succ);
    uint32_t PrevSum = Sum;

    Sum += Weight;
    assert(Sum > PrevSum); (void) PrevSum;

    if (Weight > MaxWeight) {
      MaxWeight = Weight;
      MaxSucc = Succ;
    }
  }

  // Hot probability is at least 4/5 = 80%
  if (BranchProbability(MaxWeight, Sum) > BranchProbability(4, 5))
    return MaxSucc;

  return 0;
}

/// Get the raw edge weight for the edge. If can't find it, return
/// DEFAULT_WEIGHT value. Here an edge is specified using PredBlock and an index
/// to the successors.
uint32_t BranchProbabilityInfo::
getEdgeWeight(const BasicBlock *Src, unsigned IndexInSuccessors) const {
  DenseMap<Edge, uint32_t>::const_iterator I =
      Weights.find(std::make_pair(Src, IndexInSuccessors));

  if (I != Weights.end())
    return I->second;

  return DEFAULT_WEIGHT;
}

/// Get the raw edge weight calculated for the block pair. This returns the sum
/// of all raw edge weights from Src to Dst.
uint32_t BranchProbabilityInfo::
getEdgeWeight(const BasicBlock *Src, const BasicBlock *Dst) const {
  uint32_t Weight = 0;
  DenseMap<Edge, uint32_t>::const_iterator MapI;
  for (succ_const_iterator I = succ_begin(Src), E = succ_end(Src); I != E; ++I)
    if (*I == Dst) {
      MapI = Weights.find(std::make_pair(Src, I.getSuccessorIndex()));
      if (MapI != Weights.end())
        Weight += MapI->second;
    }
  return (Weight == 0) ? DEFAULT_WEIGHT : Weight;
}

/// Set the edge weight for a given edge specified by PredBlock and an index
/// to the successors.
void BranchProbabilityInfo::
setEdgeWeight(const BasicBlock *Src, unsigned IndexInSuccessors,
              uint32_t Weight) {
  Weights[std::make_pair(Src, IndexInSuccessors)] = Weight;
  DEBUG(dbgs() << "set edge " << Src->getName() << " -> "
               << IndexInSuccessors << " successor weight to "
               << Weight << "\n");
}

/// Get an edge's probability, relative to other out-edges from Src.
BranchProbability BranchProbabilityInfo::
getEdgeProbability(const BasicBlock *Src, unsigned IndexInSuccessors) const {
  uint32_t N = getEdgeWeight(Src, IndexInSuccessors);
  uint32_t D = getSumForBlock(Src);

  return BranchProbability(N, D);
}

/// Get the probability of going from Src to Dst. It returns the sum of all
/// probabilities for edges from Src to Dst.
BranchProbability BranchProbabilityInfo::
getEdgeProbability(const BasicBlock *Src, const BasicBlock *Dst) const {

  uint32_t N = getEdgeWeight(Src, Dst);
  uint32_t D = getSumForBlock(Src);

  return BranchProbability(N, D);
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
