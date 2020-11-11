//===- BranchProbabilityInfo.cpp - Branch Probability Analysis ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "branch-prob"

static cl::opt<bool> PrintBranchProb(
    "print-bpi", cl::init(false), cl::Hidden,
    cl::desc("Print the branch probability info."));

cl::opt<std::string> PrintBranchProbFuncName(
    "print-bpi-func-name", cl::Hidden,
    cl::desc("The option to specify the name of the function "
             "whose branch probability info is printed."));

INITIALIZE_PASS_BEGIN(BranchProbabilityInfoWrapperPass, "branch-prob",
                      "Branch Probability Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(BranchProbabilityInfoWrapperPass, "branch-prob",
                    "Branch Probability Analysis", false, true)

BranchProbabilityInfoWrapperPass::BranchProbabilityInfoWrapperPass()
    : FunctionPass(ID) {
  initializeBranchProbabilityInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

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
// Unlikely edges within a loop are half as likely as other edges
static const uint32_t LBH_UNLIKELY_WEIGHT = 62;

/// Unreachable-terminating branch taken probability.
///
/// This is the probability for a branch being taken to a block that terminates
/// (eventually) in unreachable. These are predicted as unlikely as possible.
/// All reachable probability will proportionally share the remaining part.
static const BranchProbability UR_TAKEN_PROB = BranchProbability::getRaw(1);

/// Weight for a branch taken going into a cold block.
///
/// This is the weight for a branch taken toward a block marked
/// cold.  A block is marked cold if it's postdominated by a
/// block containing a call to a cold function.  Cold functions
/// are those marked with attribute 'cold'.
static const uint32_t CC_TAKEN_WEIGHT = 4;

/// Weight for a branch not-taken into a cold block.
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

/// This is the probability for an ordered floating point comparison.
static const uint32_t FPH_ORD_WEIGHT = 1024 * 1024 - 1;
/// This is the probability for an unordered floating point comparison, it means
/// one or two of the operands are NaN. Usually it is used to test for an
/// exceptional case, so the result is unlikely.
static const uint32_t FPH_UNO_WEIGHT = 1;

/// Invoke-terminating normal branch taken weight
///
/// This is the weight for branching to the normal destination of an invoke
/// instruction. We expect this to happen most of the time. Set the weight to an
/// absurdly high value so that nested loops subsume it.
static const uint32_t IH_TAKEN_WEIGHT = 1024 * 1024 - 1;

/// Invoke-terminating normal branch not-taken weight.
///
/// This is the weight for branching to the unwind destination of an invoke
/// instruction. This is essentially never taken.
static const uint32_t IH_NONTAKEN_WEIGHT = 1;

BranchProbabilityInfo::SccInfo::SccInfo(const Function &F) {
  // Record SCC numbers of blocks in the CFG to identify irreducible loops.
  // FIXME: We could only calculate this if the CFG is known to be irreducible
  // (perhaps cache this info in LoopInfo if we can easily calculate it there?).
  int SccNum = 0;
  for (scc_iterator<const Function *> It = scc_begin(&F); !It.isAtEnd();
       ++It, ++SccNum) {
    // Ignore single-block SCCs since they either aren't loops or LoopInfo will
    // catch them.
    const std::vector<const BasicBlock *> &Scc = *It;
    if (Scc.size() == 1)
      continue;

    LLVM_DEBUG(dbgs() << "BPI: SCC " << SccNum << ":");
    for (const auto *BB : Scc) {
      LLVM_DEBUG(dbgs() << " " << BB->getName());
      SccNums[BB] = SccNum;
      calculateSccBlockType(BB, SccNum);
    }
    LLVM_DEBUG(dbgs() << "\n");
  }
}

int BranchProbabilityInfo::SccInfo::getSCCNum(const BasicBlock *BB) const {
  auto SccIt = SccNums.find(BB);
  if (SccIt == SccNums.end())
    return -1;
  return SccIt->second;
}

void BranchProbabilityInfo::SccInfo::getSccEnterBlocks(
    int SccNum, SmallVectorImpl<BasicBlock *> &Enters) const {

  for (auto MapIt : SccBlocks[SccNum]) {
    const auto *BB = MapIt.first;
    if (isSCCHeader(BB, SccNum))
      for (const auto *Pred : predecessors(BB))
        if (getSCCNum(Pred) != SccNum)
          Enters.push_back(const_cast<BasicBlock *>(BB));
  }
}

void BranchProbabilityInfo::SccInfo::getSccExitBlocks(
    int SccNum, SmallVectorImpl<BasicBlock *> &Exits) const {
  for (auto MapIt : SccBlocks[SccNum]) {
    const auto *BB = MapIt.first;
    if (isSCCExitingBlock(BB, SccNum))
      for (const auto *Succ : successors(BB))
        if (getSCCNum(Succ) != SccNum)
          Exits.push_back(const_cast<BasicBlock *>(BB));
  }
}

uint32_t BranchProbabilityInfo::SccInfo::getSccBlockType(const BasicBlock *BB,
                                                         int SccNum) const {
  assert(getSCCNum(BB) == SccNum);

  assert(SccBlocks.size() > static_cast<unsigned>(SccNum) && "Unknown SCC");
  const auto &SccBlockTypes = SccBlocks[SccNum];

  auto It = SccBlockTypes.find(BB);
  if (It != SccBlockTypes.end()) {
    return It->second;
  }
  return Inner;
}

void BranchProbabilityInfo::SccInfo::calculateSccBlockType(const BasicBlock *BB,
                                                           int SccNum) {
  assert(getSCCNum(BB) == SccNum);
  uint32_t BlockType = Inner;

  if (llvm::any_of(make_range(pred_begin(BB), pred_end(BB)),
                   [&](const BasicBlock *Pred) {
        // Consider any block that is an entry point to the SCC as
        // a header.
        return getSCCNum(Pred) != SccNum;
      }))
    BlockType |= Header;

  if (llvm::any_of(
          make_range(succ_begin(BB), succ_end(BB)),
          [&](const BasicBlock *Succ) { return getSCCNum(Succ) != SccNum; }))
    BlockType |= Exiting;

  // Lazily compute the set of headers for a given SCC and cache the results
  // in the SccHeaderMap.
  if (SccBlocks.size() <= static_cast<unsigned>(SccNum))
    SccBlocks.resize(SccNum + 1);
  auto &SccBlockTypes = SccBlocks[SccNum];

  if (BlockType != Inner) {
    bool IsInserted;
    std::tie(std::ignore, IsInserted) =
        SccBlockTypes.insert(std::make_pair(BB, BlockType));
    assert(IsInserted && "Duplicated block in SCC");
  }
}

BranchProbabilityInfo::LoopBlock::LoopBlock(const BasicBlock *BB,
                                            const LoopInfo &LI,
                                            const SccInfo &SccI)
    : BB(BB) {
  LD.first = LI.getLoopFor(BB);
  if (!LD.first) {
    LD.second = SccI.getSCCNum(BB);
  }
}

bool BranchProbabilityInfo::isLoopEnteringEdge(const LoopEdge &Edge) const {
  const auto &SrcBlock = Edge.first;
  const auto &DstBlock = Edge.second;
  return (DstBlock.getLoop() &&
          !DstBlock.getLoop()->contains(SrcBlock.getLoop())) ||
         // Assume that SCCs can't be nested.
         (DstBlock.getSccNum() != -1 &&
          SrcBlock.getSccNum() != DstBlock.getSccNum());
}

bool BranchProbabilityInfo::isLoopExitingEdge(const LoopEdge &Edge) const {
  return isLoopEnteringEdge({Edge.second, Edge.first});
}

bool BranchProbabilityInfo::isLoopEnteringExitingEdge(
    const LoopEdge &Edge) const {
  return isLoopEnteringEdge(Edge) || isLoopExitingEdge(Edge);
}

bool BranchProbabilityInfo::isLoopBackEdge(const LoopEdge &Edge) const {
  const auto &SrcBlock = Edge.first;
  const auto &DstBlock = Edge.second;
  return SrcBlock.belongsToSameLoop(DstBlock) &&
         ((DstBlock.getLoop() &&
           DstBlock.getLoop()->getHeader() == DstBlock.getBlock()) ||
          (DstBlock.getSccNum() != -1 &&
           SccI->isSCCHeader(DstBlock.getBlock(), DstBlock.getSccNum())));
}

void BranchProbabilityInfo::getLoopEnterBlocks(
    const LoopBlock &LB, SmallVectorImpl<BasicBlock *> &Enters) const {
  if (LB.getLoop()) {
    auto *Header = LB.getLoop()->getHeader();
    Enters.append(pred_begin(Header), pred_end(Header));
  } else {
    assert(LB.getSccNum() != -1 && "LB doesn't belong to any loop?");
    SccI->getSccEnterBlocks(LB.getSccNum(), Enters);
  }
}

void BranchProbabilityInfo::getLoopExitBlocks(
    const LoopBlock &LB, SmallVectorImpl<BasicBlock *> &Exits) const {
  if (LB.getLoop()) {
    LB.getLoop()->getExitBlocks(Exits);
  } else {
    assert(LB.getSccNum() != -1 && "LB doesn't belong to any loop?");
    SccI->getSccExitBlocks(LB.getSccNum(), Exits);
  }
}

static void UpdatePDTWorklist(const BasicBlock *BB, PostDominatorTree *PDT,
                              SmallVectorImpl<const BasicBlock *> &WorkList,
                              SmallPtrSetImpl<const BasicBlock *> &TargetSet) {
  SmallVector<BasicBlock *, 8> Descendants;
  SmallPtrSet<const BasicBlock *, 16> NewItems;

  PDT->getDescendants(const_cast<BasicBlock *>(BB), Descendants);
  for (auto *BB : Descendants)
    if (TargetSet.insert(BB).second)
      for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
        if (!TargetSet.count(*PI))
          NewItems.insert(*PI);
  WorkList.insert(WorkList.end(), NewItems.begin(), NewItems.end());
}

/// Compute a set of basic blocks that are post-dominated by unreachables.
void BranchProbabilityInfo::computePostDominatedByUnreachable(
    const Function &F, PostDominatorTree *PDT) {
  SmallVector<const BasicBlock *, 8> WorkList;
  for (auto &BB : F) {
    const Instruction *TI = BB.getTerminator();
    if (TI->getNumSuccessors() == 0) {
      if (isa<UnreachableInst>(TI) ||
          // If this block is terminated by a call to
          // @llvm.experimental.deoptimize then treat it like an unreachable
          // since the @llvm.experimental.deoptimize call is expected to
          // practically never execute.
          BB.getTerminatingDeoptimizeCall())
        UpdatePDTWorklist(&BB, PDT, WorkList, PostDominatedByUnreachable);
    }
  }

  while (!WorkList.empty()) {
    const BasicBlock *BB = WorkList.pop_back_val();
    if (PostDominatedByUnreachable.count(BB))
      continue;
    // If the terminator is an InvokeInst, check only the normal destination
    // block as the unwind edge of InvokeInst is also very unlikely taken.
    if (auto *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      if (PostDominatedByUnreachable.count(II->getNormalDest()))
        UpdatePDTWorklist(BB, PDT, WorkList, PostDominatedByUnreachable);
    }
    // If all the successors are unreachable, BB is unreachable as well.
    else if (!successors(BB).empty() &&
             llvm::all_of(successors(BB), [this](const BasicBlock *Succ) {
               return PostDominatedByUnreachable.count(Succ);
             }))
      UpdatePDTWorklist(BB, PDT, WorkList, PostDominatedByUnreachable);
  }
}

/// compute a set of basic blocks that are post-dominated by ColdCalls.
void BranchProbabilityInfo::computePostDominatedByColdCall(
    const Function &F, PostDominatorTree *PDT) {
  SmallVector<const BasicBlock *, 8> WorkList;
  for (auto &BB : F)
    for (auto &I : BB)
      if (const CallInst *CI = dyn_cast<CallInst>(&I))
        if (CI->hasFnAttr(Attribute::Cold))
          UpdatePDTWorklist(&BB, PDT, WorkList, PostDominatedByColdCall);

  while (!WorkList.empty()) {
    const BasicBlock *BB = WorkList.pop_back_val();

    // If the terminator is an InvokeInst, check only the normal destination
    // block as the unwind edge of InvokeInst is also very unlikely taken.
    if (auto *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      if (PostDominatedByColdCall.count(II->getNormalDest()))
        UpdatePDTWorklist(BB, PDT, WorkList, PostDominatedByColdCall);
    }
    // If all of successor are post dominated then BB is also done.
    else if (!successors(BB).empty() &&
             llvm::all_of(successors(BB), [this](const BasicBlock *Succ) {
               return PostDominatedByColdCall.count(Succ);
             }))
      UpdatePDTWorklist(BB, PDT, WorkList, PostDominatedByColdCall);
  }
}

/// Calculate edge weights for successors lead to unreachable.
///
/// Predict that a successor which leads necessarily to an
/// unreachable-terminated block as extremely unlikely.
bool BranchProbabilityInfo::calcUnreachableHeuristics(const BasicBlock *BB) {
  const Instruction *TI = BB->getTerminator();
  (void) TI;
  assert(TI->getNumSuccessors() > 1 && "expected more than one successor!");
  assert(!isa<InvokeInst>(TI) &&
         "Invokes should have already been handled by calcInvokeHeuristics");

  SmallVector<unsigned, 4> UnreachableEdges;
  SmallVector<unsigned, 4> ReachableEdges;

  for (const_succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I)
    if (PostDominatedByUnreachable.count(*I))
      UnreachableEdges.push_back(I.getSuccessorIndex());
    else
      ReachableEdges.push_back(I.getSuccessorIndex());

  // Skip probabilities if all were reachable.
  if (UnreachableEdges.empty())
    return false;

  SmallVector<BranchProbability, 4> EdgeProbabilities(
      BB->getTerminator()->getNumSuccessors(), BranchProbability::getUnknown());
  if (ReachableEdges.empty()) {
    BranchProbability Prob(1, UnreachableEdges.size());
    for (unsigned SuccIdx : UnreachableEdges)
      EdgeProbabilities[SuccIdx] = Prob;
    setEdgeProbability(BB, EdgeProbabilities);
    return true;
  }

  auto UnreachableProb = UR_TAKEN_PROB;
  auto ReachableProb =
      (BranchProbability::getOne() - UR_TAKEN_PROB * UnreachableEdges.size()) /
      ReachableEdges.size();

  for (unsigned SuccIdx : UnreachableEdges)
    EdgeProbabilities[SuccIdx] = UnreachableProb;
  for (unsigned SuccIdx : ReachableEdges)
    EdgeProbabilities[SuccIdx] = ReachableProb;

  setEdgeProbability(BB, EdgeProbabilities);
  return true;
}

// Propagate existing explicit probabilities from either profile data or
// 'expect' intrinsic processing. Examine metadata against unreachable
// heuristic. The probability of the edge coming to unreachable block is
// set to min of metadata and unreachable heuristic.
bool BranchProbabilityInfo::calcMetadataWeights(const BasicBlock *BB) {
  const Instruction *TI = BB->getTerminator();
  assert(TI->getNumSuccessors() > 1 && "expected more than one successor!");
  if (!(isa<BranchInst>(TI) || isa<SwitchInst>(TI) || isa<IndirectBrInst>(TI) ||
        isa<InvokeInst>(TI)))
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
  SmallVector<unsigned, 2> UnreachableIdxs;
  SmallVector<unsigned, 2> ReachableIdxs;
  Weights.reserve(TI->getNumSuccessors());
  for (unsigned I = 1, E = WeightsNode->getNumOperands(); I != E; ++I) {
    ConstantInt *Weight =
        mdconst::dyn_extract<ConstantInt>(WeightsNode->getOperand(I));
    if (!Weight)
      return false;
    assert(Weight->getValue().getActiveBits() <= 32 &&
           "Too many bits for uint32_t");
    Weights.push_back(Weight->getZExtValue());
    WeightSum += Weights.back();
    if (PostDominatedByUnreachable.count(TI->getSuccessor(I - 1)))
      UnreachableIdxs.push_back(I - 1);
    else
      ReachableIdxs.push_back(I - 1);
  }
  assert(Weights.size() == TI->getNumSuccessors() && "Checked above");

  // If the sum of weights does not fit in 32 bits, scale every weight down
  // accordingly.
  uint64_t ScalingFactor =
      (WeightSum > UINT32_MAX) ? WeightSum / UINT32_MAX + 1 : 1;

  if (ScalingFactor > 1) {
    WeightSum = 0;
    for (unsigned I = 0, E = TI->getNumSuccessors(); I != E; ++I) {
      Weights[I] /= ScalingFactor;
      WeightSum += Weights[I];
    }
  }
  assert(WeightSum <= UINT32_MAX &&
         "Expected weights to scale down to 32 bits");

  if (WeightSum == 0 || ReachableIdxs.size() == 0) {
    for (unsigned I = 0, E = TI->getNumSuccessors(); I != E; ++I)
      Weights[I] = 1;
    WeightSum = TI->getNumSuccessors();
  }

  // Set the probability.
  SmallVector<BranchProbability, 2> BP;
  for (unsigned I = 0, E = TI->getNumSuccessors(); I != E; ++I)
    BP.push_back({ Weights[I], static_cast<uint32_t>(WeightSum) });

  // Examine the metadata against unreachable heuristic.
  // If the unreachable heuristic is more strong then we use it for this edge.
  if (UnreachableIdxs.size() == 0 || ReachableIdxs.size() == 0) {
    setEdgeProbability(BB, BP);
    return true;
  }

  auto UnreachableProb = UR_TAKEN_PROB;
  for (auto I : UnreachableIdxs)
    if (UnreachableProb < BP[I]) {
      BP[I] = UnreachableProb;
    }

  // Sum of all edge probabilities must be 1.0. If we modified the probability
  // of some edges then we must distribute the introduced difference over the
  // reachable blocks.
  //
  // Proportional distribution: the relation between probabilities of the
  // reachable edges is kept unchanged. That is for any reachable edges i and j:
  //   newBP[i] / newBP[j] == oldBP[i] / oldBP[j] =>
  //   newBP[i] / oldBP[i] == newBP[j] / oldBP[j] == K
  // Where K is independent of i,j.
  //   newBP[i] == oldBP[i] * K
  // We need to find K.
  // Make sum of all reachables of the left and right parts:
  //   sum_of_reachable(newBP) == K * sum_of_reachable(oldBP)
  // Sum of newBP must be equal to 1.0:
  //   sum_of_reachable(newBP) + sum_of_unreachable(newBP) == 1.0 =>
  //   sum_of_reachable(newBP) = 1.0 - sum_of_unreachable(newBP)
  // Where sum_of_unreachable(newBP) is what has been just changed.
  // Finally:
  //   K == sum_of_reachable(newBP) / sum_of_reachable(oldBP) =>
  //   K == (1.0 - sum_of_unreachable(newBP)) / sum_of_reachable(oldBP)
  BranchProbability NewUnreachableSum = BranchProbability::getZero();
  for (auto I : UnreachableIdxs)
    NewUnreachableSum += BP[I];

  BranchProbability NewReachableSum =
      BranchProbability::getOne() - NewUnreachableSum;

  BranchProbability OldReachableSum = BranchProbability::getZero();
  for (auto I : ReachableIdxs)
    OldReachableSum += BP[I];

  if (OldReachableSum != NewReachableSum) { // Anything to dsitribute?
    if (OldReachableSum.isZero()) {
      // If all oldBP[i] are zeroes then the proportional distribution results
      // in all zero probabilities and the error stays big. In this case we
      // evenly spread NewReachableSum over the reachable edges.
      BranchProbability PerEdge = NewReachableSum / ReachableIdxs.size();
      for (auto I : ReachableIdxs)
        BP[I] = PerEdge;
    } else {
      for (auto I : ReachableIdxs) {
        // We use uint64_t to avoid double rounding error of the following
        // calculation: BP[i] = BP[i] * NewReachableSum / OldReachableSum
        // The formula is taken from the private constructor
        // BranchProbability(uint32_t Numerator, uint32_t Denominator)
        uint64_t Mul = static_cast<uint64_t>(NewReachableSum.getNumerator()) *
                       BP[I].getNumerator();
        uint32_t Div = static_cast<uint32_t>(
            divideNearest(Mul, OldReachableSum.getNumerator()));
        BP[I] = BranchProbability::getRaw(Div);
      }
    }
  }

  setEdgeProbability(BB, BP);

  return true;
}

/// Calculate edge weights for edges leading to cold blocks.
///
/// A cold block is one post-dominated by  a block with a call to a
/// cold function.  Those edges are unlikely to be taken, so we give
/// them relatively low weight.
///
/// Return true if we could compute the weights for cold edges.
/// Return false, otherwise.
bool BranchProbabilityInfo::calcColdCallHeuristics(const BasicBlock *BB) {
  const Instruction *TI = BB->getTerminator();
  (void) TI;
  assert(TI->getNumSuccessors() > 1 && "expected more than one successor!");
  assert(!isa<InvokeInst>(TI) &&
         "Invokes should have already been handled by calcInvokeHeuristics");

  // Determine which successors are post-dominated by a cold block.
  SmallVector<unsigned, 4> ColdEdges;
  SmallVector<unsigned, 4> NormalEdges;
  for (const_succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I)
    if (PostDominatedByColdCall.count(*I))
      ColdEdges.push_back(I.getSuccessorIndex());
    else
      NormalEdges.push_back(I.getSuccessorIndex());

  // Skip probabilities if no cold edges.
  if (ColdEdges.empty())
    return false;

  SmallVector<BranchProbability, 4> EdgeProbabilities(
      BB->getTerminator()->getNumSuccessors(), BranchProbability::getUnknown());
  if (NormalEdges.empty()) {
    BranchProbability Prob(1, ColdEdges.size());
    for (unsigned SuccIdx : ColdEdges)
      EdgeProbabilities[SuccIdx] = Prob;
    setEdgeProbability(BB, EdgeProbabilities);
    return true;
  }

  auto ColdProb = BranchProbability::getBranchProbability(
      CC_TAKEN_WEIGHT,
      (CC_TAKEN_WEIGHT + CC_NONTAKEN_WEIGHT) * uint64_t(ColdEdges.size()));
  auto NormalProb = BranchProbability::getBranchProbability(
      CC_NONTAKEN_WEIGHT,
      (CC_TAKEN_WEIGHT + CC_NONTAKEN_WEIGHT) * uint64_t(NormalEdges.size()));

  for (unsigned SuccIdx : ColdEdges)
    EdgeProbabilities[SuccIdx] = ColdProb;
  for (unsigned SuccIdx : NormalEdges)
    EdgeProbabilities[SuccIdx] = NormalProb;

  setEdgeProbability(BB, EdgeProbabilities);
  return true;
}

// Calculate Edge Weights using "Pointer Heuristics". Predict a comparison
// between two pointer or pointer and NULL will fail.
bool BranchProbabilityInfo::calcPointerHeuristics(const BasicBlock *BB) {
  const BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
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

  BranchProbability TakenProb(PH_TAKEN_WEIGHT,
                              PH_TAKEN_WEIGHT + PH_NONTAKEN_WEIGHT);
  BranchProbability UntakenProb(PH_NONTAKEN_WEIGHT,
                                PH_TAKEN_WEIGHT + PH_NONTAKEN_WEIGHT);

  // p != 0   ->   isProb = true
  // p == 0   ->   isProb = false
  // p != q   ->   isProb = true
  // p == q   ->   isProb = false;
  bool isProb = CI->getPredicate() == ICmpInst::ICMP_NE;
  if (!isProb)
    std::swap(TakenProb, UntakenProb);

  setEdgeProbability(
      BB, SmallVector<BranchProbability, 2>({TakenProb, UntakenProb}));
  return true;
}

// Compute the unlikely successors to the block BB in the loop L, specifically
// those that are unlikely because this is a loop, and add them to the
// UnlikelyBlocks set.
static void
computeUnlikelySuccessors(const BasicBlock *BB, Loop *L,
                          SmallPtrSetImpl<const BasicBlock*> &UnlikelyBlocks) {
  // Sometimes in a loop we have a branch whose condition is made false by
  // taking it. This is typically something like
  //  int n = 0;
  //  while (...) {
  //    if (++n >= MAX) {
  //      n = 0;
  //    }
  //  }
  // In this sort of situation taking the branch means that at the very least it
  // won't be taken again in the next iteration of the loop, so we should
  // consider it less likely than a typical branch.
  //
  // We detect this by looking back through the graph of PHI nodes that sets the
  // value that the condition depends on, and seeing if we can reach a successor
  // block which can be determined to make the condition false.
  //
  // FIXME: We currently consider unlikely blocks to be half as likely as other
  // blocks, but if we consider the example above the likelyhood is actually
  // 1/MAX. We could therefore be more precise in how unlikely we consider
  // blocks to be, but it would require more careful examination of the form
  // of the comparison expression.
  const BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isConditional())
    return;

  // Check if the branch is based on an instruction compared with a constant
  CmpInst *CI = dyn_cast<CmpInst>(BI->getCondition());
  if (!CI || !isa<Instruction>(CI->getOperand(0)) ||
      !isa<Constant>(CI->getOperand(1)))
    return;

  // Either the instruction must be a PHI, or a chain of operations involving
  // constants that ends in a PHI which we can then collapse into a single value
  // if the PHI value is known.
  Instruction *CmpLHS = dyn_cast<Instruction>(CI->getOperand(0));
  PHINode *CmpPHI = dyn_cast<PHINode>(CmpLHS);
  Constant *CmpConst = dyn_cast<Constant>(CI->getOperand(1));
  // Collect the instructions until we hit a PHI
  SmallVector<BinaryOperator *, 1> InstChain;
  while (!CmpPHI && CmpLHS && isa<BinaryOperator>(CmpLHS) &&
         isa<Constant>(CmpLHS->getOperand(1))) {
    // Stop if the chain extends outside of the loop
    if (!L->contains(CmpLHS))
      return;
    InstChain.push_back(cast<BinaryOperator>(CmpLHS));
    CmpLHS = dyn_cast<Instruction>(CmpLHS->getOperand(0));
    if (CmpLHS)
      CmpPHI = dyn_cast<PHINode>(CmpLHS);
  }
  if (!CmpPHI || !L->contains(CmpPHI))
    return;

  // Trace the phi node to find all values that come from successors of BB
  SmallPtrSet<PHINode*, 8> VisitedInsts;
  SmallVector<PHINode*, 8> WorkList;
  WorkList.push_back(CmpPHI);
  VisitedInsts.insert(CmpPHI);
  while (!WorkList.empty()) {
    PHINode *P = WorkList.back();
    WorkList.pop_back();
    for (BasicBlock *B : P->blocks()) {
      // Skip blocks that aren't part of the loop
      if (!L->contains(B))
        continue;
      Value *V = P->getIncomingValueForBlock(B);
      // If the source is a PHI add it to the work list if we haven't
      // already visited it.
      if (PHINode *PN = dyn_cast<PHINode>(V)) {
        if (VisitedInsts.insert(PN).second)
          WorkList.push_back(PN);
        continue;
      }
      // If this incoming value is a constant and B is a successor of BB, then
      // we can constant-evaluate the compare to see if it makes the branch be
      // taken or not.
      Constant *CmpLHSConst = dyn_cast<Constant>(V);
      if (!CmpLHSConst || !llvm::is_contained(successors(BB), B))
        continue;
      // First collapse InstChain
      for (Instruction *I : llvm::reverse(InstChain)) {
        CmpLHSConst = ConstantExpr::get(I->getOpcode(), CmpLHSConst,
                                        cast<Constant>(I->getOperand(1)), true);
        if (!CmpLHSConst)
          break;
      }
      if (!CmpLHSConst)
        continue;
      // Now constant-evaluate the compare
      Constant *Result = ConstantExpr::getCompare(CI->getPredicate(),
                                                  CmpLHSConst, CmpConst, true);
      // If the result means we don't branch to the block then that block is
      // unlikely.
      if (Result &&
          ((Result->isZeroValue() && B == BI->getSuccessor(0)) ||
           (Result->isOneValue() && B == BI->getSuccessor(1))))
        UnlikelyBlocks.insert(B);
    }
  }
}

// Calculate Edge Weights using "Loop Branch Heuristics". Predict backedges
// as taken, exiting edges as not-taken.
bool BranchProbabilityInfo::calcLoopBranchHeuristics(const BasicBlock *BB,
                                                     const LoopInfo &LI) {
  LoopBlock LB(BB, LI, *SccI.get());
  if (!LB.belongsToLoop())
    return false;

  SmallPtrSet<const BasicBlock*, 8> UnlikelyBlocks;
  if (LB.getLoop())
    computeUnlikelySuccessors(BB, LB.getLoop(), UnlikelyBlocks);

  SmallVector<unsigned, 8> BackEdges;
  SmallVector<unsigned, 8> ExitingEdges;
  SmallVector<unsigned, 8> InEdges; // Edges from header to the loop.
  SmallVector<unsigned, 8> UnlikelyEdges;

  for (const_succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    LoopBlock SuccLB(*I, LI, *SccI.get());
    LoopEdge Edge(LB, SuccLB);
    bool IsUnlikelyEdge =
        LB.getLoop() && (UnlikelyBlocks.find(*I) != UnlikelyBlocks.end());

    if (IsUnlikelyEdge)
      UnlikelyEdges.push_back(I.getSuccessorIndex());
    else if (isLoopExitingEdge(Edge))
      ExitingEdges.push_back(I.getSuccessorIndex());
    else if (isLoopBackEdge(Edge))
      BackEdges.push_back(I.getSuccessorIndex());
    else {
      InEdges.push_back(I.getSuccessorIndex());
    }
  }

  if (BackEdges.empty() && ExitingEdges.empty() && UnlikelyEdges.empty())
    return false;

  // Collect the sum of probabilities of back-edges/in-edges/exiting-edges, and
  // normalize them so that they sum up to one.
  unsigned Denom = (BackEdges.empty() ? 0 : LBH_TAKEN_WEIGHT) +
                   (InEdges.empty() ? 0 : LBH_TAKEN_WEIGHT) +
                   (UnlikelyEdges.empty() ? 0 : LBH_UNLIKELY_WEIGHT) +
                   (ExitingEdges.empty() ? 0 : LBH_NONTAKEN_WEIGHT);

  SmallVector<BranchProbability, 4> EdgeProbabilities(
      BB->getTerminator()->getNumSuccessors(), BranchProbability::getUnknown());
  if (uint32_t numBackEdges = BackEdges.size()) {
    BranchProbability TakenProb = BranchProbability(LBH_TAKEN_WEIGHT, Denom);
    auto Prob = TakenProb / numBackEdges;
    for (unsigned SuccIdx : BackEdges)
      EdgeProbabilities[SuccIdx] = Prob;
  }

  if (uint32_t numInEdges = InEdges.size()) {
    BranchProbability TakenProb = BranchProbability(LBH_TAKEN_WEIGHT, Denom);
    auto Prob = TakenProb / numInEdges;
    for (unsigned SuccIdx : InEdges)
      EdgeProbabilities[SuccIdx] = Prob;
  }

  if (uint32_t numExitingEdges = ExitingEdges.size()) {
    BranchProbability NotTakenProb = BranchProbability(LBH_NONTAKEN_WEIGHT,
                                                       Denom);
    auto Prob = NotTakenProb / numExitingEdges;
    for (unsigned SuccIdx : ExitingEdges)
      EdgeProbabilities[SuccIdx] = Prob;
  }

  if (uint32_t numUnlikelyEdges = UnlikelyEdges.size()) {
    BranchProbability UnlikelyProb = BranchProbability(LBH_UNLIKELY_WEIGHT,
                                                       Denom);
    auto Prob = UnlikelyProb / numUnlikelyEdges;
    for (unsigned SuccIdx : UnlikelyEdges)
      EdgeProbabilities[SuccIdx] = Prob;
  }

  setEdgeProbability(BB, EdgeProbabilities);
  return true;
}

bool BranchProbabilityInfo::calcZeroHeuristics(const BasicBlock *BB,
                                               const TargetLibraryInfo *TLI) {
  const BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isConditional())
    return false;

  Value *Cond = BI->getCondition();
  ICmpInst *CI = dyn_cast<ICmpInst>(Cond);
  if (!CI)
    return false;

  auto GetConstantInt = [](Value *V) {
    if (auto *I = dyn_cast<BitCastInst>(V))
      return dyn_cast<ConstantInt>(I->getOperand(0));
    return dyn_cast<ConstantInt>(V);
  };

  Value *RHS = CI->getOperand(1);
  ConstantInt *CV = GetConstantInt(RHS);
  if (!CV)
    return false;

  // If the LHS is the result of AND'ing a value with a single bit bitmask,
  // we don't have information about probabilities.
  if (Instruction *LHS = dyn_cast<Instruction>(CI->getOperand(0)))
    if (LHS->getOpcode() == Instruction::And)
      if (ConstantInt *AndRHS = dyn_cast<ConstantInt>(LHS->getOperand(1)))
        if (AndRHS->getValue().isPowerOf2())
          return false;

  // Check if the LHS is the return value of a library function
  LibFunc Func = NumLibFuncs;
  if (TLI)
    if (CallInst *Call = dyn_cast<CallInst>(CI->getOperand(0)))
      if (Function *CalledFn = Call->getCalledFunction())
        TLI->getLibFunc(*CalledFn, Func);

  bool isProb;
  if (Func == LibFunc_strcasecmp ||
      Func == LibFunc_strcmp ||
      Func == LibFunc_strncasecmp ||
      Func == LibFunc_strncmp ||
      Func == LibFunc_memcmp ||
      Func == LibFunc_bcmp) {
    // strcmp and similar functions return zero, negative, or positive, if the
    // first string is equal, less, or greater than the second. We consider it
    // likely that the strings are not equal, so a comparison with zero is
    // probably false, but also a comparison with any other number is also
    // probably false given that what exactly is returned for nonzero values is
    // not specified. Any kind of comparison other than equality we know
    // nothing about.
    switch (CI->getPredicate()) {
    case CmpInst::ICMP_EQ:
      isProb = false;
      break;
    case CmpInst::ICMP_NE:
      isProb = true;
      break;
    default:
      return false;
    }
  } else if (CV->isZero()) {
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
  } else if (CV->isMinusOne()) {
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

  BranchProbability TakenProb(ZH_TAKEN_WEIGHT,
                              ZH_TAKEN_WEIGHT + ZH_NONTAKEN_WEIGHT);
  BranchProbability UntakenProb(ZH_NONTAKEN_WEIGHT,
                                ZH_TAKEN_WEIGHT + ZH_NONTAKEN_WEIGHT);
  if (!isProb)
    std::swap(TakenProb, UntakenProb);

  setEdgeProbability(
      BB, SmallVector<BranchProbability, 2>({TakenProb, UntakenProb}));
  return true;
}

bool BranchProbabilityInfo::calcFloatingPointHeuristics(const BasicBlock *BB) {
  const BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isConditional())
    return false;

  Value *Cond = BI->getCondition();
  FCmpInst *FCmp = dyn_cast<FCmpInst>(Cond);
  if (!FCmp)
    return false;

  uint32_t TakenWeight = FPH_TAKEN_WEIGHT;
  uint32_t NontakenWeight = FPH_NONTAKEN_WEIGHT;
  bool isProb;
  if (FCmp->isEquality()) {
    // f1 == f2 -> Unlikely
    // f1 != f2 -> Likely
    isProb = !FCmp->isTrueWhenEqual();
  } else if (FCmp->getPredicate() == FCmpInst::FCMP_ORD) {
    // !isnan -> Likely
    isProb = true;
    TakenWeight = FPH_ORD_WEIGHT;
    NontakenWeight = FPH_UNO_WEIGHT;
  } else if (FCmp->getPredicate() == FCmpInst::FCMP_UNO) {
    // isnan -> Unlikely
    isProb = false;
    TakenWeight = FPH_ORD_WEIGHT;
    NontakenWeight = FPH_UNO_WEIGHT;
  } else {
    return false;
  }

  BranchProbability TakenProb(TakenWeight, TakenWeight + NontakenWeight);
  BranchProbability UntakenProb(NontakenWeight, TakenWeight + NontakenWeight);
  if (!isProb)
    std::swap(TakenProb, UntakenProb);

  setEdgeProbability(
      BB, SmallVector<BranchProbability, 2>({TakenProb, UntakenProb}));
  return true;
}

bool BranchProbabilityInfo::calcInvokeHeuristics(const BasicBlock *BB) {
  const InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator());
  if (!II)
    return false;

  BranchProbability TakenProb(IH_TAKEN_WEIGHT,
                              IH_TAKEN_WEIGHT + IH_NONTAKEN_WEIGHT);
  setEdgeProbability(
      BB, SmallVector<BranchProbability, 2>({TakenProb, TakenProb.getCompl()}));
  return true;
}

void BranchProbabilityInfo::releaseMemory() {
  Probs.clear();
  Handles.clear();
}

bool BranchProbabilityInfo::invalidate(Function &, const PreservedAnalyses &PA,
                                       FunctionAnalysisManager::Invalidator &) {
  // Check whether the analysis, all analyses on functions, or the function's
  // CFG have been preserved.
  auto PAC = PA.getChecker<BranchProbabilityAnalysis>();
  return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>() ||
           PAC.preservedSet<CFGAnalyses>());
}

void BranchProbabilityInfo::print(raw_ostream &OS) const {
  OS << "---- Branch Probabilities ----\n";
  // We print the probabilities from the last function the analysis ran over,
  // or the function it is currently running over.
  assert(LastF && "Cannot print prior to running over a function");
  for (const auto &BI : *LastF) {
    for (const_succ_iterator SI = succ_begin(&BI), SE = succ_end(&BI); SI != SE;
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

const BasicBlock *
BranchProbabilityInfo::getHotSucc(const BasicBlock *BB) const {
  auto MaxProb = BranchProbability::getZero();
  const BasicBlock *MaxSucc = nullptr;

  for (const auto *Succ : successors(BB)) {
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
  assert((Probs.end() == Probs.find(std::make_pair(Src, 0))) ==
             (Probs.end() == I) &&
         "Probability for I-th successor must always be defined along with the "
         "probability for the first successor");

  if (I != Probs.end())
    return I->second;

  return {1, static_cast<uint32_t>(succ_size(Src))};
}

BranchProbability
BranchProbabilityInfo::getEdgeProbability(const BasicBlock *Src,
                                          const_succ_iterator Dst) const {
  return getEdgeProbability(Src, Dst.getSuccessorIndex());
}

/// Get the raw edge probability calculated for the block pair. This returns the
/// sum of all raw edge probabilities from Src to Dst.
BranchProbability
BranchProbabilityInfo::getEdgeProbability(const BasicBlock *Src,
                                          const BasicBlock *Dst) const {
  if (!Probs.count(std::make_pair(Src, 0)))
    return BranchProbability(llvm::count(successors(Src), Dst), succ_size(Src));

  auto Prob = BranchProbability::getZero();
  for (const_succ_iterator I = succ_begin(Src), E = succ_end(Src); I != E; ++I)
    if (*I == Dst)
      Prob += Probs.find(std::make_pair(Src, I.getSuccessorIndex()))->second;

  return Prob;
}

/// Set the edge probability for all edges at once.
void BranchProbabilityInfo::setEdgeProbability(
    const BasicBlock *Src, const SmallVectorImpl<BranchProbability> &Probs) {
  assert(Src->getTerminator()->getNumSuccessors() == Probs.size());
  eraseBlock(Src); // Erase stale data if any.
  if (Probs.size() == 0)
    return; // Nothing to set.

  Handles.insert(BasicBlockCallbackVH(Src, this));
  uint64_t TotalNumerator = 0;
  for (unsigned SuccIdx = 0; SuccIdx < Probs.size(); ++SuccIdx) {
    this->Probs[std::make_pair(Src, SuccIdx)] = Probs[SuccIdx];
    LLVM_DEBUG(dbgs() << "set edge " << Src->getName() << " -> " << SuccIdx
                      << " successor probability to " << Probs[SuccIdx]
                      << "\n");
    TotalNumerator += Probs[SuccIdx].getNumerator();
  }

  // Because of rounding errors the total probability cannot be checked to be
  // 1.0 exactly. That is TotalNumerator == BranchProbability::getDenominator.
  // Instead, every single probability in Probs must be as accurate as possible.
  // This results in error 1/denominator at most, thus the total absolute error
  // should be within Probs.size / BranchProbability::getDenominator.
  assert(TotalNumerator <= BranchProbability::getDenominator() + Probs.size());
  assert(TotalNumerator >= BranchProbability::getDenominator() - Probs.size());
}

void BranchProbabilityInfo::copyEdgeProbabilities(BasicBlock *Src,
                                                  BasicBlock *Dst) {
  eraseBlock(Dst); // Erase stale data if any.
  unsigned NumSuccessors = Src->getTerminator()->getNumSuccessors();
  assert(NumSuccessors == Dst->getTerminator()->getNumSuccessors());
  if (NumSuccessors == 0)
    return; // Nothing to set.
  if (this->Probs.find(std::make_pair(Src, 0)) == this->Probs.end())
    return; // No probability is set for edges from Src. Keep the same for Dst.

  Handles.insert(BasicBlockCallbackVH(Dst, this));
  for (unsigned SuccIdx = 0; SuccIdx < NumSuccessors; ++SuccIdx) {
    auto Prob = this->Probs[std::make_pair(Src, SuccIdx)];
    this->Probs[std::make_pair(Dst, SuccIdx)] = Prob;
    LLVM_DEBUG(dbgs() << "set edge " << Dst->getName() << " -> " << SuccIdx
                      << " successor probability to " << Prob << "\n");
  }
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

void BranchProbabilityInfo::eraseBlock(const BasicBlock *BB) {
  // Note that we cannot use successors of BB because the terminator of BB may
  // have changed when eraseBlock is called as a BasicBlockCallbackVH callback.
  // Instead we remove prob data for the block by iterating successors by their
  // indices from 0 till the last which exists. There could not be prob data for
  // a pair (BB, N) if there is no data for (BB, N-1) because the data is always
  // set for all successors from 0 to M at once by the method
  // setEdgeProbability().
  Handles.erase(BasicBlockCallbackVH(BB, this));
  for (unsigned I = 0;; ++I) {
    auto MapI = Probs.find(std::make_pair(BB, I));
    if (MapI == Probs.end()) {
      assert(Probs.count(std::make_pair(BB, I + 1)) == 0 &&
             "Must be no more successors");
      return;
    }
    Probs.erase(MapI);
  }
}

void BranchProbabilityInfo::calculate(const Function &F, const LoopInfo &LI,
                                      const TargetLibraryInfo *TLI,
                                      PostDominatorTree *PDT) {
  LLVM_DEBUG(dbgs() << "---- Branch Probability Info : " << F.getName()
                    << " ----\n\n");
  LastF = &F; // Store the last function we ran on for printing.
  assert(PostDominatedByUnreachable.empty());
  assert(PostDominatedByColdCall.empty());

  SccI = std::make_unique<SccInfo>(F);

  std::unique_ptr<PostDominatorTree> PDTPtr;

  if (!PDT) {
    PDTPtr = std::make_unique<PostDominatorTree>(const_cast<Function &>(F));
    PDT = PDTPtr.get();
  }

  computePostDominatedByUnreachable(F, PDT);
  computePostDominatedByColdCall(F, PDT);

  // Walk the basic blocks in post-order so that we can build up state about
  // the successors of a block iteratively.
  for (auto BB : post_order(&F.getEntryBlock())) {
    LLVM_DEBUG(dbgs() << "Computing probabilities for " << BB->getName()
                      << "\n");
    // If there is no at least two successors, no sense to set probability.
    if (BB->getTerminator()->getNumSuccessors() < 2)
      continue;
    if (calcMetadataWeights(BB))
      continue;
    if (calcInvokeHeuristics(BB))
      continue;
    if (calcUnreachableHeuristics(BB))
      continue;
    if (calcColdCallHeuristics(BB))
      continue;
    if (calcLoopBranchHeuristics(BB, LI))
      continue;
    if (calcPointerHeuristics(BB))
      continue;
    if (calcZeroHeuristics(BB, TLI))
      continue;
    if (calcFloatingPointHeuristics(BB))
      continue;
  }

  PostDominatedByUnreachable.clear();
  PostDominatedByColdCall.clear();
  SccI.reset();

  if (PrintBranchProb &&
      (PrintBranchProbFuncName.empty() ||
       F.getName().equals(PrintBranchProbFuncName))) {
    print(dbgs());
  }
}

void BranchProbabilityInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  // We require DT so it's available when LI is available. The LI updating code
  // asserts that DT is also present so if we don't make sure that we have DT
  // here, that assert will trigger.
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.setPreservesAll();
}

bool BranchProbabilityInfoWrapperPass::runOnFunction(Function &F) {
  const LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  const TargetLibraryInfo &TLI =
      getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  PostDominatorTree &PDT =
      getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
  BPI.calculate(F, LI, &TLI, &PDT);
  return false;
}

void BranchProbabilityInfoWrapperPass::releaseMemory() { BPI.releaseMemory(); }

void BranchProbabilityInfoWrapperPass::print(raw_ostream &OS,
                                             const Module *) const {
  BPI.print(OS);
}

AnalysisKey BranchProbabilityAnalysis::Key;
BranchProbabilityInfo
BranchProbabilityAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  BranchProbabilityInfo BPI;
  BPI.calculate(F, AM.getResult<LoopAnalysis>(F),
                &AM.getResult<TargetLibraryAnalysis>(F),
                &AM.getResult<PostDominatorTreeAnalysis>(F));
  return BPI;
}

PreservedAnalyses
BranchProbabilityPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results of BPI for function "
     << "'" << F.getName() << "':"
     << "\n";
  AM.getResult<BranchProbabilityAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}
