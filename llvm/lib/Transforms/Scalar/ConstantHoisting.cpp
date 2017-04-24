//===- ConstantHoisting.cpp - Prepare code for expensive constants --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass identifies expensive constants to hoist and coalesces them to
// better prepare it for SelectionDAG-based code generation. This works around
// the limitations of the basic-block-at-a-time approach.
//
// First it scans all instructions for integer constants and calculates its
// cost. If the constant can be folded into the instruction (the cost is
// TCC_Free) or the cost is just a simple operation (TCC_BASIC), then we don't
// consider it expensive and leave it alone. This is the default behavior and
// the default implementation of getIntImmCost will always return TCC_Free.
//
// If the cost is more than TCC_BASIC, then the integer constant can't be folded
// into the instruction and it might be beneficial to hoist the constant.
// Similar constants are coalesced to reduce register pressure and
// materialization code.
//
// When a constant is hoisted, it is also hidden behind a bitcast to force it to
// be live-out of the basic block. Otherwise the constant would be just
// duplicated and each basic block would have its own copy in the SelectionDAG.
// The SelectionDAG recognizes such constants as opaque and doesn't perform
// certain transformations on them, which would create a new expensive constant.
//
// This optimization is only applied to integer constants in instructions and
// simple (this means not nested) constant cast expressions. For example:
// %0 = load i64* inttoptr (i64 big_constant to i64*)
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include <tuple>

using namespace llvm;
using namespace consthoist;

#define DEBUG_TYPE "consthoist"

STATISTIC(NumConstantsHoisted, "Number of constants hoisted");
STATISTIC(NumConstantsRebased, "Number of constants rebased");

static cl::opt<bool> ConstHoistWithBlockFrequency(
    "consthoist-with-block-frequency", cl::init(false), cl::Hidden,
    cl::desc("Enable the use of the block frequency analysis to reduce the "
             "chance to execute const materialization more frequently than "
             "without hoisting."));

namespace {
/// \brief The constant hoisting pass.
class ConstantHoistingLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  ConstantHoistingLegacyPass() : FunctionPass(ID) {
    initializeConstantHoistingLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &Fn) override;

  StringRef getPassName() const override { return "Constant Hoisting"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    if (ConstHoistWithBlockFrequency)
      AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  void releaseMemory() override { Impl.releaseMemory(); }

private:
  ConstantHoistingPass Impl;
};
}

char ConstantHoistingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(ConstantHoistingLegacyPass, "consthoist",
                      "Constant Hoisting", false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(ConstantHoistingLegacyPass, "consthoist",
                    "Constant Hoisting", false, false)

FunctionPass *llvm::createConstantHoistingPass() {
  return new ConstantHoistingLegacyPass();
}

/// \brief Perform the constant hoisting optimization for the given function.
bool ConstantHoistingLegacyPass::runOnFunction(Function &Fn) {
  if (skipFunction(Fn))
    return false;

  DEBUG(dbgs() << "********** Begin Constant Hoisting **********\n");
  DEBUG(dbgs() << "********** Function: " << Fn.getName() << '\n');

  bool MadeChange =
      Impl.runImpl(Fn, getAnalysis<TargetTransformInfoWrapperPass>().getTTI(Fn),
                   getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
                   ConstHoistWithBlockFrequency
                       ? &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI()
                       : nullptr,
                   Fn.getEntryBlock());

  if (MadeChange) {
    DEBUG(dbgs() << "********** Function after Constant Hoisting: "
                 << Fn.getName() << '\n');
    DEBUG(dbgs() << Fn);
  }
  DEBUG(dbgs() << "********** End Constant Hoisting **********\n");

  return MadeChange;
}


/// \brief Find the constant materialization insertion point.
Instruction *ConstantHoistingPass::findMatInsertPt(Instruction *Inst,
                                                   unsigned Idx) const {
  // If the operand is a cast instruction, then we have to materialize the
  // constant before the cast instruction.
  if (Idx != ~0U) {
    Value *Opnd = Inst->getOperand(Idx);
    if (auto CastInst = dyn_cast<Instruction>(Opnd))
      if (CastInst->isCast())
        return CastInst;
  }

  // The simple and common case. This also includes constant expressions.
  if (!isa<PHINode>(Inst) && !Inst->isEHPad())
    return Inst;

  // We can't insert directly before a phi node or an eh pad. Insert before
  // the terminator of the incoming or dominating block.
  assert(Entry != Inst->getParent() && "PHI or landing pad in entry block!");
  if (Idx != ~0U && isa<PHINode>(Inst))
    return cast<PHINode>(Inst)->getIncomingBlock(Idx)->getTerminator();

  // This must be an EH pad. Iterate over immediate dominators until we find a
  // non-EH pad. We need to skip over catchswitch blocks, which are both EH pads
  // and terminators.
  auto IDom = DT->getNode(Inst->getParent())->getIDom();
  while (IDom->getBlock()->isEHPad()) {
    assert(Entry != IDom->getBlock() && "eh pad in entry block");
    IDom = IDom->getIDom();
  }

  return IDom->getBlock()->getTerminator();
}

/// \brief Given \p BBs as input, find another set of BBs which collectively
/// dominates \p BBs and have the minimal sum of frequencies. Return the BB
/// set found in \p BBs.
void findBestInsertionSet(DominatorTree &DT, BlockFrequencyInfo &BFI,
                          BasicBlock *Entry,
                          SmallPtrSet<BasicBlock *, 8> &BBs) {
  assert(!BBs.count(Entry) && "Assume Entry is not in BBs");
  // Nodes on the current path to the root.
  SmallPtrSet<BasicBlock *, 8> Path;
  // Candidates includes any block 'BB' in set 'BBs' that is not strictly
  // dominated by any other blocks in set 'BBs', and all nodes in the path
  // in the dominator tree from Entry to 'BB'.
  SmallPtrSet<BasicBlock *, 16> Candidates;
  for (auto BB : BBs) {
    Path.clear();
    // Walk up the dominator tree until Entry or another BB in BBs
    // is reached. Insert the nodes on the way to the Path.
    BasicBlock *Node = BB;
    // The "Path" is a candidate path to be added into Candidates set.
    bool isCandidate = false;
    do {
      Path.insert(Node);
      if (Node == Entry || Candidates.count(Node)) {
        isCandidate = true;
        break;
      }
      assert(DT.getNode(Node)->getIDom() &&
             "Entry doens't dominate current Node");
      Node = DT.getNode(Node)->getIDom()->getBlock();
    } while (!BBs.count(Node));

    // If isCandidate is false, Node is another Block in BBs dominating
    // current 'BB'. Drop the nodes on the Path.
    if (!isCandidate)
      continue;

    // Add nodes on the Path into Candidates.
    Candidates.insert(Path.begin(), Path.end());
  }

  // Sort the nodes in Candidates in top-down order and save the nodes
  // in Orders.
  unsigned Idx = 0;
  SmallVector<BasicBlock *, 16> Orders;
  Orders.push_back(Entry);
  while (Idx != Orders.size()) {
    BasicBlock *Node = Orders[Idx++];
    for (auto ChildDomNode : DT.getNode(Node)->getChildren()) {
      if (Candidates.count(ChildDomNode->getBlock()))
        Orders.push_back(ChildDomNode->getBlock());
    }
  }

  // Visit Orders in bottom-up order.
  typedef std::pair<SmallPtrSet<BasicBlock *, 16>, BlockFrequency>
      InsertPtsCostPair;
  // InsertPtsMap is a map from a BB to the best insertion points for the
  // subtree of BB (subtree not including the BB itself).
  DenseMap<BasicBlock *, InsertPtsCostPair> InsertPtsMap;
  InsertPtsMap.reserve(Orders.size() + 1);
  for (auto RIt = Orders.rbegin(); RIt != Orders.rend(); RIt++) {
    BasicBlock *Node = *RIt;
    bool NodeInBBs = BBs.count(Node);
    SmallPtrSet<BasicBlock *, 16> &InsertPts = InsertPtsMap[Node].first;
    BlockFrequency &InsertPtsFreq = InsertPtsMap[Node].second;

    // Return the optimal insert points in BBs.
    if (Node == Entry) {
      BBs.clear();
      if (InsertPtsFreq > BFI.getBlockFreq(Node))
        BBs.insert(Entry);
      else
        BBs.insert(InsertPts.begin(), InsertPts.end());
      break;
    }

    BasicBlock *Parent = DT.getNode(Node)->getIDom()->getBlock();
    // Initially, ParentInsertPts is empty and ParentPtsFreq is 0. Every child
    // will update its parent's ParentInsertPts and ParentPtsFreq.
    SmallPtrSet<BasicBlock *, 16> &ParentInsertPts = InsertPtsMap[Parent].first;
    BlockFrequency &ParentPtsFreq = InsertPtsMap[Parent].second;
    // Choose to insert in Node or in subtree of Node.
    if (InsertPtsFreq > BFI.getBlockFreq(Node) || NodeInBBs) {
      ParentInsertPts.insert(Node);
      ParentPtsFreq += BFI.getBlockFreq(Node);
    } else {
      ParentInsertPts.insert(InsertPts.begin(), InsertPts.end());
      ParentPtsFreq += InsertPtsFreq;
    }
  }
}

/// \brief Find an insertion point that dominates all uses.
SmallPtrSet<Instruction *, 8> ConstantHoistingPass::findConstantInsertionPoint(
    const ConstantInfo &ConstInfo) const {
  assert(!ConstInfo.RebasedConstants.empty() && "Invalid constant info entry.");
  // Collect all basic blocks.
  SmallPtrSet<BasicBlock *, 8> BBs;
  SmallPtrSet<Instruction *, 8> InsertPts;
  for (auto const &RCI : ConstInfo.RebasedConstants)
    for (auto const &U : RCI.Uses)
      BBs.insert(findMatInsertPt(U.Inst, U.OpndIdx)->getParent());

  if (BBs.count(Entry)) {
    InsertPts.insert(&Entry->front());
    return InsertPts;
  }

  if (BFI) {
    findBestInsertionSet(*DT, *BFI, Entry, BBs);
    for (auto BB : BBs) {
      BasicBlock::iterator InsertPt = BB->begin();
      for (; isa<PHINode>(InsertPt) || InsertPt->isEHPad(); ++InsertPt)
        ;
      InsertPts.insert(&*InsertPt);
    }
    return InsertPts;
  }

  while (BBs.size() >= 2) {
    BasicBlock *BB, *BB1, *BB2;
    BB1 = *BBs.begin();
    BB2 = *std::next(BBs.begin());
    BB = DT->findNearestCommonDominator(BB1, BB2);
    if (BB == Entry) {
      InsertPts.insert(&Entry->front());
      return InsertPts;
    }
    BBs.erase(BB1);
    BBs.erase(BB2);
    BBs.insert(BB);
  }
  assert((BBs.size() == 1) && "Expected only one element.");
  Instruction &FirstInst = (*BBs.begin())->front();
  InsertPts.insert(findMatInsertPt(&FirstInst));
  return InsertPts;
}


/// \brief Record constant integer ConstInt for instruction Inst at operand
/// index Idx.
///
/// The operand at index Idx is not necessarily the constant integer itself. It
/// could also be a cast instruction or a constant expression that uses the
// constant integer.
void ConstantHoistingPass::collectConstantCandidates(
    ConstCandMapType &ConstCandMap, Instruction *Inst, unsigned Idx,
    ConstantInt *ConstInt) {
  unsigned Cost;
  // Ask the target about the cost of materializing the constant for the given
  // instruction and operand index.
  if (auto IntrInst = dyn_cast<IntrinsicInst>(Inst))
    Cost = TTI->getIntImmCost(IntrInst->getIntrinsicID(), Idx,
                              ConstInt->getValue(), ConstInt->getType());
  else
    Cost = TTI->getIntImmCost(Inst->getOpcode(), Idx, ConstInt->getValue(),
                              ConstInt->getType());

  // Ignore cheap integer constants.
  if (Cost > TargetTransformInfo::TCC_Basic) {
    ConstCandMapType::iterator Itr;
    bool Inserted;
    std::tie(Itr, Inserted) = ConstCandMap.insert(std::make_pair(ConstInt, 0));
    if (Inserted) {
      ConstCandVec.push_back(ConstantCandidate(ConstInt));
      Itr->second = ConstCandVec.size() - 1;
    }
    ConstCandVec[Itr->second].addUser(Inst, Idx, Cost);
    DEBUG(if (isa<ConstantInt>(Inst->getOperand(Idx)))
            dbgs() << "Collect constant " << *ConstInt << " from " << *Inst
                   << " with cost " << Cost << '\n';
          else
          dbgs() << "Collect constant " << *ConstInt << " indirectly from "
                 << *Inst << " via " << *Inst->getOperand(Idx) << " with cost "
                 << Cost << '\n';
    );
  }
}

/// \brief Scan the instruction for expensive integer constants and record them
/// in the constant candidate vector.
void ConstantHoistingPass::collectConstantCandidates(
    ConstCandMapType &ConstCandMap, Instruction *Inst) {
  // Skip all cast instructions. They are visited indirectly later on.
  if (Inst->isCast())
    return;

  // Can't handle inline asm. Skip it.
  if (auto Call = dyn_cast<CallInst>(Inst))
    if (isa<InlineAsm>(Call->getCalledValue()))
      return;

  // Switch cases must remain constant, and if the value being tested is
  // constant the entire thing should disappear.
  if (isa<SwitchInst>(Inst))
    return;

  // Static allocas (constant size in the entry block) are handled by
  // prologue/epilogue insertion so they're free anyway. We definitely don't
  // want to make them non-constant.
  auto AI = dyn_cast<AllocaInst>(Inst);
  if (AI && AI->isStaticAlloca())
    return;

  // Scan all operands.
  for (unsigned Idx = 0, E = Inst->getNumOperands(); Idx != E; ++Idx) {
    Value *Opnd = Inst->getOperand(Idx);

    // Visit constant integers.
    if (auto ConstInt = dyn_cast<ConstantInt>(Opnd)) {
      collectConstantCandidates(ConstCandMap, Inst, Idx, ConstInt);
      continue;
    }

    // Visit cast instructions that have constant integers.
    if (auto CastInst = dyn_cast<Instruction>(Opnd)) {
      // Only visit cast instructions, which have been skipped. All other
      // instructions should have already been visited.
      if (!CastInst->isCast())
        continue;

      if (auto *ConstInt = dyn_cast<ConstantInt>(CastInst->getOperand(0))) {
        // Pretend the constant is directly used by the instruction and ignore
        // the cast instruction.
        collectConstantCandidates(ConstCandMap, Inst, Idx, ConstInt);
        continue;
      }
    }

    // Visit constant expressions that have constant integers.
    if (auto ConstExpr = dyn_cast<ConstantExpr>(Opnd)) {
      // Only visit constant cast expressions.
      if (!ConstExpr->isCast())
        continue;

      if (auto ConstInt = dyn_cast<ConstantInt>(ConstExpr->getOperand(0))) {
        // Pretend the constant is directly used by the instruction and ignore
        // the constant expression.
        collectConstantCandidates(ConstCandMap, Inst, Idx, ConstInt);
        continue;
      }
    }
  } // end of for all operands
}

/// \brief Collect all integer constants in the function that cannot be folded
/// into an instruction itself.
void ConstantHoistingPass::collectConstantCandidates(Function &Fn) {
  ConstCandMapType ConstCandMap;
  for (BasicBlock &BB : Fn)
    for (Instruction &Inst : BB)
      collectConstantCandidates(ConstCandMap, &Inst);
}

// This helper function is necessary to deal with values that have different
// bit widths (APInt Operator- does not like that). If the value cannot be
// represented in uint64 we return an "empty" APInt. This is then interpreted
// as the value is not in range.
static llvm::Optional<APInt> calculateOffsetDiff(const APInt &V1,
                                                 const APInt &V2) {
  llvm::Optional<APInt> Res = None;
  unsigned BW = V1.getBitWidth() > V2.getBitWidth() ?
                V1.getBitWidth() : V2.getBitWidth();
  uint64_t LimVal1 = V1.getLimitedValue();
  uint64_t LimVal2 = V2.getLimitedValue();

  if (LimVal1 == ~0ULL || LimVal2 == ~0ULL)
    return Res;

  uint64_t Diff = LimVal1 - LimVal2;
  return APInt(BW, Diff, true);
}

// From a list of constants, one needs to picked as the base and the other
// constants will be transformed into an offset from that base constant. The
// question is which we can pick best? For example, consider these constants
// and their number of uses:
//
//  Constants| 2 | 4 | 12 | 42 |
//  NumUses  | 3 | 2 |  8 |  7 |
//
// Selecting constant 12 because it has the most uses will generate negative
// offsets for constants 2 and 4 (i.e. -10 and -8 respectively). If negative
// offsets lead to less optimal code generation, then there might be better
// solutions. Suppose immediates in the range of 0..35 are most optimally
// supported by the architecture, then selecting constant 2 is most optimal
// because this will generate offsets: 0, 2, 10, 40. Offsets 0, 2 and 10 are in
// range 0..35, and thus 3 + 2 + 8 = 13 uses are in range. Selecting 12 would
// have only 8 uses in range, so choosing 2 as a base is more optimal. Thus, in
// selecting the base constant the range of the offsets is a very important
// factor too that we take into account here. This algorithm calculates a total
// costs for selecting a constant as the base and substract the costs if
// immediates are out of range. It has quadratic complexity, so we call this
// function only when we're optimising for size and there are less than 100
// constants, we fall back to the straightforward algorithm otherwise
// which does not do all the offset calculations.
unsigned
ConstantHoistingPass::maximizeConstantsInRange(ConstCandVecType::iterator S,
                                           ConstCandVecType::iterator E,
                                           ConstCandVecType::iterator &MaxCostItr) {
  unsigned NumUses = 0;

  if(!Entry->getParent()->optForSize() || std::distance(S,E) > 100) {
    for (auto ConstCand = S; ConstCand != E; ++ConstCand) {
      NumUses += ConstCand->Uses.size();
      if (ConstCand->CumulativeCost > MaxCostItr->CumulativeCost)
        MaxCostItr = ConstCand;
    }
    return NumUses;
  }

  DEBUG(dbgs() << "== Maximize constants in range ==\n");
  int MaxCost = -1;
  for (auto ConstCand = S; ConstCand != E; ++ConstCand) {
    auto Value = ConstCand->ConstInt->getValue();
    Type *Ty = ConstCand->ConstInt->getType();
    int Cost = 0;
    NumUses += ConstCand->Uses.size();
    DEBUG(dbgs() << "= Constant: " << ConstCand->ConstInt->getValue() << "\n");

    for (auto User : ConstCand->Uses) {
      unsigned Opcode = User.Inst->getOpcode();
      unsigned OpndIdx = User.OpndIdx;
      Cost += TTI->getIntImmCost(Opcode, OpndIdx, Value, Ty);
      DEBUG(dbgs() << "Cost: " << Cost << "\n");

      for (auto C2 = S; C2 != E; ++C2) {
        llvm::Optional<APInt> Diff = calculateOffsetDiff(
                                      C2->ConstInt->getValue(),
                                      ConstCand->ConstInt->getValue());
        if (Diff) {
          const int ImmCosts =
            TTI->getIntImmCodeSizeCost(Opcode, OpndIdx, Diff.getValue(), Ty);
          Cost -= ImmCosts;
          DEBUG(dbgs() << "Offset " << Diff.getValue() << " "
                       << "has penalty: " << ImmCosts << "\n"
                       << "Adjusted cost: " << Cost << "\n");
        }
      }
    }
    DEBUG(dbgs() << "Cumulative cost: " << Cost << "\n");
    if (Cost > MaxCost) {
      MaxCost = Cost;
      MaxCostItr = ConstCand;
      DEBUG(dbgs() << "New candidate: " << MaxCostItr->ConstInt->getValue()
                   << "\n");
    }
  }
  return NumUses;
}

/// \brief Find the base constant within the given range and rebase all other
/// constants with respect to the base constant.
void ConstantHoistingPass::findAndMakeBaseConstant(
    ConstCandVecType::iterator S, ConstCandVecType::iterator E) {
  auto MaxCostItr = S;
  unsigned NumUses = maximizeConstantsInRange(S, E, MaxCostItr);

  // Don't hoist constants that have only one use.
  if (NumUses <= 1)
    return;

  ConstantInfo ConstInfo;
  ConstInfo.BaseConstant = MaxCostItr->ConstInt;
  Type *Ty = ConstInfo.BaseConstant->getType();

  // Rebase the constants with respect to the base constant.
  for (auto ConstCand = S; ConstCand != E; ++ConstCand) {
    APInt Diff = ConstCand->ConstInt->getValue() -
                 ConstInfo.BaseConstant->getValue();
    Constant *Offset = Diff == 0 ? nullptr : ConstantInt::get(Ty, Diff);
    ConstInfo.RebasedConstants.push_back(
      RebasedConstantInfo(std::move(ConstCand->Uses), Offset));
  }
  ConstantVec.push_back(std::move(ConstInfo));
}

/// \brief Finds and combines constant candidates that can be easily
/// rematerialized with an add from a common base constant.
void ConstantHoistingPass::findBaseConstants() {
  // Sort the constants by value and type. This invalidates the mapping!
  std::sort(ConstCandVec.begin(), ConstCandVec.end(),
            [](const ConstantCandidate &LHS, const ConstantCandidate &RHS) {
    if (LHS.ConstInt->getType() != RHS.ConstInt->getType())
      return LHS.ConstInt->getType()->getBitWidth() <
             RHS.ConstInt->getType()->getBitWidth();
    return LHS.ConstInt->getValue().ult(RHS.ConstInt->getValue());
  });

  // Simple linear scan through the sorted constant candidate vector for viable
  // merge candidates.
  auto MinValItr = ConstCandVec.begin();
  for (auto CC = std::next(ConstCandVec.begin()), E = ConstCandVec.end();
       CC != E; ++CC) {
    if (MinValItr->ConstInt->getType() == CC->ConstInt->getType()) {
      // Check if the constant is in range of an add with immediate.
      APInt Diff = CC->ConstInt->getValue() - MinValItr->ConstInt->getValue();
      if ((Diff.getBitWidth() <= 64) &&
          TTI->isLegalAddImmediate(Diff.getSExtValue()))
        continue;
    }
    // We either have now a different constant type or the constant is not in
    // range of an add with immediate anymore.
    findAndMakeBaseConstant(MinValItr, CC);
    // Start a new base constant search.
    MinValItr = CC;
  }
  // Finalize the last base constant search.
  findAndMakeBaseConstant(MinValItr, ConstCandVec.end());
}

/// \brief Updates the operand at Idx in instruction Inst with the result of
///        instruction Mat. If the instruction is a PHI node then special
///        handling for duplicate values form the same incoming basic block is
///        required.
/// \return The update will always succeed, but the return value indicated if
///         Mat was used for the update or not.
static bool updateOperand(Instruction *Inst, unsigned Idx, Instruction *Mat) {
  if (auto PHI = dyn_cast<PHINode>(Inst)) {
    // Check if any previous operand of the PHI node has the same incoming basic
    // block. This is a very odd case that happens when the incoming basic block
    // has a switch statement. In this case use the same value as the previous
    // operand(s), otherwise we will fail verification due to different values.
    // The values are actually the same, but the variable names are different
    // and the verifier doesn't like that.
    BasicBlock *IncomingBB = PHI->getIncomingBlock(Idx);
    for (unsigned i = 0; i < Idx; ++i) {
      if (PHI->getIncomingBlock(i) == IncomingBB) {
        Value *IncomingVal = PHI->getIncomingValue(i);
        Inst->setOperand(Idx, IncomingVal);
        return false;
      }
    }
  }

  Inst->setOperand(Idx, Mat);
  return true;
}

/// \brief Emit materialization code for all rebased constants and update their
/// users.
void ConstantHoistingPass::emitBaseConstants(Instruction *Base,
                                             Constant *Offset,
                                             const ConstantUser &ConstUser) {
  Instruction *Mat = Base;
  if (Offset) {
    Instruction *InsertionPt = findMatInsertPt(ConstUser.Inst,
                                               ConstUser.OpndIdx);
    Mat = BinaryOperator::Create(Instruction::Add, Base, Offset,
                                 "const_mat", InsertionPt);

    DEBUG(dbgs() << "Materialize constant (" << *Base->getOperand(0)
                 << " + " << *Offset << ") in BB "
                 << Mat->getParent()->getName() << '\n' << *Mat << '\n');
    Mat->setDebugLoc(ConstUser.Inst->getDebugLoc());
  }
  Value *Opnd = ConstUser.Inst->getOperand(ConstUser.OpndIdx);

  // Visit constant integer.
  if (isa<ConstantInt>(Opnd)) {
    DEBUG(dbgs() << "Update: " << *ConstUser.Inst << '\n');
    if (!updateOperand(ConstUser.Inst, ConstUser.OpndIdx, Mat) && Offset)
      Mat->eraseFromParent();
    DEBUG(dbgs() << "To    : " << *ConstUser.Inst << '\n');
    return;
  }

  // Visit cast instruction.
  if (auto CastInst = dyn_cast<Instruction>(Opnd)) {
    assert(CastInst->isCast() && "Expected an cast instruction!");
    // Check if we already have visited this cast instruction before to avoid
    // unnecessary cloning.
    Instruction *&ClonedCastInst = ClonedCastMap[CastInst];
    if (!ClonedCastInst) {
      ClonedCastInst = CastInst->clone();
      ClonedCastInst->setOperand(0, Mat);
      ClonedCastInst->insertAfter(CastInst);
      // Use the same debug location as the original cast instruction.
      ClonedCastInst->setDebugLoc(CastInst->getDebugLoc());
      DEBUG(dbgs() << "Clone instruction: " << *CastInst << '\n'
                   << "To               : " << *ClonedCastInst << '\n');
    }

    DEBUG(dbgs() << "Update: " << *ConstUser.Inst << '\n');
    updateOperand(ConstUser.Inst, ConstUser.OpndIdx, ClonedCastInst);
    DEBUG(dbgs() << "To    : " << *ConstUser.Inst << '\n');
    return;
  }

  // Visit constant expression.
  if (auto ConstExpr = dyn_cast<ConstantExpr>(Opnd)) {
    Instruction *ConstExprInst = ConstExpr->getAsInstruction();
    ConstExprInst->setOperand(0, Mat);
    ConstExprInst->insertBefore(findMatInsertPt(ConstUser.Inst,
                                                ConstUser.OpndIdx));

    // Use the same debug location as the instruction we are about to update.
    ConstExprInst->setDebugLoc(ConstUser.Inst->getDebugLoc());

    DEBUG(dbgs() << "Create instruction: " << *ConstExprInst << '\n'
                 << "From              : " << *ConstExpr << '\n');
    DEBUG(dbgs() << "Update: " << *ConstUser.Inst << '\n');
    if (!updateOperand(ConstUser.Inst, ConstUser.OpndIdx, ConstExprInst)) {
      ConstExprInst->eraseFromParent();
      if (Offset)
        Mat->eraseFromParent();
    }
    DEBUG(dbgs() << "To    : " << *ConstUser.Inst << '\n');
    return;
  }
}

/// \brief Hoist and hide the base constant behind a bitcast and emit
/// materialization code for derived constants.
bool ConstantHoistingPass::emitBaseConstants() {
  bool MadeChange = false;
  for (auto const &ConstInfo : ConstantVec) {
    // Hoist and hide the base constant behind a bitcast.
    SmallPtrSet<Instruction *, 8> IPSet = findConstantInsertionPoint(ConstInfo);
    assert(!IPSet.empty() && "IPSet is empty");

    unsigned UsesNum = 0;
    unsigned ReBasesNum = 0;
    for (Instruction *IP : IPSet) {
      IntegerType *Ty = ConstInfo.BaseConstant->getType();
      Instruction *Base =
          new BitCastInst(ConstInfo.BaseConstant, Ty, "const", IP);
      DEBUG(dbgs() << "Hoist constant (" << *ConstInfo.BaseConstant
                   << ") to BB " << IP->getParent()->getName() << '\n'
                   << *Base << '\n');

      // Emit materialization code for all rebased constants.
      unsigned Uses = 0;
      for (auto const &RCI : ConstInfo.RebasedConstants) {
        for (auto const &U : RCI.Uses) {
          Uses++;
          BasicBlock *OrigMatInsertBB =
              findMatInsertPt(U.Inst, U.OpndIdx)->getParent();
          // If Base constant is to be inserted in multiple places,
          // generate rebase for U using the Base dominating U.
          if (IPSet.size() == 1 ||
              DT->dominates(Base->getParent(), OrigMatInsertBB)) {
            emitBaseConstants(Base, RCI.Offset, U);
            ReBasesNum++;
          }
        }
      }
      UsesNum = Uses;

      // Use the same debug location as the last user of the constant.
      assert(!Base->use_empty() && "The use list is empty!?");
      assert(isa<Instruction>(Base->user_back()) &&
             "All uses should be instructions.");
      Base->setDebugLoc(cast<Instruction>(Base->user_back())->getDebugLoc());
    }
    (void)UsesNum;
    (void)ReBasesNum;
    // Expect all uses are rebased after rebase is done.
    assert(UsesNum == ReBasesNum && "Not all uses are rebased");

    NumConstantsHoisted++;

    // Base constant is also included in ConstInfo.RebasedConstants, so
    // deduct 1 from ConstInfo.RebasedConstants.size().
    NumConstantsRebased = ConstInfo.RebasedConstants.size() - 1;

    MadeChange = true;
  }
  return MadeChange;
}

/// \brief Check all cast instructions we made a copy of and remove them if they
/// have no more users.
void ConstantHoistingPass::deleteDeadCastInst() const {
  for (auto const &I : ClonedCastMap)
    if (I.first->use_empty())
      I.first->eraseFromParent();
}

/// \brief Optimize expensive integer constants in the given function.
bool ConstantHoistingPass::runImpl(Function &Fn, TargetTransformInfo &TTI,
                                   DominatorTree &DT, BlockFrequencyInfo *BFI,
                                   BasicBlock &Entry) {
  this->TTI = &TTI;
  this->DT = &DT;
  this->BFI = BFI;
  this->Entry = &Entry;  
  // Collect all constant candidates.
  collectConstantCandidates(Fn);

  // There are no constant candidates to worry about.
  if (ConstCandVec.empty())
    return false;

  // Combine constants that can be easily materialized with an add from a common
  // base constant.
  findBaseConstants();

  // There are no constants to emit.
  if (ConstantVec.empty())
    return false;

  // Finally hoist the base constant and emit materialization code for dependent
  // constants.
  bool MadeChange = emitBaseConstants();

  // Cleanup dead instructions.
  deleteDeadCastInst();

  return MadeChange;
}

PreservedAnalyses ConstantHoistingPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto BFI = ConstHoistWithBlockFrequency
                 ? &AM.getResult<BlockFrequencyAnalysis>(F)
                 : nullptr;
  if (!runImpl(F, TTI, DT, BFI, F.getEntryBlock()))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
