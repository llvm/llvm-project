//===- LowerSwitch.cpp - Eliminate Switch instructions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LowerSwitch transformation rewrites switch instructions with a sequence
// of branches, which allows targets to get away with not implementing the
// switch instruction until it is convenient.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cassert>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "lower-switch"

namespace {

struct IntRange {
  APInt Low, High;
};

} // end anonymous namespace

namespace {
// Return true iff R is covered by Ranges.
bool IsInRanges(const IntRange &R, const std::vector<IntRange> &Ranges) {
  // Note: Ranges must be sorted, non-overlapping and non-adjacent.

  // Find the first range whose High field is >= R.High,
  // then check if the Low field is <= R.Low. If so, we
  // have a Range that covers R.
  auto I = llvm::lower_bound(
      Ranges, R, [](IntRange A, IntRange B) { return A.High.slt(B.High); });
  return I != Ranges.end() && I->Low.sle(R.Low);
}

// A case that already goes to the default, and its count.
//
//   switch i32 %x, label %def [
//     i32 1, label %a
//     i32 4, label %def   ; {Value: 4, Weight: 15}
//   ], !prof !{!"branch_weights", i32 0, i32 20, i32 15}
struct DefaultCase {
  APInt Value;
  uint64_t Weight = 0;
};

// The one leaf that gets the whole default count.
//
//   %v = and i32 %x, 3
//   switch i32 %v, label %def [
//     i32 0, label %a
//     i32 1, label %b
//   ], !prof !{!"branch_weights", i32 100, i32 10, i32 20}
//
// lowers to:
//
//   %pivot = icmp slt i32 %v, 1
//   br i1 %pivot, label %a, label %check
// check:
//   %case = icmp eq i32 %v, 1
//   br i1 %case, label %b, label %def
//
// Only %check can reach %def, so it keeps the whole default count:
// {Low: 1, Weight: 100}.
struct UniqueMissLeaf {
  ConstantInt *Low = nullptr;
  uint64_t Weight = 0;
};

// Numeric weights for the compare tree.
struct SwitchBranchWeights {
  // Default count for one remaining case range. Otherwise 0.
  uint64_t MissWeight = 0;
  // Sorted by signed value before the tree walk.
  SmallVector<DefaultCase, 8> DefaultCases;
  // True when the switch weights were marked !"expected".
  bool IsExpected = false;
  std::optional<UniqueMissLeaf> UniqueMiss;
};

/// fitWeights() may scale large counts down to fit into uint32_t values.
/// A non-zero weight that becomes 0 is stored as 1. A real 0 stays 0.
static void fitAndSetBranchWeights(Instruction &I, ArrayRef<uint64_t> Weights,
                                   bool IsExpected) {
  SmallVector<uint32_t> Fitted = fitWeights(Weights);
  for (size_t Idx = 0; Idx < Weights.size() && Idx < Fitted.size(); ++Idx)
    if (Weights[Idx] != 0 && Fitted[Idx] == 0)
      Fitted[Idx] = 1;
  setBranchWeights(I, Fitted, IsExpected);
}

/// Sum of default-case weights whose value lies in [Lo, Hi].
/// \p Cases is sorted by signed value.
/// Example:
///   DefaultCases = {{2, 5}, {7, 10}, {9, 20}}
///   sumDefaultCases(..., 7, 9) == 30
/// because only values 7 and 9 fall in the queried range.
static uint64_t sumDefaultCases(ArrayRef<DefaultCase> Cases, const APInt &Lo,
                                const APInt &Hi) {
  auto It = llvm::lower_bound(Cases, Lo,
                              [](const DefaultCase &Case, const APInt &Value) {
                                return Case.Value.slt(Value);
                              });
  uint64_t Weight = 0;
  for (; It != Cases.end() && It->Value.sle(Hi); ++It)
    Weight = SaturatingAdd(Weight, It->Weight);
  return Weight;
}

/// True when KnownBits proves that V cannot occur.
/// Example:
///   Known = xxxx1   (bit 0 is known one)
///   V     = xxxx0
/// Then V conflicts with KnownBits and cannot be produced.
static bool conflictsWithKnownBits(const APInt &V, const KnownBits &Known) {
  return Known.Zero.intersects(V) || Known.One.intersects(~V);
}

struct CaseRange {
  ConstantInt *Low;
  ConstantInt *High;
  BasicBlock *BB;
  uint64_t Weight;

  CaseRange(ConstantInt *low, ConstantInt *high, BasicBlock *bb,
            uint64_t weight = 0)
      : Low(low), High(high), BB(bb), Weight(weight) {}
};

using CaseVector = std::vector<CaseRange>;
using CaseItr = std::vector<CaseRange>::iterator;

/// The comparison function for sorting the switch case values in the vector.
/// WARNING: Case ranges should be disjoint!
struct CaseCmp {
  bool operator()(const CaseRange &C1, const CaseRange &C2) {
    const ConstantInt *CI1 = cast<const ConstantInt>(C1.Low);
    const ConstantInt *CI2 = cast<const ConstantInt>(C2.High);
    return CI1->getValue().slt(CI2->getValue());
  }
};

/// Used for debugging purposes.
LLVM_ATTRIBUTE_USED
raw_ostream &operator<<(raw_ostream &O, const CaseVector &C) {
  O << "[";

  for (CaseVector::const_iterator B = C.begin(), E = C.end(); B != E;) {
    O << "[" << B->Low->getValue() << ", " << B->High->getValue() << "]";
    if (++B != E)
      O << ", ";
  }

  return O << "]";
}

/// Update the first occurrence of the "switch statement" BB in the PHI
/// node with the "new" BB. The other occurrences will:
///
/// 1) Be updated by subsequent calls to this function.  Switch statements may
/// have more than one outcoming edge into the same BB if they all have the same
/// value. When the switch statement is converted these incoming edges are now
/// coming from multiple BBs.
/// 2) Removed if subsequent incoming values now share the same case, i.e.,
/// multiple outcome edges are condensed into one. This is necessary to keep the
/// number of phi values equal to the number of branches to SuccBB.
void FixPhis(BasicBlock *SuccBB, BasicBlock *OrigBB, BasicBlock *NewBB,
             const APInt &NumMergedCases) {
  for (auto &I : SuccBB->phis()) {
    PHINode *PN = cast<PHINode>(&I);

    // Only update the first occurrence if NewBB exists.
    unsigned Idx = 0, E = PN->getNumIncomingValues();
    APInt LocalNumMergedCases = NumMergedCases;
    for (; Idx != E && NewBB; ++Idx) {
      if (PN->getIncomingBlock(Idx) == OrigBB) {
        PN->setIncomingBlock(Idx, NewBB);
        break;
      }
    }

    // Skip the updated incoming block so that it will not be removed.
    if (NewBB)
      ++Idx;

    // Remove additional occurrences coming from condensed cases and keep the
    // number of incoming values equal to the number of branches to SuccBB.
    SmallVector<unsigned, 8> Indices;
    for (; LocalNumMergedCases.ugt(0) && Idx < E; ++Idx)
      if (PN->getIncomingBlock(Idx) == OrigBB) {
        Indices.push_back(Idx);
        LocalNumMergedCases -= 1;
      }
    // Remove incoming values in the reverse order to prevent invalidating
    // *successive* index.
    for (unsigned III : llvm::reverse(Indices))
      PN->removeIncomingValue(III);
  }
}

/// Sum the weights of cases in [Begin, End).
static uint64_t sumCaseWeights(CaseItr Begin, CaseItr End) {
  uint64_t Weight = 0;
  for (; Begin != End; ++Begin)
    Weight = SaturatingAdd(Weight, Begin->Weight);
  return Weight;
}

/// Create a new leaf block for the binary lookup tree. It checks if the
/// switch's value == the case's value. If not, then it jumps to the default
/// branch. At this point in the tree, the value can't be another valid case
/// value, so the jump to the "default" branch is warranted.
BasicBlock *NewLeafBlock(CaseRange &Leaf, Value *Val, ConstantInt *LowerBound,
                         ConstantInt *UpperBound, BasicBlock *OrigBlock,
                         BasicBlock *Default,
                         const std::optional<SwitchBranchWeights> &Profile,
                         bool MarkUnknown) {
  Function *F = OrigBlock->getParent();
  BasicBlock *NewLeaf = BasicBlock::Create(Val->getContext(), "LeafBlock");
  F->insert(++OrigBlock->getIterator(), NewLeaf);

  // Emit comparison
  ICmpInst *Comp = nullptr;
  if (Leaf.Low == Leaf.High) {
    // Make the seteq instruction...
    Comp =
        new ICmpInst(NewLeaf, ICmpInst::ICMP_EQ, Val, Leaf.Low, "SwitchLeaf");
  } else {
    // Make range comparison
    if (Leaf.Low == LowerBound) {
      // Val >= Min && Val <= Hi --> Val <= Hi
      Comp = new ICmpInst(NewLeaf, ICmpInst::ICMP_SLE, Val, Leaf.High,
                          "SwitchLeaf");
    } else if (Leaf.High == UpperBound) {
      // Val <= Max && Val >= Lo --> Val >= Lo
      Comp = new ICmpInst(NewLeaf, ICmpInst::ICMP_SGE, Val, Leaf.Low,
                          "SwitchLeaf");
    } else if (Leaf.Low->isZero()) {
      // Val >= 0 && Val <= Hi --> Val <=u Hi
      Comp = new ICmpInst(NewLeaf, ICmpInst::ICMP_ULE, Val, Leaf.High,
                          "SwitchLeaf");
    } else {
      // Emit V-Lo <=u Hi-Lo
      Constant *NegLo = ConstantExpr::getNeg(Leaf.Low);
      Instruction *Add = BinaryOperator::CreateAdd(
          Val, NegLo, Val->getName() + ".off", NewLeaf);
      Constant *UpperBound = ConstantExpr::getAdd(NegLo, Leaf.High);
      Comp = new ICmpInst(NewLeaf, ICmpInst::ICMP_ULE, Add, UpperBound,
                          "SwitchLeaf");
    }
  }

  // Make the conditional branch...
  BasicBlock *Succ = Leaf.BB;
  auto *Br = CondBrInst::Create(Comp, Succ, Default, NewLeaf);
  if (Profile) {
    // True edge is the matched case. The false edge starts at MissWeight:
    // the default count when one case range remains, otherwise 0.
    // UniqueMiss adds that count when this is the one miss leaf. Default
    // cases in this leaf's range are added on top.
    //
    //   %v = and i32 %x, 3
    //   switch i32 %v, label %def [
    //     i32 0, label %a    ; 10
    //     i32 1, label %b    ; 20
    //     i32 3, label %def  ; 5
    //   ], !prof !{!"branch_weights", i32 100, i32 10, i32 20, i32 5}
    //
    // Case 0 is covered, so only the compare for 1 can miss. Two case ranges
    // remain, so value 3 stays a default case and the default count is
    // UniqueMiss. At that leaf:
    //   true  = 20
    //   false = 0 + 100 + 5 = 105
    uint64_t Miss = Profile->MissWeight;
    if (Profile->UniqueMiss && Leaf.Low == Profile->UniqueMiss->Low)
      Miss = SaturatingAdd(Miss, Profile->UniqueMiss->Weight);
    if (!Profile->DefaultCases.empty())
      Miss = SaturatingAdd(Miss, sumDefaultCases(Profile->DefaultCases,
                                                 LowerBound->getValue(),
                                                 UpperBound->getValue()));

    fitAndSetBranchWeights(*Br, {Leaf.Weight, Miss}, Profile->IsExpected);
  } else if (MarkUnknown) {
    // A switch has a single default edge, but after lowering its executions
    // may contribute to multiple comparison-tree branches. If those counts
    // cannot be redistributed reliably, preserve the profile but mark the
    // generated branch weights as unknown.
    setExplicitlyUnknownBranchWeightsIfProfiled(*Br, DEBUG_TYPE);
  }

  // Update the PHI incoming value/block for the default.
  for (auto &I : Default->phis()) {
    PHINode *PN = cast<PHINode>(&I);
    auto *V = PN->getIncomingValueForBlock(OrigBlock);
    PN->addIncoming(V, NewLeaf);
  }

  // If there were any PHI nodes in this successor, rewrite one entry
  // from OrigBlock to come from NewLeaf.
  for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    // Remove all but one incoming entries from the cluster
    APInt Range = Leaf.High->getValue() - Leaf.Low->getValue();
    for (APInt j(Range.getBitWidth(), 0, false); j.ult(Range); ++j) {
      PN->removeIncomingValue(OrigBlock);
    }

    int BlockIdx = PN->getBasicBlockIndex(OrigBlock);
    assert(BlockIdx != -1 && "Switch didn't go to this successor??");
    PN->setIncomingBlock((unsigned)BlockIdx, NewLeaf);
  }

  return NewLeaf;
}

/// Upper bound of the values that fall left of \p PivotLow.
static ConstantInt *upperBeforePivot(ConstantInt *PivotLow,
                                     const CaseRange &PrevCase,
                                     const std::vector<IntRange> &Unreachable) {
  ConstantInt *LeftHigh =
      ConstantInt::get(PivotLow->getContext(), PivotLow->getValue() - 1);
  if (!Unreachable.empty()) {
    APInt GapLow = PrevCase.High->getValue() + 1;
    APInt GapHigh = PivotLow->getValue() - 1;
    IntRange Gap = {GapLow, GapHigh};
    if (GapHigh.sge(GapLow) && IsInRanges(Gap, Unreachable))
      LeftHigh = PrevCase.High;
  }
  return LeftHigh;
}

/// One node of the balanced case tree. The split is recorded here, and both
/// the miss count and the branch builder walk this same tree.
struct SwitchTree {
  CaseItr Begin;
  CaseItr End;
  ConstantInt *Lower = nullptr;
  ConstantInt *Upper = nullptr;
  // Parent bounds already cover this case, so the leaf needs no compare.
  bool Covered = false;
  ConstantInt *Pivot = nullptr;
  ConstantInt *LeftHigh = nullptr;
  // Both null on a leaf. An internal node has both.
  std::unique_ptr<SwitchTree> Left;
  std::unique_ptr<SwitchTree> Right;
};

/// Record the balanced compare splits for [Begin, End).
static std::unique_ptr<SwitchTree>
buildSwitchTree(CaseItr Begin, CaseItr End, ConstantInt *Lower,
                ConstantInt *Upper, const std::vector<IntRange> &Unreachable) {
  auto Node = std::make_unique<SwitchTree>();
  Node->Begin = Begin;
  Node->End = End;
  Node->Lower = Lower;
  Node->Upper = Upper;
  unsigned Size = End - Begin;
  if (Size == 1) {
    Node->Covered = Begin->Low == Lower && Begin->High == Upper;
    return Node;
  }

  unsigned Mid = Size / 2;
  // The pivot case is never the lowest, so subtracting one below is safe.
  Node->Pivot = (Begin + Mid)->Low;
  Node->LeftHigh =
      upperBeforePivot(Node->Pivot, *(Begin + Mid - 1), Unreachable);
  LLVM_DEBUG({
    std::vector<CaseRange> LHS(Begin, Begin + Mid);
    std::vector<CaseRange> RHS(Begin + Mid, End);
    dbgs() << "LHS: " << LHS << "\n"
           << "RHS: " << RHS << "\n"
           << "Pivot ==> [" << Node->Pivot->getValue() << ", "
           << (Begin + Mid)->High->getValue() << "]\n"
           << "LHS Bounds ==> [" << Lower->getValue() << ", "
           << Node->LeftHigh->getValue() << "]\n"
           << "RHS Bounds ==> [" << Node->Pivot->getValue() << ", "
           << Upper->getValue() << "]\n";
  });
  Node->Left =
      buildSwitchTree(Begin, Begin + Mid, Lower, Node->LeftHigh, Unreachable);
  Node->Right =
      buildSwitchTree(Begin + Mid, End, Node->Pivot, Upper, Unreachable);
  assert(Node->Left && Node->Right && "internal node missing a child");
  return Node;
}

/// Count leaves that still compare against the default.
/// If exactly one miss leaf exists, \p OnlyLow is set to that leaf's low value.
static void countDefaultLeaves(const SwitchTree &Node, unsigned &Misses,
                               ConstantInt *&OnlyLow) {
  if (!Node.Left) {
    if (Node.Covered)
      return;
    // Keep OnlyLow only while this is the sole miss leaf.
    if (++Misses == 1)
      OnlyLow = Node.Begin->Low;
    else
      OnlyLow = nullptr;
    return;
  }
  countDefaultLeaves(*Node.Left, Misses, OnlyLow);
  countDefaultLeaves(*Node.Right, Misses, OnlyLow);
}

/// Convert the switch statement into a binary lookup of the case values.
/// \p Node is the tree from buildSwitchTree. Its bounds are the values already
/// checked by a block emitted higher in the call stack. \p Profile is set when
/// the tree keeps a numeric !prof.
BasicBlock *SwitchConvert(const SwitchTree &Node, Value *Val,
                          BasicBlock *Predecessor, BasicBlock *OrigBlock,
                          BasicBlock *Default,
                          const std::optional<SwitchBranchWeights> &Profile,
                          bool MarkUnknown) {
  assert(Node.Lower && Node.Upper && "Bounds must be initialized");

  if (!Node.Left) {
    // A covered range is already checked by the parent bounds, so the case
    // block is reached directly.
    if (Node.Covered) {
      APInt NumMergedCases = Node.Upper->getValue() - Node.Lower->getValue();
      FixPhis(Node.Begin->BB, OrigBlock, Predecessor, NumMergedCases);
      return Node.Begin->BB;
    }
    return NewLeafBlock(*Node.Begin, Val, Node.Lower, Node.Upper, OrigBlock,
                        Default, Profile, MarkUnknown);
  }

  // Create a new node that checks if the value is < pivot. Go to the
  // left branch if it is and right branch if not.
  Function *F = OrigBlock->getParent();
  BasicBlock *NewNode = BasicBlock::Create(Val->getContext(), "NodeBlock");
  ICmpInst *Comp = new ICmpInst(ICmpInst::ICMP_SLT, Val, Node.Pivot, "Pivot");

  BasicBlock *LBranch = SwitchConvert(*Node.Left, Val, NewNode, OrigBlock,
                                      Default, Profile, MarkUnknown);
  BasicBlock *RBranch = SwitchConvert(*Node.Right, Val, NewNode, OrigBlock,
                                      Default, Profile, MarkUnknown);

  F->insert(++OrigBlock->getIterator(), NewNode);
  Comp->insertInto(NewNode, NewNode->end());

  auto *Br = CondBrInst::Create(Comp, LBranch, RBranch, NewNode);

  if (Profile) {
    // Each side is its case weights, plus default cases in that range, plus
    // UniqueMiss when that side contains the miss leaf.
    //
    //   %v = and i32 %x, 3
    //   switch i32 %v, label %def [
    //     i32 0, label %a    ; 10
    //     i32 1, label %b    ; 20
    //     i32 3, label %def  ; 5
    //   ], !prof !{!"branch_weights", i32 100, i32 10, i32 20, i32 5}
    //
    // Pivot is slt 1. Case 0 is covered, so the default count 100 and
    // value 3 both fall on the right:
    //   left  = 10
    //   right = 20 + 5 + 100 = 125
    CaseItr Mid = Node.Right->Begin;

    uint64_t LWeight = sumCaseWeights(Node.Begin, Mid);
    uint64_t RWeight = sumCaseWeights(Mid, Node.End);

    if (!Profile->DefaultCases.empty()) {
      // Add default-case counts that belong to each child's value range.
      LWeight =
          SaturatingAdd(LWeight, sumDefaultCases(Profile->DefaultCases,
                                                 Node.Lower->getValue(),
                                                 Node.LeftHigh->getValue()));
      RWeight = SaturatingAdd(RWeight, sumDefaultCases(Profile->DefaultCases,
                                                       Node.Pivot->getValue(),
                                                       Node.Upper->getValue()));
    }

    if (Profile->UniqueMiss) {
      ConstantInt *EdgeLow = Profile->UniqueMiss->Low;
      auto ContainsEdge = [&](CaseItr B, CaseItr E) {
        return any_of(iterator_range(B, E),
                      [&](const CaseRange &R) { return R.Low == EdgeLow; });
      };
      // A UniqueMiss count belongs to exactly one child. Add it to the
      // subtree containing the corresponding case.
      if (ContainsEdge(Node.Begin, Mid))
        LWeight = SaturatingAdd(LWeight, Profile->UniqueMiss->Weight);
      else if (ContainsEdge(Mid, Node.End))
        RWeight = SaturatingAdd(RWeight, Profile->UniqueMiss->Weight);
    }

    fitAndSetBranchWeights(*Br, {LWeight, RWeight}, Profile->IsExpected);
  } else if (MarkUnknown) {
    // Several miss leaves and a nonzero default, so the count is not split.
    setExplicitlyUnknownBranchWeightsIfProfiled(*Br, DEBUG_TYPE);
  }
  return NewNode;
}

/// Cases with the default removed, plus the weights gathered from the switch.
struct ClusteredSwitch {
  CaseVector Cases;
  unsigned NumSimpleCases = 0;
  // branch_weights was present, with one count per successor.
  bool HadWeights = false;
  // Numeric !prof is still kept on the lowered compares. Cleared when the
  // default count cannot be placed. HadWeights stays set.
  bool HasProfile = false;
  // Original default count before the tree exists. One case range stores it
  // in Weights.MissWeight. One miss leaf moves it to UniqueMiss. Several
  // miss leaves drop the numeric profile.
  //
  //   switch i32 %x, label %def [
  //     i32 1, label %a
  //     i32 2, label %b
  //   ], !prof !{!"branch_weights", i32 10, i32 20, i32 30}
  //
  // Both leaves can miss, so this 10 is not split.
  uint64_t DefaultWeight = 0;
  SwitchBranchWeights Weights;
};

/// Transform simple list of \p SI's cases into list of CaseRange's.
/// The result cases do not reference \p SI's default block.
static ClusteredSwitch Clusterify(SwitchInst *SI) {
  ClusteredSwitch Result;
  SmallVector<uint64_t> Weights;
  MDNode *ProfileData = getBranchWeightMDNode(*SI);
  bool HaveWeights = ProfileData && isBranchWeightMD(ProfileData);
  if (HaveWeights) {
    extractFromBranchWeightMD64(ProfileData, Weights);
    HaveWeights = Weights.size() == SI->getNumSuccessors();
  }
  Result.HadWeights = HaveWeights;

  // Start with "simple" cases
  for (auto Case : SI->cases()) {
    if (Case.getCaseSuccessor() == SI->getDefaultDest()) {
      // This case already targets the default. Its value picks the miss edge.
      //
      //   i32 4, label %def   ; DefaultCase {Value: 4, Weight: 15}
      if (HaveWeights)
        Result.Weights.DefaultCases.push_back(
            {Case.getCaseValue()->getValue(),
             Weights[Case.getSuccessorIndex()]});
      continue;
    }
    uint64_t Weight = HaveWeights ? Weights[Case.getSuccessorIndex()] : 0;
    Result.Cases.push_back(CaseRange(Case.getCaseValue(), Case.getCaseValue(),
                                     Case.getCaseSuccessor(), Weight));
    ++Result.NumSimpleCases;
  }

  llvm::sort(Result.Cases, CaseCmp());

  // Merge case into clusters
  if (Result.Cases.size() >= 2) {
    CaseItr I = Result.Cases.begin();
    for (CaseItr J = std::next(I), E = Result.Cases.end(); J != E; ++J) {
      const APInt &nextValue = J->Low->getValue();
      const APInt &currentValue = I->High->getValue();
      BasicBlock *nextBB = J->BB;
      BasicBlock *currentBB = I->BB;

      // If the two neighboring cases go to the same destination, merge them
      // into a single case.
      assert(nextValue.sgt(currentValue) &&
             "Cases should be strictly ascending");
      if ((nextValue == currentValue + 1) && (currentBB == nextBB)) {
        I->High = J->High;
        I->Weight = SaturatingAdd(I->Weight, J->Weight);
      } else if (++I != J) {
        *I = *J;
      }
    }
    Result.Cases.erase(std::next(I), Result.Cases.end());
  }

  // Weights[0] is the original default edge count.
  if (HaveWeights && SI->defaultDestUnreachable()) {
    // Cases that target this block cannot fire.
    Result.Weights.DefaultCases.clear();
    Result.HasProfile = true;
  } else if (HaveWeights && Result.Cases.size() <= 1) {
    // One miss edge, so it keeps the default count.
    Result.Weights.MissWeight = Weights[0];
    Result.HasProfile = true;
  } else if (HaveWeights && Weights[0] == 0) {
    // Nothing to split across the miss edges.
    Result.HasProfile = true;
  } else if (HaveWeights) {
    // Hold the count until the tree shows exactly one miss leaf.
    Result.DefaultWeight = Weights[0];
    Result.HasProfile = true;
  }

  return Result;
}

/// Replace the specified switch instruction with a sequence of chained if-then
/// insts in a balanced binary search.
void ProcessSwitchInst(SwitchInst *SI,
                       SmallPtrSetImpl<BasicBlock *> &DeleteList,
                       AssumptionCache *AC, LazyValueInfo *LVI) {
  BasicBlock *OrigBlock = SI->getParent();
  Function *F = OrigBlock->getParent();
  Value *Val = SI->getCondition(); // The value we are switching on...
  BasicBlock *Default = SI->getDefaultDest();

  // Don't handle unreachable blocks. If there are successors with phis, this
  // would leave them behind with missing predecessors.
  if ((OrigBlock != &F->getEntryBlock() && pred_empty(OrigBlock)) ||
      OrigBlock->getSinglePredecessor() == OrigBlock) {
    DeleteList.insert(OrigBlock);
    return;
  }

  ClusteredSwitch Cluster = Clusterify(SI);
  CaseVector &Cases = Cluster.Cases;
  bool &HasProfile = Cluster.HasProfile;
  uint64_t &MissWeight = Cluster.Weights.MissWeight;
  uint64_t &DefaultWeight = Cluster.DefaultWeight;
  const bool HadWeights = Cluster.HadWeights;
  SmallVector<DefaultCase, 8> &DefaultCases = Cluster.Weights.DefaultCases;
  const unsigned NumSimpleCases = Cluster.NumSimpleCases;
  IntegerType *IT = cast<IntegerType>(SI->getCondition()->getType());
  const unsigned BitWidth = IT->getBitWidth();
  // Explicitly use higher precision to prevent unsigned overflow where
  // `UnsignedMax - 0 + 1 == 0`
  APInt UnsignedZero(BitWidth + 1, 0);
  APInt UnsignedMax = APInt::getMaxValue(BitWidth);
  LLVM_DEBUG(dbgs() << "Clusterify finished. Total clusters: " << Cases.size()
                    << ". Total non-default cases: " << NumSimpleCases
                    << "\nCase clusters: " << Cases << "\n");

  // If there is only the default destination, just branch.
  if (Cases.empty()) {
    UncondBrInst::Create(Default, OrigBlock);
    // Remove all the references from Default's PHIs to OrigBlock, but one.
    FixPhis(Default, OrigBlock, OrigBlock, UnsignedMax);
    SI->eraseFromParent();
    return;
  }

  ConstantInt *LowerBound = nullptr;
  ConstantInt *UpperBound = nullptr;
  bool DefaultIsUnreachableFromSwitch = false;
  // Known bits are filled in only when the default is reachable.
  KnownBits Known(BitWidth);

  if (SI->defaultDestUnreachable()) {
    // Make the bounds tightly fitted around the case value range, because we
    // know that the value passed to the switch must be exactly one of the case
    // values.
    LowerBound = Cases.front().Low;
    UpperBound = Cases.back().High;
    DefaultIsUnreachableFromSwitch = true;
  } else {
    // Constraining the range of the value being switched over helps eliminating
    // unreachable BBs and minimizing the number of `add` instructions
    // newLeafBlock ends up emitting. Running CorrelatedValuePropagation after
    // LowerSwitch isn't as good, and also much more expensive in terms of
    // compile time for the following reasons:
    // 1. it processes many kinds of instructions, not just switches;
    // 2. even if limited to icmp instructions only, it will have to process
    //    roughly C icmp's per switch, where C is the number of cases in the
    //    switch, while LowerSwitch only needs to call LVI once per switch.
    const DataLayout &DL = F->getDataLayout();
    Known = computeKnownBits(Val, DL, AC, SI);
    // TODO Shouldn't this create a signed range?
    ConstantRange KnownBitsRange =
        ConstantRange::fromKnownBits(Known, /*IsSigned=*/false);
    const ConstantRange LVIRange =
        LVI->getConstantRange(Val, SI, /*UndefAllowed*/ false);
    ConstantRange ValRange = KnownBitsRange.intersectWith(LVIRange);
    // We delegate removal of unreachable non-default cases to other passes. In
    // the unlikely event that some of them survived, we just conservatively
    // maintain the invariant that all the cases lie between the bounds. This
    // may, however, still render the default case effectively unreachable.
    const APInt &Low = Cases.front().Low->getValue();
    const APInt &High = Cases.back().High->getValue();
    APInt Min = APIntOps::smin(ValRange.getSignedMin(), Low);
    APInt Max = APIntOps::smax(ValRange.getSignedMax(), High);

    LowerBound = ConstantInt::get(SI->getContext(), Min);
    UpperBound = ConstantInt::get(SI->getContext(), Max);
    DefaultIsUnreachableFromSwitch = (Min + (NumSimpleCases - 1) == Max);
  }

  // Remove a default-targeting value the switch cannot reach. That count
  // never lands on a miss edge.
  //
  // Example:
  //
  //   switch %v, label %def [
  //     i32 0, label %zero
  //     i32 1, label %one
  //     i32 4, label %def
  //   ], !prof !{!"branch_weights", 0, 20, 30, 15}
  //
  // If %v is known to be in [0, 3], value 4 is impossible, so its count (15)
  // cannot contribute to any miss edge and is dropped.
  //
  // When only one case range remains:
  //
  //   switch %v, label %def [
  //     i32 0, label %one
  //     i32 4, label %def
  //   ], !prof !{!"branch_weights", i32 7, i32 20, i32 15}
  //
  // the lowered compare has a single miss edge:
  //
  //   %cmp = icmp eq i32 %v, 0
  //   br i1 %cmp, label %one, label %def
  //
  // If 4 can fire, its 15 is added to the default count 7 and DefaultCases
  // is cleared, so the miss is 22 and is not counted again. A value outside
  // the range is dropped.
  if (HasProfile && !DefaultCases.empty()) {
    auto CannotReach = [&](const DefaultCase &Case) {
      return Case.Value.slt(LowerBound->getValue()) ||
             Case.Value.sgt(UpperBound->getValue()) ||
             conflictsWithKnownBits(Case.Value, Known);
    };
    if (Cases.size() <= 1) {
      for (const DefaultCase &Case : DefaultCases)
        if (!CannotReach(Case))
          MissWeight = SaturatingAdd(MissWeight, Case.Weight);
      // Now part of MissWeight. Later range queries must not see them.
      DefaultCases.clear();
    } else {
      llvm::erase_if(DefaultCases, CannotReach);
    }
  }

  std::vector<IntRange> UnreachableRanges;

  // Fold the successor with the most values into the default. Its weight
  // moves with those cases. A dead default count stays 0.
  if (DefaultIsUnreachableFromSwitch) {
    DefaultWeight = 0;
    DenseMap<BasicBlock *, APInt> Popularity;
    APInt MaxPop(UnsignedZero);
    BasicBlock *PopSucc = nullptr;

    APInt SignedMax = APInt::getSignedMaxValue(BitWidth);
    APInt SignedMin = APInt::getSignedMinValue(BitWidth);
    IntRange R = {SignedMin, SignedMax};
    UnreachableRanges.push_back(R);
    for (const auto &I : Cases) {
      const APInt &Low = I.Low->getValue();
      const APInt &High = I.High->getValue();

      IntRange &LastRange = UnreachableRanges.back();
      if (LastRange.Low.eq(Low)) {
        // There is nothing left of the previous range.
        UnreachableRanges.pop_back();
      } else {
        // Terminate the previous range.
        assert(Low.sgt(LastRange.Low));
        LastRange.High = Low - 1;
      }
      if (High.ne(SignedMax)) {
        IntRange R = {High + 1, SignedMax};
        UnreachableRanges.push_back(R);
      }

      // Count popularity.
      assert(High.sge(Low) && "Popularity shouldn't be negative.");
      APInt N = High.sext(BitWidth + 1) - Low.sext(BitWidth + 1) + 1;
      // Explict insert to make sure the bitwidth of APInts match
      APInt &Pop = Popularity.insert({I.BB, APInt(UnsignedZero)}).first->second;
      if ((Pop += N).ugt(MaxPop)) {
        MaxPop = Pop;
        PopSucc = I.BB;
      }
    }
#ifndef NDEBUG
    /* UnreachableRanges should be sorted and the ranges non-adjacent. */
    for (auto I = UnreachableRanges.begin(), E = UnreachableRanges.end();
         I != E; ++I) {
      assert(I->Low.sle(I->High));
      auto Next = I + 1;
      if (Next != E) {
        assert(Next->Low.sgt(I->High));
      }
    }
#endif

    // As the default block in the switch is unreachable, update the PHI nodes
    // (remove all of the references to the default block) to reflect this.
    const unsigned NumDefaultEdges = SI->getNumCases() + 1 - NumSimpleCases;
    for (unsigned I = 0; I < NumDefaultEdges; ++I)
      Default->removePredecessor(OrigBlock);

    // Use the most popular block as the new default, reducing the number of
    // cases. Keep each removed cluster's weight for the edge that now reaches
    // it.
    if (HasProfile) {
      for (const CaseRange &R : Cases)
        if (R.BB == PopSucc)
          // Clusters do not overlap, and every pivot is a kept case's Low, so
          // the whole cluster takes the same side as Low.
          //
          //   switch i32 %x, label %unreachable [
          //     i32 1, label %one     ; 20, peeled
          //     i32 2, label %two     ; 30
          //     i32 3, label %three   ; 50
          //   ], !prof !{!"branch_weights", i32 0, i32 20, i32 30, i32 50}
          //
          // Case 1 becomes DefaultCase {1, 20}. The leaf for 2 misses to
          // %one with that 20. DefaultWeight was cleared, so 0 is not added.
          DefaultCases.push_back({R.Low->getValue(), R.Weight});
    }
    Default = PopSucc;
    llvm::erase_if(Cases,
                   [PopSucc](const CaseRange &R) { return R.BB == PopSucc; });

    // If there are no cases left, just branch.
    if (Cases.empty()) {
      UncondBrInst::Create(Default, OrigBlock);
      SI->eraseFromParent();
      // As all the cases have been replaced with a single branch, only keep
      // one entry in the PHI nodes.
      if (!MaxPop.isZero())
        for (APInt I(UnsignedZero); I.ult(MaxPop - 1); ++I)
          PopSucc->removePredecessor(OrigBlock);
      return;
    }

    // If the condition was a PHI node with the switch block as a predecessor
    // removing predecessors may have caused the condition to be erased.
    // Getting the condition value again here protects against that.
    Val = SI->getCondition();
  }

  std::unique_ptr<SwitchTree> Tree = buildSwitchTree(
      Cases.begin(), Cases.end(), LowerBound, UpperBound, UnreachableRanges);

  ConstantInt *MissLow = nullptr;
  if (HasProfile && DefaultWeight != 0) {
    unsigned Misses = 0;
    ConstantInt *OnlyLow = nullptr;
    countDefaultLeaves(*Tree, Misses, OnlyLow);
    if (Misses == 1)
      MissLow = OnlyLow;
    else
      // Several miss leaves would need an invented split of the default count.
      // Keep only that profile data existed.
      HasProfile = false;
  }
  // Weights existed, and a numeric !prof was not kept.
  const bool MarkUnknown = HadWeights && !HasProfile;
  std::optional<SwitchBranchWeights> Profile;
  if (HasProfile) {
    // One signed sort, after folded clusters are appended and dead cases
    // are removed. Later range queries binary-search this order.
    llvm::sort(DefaultCases, [](const DefaultCase &A, const DefaultCase &B) {
      return A.Value.slt(B.Value);
    });
    Cluster.Weights.IsExpected = hasBranchWeightOrigin(*SI);
    if (MissLow)
      // DefaultWeight moves onto the one remaining miss leaf.
      //
      //   %v = and i32 %x, 3
      //   switch i32 %v, label %def [
      //     i32 0, label %a
      //     i32 1, label %b
      //   ], !prof !{!"branch_weights", i32 100, i32 10, i32 20}
      //
      // Only the compare for 1 reaches %def:
      // UniqueMiss is {Low: 1, Weight: 100}.
      Cluster.Weights.UniqueMiss = UniqueMissLeaf{MissLow, DefaultWeight};
    Profile = std::move(Cluster.Weights);
  }
  BasicBlock *SwitchBlock = SwitchConvert(*Tree, Val, OrigBlock, OrigBlock,
                                          Default, Profile, MarkUnknown);

  // We have added incoming values for newly-created predecessors in
  // NewLeafBlock(). The only meaningful work we offload to FixPhis() is to
  // remove the incoming values from OrigBlock. There might be a special case
  // that SwitchBlock is the same as Default, under which the PHIs in Default
  // are fixed inside SwitchConvert().
  if (SwitchBlock != Default)
    FixPhis(Default, OrigBlock, nullptr, UnsignedMax);

  // Branch to our shiny new if-then stuff...
  UncondBrInst::Create(SwitchBlock, OrigBlock);

  // We are now done with the switch instruction, delete it.
  BasicBlock *OldDefault = SI->getDefaultDest();
  SI->eraseFromParent();

  // If the Default block has no more predecessors just add it to DeleteList.
  if (pred_empty(OldDefault))
    DeleteList.insert(OldDefault);
}

bool LowerSwitch(Function &F, LazyValueInfo *LVI, AssumptionCache *AC) {
  bool Changed = false;
  SmallPtrSet<BasicBlock *, 8> DeleteList;

  // We use make_early_inc_range here so that we don't traverse new blocks.
  for (BasicBlock &Cur : llvm::make_early_inc_range(F)) {
    // If the block is a dead Default block that will be deleted later, don't
    // waste time processing it.
    if (DeleteList.count(&Cur))
      continue;

    if (SwitchInst *SI = dyn_cast<SwitchInst>(Cur.getTerminator())) {
      Changed = true;
      ProcessSwitchInst(SI, DeleteList, AC, LVI);
    }
  }

  for (BasicBlock *BB : DeleteList) {
    LVI->eraseBlock(BB);
    DeleteDeadBlock(BB);
  }

  return Changed;
}

/// Replace all SwitchInst instructions with chained branch instructions.
class LowerSwitchLegacyPass : public FunctionPass {
public:
  // Pass identification, replacement for typeid
  static char ID;

  LowerSwitchLegacyPass() : FunctionPass(ID) {
    initializeLowerSwitchLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LazyValueInfoWrapperPass>();
  }
};

} // end anonymous namespace

char LowerSwitchLegacyPass::ID = 0;

// Publicly exposed interface to pass...
char &llvm::LowerSwitchID = LowerSwitchLegacyPass::ID;

INITIALIZE_PASS_BEGIN(LowerSwitchLegacyPass, "lowerswitch",
                      "Lower SwitchInst's to branches", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LazyValueInfoWrapperPass)
INITIALIZE_PASS_END(LowerSwitchLegacyPass, "lowerswitch",
                    "Lower SwitchInst's to branches", false, false)

// createLowerSwitchPass - Interface to this file...
FunctionPass *llvm::createLowerSwitchPass() {
  return new LowerSwitchLegacyPass();
}

bool LowerSwitchLegacyPass::runOnFunction(Function &F) {
  LazyValueInfo *LVI = &getAnalysis<LazyValueInfoWrapperPass>().getLVI();
  auto *ACT = getAnalysisIfAvailable<AssumptionCacheTracker>();
  AssumptionCache *AC = ACT ? &ACT->getAssumptionCache(F) : nullptr;
  return LowerSwitch(F, LVI, AC);
}

PreservedAnalyses LowerSwitchPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  LazyValueInfo *LVI = &AM.getResult<LazyValueAnalysis>(F);
  AssumptionCache *AC = AM.getCachedResult<AssumptionAnalysis>(F);
  return LowerSwitch(F, LVI, AC) ? PreservedAnalyses::none()
                                 : PreservedAnalyses::all();
}
