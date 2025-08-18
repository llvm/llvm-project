//===- StructurizeCFG.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <cassert>
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "structurizecfg"

// The name for newly created blocks.
const char FlowBlockName[] = "Flow";

namespace {

static cl::opt<bool> ForceSkipUniformRegions(
  "structurizecfg-skip-uniform-regions",
  cl::Hidden,
  cl::desc("Force whether the StructurizeCFG pass skips uniform regions"),
  cl::init(false));

static cl::opt<bool>
    RelaxedUniformRegions("structurizecfg-relaxed-uniform-regions", cl::Hidden,
                          cl::desc("Allow relaxed uniform region checks"),
                          cl::init(true));

// Definition of the complex types used in this pass.

using BBValuePair = std::pair<BasicBlock *, Value *>;

using RNVector = SmallVector<RegionNode *, 8>;
using BBVector = SmallVector<BasicBlock *, 8>;
using BranchVector = SmallVector<BranchInst *, 8>;
using BBValueVector = SmallVector<BBValuePair, 2>;

using BBSet = SmallPtrSet<BasicBlock *, 8>;

using PhiMap = MapVector<PHINode *, BBValueVector>;
using BB2BBVecMap = MapVector<BasicBlock *, BBVector>;

using BBPhiMap = DenseMap<BasicBlock *, PhiMap>;

using MaybeCondBranchWeights = std::optional<class CondBranchWeights>;

class CondBranchWeights {
  uint32_t TrueWeight;
  uint32_t FalseWeight;

  CondBranchWeights(uint32_t T, uint32_t F) : TrueWeight(T), FalseWeight(F) {}

public:
  static MaybeCondBranchWeights tryParse(const BranchInst &Br) {
    assert(Br.isConditional());

    uint64_t T, F;
    if (!extractBranchWeights(Br, T, F))
      return std::nullopt;

    return CondBranchWeights(T, F);
  }

  static void setMetadata(BranchInst &Br,
                          const MaybeCondBranchWeights &Weights) {
    assert(Br.isConditional());
    if (!Weights)
      return;
    uint32_t Arr[] = {Weights->TrueWeight, Weights->FalseWeight};
    setBranchWeights(Br, Arr, false);
  }

  CondBranchWeights invert() const {
    return CondBranchWeights{FalseWeight, TrueWeight};
  }
};

struct PredInfo {
  Value *Pred;
  MaybeCondBranchWeights Weights;
};

using BBPredicates = DenseMap<BasicBlock *, PredInfo>;
using PredMap = DenseMap<BasicBlock *, BBPredicates>;
using BB2BBMap = DenseMap<BasicBlock *, BasicBlock *>;
using Val2BBMap = DenseMap<Value *, BasicBlock *>;

// A traits type that is intended to be used in graph algorithms. The graph
// traits starts at an entry node, and traverses the RegionNodes that are in
// the Nodes set.
struct SubGraphTraits {
  using NodeRef = std::pair<RegionNode *, SmallDenseSet<RegionNode *> *>;
  using BaseSuccIterator = GraphTraits<RegionNode *>::ChildIteratorType;

  // This wraps a set of Nodes into the iterator, so we know which edges to
  // filter out.
  class WrappedSuccIterator
      : public iterator_adaptor_base<
            WrappedSuccIterator, BaseSuccIterator,
            typename std::iterator_traits<BaseSuccIterator>::iterator_category,
            NodeRef, std::ptrdiff_t, NodeRef *, NodeRef> {
    SmallDenseSet<RegionNode *> *Nodes;

  public:
    WrappedSuccIterator(BaseSuccIterator It, SmallDenseSet<RegionNode *> *Nodes)
        : iterator_adaptor_base(It), Nodes(Nodes) {}

    NodeRef operator*() const { return {*I, Nodes}; }
  };

  static bool filterAll(const NodeRef &N) { return true; }
  static bool filterSet(const NodeRef &N) { return N.second->count(N.first); }

  using ChildIteratorType =
      filter_iterator<WrappedSuccIterator, bool (*)(const NodeRef &)>;

  static NodeRef getEntryNode(Region *R) {
    return {GraphTraits<Region *>::getEntryNode(R), nullptr};
  }

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static iterator_range<ChildIteratorType> children(const NodeRef &N) {
    auto *filter = N.second ? &filterSet : &filterAll;
    return make_filter_range(
        make_range<WrappedSuccIterator>(
            {GraphTraits<RegionNode *>::child_begin(N.first), N.second},
            {GraphTraits<RegionNode *>::child_end(N.first), N.second}),
        filter);
  }

  static ChildIteratorType child_begin(const NodeRef &N) {
    return children(N).begin();
  }

  static ChildIteratorType child_end(const NodeRef &N) {
    return children(N).end();
  }
};

/// Finds the nearest common dominator of a set of BasicBlocks.
///
/// For every BB you add to the set, you can specify whether we "remember" the
/// block.  When you get the common dominator, you can also ask whether it's one
/// of the blocks we remembered.
class NearestCommonDominator {
  DominatorTree *DT;
  BasicBlock *Result = nullptr;
  bool ResultIsRemembered = false;

  /// Add BB to the resulting dominator.
  void addBlock(BasicBlock *BB, bool Remember) {
    if (!Result) {
      Result = BB;
      ResultIsRemembered = Remember;
      return;
    }

    BasicBlock *NewResult = DT->findNearestCommonDominator(Result, BB);
    if (NewResult != Result)
      ResultIsRemembered = false;
    if (NewResult == BB)
      ResultIsRemembered |= Remember;
    Result = NewResult;
  }

public:
  explicit NearestCommonDominator(DominatorTree *DomTree) : DT(DomTree) {}

  void addBlock(BasicBlock *BB) {
    addBlock(BB, /* Remember = */ false);
  }

  void addAndRememberBlock(BasicBlock *BB) {
    addBlock(BB, /* Remember = */ true);
  }

  /// Get the nearest common dominator of all the BBs added via addBlock() and
  /// addAndRememberBlock().
  BasicBlock *result() { return Result; }

  /// Is the BB returned by getResult() one of the blocks we added to the set
  /// with addAndRememberBlock()?
  bool resultIsRememberedBlock() { return ResultIsRemembered; }
};

/// Transforms the control flow graph on one single entry/exit region
/// at a time.
///
/// After the transform all "If"/"Then"/"Else" style control flow looks like
/// this:
///
/// \verbatim
/// 1
/// ||
/// | |
/// 2 |
/// | /
/// |/
/// 3
/// ||   Where:
/// | |  1 = "If" block, calculates the condition
/// 4 |  2 = "Then" subregion, runs if the condition is true
/// | /  3 = "Flow" blocks, newly inserted flow blocks, rejoins the flow
/// |/   4 = "Else" optional subregion, runs if the condition is false
/// 5    5 = "End" block, also rejoins the control flow
/// \endverbatim
///
/// Control flow is expressed as a branch where the true exit goes into the
/// "Then"/"Else" region, while the false exit skips the region
/// The condition for the optional "Else" region is expressed as a PHI node.
/// The incoming values of the PHI node are true for the "If" edge and false
/// for the "Then" edge.
///
/// Additionally to that even complicated loops look like this:
///
/// \verbatim
/// 1
/// ||
/// | |
/// 2 ^  Where:
/// | /  1 = "Entry" block
/// |/   2 = "Loop" optional subregion, with all exits at "Flow" block
/// 3    3 = "Flow" block, with back edge to entry block
/// |
/// \endverbatim
///
/// The back edge of the "Flow" block is always on the false side of the branch
/// while the true side continues the general flow. So the loop condition
/// consist of a network of PHI nodes where the true incoming values expresses
/// breaks and the false values expresses continue states.

class StructurizeCFG {
  Type *Boolean;
  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;
  Value *BoolPoison;
  const TargetTransformInfo *TTI;
  Function *Func;
  Region *ParentRegion;

  UniformityInfo *UA = nullptr;
  DominatorTree *DT;

  SmallVector<RegionNode *, 8> Order;
  BBSet Visited;
  BBSet FlowSet;

  SmallVector<WeakVH, 8> AffectedPhis;
  BBPhiMap DeletedPhis;
  BB2BBVecMap AddedPhis;

  PredMap Predicates;
  BranchVector Conditions;

  BB2BBMap Loops;
  PredMap LoopPreds;
  BranchVector LoopConds;

  Val2BBMap HoistedValues;

  RegionNode *PrevNode;

  void hoistZeroCostElseBlockPhiValues(BasicBlock *ElseBB, BasicBlock *ThenBB);

  void orderNodes();

  void analyzeLoops(RegionNode *N);

  PredInfo buildCondition(BranchInst *Term, unsigned Idx, bool Invert);

  void gatherPredicates(RegionNode *N);

  void collectInfos();

  void insertConditions(bool Loops);

  void simplifyConditions();

  void delPhiValues(BasicBlock *From, BasicBlock *To);

  void addPhiValues(BasicBlock *From, BasicBlock *To);

  void findUndefBlocks(BasicBlock *PHIBlock,
                       const SmallPtrSet<BasicBlock *, 8> &Incomings,
                       SmallVector<BasicBlock *> &UndefBlks) const;

  void mergeIfCompatible(EquivalenceClasses<PHINode *> &PhiClasses, PHINode *A,
                         PHINode *B);

  void setPhiValues();

  void simplifyAffectedPhis();

  void simplifyHoistedPhis();

  DebugLoc killTerminator(BasicBlock *BB);

  void changeExit(RegionNode *Node, BasicBlock *NewExit,
                  bool IncludeDominator);

  BasicBlock *getNextFlow(BasicBlock *Dominator);

  std::pair<BasicBlock *, DebugLoc> needPrefix(bool NeedEmpty);

  BasicBlock *needPostfix(BasicBlock *Flow, bool ExitUseAllowed);

  void setPrevNode(BasicBlock *BB);

  bool dominatesPredicates(BasicBlock *BB, RegionNode *Node);

  bool isPredictableTrue(RegionNode *Node);

  void wireFlow(bool ExitUseAllowed, BasicBlock *LoopEnd);

  void handleLoops(bool ExitUseAllowed, BasicBlock *LoopEnd);

  void createFlow();

  void rebuildSSA();

public:
  void init(Region *R);
  bool run(Region *R, DominatorTree *DT, const TargetTransformInfo *TTI);
  bool makeUniformRegion(Region *R, UniformityInfo &UA);
};

class StructurizeCFGLegacyPass : public RegionPass {
  bool SkipUniformRegions;

public:
  static char ID;

  explicit StructurizeCFGLegacyPass(bool SkipUniformRegions_ = false)
      : RegionPass(ID), SkipUniformRegions(SkipUniformRegions_) {
    if (ForceSkipUniformRegions.getNumOccurrences())
      SkipUniformRegions = ForceSkipUniformRegions.getValue();
    initializeStructurizeCFGLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnRegion(Region *R, RGPassManager &RGM) override {
    StructurizeCFG SCFG;
    SCFG.init(R);
    if (SkipUniformRegions) {
      UniformityInfo &UA =
          getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
      if (SCFG.makeUniformRegion(R, UA))
        return false;
    }
    Function *F = R->getEntry()->getParent();
    const TargetTransformInfo *TTI =
        &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*F);
    DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    return SCFG.run(R, DT, TTI);
  }

  StringRef getPassName() const override { return "Structurize control flow"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    if (SkipUniformRegions)
      AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();

    AU.addPreserved<DominatorTreeWrapperPass>();
    RegionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

/// Checks whether an instruction is zero cost instruction and checks if the
/// operands are from different BB. If so, this instruction can be coalesced
/// if its hoisted to predecessor block. So, this returns true.
static bool isHoistableInstruction(Instruction *I, BasicBlock *BB,
                                   const TargetTransformInfo *TTI) {
  if (I->getParent() != BB || isa<PHINode>(I))
    return false;

  // If the instruction is not a zero cost instruction, return false.
  auto Cost = TTI->getInstructionCost(I, TargetTransformInfo::TCK_Latency);
  InstructionCost::CostType CostVal =
      Cost.isValid()
          ? Cost.getValue()
          : (InstructionCost::CostType)TargetTransformInfo::TCC_Expensive;
  if (CostVal != 0)
    return false;

  // Check if any operands are instructions defined in the same block.
  for (auto &Op : I->operands()) {
    if (auto *OpI = dyn_cast<Instruction>(Op)) {
      if (OpI->getParent() == BB)
        return false;
    }
  }

  return true;
}

char StructurizeCFGLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(StructurizeCFGLegacyPass, "structurizecfg",
                      "Structurize the CFG", false, false)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass)
INITIALIZE_PASS_END(StructurizeCFGLegacyPass, "structurizecfg",
                    "Structurize the CFG", false, false)

/// Structurization can introduce unnecessary VGPR copies due to register
/// coalescing interference. For example, if the Else block has a zero-cost
/// instruction and the Then block modifies the VGPR value, only one value is
/// live at a time in merge block before structurization. After structurization,
/// the coalescer may incorrectly treat the Then value as live in the Else block
/// (via the path Then → Flow → Else), leading to unnecessary VGPR copies.
///
/// This function examines phi nodes whose incoming values are zero-cost
/// instructions in the Else block. It identifies such values that can be safely
/// hoisted and moves them to the nearest common dominator of Then and Else
/// blocks. A follow-up function after setting PhiNodes assigns the hoisted
/// value to poison phi nodes along the if→flow edge, aiding register coalescing
/// and minimizing unnecessary live ranges.
void StructurizeCFG::hoistZeroCostElseBlockPhiValues(BasicBlock *ElseBB,
                                                     BasicBlock *ThenBB) {

  BasicBlock *ElseSucc = ElseBB->getSingleSuccessor();
  BasicBlock *CommonDominator = DT->findNearestCommonDominator(ElseBB, ThenBB);

  if (!ElseSucc || !CommonDominator)
    return;
  Instruction *Term = CommonDominator->getTerminator();
  for (PHINode &Phi : ElseSucc->phis()) {
    Value *ElseVal = Phi.getIncomingValueForBlock(ElseBB);
    auto *Inst = dyn_cast<Instruction>(ElseVal);
    if (!Inst || !isHoistableInstruction(Inst, ElseBB, TTI))
      continue;
    Inst->removeFromParent();
    Inst->insertInto(CommonDominator, Term->getIterator());
    HoistedValues[Inst] = CommonDominator;
  }
}

/// Build up the general order of nodes, by performing a topological sort of the
/// parent region's nodes, while ensuring that there is no outer cycle node
/// between any two inner cycle nodes.
void StructurizeCFG::orderNodes() {
  Order.resize(std::distance(GraphTraits<Region *>::nodes_begin(ParentRegion),
                             GraphTraits<Region *>::nodes_end(ParentRegion)));
  if (Order.empty())
    return;

  SmallDenseSet<RegionNode *> Nodes;
  auto EntryNode = SubGraphTraits::getEntryNode(ParentRegion);

  // A list of range indices of SCCs in Order, to be processed.
  SmallVector<std::pair<unsigned, unsigned>, 8> WorkList;
  unsigned I = 0, E = Order.size();
  while (true) {
    // Run through all the SCCs in the subgraph starting with Entry.
    for (auto SCCI =
             scc_iterator<SubGraphTraits::NodeRef, SubGraphTraits>::begin(
                 EntryNode);
         !SCCI.isAtEnd(); ++SCCI) {
      auto &SCC = *SCCI;

      // An SCC up to the size of 2, can be reduced to an entry (the last node),
      // and a possible additional node. Therefore, it is already in order, and
      // there is no need to add it to the work-list.
      unsigned Size = SCC.size();
      if (Size > 2)
        WorkList.emplace_back(I, I + Size);

      // Add the SCC nodes to the Order array.
      for (const auto &N : SCC) {
        assert(I < E && "SCC size mismatch!");
        Order[I++] = N.first;
      }
    }
    assert(I == E && "SCC size mismatch!");

    // If there are no more SCCs to order, then we are done.
    if (WorkList.empty())
      break;

    std::tie(I, E) = WorkList.pop_back_val();

    // Collect the set of nodes in the SCC's subgraph. These are only the
    // possible child nodes; we do not add the entry (last node) otherwise we
    // will have the same exact SCC all over again.
    Nodes.clear();
    Nodes.insert(Order.begin() + I, Order.begin() + E - 1);

    // Update the entry node.
    EntryNode.first = Order[E - 1];
    EntryNode.second = &Nodes;
  }
}

/// Determine the end of the loops
void StructurizeCFG::analyzeLoops(RegionNode *N) {
  if (N->isSubRegion()) {
    // Test for exit as back edge
    BasicBlock *Exit = N->getNodeAs<Region>()->getExit();
    if (Visited.count(Exit))
      Loops[Exit] = N->getEntry();

  } else {
    // Test for successors as back edge
    BasicBlock *BB = N->getNodeAs<BasicBlock>();
    BranchInst *Term = cast<BranchInst>(BB->getTerminator());

    for (BasicBlock *Succ : Term->successors())
      if (Visited.count(Succ))
        Loops[Succ] = BB;
  }
}

/// Build the condition for one edge
PredInfo StructurizeCFG::buildCondition(BranchInst *Term, unsigned Idx,
                                        bool Invert) {
  Value *Cond = Invert ? BoolFalse : BoolTrue;
  MaybeCondBranchWeights Weights;

  if (Term->isConditional()) {
    Cond = Term->getCondition();
    Weights = CondBranchWeights::tryParse(*Term);

    if (Idx != (unsigned)Invert) {
      Cond = invertCondition(Cond);
      if (Weights)
        Weights = Weights->invert();
    }
  }
  return {Cond, Weights};
}

/// Analyze the predecessors of each block and build up predicates
void StructurizeCFG::gatherPredicates(RegionNode *N) {
  RegionInfo *RI = ParentRegion->getRegionInfo();
  BasicBlock *BB = N->getEntry();
  BBPredicates &Pred = Predicates[BB];
  BBPredicates &LPred = LoopPreds[BB];

  for (BasicBlock *P : predecessors(BB)) {
    // Ignore it if it's a branch from outside into our region entry
    if (!ParentRegion->contains(P))
      continue;

    Region *R = RI->getRegionFor(P);
    if (R == ParentRegion) {
      // It's a top level block in our region
      BranchInst *Term = cast<BranchInst>(P->getTerminator());
      for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
        BasicBlock *Succ = Term->getSuccessor(i);
        if (Succ != BB)
          continue;

        if (Visited.count(P)) {
          // Normal forward edge
          if (Term->isConditional()) {
            // Try to treat it like an ELSE block
            BasicBlock *Other = Term->getSuccessor(!i);
            if (Visited.count(Other) && !Loops.count(Other) &&
                !Pred.count(Other) && !Pred.count(P)) {
              hoistZeroCostElseBlockPhiValues(Succ, Other);
              Pred[Other] = {BoolFalse, std::nullopt};
              Pred[P] = {BoolTrue, std::nullopt};
              continue;
            }
          }
          Pred[P] = buildCondition(Term, i, false);
        } else {
          // Back edge
          LPred[P] = buildCondition(Term, i, true);
        }
      }
    } else {
      // It's an exit from a sub region
      while (R->getParent() != ParentRegion)
        R = R->getParent();

      // Edge from inside a subregion to its entry, ignore it
      if (*R == *N)
        continue;

      BasicBlock *Entry = R->getEntry();
      if (Visited.count(Entry))
        Pred[Entry] = {BoolTrue, std::nullopt};
      else
        LPred[Entry] = {BoolFalse, std::nullopt};
    }
  }
}

/// Collect various loop and predicate infos
void StructurizeCFG::collectInfos() {
  // Reset predicate
  Predicates.clear();

  // and loop infos
  Loops.clear();
  LoopPreds.clear();

  // Reset the visited nodes
  Visited.clear();

  for (RegionNode *RN : reverse(Order)) {
    LLVM_DEBUG(dbgs() << "Visiting: "
                      << (RN->isSubRegion() ? "SubRegion with entry: " : "")
                      << RN->getEntry()->getName() << "\n");

    // Analyze all the conditions leading to a node
    gatherPredicates(RN);

    // Remember that we've seen this node
    Visited.insert(RN->getEntry());

    // Find the last back edges
    analyzeLoops(RN);
  }
}

/// Insert the missing branch conditions
void StructurizeCFG::insertConditions(bool Loops) {
  BranchVector &Conds = Loops ? LoopConds : Conditions;
  Value *Default = Loops ? BoolTrue : BoolFalse;
  SSAUpdater PhiInserter;

  for (BranchInst *Term : Conds) {
    assert(Term->isConditional());

    BasicBlock *Parent = Term->getParent();
    BasicBlock *SuccTrue = Term->getSuccessor(0);
    BasicBlock *SuccFalse = Term->getSuccessor(1);

    PhiInserter.Initialize(Boolean, "");
    PhiInserter.AddAvailableValue(Loops ? SuccFalse : Parent, Default);

    BBPredicates &Preds = Loops ? LoopPreds[SuccFalse] : Predicates[SuccTrue];

    NearestCommonDominator Dominator(DT);
    Dominator.addBlock(Parent);

    PredInfo ParentInfo{nullptr, std::nullopt};
    for (auto [BB, PI] : Preds) {
      if (BB == Parent) {
        ParentInfo = PI;
        break;
      }
      PhiInserter.AddAvailableValue(BB, PI.Pred);
      Dominator.addAndRememberBlock(BB);
    }

    if (ParentInfo.Pred) {
      Term->setCondition(ParentInfo.Pred);
      CondBranchWeights::setMetadata(*Term, ParentInfo.Weights);
    } else {
      if (!Dominator.resultIsRememberedBlock())
        PhiInserter.AddAvailableValue(Dominator.result(), Default);

      Term->setCondition(PhiInserter.GetValueInMiddleOfBlock(Parent));
    }
  }
}

/// Simplify any inverted conditions that were built by buildConditions.
void StructurizeCFG::simplifyConditions() {
  SmallVector<Instruction *> InstToErase;
  for (auto &I : concat<PredMap::value_type>(Predicates, LoopPreds)) {
    auto &Preds = I.second;
    for (auto [BB, PI] : Preds) {
      Instruction *Inverted;
      if (match(PI.Pred, m_Not(m_OneUse(m_Instruction(Inverted)))) &&
          !PI.Pred->use_empty()) {
        if (auto *InvertedCmp = dyn_cast<CmpInst>(Inverted)) {
          InvertedCmp->setPredicate(InvertedCmp->getInversePredicate());
          PI.Pred->replaceAllUsesWith(InvertedCmp);
          InstToErase.push_back(cast<Instruction>(PI.Pred));
        }
      }
    }
  }
  for (auto *I : InstToErase)
    I->eraseFromParent();
}

/// Remove all PHI values coming from "From" into "To" and remember
/// them in DeletedPhis
void StructurizeCFG::delPhiValues(BasicBlock *From, BasicBlock *To) {
  PhiMap &Map = DeletedPhis[To];
  for (PHINode &Phi : To->phis()) {
    bool Recorded = false;
    while (Phi.getBasicBlockIndex(From) != -1) {
      Value *Deleted = Phi.removeIncomingValue(From, false);
      Map[&Phi].push_back(std::make_pair(From, Deleted));
      if (!Recorded) {
        AffectedPhis.push_back(&Phi);
        Recorded = true;
      }
    }
  }
}

/// Add a dummy PHI value as soon as we knew the new predecessor
void StructurizeCFG::addPhiValues(BasicBlock *From, BasicBlock *To) {
  for (PHINode &Phi : To->phis()) {
    Value *Poison = PoisonValue::get(Phi.getType());
    Phi.addIncoming(Poison, From);
  }
  AddedPhis[To].push_back(From);
}

/// When we are reconstructing a PHI inside \p PHIBlock with incoming values
/// from predecessors \p Incomings, we have a chance to mark the available value
/// from some blocks as undefined. The function will find out all such blocks
/// and return in \p UndefBlks.
void StructurizeCFG::findUndefBlocks(
    BasicBlock *PHIBlock, const SmallPtrSet<BasicBlock *, 8> &Incomings,
    SmallVector<BasicBlock *> &UndefBlks) const {
  //  We may get a post-structured CFG like below:
  //
  //  | P1
  //  |/
  //  F1
  //  |\
  //  | N
  //  |/
  //  F2
  //  |\
  //  | P2
  //  |/
  //  F3
  //  |\
  //  B
  //
  // B is the block that has a PHI being reconstructed. P1/P2 are predecessors
  // of B before structurization. F1/F2/F3 are flow blocks inserted during
  // structurization process. Block N is not a predecessor of B before
  // structurization, but are placed between the predecessors(P1/P2) of B after
  // structurization. This usually means that threads went to N never take the
  // path N->F2->F3->B. For example, the threads take the branch F1->N may
  // always take the branch F2->P2. So, when we are reconstructing a PHI
  // originally in B, we can safely say the incoming value from N is undefined.
  SmallPtrSet<BasicBlock *, 8> VisitedBlock;
  SmallVector<BasicBlock *, 8> Stack;
  if (PHIBlock == ParentRegion->getExit()) {
    for (auto P : predecessors(PHIBlock)) {
      if (ParentRegion->contains(P))
        Stack.push_back(P);
    }
  } else {
    append_range(Stack, predecessors(PHIBlock));
  }

  // Do a backward traversal over the CFG, and stop further searching if
  // the block is not a Flow. If a block is neither flow block nor the
  // incoming predecessor, then the incoming value from the block is
  // undefined value for the PHI being reconstructed.
  while (!Stack.empty()) {
    BasicBlock *Current = Stack.pop_back_val();
    if (!VisitedBlock.insert(Current).second)
      continue;

    if (FlowSet.contains(Current))
      llvm::append_range(Stack, predecessors(Current));
    else if (!Incomings.contains(Current))
      UndefBlks.push_back(Current);
  }
}

// If two phi nodes have compatible incoming values (for each
// incoming block, either they have the same incoming value or only one phi
// node has an incoming value), let them share the merged incoming values. The
// merge process is guided by the equivalence information from \p PhiClasses.
// The function will possibly update the incoming values of leader phi in
// DeletedPhis.
void StructurizeCFG::mergeIfCompatible(
    EquivalenceClasses<PHINode *> &PhiClasses, PHINode *A, PHINode *B) {
  auto ItA = PhiClasses.findLeader(PhiClasses.insert(A));
  auto ItB = PhiClasses.findLeader(PhiClasses.insert(B));
  // They are already in the same class, no work needed.
  if (ItA == ItB)
    return;

  PHINode *LeaderA = *ItA;
  PHINode *LeaderB = *ItB;
  BBValueVector &IncomingA = DeletedPhis[LeaderA->getParent()][LeaderA];
  BBValueVector &IncomingB = DeletedPhis[LeaderB->getParent()][LeaderB];

  DenseMap<BasicBlock *, Value *> Mergeable(IncomingA.begin(), IncomingA.end());
  for (auto [BB, V] : IncomingB) {
    auto BBIt = Mergeable.find(BB);
    if (BBIt != Mergeable.end() && BBIt->second != V)
      return;
    // Either IncomingA does not have this value or IncomingA has the same
    // value.
    Mergeable.insert({BB, V});
  }

  // Update the incoming value of leaderA.
  IncomingA.assign(Mergeable.begin(), Mergeable.end());
  PhiClasses.unionSets(ItA, ItB);
}

/// Add the real PHI value as soon as everything is set up
void StructurizeCFG::setPhiValues() {
  SmallVector<PHINode *, 8> InsertedPhis;
  SSAUpdater Updater(&InsertedPhis);
  DenseMap<BasicBlock *, SmallVector<BasicBlock *>> UndefBlksMap;

  // Find phi nodes that have compatible incoming values (either they have
  // the same value for the same block or only one phi node has an incoming
  // value, see example below). We only search again the phi's that are
  // referenced by another phi, which is the case we care about.
  //
  // For example (-- means no incoming value):
  // phi1 : BB1:phi2   BB2:v  BB3:--
  // phi2:  BB1:--     BB2:v  BB3:w
  //
  // Then we can merge these incoming values and let phi1, phi2 use the
  // same set of incoming values:
  //
  // phi1&phi2: BB1:phi2  BB2:v  BB3:w
  //
  // By doing this, phi1 and phi2 would share more intermediate phi nodes.
  // This would help reduce the number of phi nodes during SSA reconstruction
  // and ultimately result in fewer COPY instructions.
  //
  // This should be correct, because if a phi node does not have incoming
  // value from certain block, this means the block is not the predecessor
  // of the parent block, so we actually don't care about its incoming value.
  EquivalenceClasses<PHINode *> PhiClasses;
  for (const auto &[To, From] : AddedPhis) {
    auto OldPhiIt = DeletedPhis.find(To);
    if (OldPhiIt == DeletedPhis.end())
      continue;

    PhiMap &BlkPhis = OldPhiIt->second;
    SmallVector<BasicBlock *> &UndefBlks = UndefBlksMap[To];
    SmallPtrSet<BasicBlock *, 8> Incomings;

    // Get the undefined blocks shared by all the phi nodes.
    if (!BlkPhis.empty()) {
      Incomings.insert_range(llvm::make_first_range(BlkPhis.front().second));
      findUndefBlocks(To, Incomings, UndefBlks);
    }

    for (const auto &[Phi, Incomings] : OldPhiIt->second) {
      SmallVector<PHINode *> IncomingPHIs;
      for (const auto &[BB, V] : Incomings) {
        // First, for each phi, check whether it has incoming value which is
        // another phi.
        if (PHINode *P = dyn_cast<PHINode>(V))
          IncomingPHIs.push_back(P);
      }

      for (auto *OtherPhi : IncomingPHIs) {
        // Skip phis that are unrelated to the phi reconstruction for now.
        if (!DeletedPhis.contains(OtherPhi->getParent()))
          continue;
        mergeIfCompatible(PhiClasses, Phi, OtherPhi);
      }
    }
  }

  for (const auto &AddedPhi : AddedPhis) {
    BasicBlock *To = AddedPhi.first;
    const BBVector &From = AddedPhi.second;

    auto It = DeletedPhis.find(To);
    if (It == DeletedPhis.end())
      continue;

    PhiMap &Map = It->second;
    SmallVector<BasicBlock *> &UndefBlks = UndefBlksMap[To];
    for (const auto &[Phi, Incoming] : Map) {
      Value *Poison = PoisonValue::get(Phi->getType());
      Updater.Initialize(Phi->getType(), "");
      Updater.AddAvailableValue(&Func->getEntryBlock(), Poison);
      Updater.AddAvailableValue(To, Poison);

      // Use leader phi's incoming if there is.
      auto LeaderIt = PhiClasses.findLeader(Phi);
      bool UseIncomingOfLeader =
          LeaderIt != PhiClasses.member_end() && *LeaderIt != Phi;
      const auto &IncomingMap =
          UseIncomingOfLeader ? DeletedPhis[(*LeaderIt)->getParent()][*LeaderIt]
                              : Incoming;

      SmallVector<BasicBlock *> ConstantPreds;
      for (const auto &[BB, V] : IncomingMap) {
        Updater.AddAvailableValue(BB, V);
        if (isa<Constant>(V))
          ConstantPreds.push_back(BB);
      }

      for (auto UB : UndefBlks) {
        // If this undef block is dominated by any predecessor(before
        // structurization) of reconstructed PHI with constant incoming value,
        // don't mark the available value as undefined. Setting undef to such
        // block will stop us from getting optimal phi insertion.
        if (any_of(ConstantPreds,
                   [&](BasicBlock *CP) { return DT->dominates(CP, UB); }))
          continue;
        // Maybe already get a value through sharing with other phi nodes.
        if (Updater.HasValueForBlock(UB))
          continue;

        Updater.AddAvailableValue(UB, Poison);
      }

      for (BasicBlock *FI : From)
        Phi->setIncomingValueForBlock(FI, Updater.GetValueAtEndOfBlock(FI));
      AffectedPhis.push_back(Phi);
    }
  }

  AffectedPhis.append(InsertedPhis.begin(), InsertedPhis.end());
}

/// Updates PHI nodes after hoisted zero cost instructions by replacing poison
/// entries on Flow nodes with the appropriate hoisted values
void StructurizeCFG::simplifyHoistedPhis() {
  for (WeakVH VH : AffectedPhis) {
    PHINode *Phi = dyn_cast_or_null<PHINode>(VH);
    if (!Phi || Phi->getNumIncomingValues() != 2)
      continue;

    for (int i = 0; i < 2; i++) {
      Value *V = Phi->getIncomingValue(i);
      auto BBIt = HoistedValues.find(V);

      if (BBIt == HoistedValues.end())
        continue;

      Value *OtherV = Phi->getIncomingValue(!i);
      PHINode *OtherPhi = dyn_cast<PHINode>(OtherV);
      if (!OtherPhi)
        continue;

      int PoisonValBBIdx = -1;
      for (size_t i = 0; i < OtherPhi->getNumIncomingValues(); i++) {
        if (!isa<PoisonValue>(OtherPhi->getIncomingValue(i)))
          continue;
        PoisonValBBIdx = i;
        break;
      }
      if (PoisonValBBIdx == -1 ||
          !DT->dominates(BBIt->second,
                         OtherPhi->getIncomingBlock(PoisonValBBIdx)))
        continue;

      OtherPhi->setIncomingValue(PoisonValBBIdx, V);
      Phi->setIncomingValue(i, OtherV);
    }
  }
}

void StructurizeCFG::simplifyAffectedPhis() {
  bool Changed;
  do {
    Changed = false;
    SimplifyQuery Q(Func->getDataLayout());
    Q.DT = DT;
    // Setting CanUseUndef to true might extend value liveness, set it to false
    // to achieve better register pressure.
    Q.CanUseUndef = false;
    for (WeakVH VH : AffectedPhis) {
      if (auto Phi = dyn_cast_or_null<PHINode>(VH)) {
        if (auto NewValue = simplifyInstruction(Phi, Q)) {
          Phi->replaceAllUsesWith(NewValue);
          Phi->eraseFromParent();
          Changed = true;
        }
      }
    }
  } while (Changed);
}

/// Remove phi values from all successors and then remove the terminator.
DebugLoc StructurizeCFG::killTerminator(BasicBlock *BB) {
  Instruction *Term = BB->getTerminator();
  if (!Term)
    return DebugLoc();

  for (BasicBlock *Succ : successors(BB))
    delPhiValues(BB, Succ);

  DebugLoc DL = Term->getDebugLoc();
  Term->eraseFromParent();
  return DL;
}

/// Let node exit(s) point to NewExit
void StructurizeCFG::changeExit(RegionNode *Node, BasicBlock *NewExit,
                                bool IncludeDominator) {
  if (Node->isSubRegion()) {
    Region *SubRegion = Node->getNodeAs<Region>();
    BasicBlock *OldExit = SubRegion->getExit();
    BasicBlock *Dominator = nullptr;

    // Find all the edges from the sub region to the exit.
    // We use make_early_inc_range here because we modify BB's terminator.
    for (BasicBlock *BB : llvm::make_early_inc_range(predecessors(OldExit))) {
      if (!SubRegion->contains(BB))
        continue;

      // Modify the edges to point to the new exit
      delPhiValues(BB, OldExit);
      BB->getTerminator()->replaceUsesOfWith(OldExit, NewExit);
      addPhiValues(BB, NewExit);

      // Find the new dominator (if requested)
      if (IncludeDominator) {
        if (!Dominator)
          Dominator = BB;
        else
          Dominator = DT->findNearestCommonDominator(Dominator, BB);
      }
    }

    // Change the dominator (if requested)
    if (Dominator)
      DT->changeImmediateDominator(NewExit, Dominator);

    // Update the region info
    SubRegion->replaceExit(NewExit);
  } else {
    BasicBlock *BB = Node->getNodeAs<BasicBlock>();
    DebugLoc DL = killTerminator(BB);
    BranchInst *Br = BranchInst::Create(NewExit, BB);
    Br->setDebugLoc(DL);
    addPhiValues(BB, NewExit);
    if (IncludeDominator)
      DT->changeImmediateDominator(NewExit, BB);
  }
}

/// Create a new flow node and update dominator tree and region info
BasicBlock *StructurizeCFG::getNextFlow(BasicBlock *Dominator) {
  LLVMContext &Context = Func->getContext();
  BasicBlock *Insert = Order.empty() ? ParentRegion->getExit() :
                       Order.back()->getEntry();
  BasicBlock *Flow = BasicBlock::Create(Context, FlowBlockName,
                                        Func, Insert);
  FlowSet.insert(Flow);
  DT->addNewBlock(Flow, Dominator);
  ParentRegion->getRegionInfo()->setRegionFor(Flow, ParentRegion);
  return Flow;
}

/// Create a new or reuse the previous node as flow node. Returns a block and a
/// debug location to be used for new instructions in that block.
std::pair<BasicBlock *, DebugLoc> StructurizeCFG::needPrefix(bool NeedEmpty) {
  BasicBlock *Entry = PrevNode->getEntry();

  if (!PrevNode->isSubRegion()) {
    DebugLoc DL = killTerminator(Entry);
    if (!NeedEmpty || Entry->getFirstInsertionPt() == Entry->end())
      return {Entry, DL};
  }

  // create a new flow node
  BasicBlock *Flow = getNextFlow(Entry);

  // and wire it up
  changeExit(PrevNode, Flow, true);
  PrevNode = ParentRegion->getBBNode(Flow);
  return {Flow, DebugLoc()};
}

/// Returns the region exit if possible, otherwise just a new flow node
BasicBlock *StructurizeCFG::needPostfix(BasicBlock *Flow,
                                        bool ExitUseAllowed) {
  if (!Order.empty() || !ExitUseAllowed)
    return getNextFlow(Flow);

  BasicBlock *Exit = ParentRegion->getExit();
  DT->changeImmediateDominator(Exit, Flow);
  addPhiValues(Flow, Exit);
  return Exit;
}

/// Set the previous node
void StructurizeCFG::setPrevNode(BasicBlock *BB) {
  PrevNode = ParentRegion->contains(BB) ? ParentRegion->getBBNode(BB)
                                        : nullptr;
}

/// Does BB dominate all the predicates of Node?
bool StructurizeCFG::dominatesPredicates(BasicBlock *BB, RegionNode *Node) {
  BBPredicates &Preds = Predicates[Node->getEntry()];
  return llvm::all_of(Preds, [&](std::pair<BasicBlock *, PredInfo> Pred) {
    return DT->dominates(BB, Pred.first);
  });
}

/// Can we predict that this node will always be called?
bool StructurizeCFG::isPredictableTrue(RegionNode *Node) {
  BBPredicates &Preds = Predicates[Node->getEntry()];
  bool Dominated = false;

  // Regionentry is always true
  if (!PrevNode)
    return true;

  for (auto [BB, PI] : Preds) {
    if (PI.Pred != BoolTrue)
      return false;

    if (!Dominated && DT->dominates(BB, PrevNode->getEntry()))
      Dominated = true;
  }

  // TODO: The dominator check is too strict
  return Dominated;
}

/// Take one node from the order vector and wire it up
void StructurizeCFG::wireFlow(bool ExitUseAllowed,
                              BasicBlock *LoopEnd) {
  RegionNode *Node = Order.pop_back_val();
  Visited.insert(Node->getEntry());

  if (isPredictableTrue(Node)) {
    // Just a linear flow
    if (PrevNode) {
      changeExit(PrevNode, Node->getEntry(), true);
    }
    PrevNode = Node;
  } else {
    // Insert extra prefix node (or reuse last one)
    auto [Flow, DL] = needPrefix(false);

    // Insert extra postfix node (or use exit instead)
    BasicBlock *Entry = Node->getEntry();
    BasicBlock *Next = needPostfix(Flow, ExitUseAllowed);

    // let it point to entry and next block
    BranchInst *Br = BranchInst::Create(Entry, Next, BoolPoison, Flow);
    Br->setDebugLoc(DL);
    Conditions.push_back(Br);
    addPhiValues(Flow, Entry);
    DT->changeImmediateDominator(Entry, Flow);

    PrevNode = Node;
    while (!Order.empty() && !Visited.count(LoopEnd) &&
           dominatesPredicates(Entry, Order.back())) {
      handleLoops(false, LoopEnd);
    }

    changeExit(PrevNode, Next, false);
    setPrevNode(Next);
  }
}

void StructurizeCFG::handleLoops(bool ExitUseAllowed,
                                 BasicBlock *LoopEnd) {
  RegionNode *Node = Order.back();
  BasicBlock *LoopStart = Node->getEntry();

  if (!Loops.count(LoopStart)) {
    wireFlow(ExitUseAllowed, LoopEnd);
    return;
  }

  if (!isPredictableTrue(Node))
    LoopStart = needPrefix(true).first;

  LoopEnd = Loops[Node->getEntry()];
  wireFlow(false, LoopEnd);
  while (!Visited.count(LoopEnd)) {
    handleLoops(false, LoopEnd);
  }

  assert(LoopStart != &LoopStart->getParent()->getEntryBlock());

  // Create an extra loop end node
  DebugLoc DL;
  std::tie(LoopEnd, DL) = needPrefix(false);
  BasicBlock *Next = needPostfix(LoopEnd, ExitUseAllowed);
  BranchInst *Br = BranchInst::Create(Next, LoopStart, BoolPoison, LoopEnd);
  Br->setDebugLoc(DL);
  LoopConds.push_back(Br);
  addPhiValues(LoopEnd, LoopStart);
  setPrevNode(Next);
}

/// After this function control flow looks like it should be, but
/// branches and PHI nodes only have undefined conditions.
void StructurizeCFG::createFlow() {
  BasicBlock *Exit = ParentRegion->getExit();
  bool EntryDominatesExit = DT->dominates(ParentRegion->getEntry(), Exit);

  AffectedPhis.clear();
  DeletedPhis.clear();
  AddedPhis.clear();
  Conditions.clear();
  LoopConds.clear();

  PrevNode = nullptr;
  Visited.clear();

  while (!Order.empty()) {
    handleLoops(EntryDominatesExit, nullptr);
  }

  if (PrevNode)
    changeExit(PrevNode, Exit, EntryDominatesExit);
  else
    assert(EntryDominatesExit);
}

/// Handle a rare case where the disintegrated nodes instructions
/// no longer dominate all their uses. Not sure if this is really necessary
void StructurizeCFG::rebuildSSA() {
  SSAUpdater Updater;
  for (BasicBlock *BB : ParentRegion->blocks())
    for (Instruction &I : *BB) {
      bool Initialized = false;
      // We may modify the use list as we iterate over it, so we use
      // make_early_inc_range.
      for (Use &U : llvm::make_early_inc_range(I.uses())) {
        Instruction *User = cast<Instruction>(U.getUser());
        if (User->getParent() == BB) {
          continue;
        } else if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
          if (UserPN->getIncomingBlock(U) == BB)
            continue;
        }

        if (DT->dominates(&I, User))
          continue;

        if (!Initialized) {
          Value *Poison = PoisonValue::get(I.getType());
          Updater.Initialize(I.getType(), "");
          Updater.AddAvailableValue(&Func->getEntryBlock(), Poison);
          Updater.AddAvailableValue(BB, &I);
          Initialized = true;
        }
        Updater.RewriteUseAfterInsertions(U);
      }
    }
}

static bool hasOnlyUniformBranches(Region *R, unsigned UniformMDKindID,
                                   const UniformityInfo &UA) {
  // Bool for if all sub-regions are uniform.
  bool SubRegionsAreUniform = true;
  // Count of how many direct children are conditional.
  unsigned ConditionalDirectChildren = 0;

  for (auto *E : R->elements()) {
    if (!E->isSubRegion()) {
      auto Br = dyn_cast<BranchInst>(E->getEntry()->getTerminator());
      if (!Br || !Br->isConditional())
        continue;

      if (!UA.isUniform(Br))
        return false;

      // One of our direct children is conditional.
      ConditionalDirectChildren++;

      LLVM_DEBUG(dbgs() << "BB: " << Br->getParent()->getName()
                        << " has uniform terminator\n");
    } else {
      // Explicitly refuse to treat regions as uniform if they have non-uniform
      // subregions. We cannot rely on UniformityAnalysis for branches in
      // subregions because those branches may have been removed and re-created,
      // so we look for our metadata instead.
      //
      // Warning: It would be nice to treat regions as uniform based only on
      // their direct child basic blocks' terminators, regardless of whether
      // subregions are uniform or not. However, this requires a very careful
      // look at SIAnnotateControlFlow to make sure nothing breaks there.
      for (auto *BB : E->getNodeAs<Region>()->blocks()) {
        auto Br = dyn_cast<BranchInst>(BB->getTerminator());
        if (!Br || !Br->isConditional())
          continue;

        if (!Br->getMetadata(UniformMDKindID)) {
          // Early exit if we cannot have relaxed uniform regions.
          if (!RelaxedUniformRegions)
            return false;

          SubRegionsAreUniform = false;
          break;
        }
      }
    }
  }

  // Our region is uniform if:
  // 1. All conditional branches that are direct children are uniform (checked
  // above).
  // 2. And either:
  //   a. All sub-regions are uniform.
  //   b. There is one or less conditional branches among the direct children.
  return SubRegionsAreUniform || (ConditionalDirectChildren <= 1);
}

void StructurizeCFG::init(Region *R) {
  LLVMContext &Context = R->getEntry()->getContext();

  Boolean = Type::getInt1Ty(Context);
  BoolTrue = ConstantInt::getTrue(Context);
  BoolFalse = ConstantInt::getFalse(Context);
  BoolPoison = PoisonValue::get(Boolean);

  this->UA = nullptr;
}

bool StructurizeCFG::makeUniformRegion(Region *R, UniformityInfo &UA) {
  if (R->isTopLevelRegion())
    return false;

  this->UA = &UA;

  // TODO: We could probably be smarter here with how we handle sub-regions.
  // We currently rely on the fact that metadata is set by earlier invocations
  // of the pass on sub-regions, and that this metadata doesn't get lost --
  // but we shouldn't rely on metadata for correctness!
  unsigned UniformMDKindID =
      R->getEntry()->getContext().getMDKindID("structurizecfg.uniform");

  if (hasOnlyUniformBranches(R, UniformMDKindID, UA)) {
    LLVM_DEBUG(dbgs() << "Skipping region with uniform control flow: " << *R
                      << '\n');

    // Mark all direct child block terminators as having been treated as
    // uniform. To account for a possible future in which non-uniform
    // sub-regions are treated more cleverly, indirect children are not
    // marked as uniform.
    MDNode *MD = MDNode::get(R->getEntry()->getParent()->getContext(), {});
    for (RegionNode *E : R->elements()) {
      if (E->isSubRegion())
        continue;

      if (Instruction *Term = E->getEntry()->getTerminator())
        Term->setMetadata(UniformMDKindID, MD);
    }

    return true;
  }
  return false;
}

/// Run the transformation for each region found
bool StructurizeCFG::run(Region *R, DominatorTree *DT,
                         const TargetTransformInfo *TTI) {
  if (R->isTopLevelRegion())
    return false;

  this->DT = DT;
  this->TTI = TTI;
  Func = R->getEntry()->getParent();
  assert(hasOnlySimpleTerminator(*Func) && "Unsupported block terminator.");

  ParentRegion = R;

  orderNodes();
  collectInfos();
  createFlow();
  insertConditions(false);
  insertConditions(true);
  setPhiValues();
  simplifyHoistedPhis();
  simplifyConditions();
  simplifyAffectedPhis();
  rebuildSSA();

  // Cleanup
  Order.clear();
  Visited.clear();
  DeletedPhis.clear();
  AddedPhis.clear();
  Predicates.clear();
  Conditions.clear();
  Loops.clear();
  LoopPreds.clear();
  LoopConds.clear();
  FlowSet.clear();

  return true;
}

Pass *llvm::createStructurizeCFGPass(bool SkipUniformRegions) {
  return new StructurizeCFGLegacyPass(SkipUniformRegions);
}

static void addRegionIntoQueue(Region &R, std::vector<Region *> &Regions) {
  Regions.push_back(&R);
  for (const auto &E : R)
    addRegionIntoQueue(*E, Regions);
}

StructurizeCFGPass::StructurizeCFGPass(bool SkipUniformRegions_)
    : SkipUniformRegions(SkipUniformRegions_) {
  if (ForceSkipUniformRegions.getNumOccurrences())
    SkipUniformRegions = ForceSkipUniformRegions.getValue();
}

void StructurizeCFGPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<StructurizeCFGPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  if (SkipUniformRegions)
    OS << "<skip-uniform-regions>";
}

PreservedAnalyses StructurizeCFGPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {

  bool Changed = false;
  DominatorTree *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto &RI = AM.getResult<RegionInfoAnalysis>(F);
  TargetTransformInfo *TTI = &AM.getResult<TargetIRAnalysis>(F);
  UniformityInfo *UI = nullptr;
  if (SkipUniformRegions)
    UI = &AM.getResult<UniformityInfoAnalysis>(F);

  std::vector<Region *> Regions;
  addRegionIntoQueue(*RI.getTopLevelRegion(), Regions);
  while (!Regions.empty()) {
    Region *R = Regions.back();
    Regions.pop_back();

    StructurizeCFG SCFG;
    SCFG.init(R);

    if (SkipUniformRegions && SCFG.makeUniformRegion(R, *UI)) {
      Changed = true; // May have added metadata.
      continue;
    }

    Changed |= SCFG.run(R, DT, TTI);
  }
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
