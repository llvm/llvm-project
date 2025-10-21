//===- DFAJumpThreading.cpp - Threads a switch statement inside a loop ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform each threading path to effectively jump thread the DFA. For
// example, the CFG below could be transformed as follows, where the cloned
// blocks unconditionally branch to the next correct case based on what is
// identified in the analysis.
//
//          sw.bb                        sw.bb
//        /   |   \                    /   |   \
//   case1  case2  case3          case1  case2  case3
//        \   |   /                 |      |      |
//       determinator            det.2   det.3  det.1
//        br sw.bb                /        |        \
//                          sw.bb.2     sw.bb.3     sw.bb.1
//                           br case2    br case3    br case1§
//
// Definitions and Terminology:
//
// * Threading path:
//   a list of basic blocks, the exit state, and the block that determines
//   the next state, for which the following notation will be used:
//   < path of BBs that form a cycle > [ state, determinator ]
//
// * Predictable switch:
//   The switch variable is always a known constant so that all conditional
//   jumps based on switch variable can be converted to unconditional jump.
//
// * Determinator:
//   The basic block that determines the next state of the DFA.
//
// Representing the optimization in C-like pseudocode: the code pattern on the
// left could functionally be transformed to the right pattern if the switch
// condition is predictable.
//
//  X = A                       goto A
//  for (...)                   A:
//    switch (X)                  ...
//      case A                    goto B
//        X = B                 B:
//      case B                    ...
//        X = C                   goto C
//
// The pass first checks that switch variable X is decided by the control flow
// path taken in the loop; for example, in case B, the next value of X is
// decided to be C. It then enumerates through all paths in the loop and labels
// the basic blocks where the next state is decided.
//
// Using this information it creates new paths that unconditionally branch to
// the next case. This involves cloning code, so it only gets triggered if the
// amount of code duplicated is below a threshold.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DFAJumpThreading.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SSAUpdaterBulk.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <deque>

#ifdef EXPENSIVE_CHECKS
#include "llvm/IR/Verifier.h"
#endif

using namespace llvm;

#define DEBUG_TYPE "dfa-jump-threading"

STATISTIC(NumTransforms, "Number of transformations done");
STATISTIC(NumCloned, "Number of blocks cloned");
STATISTIC(NumPaths, "Number of individual paths threaded");

namespace llvm {
static cl::opt<bool>
    ClViewCfgBefore("dfa-jump-view-cfg-before",
                    cl::desc("View the CFG before DFA Jump Threading"),
                    cl::Hidden, cl::init(false));

static cl::opt<bool> EarlyExitHeuristic(
    "dfa-early-exit-heuristic",
    cl::desc("Exit early if an unpredictable value come from the same loop"),
    cl::Hidden, cl::init(true));

static cl::opt<unsigned> MaxPathLength(
    "dfa-max-path-length",
    cl::desc("Max number of blocks searched to find a threading path"),
    cl::Hidden, cl::init(20));

static cl::opt<unsigned> MaxNumVisitiedPaths(
    "dfa-max-num-visited-paths",
    cl::desc(
        "Max number of blocks visited while enumerating paths around a switch"),
    cl::Hidden, cl::init(2500));

static cl::opt<unsigned>
    MaxNumPaths("dfa-max-num-paths",
                cl::desc("Max number of paths enumerated around a switch"),
                cl::Hidden, cl::init(200));

static cl::opt<unsigned>
    CostThreshold("dfa-cost-threshold",
                  cl::desc("Maximum cost accepted for the transformation"),
                  cl::Hidden, cl::init(50));

extern cl::opt<bool> ProfcheckDisableMetadataFixes;

} // namespace llvm

static cl::opt<double> MaxClonedRate(
    "dfa-max-cloned-rate",
    cl::desc(
        "Maximum cloned instructions rate accepted for the transformation"),
    cl::Hidden, cl::init(7.5));

namespace {
class SelectInstToUnfold {
  SelectInst *SI;
  PHINode *SIUse;

public:
  SelectInstToUnfold(SelectInst *SI, PHINode *SIUse) : SI(SI), SIUse(SIUse) {}

  SelectInst *getInst() { return SI; }
  PHINode *getUse() { return SIUse; }

  explicit operator bool() const { return SI && SIUse; }
};

class DFAJumpThreading {
public:
  DFAJumpThreading(AssumptionCache *AC, DomTreeUpdater *DTU, LoopInfo *LI,
                   TargetTransformInfo *TTI, OptimizationRemarkEmitter *ORE)
      : AC(AC), DTU(DTU), LI(LI), TTI(TTI), ORE(ORE) {}

  bool run(Function &F);
  bool LoopInfoBroken;

private:
  void
  unfoldSelectInstrs(const SmallVector<SelectInstToUnfold, 4> &SelectInsts) {
    SmallVector<SelectInstToUnfold, 4> Stack(SelectInsts);

    while (!Stack.empty()) {
      SelectInstToUnfold SIToUnfold = Stack.pop_back_val();

      std::vector<SelectInstToUnfold> NewSIsToUnfold;
      std::vector<BasicBlock *> NewBBs;
      unfold(DTU, LI, SIToUnfold, &NewSIsToUnfold, &NewBBs);

      // Put newly discovered select instructions into the work list.
      llvm::append_range(Stack, NewSIsToUnfold);
    }
  }

  static void unfold(DomTreeUpdater *DTU, LoopInfo *LI,
                     SelectInstToUnfold SIToUnfold,
                     std::vector<SelectInstToUnfold> *NewSIsToUnfold,
                     std::vector<BasicBlock *> *NewBBs);

  AssumptionCache *AC;
  DomTreeUpdater *DTU;
  LoopInfo *LI;
  TargetTransformInfo *TTI;
  OptimizationRemarkEmitter *ORE;
};
} // namespace

/// Unfold the select instruction held in \p SIToUnfold by replacing it with
/// control flow.
///
/// Put newly discovered select instructions into \p NewSIsToUnfold. Put newly
/// created basic blocks into \p NewBBs.
///
/// TODO: merge it with CodeGenPrepare::optimizeSelectInst() if possible.
void DFAJumpThreading::unfold(DomTreeUpdater *DTU, LoopInfo *LI,
                              SelectInstToUnfold SIToUnfold,
                              std::vector<SelectInstToUnfold> *NewSIsToUnfold,
                              std::vector<BasicBlock *> *NewBBs) {
  SelectInst *SI = SIToUnfold.getInst();
  PHINode *SIUse = SIToUnfold.getUse();
  assert(SI->hasOneUse());
  // The select may come indirectly, instead of from where it is defined.
  BasicBlock *StartBlock = SIUse->getIncomingBlock(*SI->use_begin());
  BranchInst *StartBlockTerm =
      dyn_cast<BranchInst>(StartBlock->getTerminator());
  assert(StartBlockTerm);

  if (StartBlockTerm->isUnconditional()) {
    BasicBlock *EndBlock = StartBlock->getUniqueSuccessor();
    // Arbitrarily choose the 'false' side for a new input value to the PHI.
    BasicBlock *NewBlock = BasicBlock::Create(
        SI->getContext(), Twine(SI->getName(), ".si.unfold.false"),
        EndBlock->getParent(), EndBlock);
    NewBBs->push_back(NewBlock);
    BranchInst::Create(EndBlock, NewBlock);
    DTU->applyUpdates({{DominatorTree::Insert, NewBlock, EndBlock}});

    // StartBlock
    //   |  \
    //   |  NewBlock
    //   |  /
    // EndBlock
    Value *SIOp1 = SI->getTrueValue();
    Value *SIOp2 = SI->getFalseValue();

    PHINode *NewPhi = PHINode::Create(SIUse->getType(), 1,
                                      Twine(SIOp2->getName(), ".si.unfold.phi"),
                                      NewBlock->getFirstInsertionPt());
    NewPhi->addIncoming(SIOp2, StartBlock);

    // Update any other PHI nodes in EndBlock.
    for (PHINode &Phi : EndBlock->phis()) {
      if (SIUse == &Phi)
        continue;
      Phi.addIncoming(Phi.getIncomingValueForBlock(StartBlock), NewBlock);
    }

    // Update the phi node of SI, which is its only use.
    if (EndBlock == SIUse->getParent()) {
      SIUse->addIncoming(NewPhi, NewBlock);
      SIUse->replaceUsesOfWith(SI, SIOp1);
    } else {
      PHINode *EndPhi = PHINode::Create(SIUse->getType(), pred_size(EndBlock),
                                        Twine(SI->getName(), ".si.unfold.phi"),
                                        EndBlock->getFirstInsertionPt());
      for (BasicBlock *Pred : predecessors(EndBlock)) {
        if (Pred != StartBlock && Pred != NewBlock)
          EndPhi->addIncoming(EndPhi, Pred);
      }

      EndPhi->addIncoming(SIOp1, StartBlock);
      EndPhi->addIncoming(NewPhi, NewBlock);
      SIUse->replaceUsesOfWith(SI, EndPhi);
      SIUse = EndPhi;
    }

    if (auto *OpSi = dyn_cast<SelectInst>(SIOp1))
      NewSIsToUnfold->push_back(SelectInstToUnfold(OpSi, SIUse));
    if (auto *OpSi = dyn_cast<SelectInst>(SIOp2))
      NewSIsToUnfold->push_back(SelectInstToUnfold(OpSi, NewPhi));

    // Insert the real conditional branch based on the original condition.
    StartBlockTerm->eraseFromParent();
    auto *BI =
        BranchInst::Create(EndBlock, NewBlock, SI->getCondition(), StartBlock);
    if (!ProfcheckDisableMetadataFixes)
      BI->setMetadata(LLVMContext::MD_prof,
                      SI->getMetadata(LLVMContext::MD_prof));
    DTU->applyUpdates({{DominatorTree::Insert, StartBlock, NewBlock}});
  } else {
    BasicBlock *EndBlock = SIUse->getParent();
    BasicBlock *NewBlockT = BasicBlock::Create(
        SI->getContext(), Twine(SI->getName(), ".si.unfold.true"),
        EndBlock->getParent(), EndBlock);
    BasicBlock *NewBlockF = BasicBlock::Create(
        SI->getContext(), Twine(SI->getName(), ".si.unfold.false"),
        EndBlock->getParent(), EndBlock);

    NewBBs->push_back(NewBlockT);
    NewBBs->push_back(NewBlockF);

    // Def only has one use in EndBlock.
    // Before transformation:
    // StartBlock(Def)
    //   |      \
    // EndBlock  OtherBlock
    //  (Use)
    //
    // After transformation:
    // StartBlock(Def)
    //   |      \
    //   |       OtherBlock
    // NewBlockT
    //   |     \
    //   |   NewBlockF
    //   |      /
    //   |     /
    // EndBlock
    //  (Use)
    BranchInst::Create(EndBlock, NewBlockF);
    // Insert the real conditional branch based on the original condition.
    auto *BI =
        BranchInst::Create(EndBlock, NewBlockF, SI->getCondition(), NewBlockT);
    if (!ProfcheckDisableMetadataFixes)
      BI->setMetadata(LLVMContext::MD_prof,
                      SI->getMetadata(LLVMContext::MD_prof));
    DTU->applyUpdates({{DominatorTree::Insert, NewBlockT, NewBlockF},
                       {DominatorTree::Insert, NewBlockT, EndBlock},
                       {DominatorTree::Insert, NewBlockF, EndBlock}});

    Value *TrueVal = SI->getTrueValue();
    Value *FalseVal = SI->getFalseValue();

    PHINode *NewPhiT = PHINode::Create(
        SIUse->getType(), 1, Twine(TrueVal->getName(), ".si.unfold.phi"),
        NewBlockT->getFirstInsertionPt());
    PHINode *NewPhiF = PHINode::Create(
        SIUse->getType(), 1, Twine(FalseVal->getName(), ".si.unfold.phi"),
        NewBlockF->getFirstInsertionPt());
    NewPhiT->addIncoming(TrueVal, StartBlock);
    NewPhiF->addIncoming(FalseVal, NewBlockT);

    if (auto *TrueSI = dyn_cast<SelectInst>(TrueVal))
      NewSIsToUnfold->push_back(SelectInstToUnfold(TrueSI, NewPhiT));
    if (auto *FalseSi = dyn_cast<SelectInst>(FalseVal))
      NewSIsToUnfold->push_back(SelectInstToUnfold(FalseSi, NewPhiF));

    SIUse->addIncoming(NewPhiT, NewBlockT);
    SIUse->addIncoming(NewPhiF, NewBlockF);
    SIUse->removeIncomingValue(StartBlock);

    // Update any other PHI nodes in EndBlock.
    for (PHINode &Phi : EndBlock->phis()) {
      if (SIUse == &Phi)
        continue;
      Phi.addIncoming(Phi.getIncomingValueForBlock(StartBlock), NewBlockT);
      Phi.addIncoming(Phi.getIncomingValueForBlock(StartBlock), NewBlockF);
      Phi.removeIncomingValue(StartBlock);
    }

    // Update the appropriate successor of the start block to point to the new
    // unfolded block.
    unsigned SuccNum = StartBlockTerm->getSuccessor(1) == EndBlock ? 1 : 0;
    StartBlockTerm->setSuccessor(SuccNum, NewBlockT);
    DTU->applyUpdates({{DominatorTree::Delete, StartBlock, EndBlock},
                       {DominatorTree::Insert, StartBlock, NewBlockT}});
  }

  // Preserve loop info
  if (Loop *L = LI->getLoopFor(StartBlock)) {
    for (BasicBlock *NewBB : *NewBBs)
      L->addBasicBlockToLoop(NewBB, *LI);
  }

  // The select is now dead.
  assert(SI->use_empty() && "Select must be dead now");
  SI->eraseFromParent();
}

namespace {
struct ClonedBlock {
  BasicBlock *BB;
  APInt State; ///< \p State corresponds to the next value of a switch stmnt.
};
} // namespace

typedef std::deque<BasicBlock *> PathType;
typedef std::vector<PathType> PathsType;
typedef SmallPtrSet<const BasicBlock *, 8> VisitedBlocks;
typedef std::vector<ClonedBlock> CloneList;

// This data structure keeps track of all blocks that have been cloned.  If two
// different ThreadingPaths clone the same block for a certain state it should
// be reused, and it can be looked up in this map.
typedef DenseMap<BasicBlock *, CloneList> DuplicateBlockMap;

// This map keeps track of all the new definitions for an instruction. This
// information is needed when restoring SSA form after cloning blocks.
typedef MapVector<Instruction *, std::vector<Instruction *>> DefMap;

inline raw_ostream &operator<<(raw_ostream &OS, const PathType &Path) {
  auto BBNames = llvm::map_range(
      Path, [](const BasicBlock *BB) { return BB->getNameOrAsOperand(); });
  OS << "< " << llvm::join(BBNames, ", ") << " >";
  return OS;
}

namespace {
/// ThreadingPath is a path in the control flow of a loop that can be threaded
/// by cloning necessary basic blocks and replacing conditional branches with
/// unconditional ones. A threading path includes a list of basic blocks, the
/// exit state, and the block that determines the next state.
struct ThreadingPath {
  /// Exit value is DFA's exit state for the given path.
  APInt getExitValue() const { return ExitVal; }
  void setExitValue(const ConstantInt *V) {
    ExitVal = V->getValue();
    IsExitValSet = true;
  }
  void setExitValue(const APInt &V) {
    ExitVal = V;
    IsExitValSet = true;
  }
  bool isExitValueSet() const { return IsExitValSet; }

  /// Determinator is the basic block that determines the next state of the DFA.
  const BasicBlock *getDeterminatorBB() const { return DBB; }
  void setDeterminator(const BasicBlock *BB) { DBB = BB; }

  /// Path is a list of basic blocks.
  const PathType &getPath() const { return Path; }
  void setPath(const PathType &NewPath) { Path = NewPath; }
  void push_back(BasicBlock *BB) { Path.push_back(BB); }
  void push_front(BasicBlock *BB) { Path.push_front(BB); }
  void appendExcludingFirst(const PathType &OtherPath) {
    llvm::append_range(Path, llvm::drop_begin(OtherPath));
  }

  void print(raw_ostream &OS) const {
    OS << Path << " [ " << ExitVal << ", " << DBB->getNameOrAsOperand() << " ]";
  }

private:
  PathType Path;
  APInt ExitVal;
  const BasicBlock *DBB = nullptr;
  bool IsExitValSet = false;
};

#ifndef NDEBUG
inline raw_ostream &operator<<(raw_ostream &OS, const ThreadingPath &TPath) {
  TPath.print(OS);
  return OS;
}
#endif

struct MainSwitch {
  MainSwitch(SwitchInst *SI, LoopInfo *LI, OptimizationRemarkEmitter *ORE)
      : LI(LI) {
    if (isCandidate(SI)) {
      Instr = SI;
    } else {
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "SwitchNotPredictable", SI)
               << "Switch instruction is not predictable.";
      });
    }
  }

  virtual ~MainSwitch() = default;

  SwitchInst *getInstr() const { return Instr; }
  const SmallVector<SelectInstToUnfold, 4> getSelectInsts() {
    return SelectInsts;
  }

private:
  /// Do a use-def chain traversal starting from the switch condition to see if
  /// \p SI is a potential condidate.
  ///
  /// Also, collect select instructions to unfold.
  bool isCandidate(const SwitchInst *SI) {
    std::deque<std::pair<Value *, BasicBlock *>> Q;
    SmallPtrSet<Value *, 16> SeenValues;
    SelectInsts.clear();

    Value *SICond = SI->getCondition();
    LLVM_DEBUG(dbgs() << "\tSICond: " << *SICond << "\n");
    if (!isa<PHINode>(SICond))
      return false;

    // The switch must be in a loop.
    const Loop *L = LI->getLoopFor(SI->getParent());
    if (!L)
      return false;

    addToQueue(SICond, nullptr, Q, SeenValues);

    while (!Q.empty()) {
      Value *Current = Q.front().first;
      BasicBlock *CurrentIncomingBB = Q.front().second;
      Q.pop_front();

      if (auto *Phi = dyn_cast<PHINode>(Current)) {
        for (BasicBlock *IncomingBB : Phi->blocks()) {
          Value *Incoming = Phi->getIncomingValueForBlock(IncomingBB);
          addToQueue(Incoming, IncomingBB, Q, SeenValues);
        }
        LLVM_DEBUG(dbgs() << "\tphi: " << *Phi << "\n");
      } else if (SelectInst *SelI = dyn_cast<SelectInst>(Current)) {
        if (!isValidSelectInst(SelI))
          return false;
        addToQueue(SelI->getTrueValue(), CurrentIncomingBB, Q, SeenValues);
        addToQueue(SelI->getFalseValue(), CurrentIncomingBB, Q, SeenValues);
        LLVM_DEBUG(dbgs() << "\tselect: " << *SelI << "\n");
        if (auto *SelIUse = dyn_cast<PHINode>(SelI->user_back()))
          SelectInsts.push_back(SelectInstToUnfold(SelI, SelIUse));
      } else if (isa<Constant>(Current)) {
        LLVM_DEBUG(dbgs() << "\tconst: " << *Current << "\n");
        continue;
      } else {
        LLVM_DEBUG(dbgs() << "\tother: " << *Current << "\n");
        // Allow unpredictable values. The hope is that those will be the
        // initial switch values that can be ignored (they will hit the
        // unthreaded switch) but this assumption will get checked later after
        // paths have been enumerated (in function getStateDefMap).

        // If the unpredictable value comes from the same inner loop it is
        // likely that it will also be on the enumerated paths, causing us to
        // exit after we have enumerated all the paths. This heuristic save
        // compile time because a search for all the paths can become expensive.
        if (EarlyExitHeuristic &&
            L->contains(LI->getLoopFor(CurrentIncomingBB))) {
          LLVM_DEBUG(dbgs()
                     << "\tExiting early due to unpredictability heuristic.\n");
          return false;
        }

        continue;
      }
    }

    return true;
  }

  void addToQueue(Value *Val, BasicBlock *BB,
                  std::deque<std::pair<Value *, BasicBlock *>> &Q,
                  SmallPtrSet<Value *, 16> &SeenValues) {
    if (SeenValues.insert(Val).second)
      Q.push_back({Val, BB});
  }

  bool isValidSelectInst(SelectInst *SI) {
    if (!SI->hasOneUse())
      return false;

    Instruction *SIUse = dyn_cast<Instruction>(SI->user_back());
    // The use of the select inst should be either a phi or another select.
    if (!SIUse || !(isa<PHINode>(SIUse) || isa<SelectInst>(SIUse)))
      return false;

    BasicBlock *SIBB = SI->getParent();

    // Currently, we can only expand select instructions in basic blocks with
    // one successor.
    BranchInst *SITerm = dyn_cast<BranchInst>(SIBB->getTerminator());
    if (!SITerm || !SITerm->isUnconditional())
      return false;

    // Only fold the select coming from directly where it is defined.
    // TODO: We have dealt with the select coming indirectly now. This
    // constraint can be relaxed.
    PHINode *PHIUser = dyn_cast<PHINode>(SIUse);
    if (PHIUser && PHIUser->getIncomingBlock(*SI->use_begin()) != SIBB)
      return false;

    // If select will not be sunk during unfolding, and it is in the same basic
    // block as another state defining select, then cannot unfold both.
    for (SelectInstToUnfold SIToUnfold : SelectInsts) {
      SelectInst *PrevSI = SIToUnfold.getInst();
      if (PrevSI->getTrueValue() != SI && PrevSI->getFalseValue() != SI &&
          PrevSI->getParent() == SI->getParent())
        return false;
    }

    return true;
  }

  LoopInfo *LI;
  SwitchInst *Instr = nullptr;
  SmallVector<SelectInstToUnfold, 4> SelectInsts;
};

struct AllSwitchPaths {
  AllSwitchPaths(const MainSwitch *MSwitch, OptimizationRemarkEmitter *ORE,
                 LoopInfo *LI, Loop *L)
      : Switch(MSwitch->getInstr()), SwitchBlock(Switch->getParent()), ORE(ORE),
        LI(LI), SwitchOuterLoop(L) {}

  std::vector<ThreadingPath> &getThreadingPaths() { return TPaths; }
  unsigned getNumThreadingPaths() { return TPaths.size(); }
  SwitchInst *getSwitchInst() { return Switch; }
  BasicBlock *getSwitchBlock() { return SwitchBlock; }

  void run() {
    findTPaths();
    unifyTPaths();
  }

private:
  // Value: an instruction that defines a switch state;
  // Key: the parent basic block of that instruction.
  typedef DenseMap<const BasicBlock *, const PHINode *> StateDefMap;
  std::vector<ThreadingPath> getPathsFromStateDefMap(StateDefMap &StateDef,
                                                     PHINode *Phi,
                                                     VisitedBlocks &VB,
                                                     unsigned PathsLimit) {
    std::vector<ThreadingPath> Res;
    auto *PhiBB = Phi->getParent();
    VB.insert(PhiBB);

    VisitedBlocks UniqueBlocks;
    for (auto *IncomingBB : Phi->blocks()) {
      if (Res.size() >= PathsLimit)
        break;
      if (!UniqueBlocks.insert(IncomingBB).second)
        continue;
      if (!SwitchOuterLoop->contains(IncomingBB))
        continue;

      Value *IncomingValue = Phi->getIncomingValueForBlock(IncomingBB);
      // We found the determinator. This is the start of our path.
      if (auto *C = dyn_cast<ConstantInt>(IncomingValue)) {
        // SwitchBlock is the determinator, unsupported unless its also the def.
        if (PhiBB == SwitchBlock &&
            SwitchBlock != cast<PHINode>(Switch->getOperand(0))->getParent())
          continue;
        ThreadingPath NewPath;
        NewPath.setDeterminator(PhiBB);
        NewPath.setExitValue(C);
        // Don't add SwitchBlock at the start, this is handled later.
        if (IncomingBB != SwitchBlock)
          NewPath.push_back(IncomingBB);
        NewPath.push_back(PhiBB);
        Res.push_back(NewPath);
        continue;
      }
      // Don't get into a cycle.
      if (VB.contains(IncomingBB) || IncomingBB == SwitchBlock)
        continue;
      // Recurse up the PHI chain.
      auto *IncomingPhi = dyn_cast<PHINode>(IncomingValue);
      if (!IncomingPhi)
        continue;
      auto *IncomingPhiDefBB = IncomingPhi->getParent();
      if (!StateDef.contains(IncomingPhiDefBB))
        continue;

      // Direct predecessor, just add to the path.
      if (IncomingPhiDefBB == IncomingBB) {
        assert(PathsLimit > Res.size());
        std::vector<ThreadingPath> PredPaths = getPathsFromStateDefMap(
            StateDef, IncomingPhi, VB, PathsLimit - Res.size());
        for (ThreadingPath &Path : PredPaths) {
          Path.push_back(PhiBB);
          Res.push_back(std::move(Path));
        }
        continue;
      }
      // Not a direct predecessor, find intermediate paths to append to the
      // existing path.
      if (VB.contains(IncomingPhiDefBB))
        continue;

      PathsType IntermediatePaths;
      assert(PathsLimit > Res.size());
      auto InterPathLimit = PathsLimit - Res.size();
      IntermediatePaths = paths(IncomingPhiDefBB, IncomingBB, VB,
                                /* PathDepth = */ 1, InterPathLimit);
      if (IntermediatePaths.empty())
        continue;

      assert(InterPathLimit >= IntermediatePaths.size());
      auto PredPathLimit = InterPathLimit / IntermediatePaths.size();
      std::vector<ThreadingPath> PredPaths =
          getPathsFromStateDefMap(StateDef, IncomingPhi, VB, PredPathLimit);
      for (const ThreadingPath &Path : PredPaths) {
        for (const PathType &IPath : IntermediatePaths) {
          ThreadingPath NewPath(Path);
          NewPath.appendExcludingFirst(IPath);
          NewPath.push_back(PhiBB);
          Res.push_back(NewPath);
        }
      }
    }
    VB.erase(PhiBB);
    return Res;
  }

  PathsType paths(BasicBlock *BB, BasicBlock *ToBB, VisitedBlocks &Visited,
                  unsigned PathDepth, unsigned PathsLimit) {
    PathsType Res;

    // Stop exploring paths after visiting MaxPathLength blocks
    if (PathDepth > MaxPathLength) {
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "MaxPathLengthReached",
                                          Switch)
               << "Exploration stopped after visiting MaxPathLength="
               << ore::NV("MaxPathLength", MaxPathLength) << " blocks.";
      });
      return Res;
    }

    Visited.insert(BB);
    if (++NumVisited > MaxNumVisitiedPaths)
      return Res;

    // Stop if we have reached the BB out of loop, since its successors have no
    // impact on the DFA.
    if (!SwitchOuterLoop->contains(BB))
      return Res;

    // Some blocks have multiple edges to the same successor, and this set
    // is used to prevent a duplicate path from being generated
    SmallPtrSet<BasicBlock *, 4> Successors;
    for (BasicBlock *Succ : successors(BB)) {
      if (Res.size() >= PathsLimit)
        break;
      if (!Successors.insert(Succ).second)
        continue;

      // Found a cycle through the final block.
      if (Succ == ToBB) {
        Res.push_back({BB, ToBB});
        continue;
      }

      // We have encountered a cycle, do not get caught in it
      if (Visited.contains(Succ))
        continue;

      auto *CurrLoop = LI->getLoopFor(BB);
      // Unlikely to be beneficial.
      if (Succ == CurrLoop->getHeader())
        continue;
      // Skip for now, revisit this condition later to see the impact on
      // coverage and compile time.
      if (LI->getLoopFor(Succ) != CurrLoop)
        continue;
      assert(PathsLimit > Res.size());
      PathsType SuccPaths =
          paths(Succ, ToBB, Visited, PathDepth + 1, PathsLimit - Res.size());
      for (PathType &Path : SuccPaths) {
        Path.push_front(BB);
        Res.push_back(Path);
      }
    }
    // This block could now be visited again from a different predecessor. Note
    // that this will result in exponential runtime. Subpaths could possibly be
    // cached but it takes a lot of memory to store them.
    Visited.erase(BB);
    return Res;
  }

  /// Walk the use-def chain and collect all the state-defining blocks and the
  /// PHI nodes in those blocks that define the state.
  StateDefMap getStateDefMap() const {
    StateDefMap Res;
    PHINode *FirstDef = dyn_cast<PHINode>(Switch->getOperand(0));
    assert(FirstDef && "The first definition must be a phi.");

    SmallVector<PHINode *, 8> Stack;
    Stack.push_back(FirstDef);
    SmallPtrSet<Value *, 16> SeenValues;

    while (!Stack.empty()) {
      PHINode *CurPhi = Stack.pop_back_val();

      Res[CurPhi->getParent()] = CurPhi;
      SeenValues.insert(CurPhi);

      for (BasicBlock *IncomingBB : CurPhi->blocks()) {
        PHINode *IncomingPhi =
            dyn_cast<PHINode>(CurPhi->getIncomingValueForBlock(IncomingBB));
        if (!IncomingPhi)
          continue;
        bool IsOutsideLoops = !SwitchOuterLoop->contains(IncomingBB);
        if (SeenValues.contains(IncomingPhi) || IsOutsideLoops)
          continue;

        Stack.push_back(IncomingPhi);
      }
    }

    return Res;
  }

  // Find all threadable paths.
  void findTPaths() {
    StateDefMap StateDef = getStateDefMap();
    if (StateDef.empty()) {
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "SwitchNotPredictable",
                                        Switch)
               << "Switch instruction is not predictable.";
      });
      return;
    }

    auto *SwitchPhi = cast<PHINode>(Switch->getOperand(0));
    auto *SwitchPhiDefBB = SwitchPhi->getParent();
    VisitedBlocks VB;
    // Get paths from the determinator BBs to SwitchPhiDefBB
    std::vector<ThreadingPath> PathsToPhiDef =
        getPathsFromStateDefMap(StateDef, SwitchPhi, VB, MaxNumPaths);
    if (SwitchPhiDefBB == SwitchBlock || PathsToPhiDef.empty()) {
      TPaths = std::move(PathsToPhiDef);
      return;
    }

    assert(MaxNumPaths >= PathsToPhiDef.size() && !PathsToPhiDef.empty());
    auto PathsLimit = MaxNumPaths / PathsToPhiDef.size();
    // Find and append paths from SwitchPhiDefBB to SwitchBlock.
    PathsType PathsToSwitchBB =
        paths(SwitchPhiDefBB, SwitchBlock, VB, /* PathDepth = */ 1, PathsLimit);
    if (PathsToSwitchBB.empty())
      return;

    std::vector<ThreadingPath> TempList;
    for (const ThreadingPath &Path : PathsToPhiDef) {
      for (const PathType &PathToSw : PathsToSwitchBB) {
        ThreadingPath PathCopy(Path);
        PathCopy.appendExcludingFirst(PathToSw);
        TempList.push_back(PathCopy);
      }
    }
    TPaths = std::move(TempList);
  }

  /// Fast helper to get the successor corresponding to a particular case value
  /// for a switch statement.
  BasicBlock *getNextCaseSuccessor(const APInt &NextState) {
    // Precompute the value => successor mapping
    if (CaseValToDest.empty()) {
      for (auto Case : Switch->cases()) {
        APInt CaseVal = Case.getCaseValue()->getValue();
        CaseValToDest[CaseVal] = Case.getCaseSuccessor();
      }
    }

    auto SuccIt = CaseValToDest.find(NextState);
    return SuccIt == CaseValToDest.end() ? Switch->getDefaultDest()
                                         : SuccIt->second;
  }

  // Two states are equivalent if they have the same switch destination.
  // Unify the states in different threading path if the states are equivalent.
  void unifyTPaths() {
    SmallDenseMap<BasicBlock *, APInt> DestToState;
    for (ThreadingPath &Path : TPaths) {
      APInt NextState = Path.getExitValue();
      BasicBlock *Dest = getNextCaseSuccessor(NextState);
      auto [StateIt, Inserted] = DestToState.try_emplace(Dest, NextState);
      if (Inserted)
        continue;
      if (NextState != StateIt->second) {
        LLVM_DEBUG(dbgs() << "Next state in " << Path << " is equivalent to "
                          << StateIt->second << "\n");
        Path.setExitValue(StateIt->second);
      }
    }
  }

  unsigned NumVisited = 0;
  SwitchInst *Switch;
  BasicBlock *SwitchBlock;
  OptimizationRemarkEmitter *ORE;
  std::vector<ThreadingPath> TPaths;
  DenseMap<APInt, BasicBlock *> CaseValToDest;
  LoopInfo *LI;
  Loop *SwitchOuterLoop;
};

struct TransformDFA {
  TransformDFA(AllSwitchPaths *SwitchPaths, DomTreeUpdater *DTU,
               AssumptionCache *AC, TargetTransformInfo *TTI,
               OptimizationRemarkEmitter *ORE,
               SmallPtrSet<const Value *, 32> EphValues)
      : SwitchPaths(SwitchPaths), DTU(DTU), AC(AC), TTI(TTI), ORE(ORE),
        EphValues(EphValues) {}

  bool run() {
    if (isLegalAndProfitableToTransform()) {
      createAllExitPaths();
      NumTransforms++;
      return true;
    }
    return false;
  }

private:
  /// This function performs both a legality check and profitability check at
  /// the same time since it is convenient to do so. It iterates through all
  /// blocks that will be cloned, and keeps track of the duplication cost. It
  /// also returns false if it is illegal to clone some required block.
  bool isLegalAndProfitableToTransform() {
    CodeMetrics Metrics;
    uint64_t NumClonedInst = 0;
    SwitchInst *Switch = SwitchPaths->getSwitchInst();

    // Don't thread switch without multiple successors.
    if (Switch->getNumSuccessors() <= 1)
      return false;

    // Note that DuplicateBlockMap is not being used as intended here. It is
    // just being used to ensure (BB, State) pairs are only counted once.
    DuplicateBlockMap DuplicateMap;
    for (ThreadingPath &TPath : SwitchPaths->getThreadingPaths()) {
      PathType PathBBs = TPath.getPath();
      APInt NextState = TPath.getExitValue();
      const BasicBlock *Determinator = TPath.getDeterminatorBB();

      // Update Metrics for the Switch block, this is always cloned
      BasicBlock *BB = SwitchPaths->getSwitchBlock();
      BasicBlock *VisitedBB = getClonedBB(BB, NextState, DuplicateMap);
      if (!VisitedBB) {
        Metrics.analyzeBasicBlock(BB, *TTI, EphValues);
        NumClonedInst += BB->sizeWithoutDebug();
        DuplicateMap[BB].push_back({BB, NextState});
      }

      // If the Switch block is the Determinator, then we can continue since
      // this is the only block that is cloned and we already counted for it.
      if (PathBBs.front() == Determinator)
        continue;

      // Otherwise update Metrics for all blocks that will be cloned. If any
      // block is already cloned and would be reused, don't double count it.
      auto DetIt = llvm::find(PathBBs, Determinator);
      for (auto BBIt = DetIt; BBIt != PathBBs.end(); BBIt++) {
        BB = *BBIt;
        VisitedBB = getClonedBB(BB, NextState, DuplicateMap);
        if (VisitedBB)
          continue;
        Metrics.analyzeBasicBlock(BB, *TTI, EphValues);
        NumClonedInst += BB->sizeWithoutDebug();
        DuplicateMap[BB].push_back({BB, NextState});
      }

      if (Metrics.notDuplicatable) {
        LLVM_DEBUG(dbgs() << "DFA Jump Threading: Not jump threading, contains "
                          << "non-duplicatable instructions.\n");
        ORE->emit([&]() {
          return OptimizationRemarkMissed(DEBUG_TYPE, "NonDuplicatableInst",
                                          Switch)
                 << "Contains non-duplicatable instructions.";
        });
        return false;
      }

      // FIXME: Allow jump threading with controlled convergence.
      if (Metrics.Convergence != ConvergenceKind::None) {
        LLVM_DEBUG(dbgs() << "DFA Jump Threading: Not jump threading, contains "
                          << "convergent instructions.\n");
        ORE->emit([&]() {
          return OptimizationRemarkMissed(DEBUG_TYPE, "ConvergentInst", Switch)
                 << "Contains convergent instructions.";
        });
        return false;
      }

      if (!Metrics.NumInsts.isValid()) {
        LLVM_DEBUG(dbgs() << "DFA Jump Threading: Not jump threading, contains "
                          << "instructions with invalid cost.\n");
        ORE->emit([&]() {
          return OptimizationRemarkMissed(DEBUG_TYPE, "ConvergentInst", Switch)
                 << "Contains instructions with invalid cost.";
        });
        return false;
      }
    }

    // Too much cloned instructions slow down later optimizations, especially
    // SLPVectorizer.
    // TODO: Thread the switch partially before reaching the threshold.
    uint64_t NumOrigInst = 0;
    for (auto *BB : DuplicateMap.keys())
      NumOrigInst += BB->sizeWithoutDebug();
    if (double(NumClonedInst) / double(NumOrigInst) > MaxClonedRate) {
      LLVM_DEBUG(dbgs() << "DFA Jump Threading: Not jump threading, too much "
                           "instructions wll be cloned\n");
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "NotProfitable", Switch)
               << "Too much instructions will be cloned.";
      });
      return false;
    }

    InstructionCost DuplicationCost = 0;

    unsigned JumpTableSize = 0;
    TTI->getEstimatedNumberOfCaseClusters(*Switch, JumpTableSize, nullptr,
                                          nullptr);
    if (JumpTableSize == 0) {
      // Factor in the number of conditional branches reduced from jump
      // threading. Assume that lowering the switch block is implemented by
      // using binary search, hence the LogBase2().
      unsigned CondBranches =
          APInt(32, Switch->getNumSuccessors()).ceilLogBase2();
      assert(CondBranches > 0 &&
             "The threaded switch must have multiple branches");
      DuplicationCost = Metrics.NumInsts / CondBranches;
    } else {
      // Compared with jump tables, the DFA optimizer removes an indirect branch
      // on each loop iteration, thus making branch prediction more precise. The
      // more branch targets there are, the more likely it is for the branch
      // predictor to make a mistake, and the more benefit there is in the DFA
      // optimizer. Thus, the more branch targets there are, the lower is the
      // cost of the DFA opt.
      DuplicationCost = Metrics.NumInsts / JumpTableSize;
    }

    LLVM_DEBUG(dbgs() << "\nDFA Jump Threading: Cost to jump thread block "
                      << SwitchPaths->getSwitchBlock()->getName()
                      << " is: " << DuplicationCost << "\n\n");

    if (DuplicationCost > CostThreshold) {
      LLVM_DEBUG(dbgs() << "Not jump threading, duplication cost exceeds the "
                        << "cost threshold.\n");
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "NotProfitable", Switch)
               << "Duplication cost exceeds the cost threshold (cost="
               << ore::NV("Cost", DuplicationCost)
               << ", threshold=" << ore::NV("Threshold", CostThreshold) << ").";
      });
      return false;
    }

    ORE->emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "JumpThreaded", Switch)
             << "Switch statement jump-threaded.";
    });

    return true;
  }

  /// Transform each threading path to effectively jump thread the DFA.
  void createAllExitPaths() {
    // Move the switch block to the end of the path, since it will be duplicated
    BasicBlock *SwitchBlock = SwitchPaths->getSwitchBlock();
    for (ThreadingPath &TPath : SwitchPaths->getThreadingPaths()) {
      LLVM_DEBUG(dbgs() << TPath << "\n");
      // TODO: Fix exit path creation logic so that we dont need this
      // placeholder.
      TPath.push_front(SwitchBlock);
    }

    // Transform the ThreadingPaths and keep track of the cloned values
    DuplicateBlockMap DuplicateMap;
    DefMap NewDefs;

    SmallPtrSet<BasicBlock *, 16> BlocksToClean;
    BlocksToClean.insert_range(successors(SwitchBlock));

    for (const ThreadingPath &TPath : SwitchPaths->getThreadingPaths()) {
      createExitPath(NewDefs, TPath, DuplicateMap, BlocksToClean, DTU);
      NumPaths++;
    }

    // After all paths are cloned, now update the last successor of the cloned
    // path so it skips over the switch statement
    for (const ThreadingPath &TPath : SwitchPaths->getThreadingPaths())
      updateLastSuccessor(TPath, DuplicateMap, DTU);

    // For each instruction that was cloned and used outside, update its uses
    updateSSA(NewDefs);

    // Clean PHI Nodes for the newly created blocks
    for (BasicBlock *BB : BlocksToClean)
      cleanPhiNodes(BB);
  }

  /// For a specific ThreadingPath \p Path, create an exit path starting from
  /// the determinator block.
  ///
  /// To remember the correct destination, we have to duplicate blocks
  /// corresponding to each state. Also update the terminating instruction of
  /// the predecessors, and phis in the successor blocks.
  void createExitPath(DefMap &NewDefs, const ThreadingPath &Path,
                      DuplicateBlockMap &DuplicateMap,
                      SmallPtrSet<BasicBlock *, 16> &BlocksToClean,
                      DomTreeUpdater *DTU) {
    APInt NextState = Path.getExitValue();
    const BasicBlock *Determinator = Path.getDeterminatorBB();
    PathType PathBBs = Path.getPath();

    // Don't select the placeholder block in front
    if (PathBBs.front() == Determinator)
      PathBBs.pop_front();

    auto DetIt = llvm::find(PathBBs, Determinator);
    // When there is only one BB in PathBBs, the determinator takes itself as a
    // direct predecessor.
    BasicBlock *PrevBB = PathBBs.size() == 1 ? *DetIt : *std::prev(DetIt);
    for (auto BBIt = DetIt; BBIt != PathBBs.end(); BBIt++) {
      BasicBlock *BB = *BBIt;
      BlocksToClean.insert(BB);

      // We already cloned BB for this NextState, now just update the branch
      // and continue.
      BasicBlock *NextBB = getClonedBB(BB, NextState, DuplicateMap);
      if (NextBB) {
        updatePredecessor(PrevBB, BB, NextBB, DTU);
        PrevBB = NextBB;
        continue;
      }

      // Clone the BB and update the successor of Prev to jump to the new block
      BasicBlock *NewBB = cloneBlockAndUpdatePredecessor(
          BB, PrevBB, NextState, DuplicateMap, NewDefs, DTU);
      DuplicateMap[BB].push_back({NewBB, NextState});
      BlocksToClean.insert(NewBB);
      PrevBB = NewBB;
    }
  }

  /// Restore SSA form after cloning blocks.
  ///
  /// Each cloned block creates new defs for a variable, and the uses need to be
  /// updated to reflect this. The uses may be replaced with a cloned value, or
  /// some derived phi instruction. Note that all uses of a value defined in the
  /// same block were already remapped when cloning the block.
  void updateSSA(DefMap &NewDefs) {
    SSAUpdaterBulk SSAUpdate;
    SmallVector<Use *, 16> UsesToRename;

    for (const auto &KV : NewDefs) {
      Instruction *I = KV.first;
      BasicBlock *BB = I->getParent();
      std::vector<Instruction *> Cloned = KV.second;

      // Scan all uses of this instruction to see if it is used outside of its
      // block, and if so, record them in UsesToRename.
      for (Use &U : I->uses()) {
        Instruction *User = cast<Instruction>(U.getUser());
        if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
          if (UserPN->getIncomingBlock(U) == BB)
            continue;
        } else if (User->getParent() == BB) {
          continue;
        }

        UsesToRename.push_back(&U);
      }

      // If there are no uses outside the block, we're done with this
      // instruction.
      if (UsesToRename.empty())
        continue;
      LLVM_DEBUG(dbgs() << "DFA-JT: Renaming non-local uses of: " << *I
                        << "\n");

      // We found a use of I outside of BB.  Rename all uses of I that are
      // outside its block to be uses of the appropriate PHI node etc.  See
      // ValuesInBlocks with the values we know.
      unsigned VarNum = SSAUpdate.AddVariable(I->getName(), I->getType());
      SSAUpdate.AddAvailableValue(VarNum, BB, I);
      for (Instruction *New : Cloned)
        SSAUpdate.AddAvailableValue(VarNum, New->getParent(), New);

      while (!UsesToRename.empty())
        SSAUpdate.AddUse(VarNum, UsesToRename.pop_back_val());

      LLVM_DEBUG(dbgs() << "\n");
    }
    // SSAUpdater handles phi placement and renaming uses with the appropriate
    // value.
    SSAUpdate.RewriteAllUses(&DTU->getDomTree());
  }

  /// Helper to get the successor corresponding to a particular case value for
  /// a switch statement.
  /// TODO: Unify it with SwitchPaths->getNextCaseSuccessor(SwitchInst *Switch)
  /// by updating cached value => successor mapping during threading.
  static BasicBlock *getNextCaseSuccessor(SwitchInst *Switch,
                                          const APInt &NextState) {
    BasicBlock *NextCase = nullptr;
    for (auto Case : Switch->cases()) {
      if (Case.getCaseValue()->getValue() == NextState) {
        NextCase = Case.getCaseSuccessor();
        break;
      }
    }
    if (!NextCase)
      NextCase = Switch->getDefaultDest();
    return NextCase;
  }

  /// Clones a basic block, and adds it to the CFG.
  ///
  /// This function also includes updating phi nodes in the successors of the
  /// BB, and remapping uses that were defined locally in the cloned BB.
  BasicBlock *cloneBlockAndUpdatePredecessor(BasicBlock *BB, BasicBlock *PrevBB,
                                             const APInt &NextState,
                                             DuplicateBlockMap &DuplicateMap,
                                             DefMap &NewDefs,
                                             DomTreeUpdater *DTU) {
    ValueToValueMapTy VMap;
    BasicBlock *NewBB = CloneBasicBlock(
        BB, VMap, ".jt" + std::to_string(NextState.getLimitedValue()),
        BB->getParent());
    NewBB->moveAfter(BB);
    NumCloned++;

    for (Instruction &I : *NewBB) {
      // Do not remap operands of PHINode in case a definition in BB is an
      // incoming value to a phi in the same block. This incoming value will
      // be renamed later while restoring SSA.
      if (isa<PHINode>(&I))
        continue;
      RemapInstruction(&I, VMap,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
      if (AssumeInst *II = dyn_cast<AssumeInst>(&I))
        AC->registerAssumption(II);
    }

    updateSuccessorPhis(BB, NewBB, NextState, VMap, DuplicateMap);
    updatePredecessor(PrevBB, BB, NewBB, DTU);
    updateDefMap(NewDefs, VMap);

    // Add all successors to the DominatorTree
    SmallPtrSet<BasicBlock *, 4> SuccSet;
    for (auto *SuccBB : successors(NewBB)) {
      if (SuccSet.insert(SuccBB).second)
        DTU->applyUpdates({{DominatorTree::Insert, NewBB, SuccBB}});
    }
    SuccSet.clear();
    return NewBB;
  }

  /// Update the phi nodes in BB's successors.
  ///
  /// This means creating a new incoming value from NewBB with the new
  /// instruction wherever there is an incoming value from BB.
  void updateSuccessorPhis(BasicBlock *BB, BasicBlock *ClonedBB,
                           const APInt &NextState, ValueToValueMapTy &VMap,
                           DuplicateBlockMap &DuplicateMap) {
    std::vector<BasicBlock *> BlocksToUpdate;

    // If BB is the last block in the path, we can simply update the one case
    // successor that will be reached.
    if (BB == SwitchPaths->getSwitchBlock()) {
      SwitchInst *Switch = SwitchPaths->getSwitchInst();
      BasicBlock *NextCase = getNextCaseSuccessor(Switch, NextState);
      BlocksToUpdate.push_back(NextCase);
      BasicBlock *ClonedSucc = getClonedBB(NextCase, NextState, DuplicateMap);
      if (ClonedSucc)
        BlocksToUpdate.push_back(ClonedSucc);
    }
    // Otherwise update phis in all successors.
    else {
      for (BasicBlock *Succ : successors(BB)) {
        BlocksToUpdate.push_back(Succ);

        // Check if a successor has already been cloned for the particular exit
        // value. In this case if a successor was already cloned, the phi nodes
        // in the cloned block should be updated directly.
        BasicBlock *ClonedSucc = getClonedBB(Succ, NextState, DuplicateMap);
        if (ClonedSucc)
          BlocksToUpdate.push_back(ClonedSucc);
      }
    }

    // If there is a phi with an incoming value from BB, create a new incoming
    // value for the new predecessor ClonedBB. The value will either be the same
    // value from BB or a cloned value.
    for (BasicBlock *Succ : BlocksToUpdate) {
      for (PHINode &Phi : Succ->phis()) {
        Value *Incoming = Phi.getIncomingValueForBlock(BB);
        if (Incoming) {
          if (isa<Constant>(Incoming)) {
            Phi.addIncoming(Incoming, ClonedBB);
            continue;
          }
          Value *ClonedVal = VMap[Incoming];
          if (ClonedVal)
            Phi.addIncoming(ClonedVal, ClonedBB);
          else
            Phi.addIncoming(Incoming, ClonedBB);
        }
      }
    }
  }

  /// Sets the successor of PrevBB to be NewBB instead of OldBB. Note that all
  /// other successors are kept as well.
  void updatePredecessor(BasicBlock *PrevBB, BasicBlock *OldBB,
                         BasicBlock *NewBB, DomTreeUpdater *DTU) {
    // When a path is reused, there is a chance that predecessors were already
    // updated before. Check if the predecessor needs to be updated first.
    if (!isPredecessor(OldBB, PrevBB))
      return;

    Instruction *PrevTerm = PrevBB->getTerminator();
    for (unsigned Idx = 0; Idx < PrevTerm->getNumSuccessors(); Idx++) {
      if (PrevTerm->getSuccessor(Idx) == OldBB) {
        OldBB->removePredecessor(PrevBB, /* KeepOneInputPHIs = */ true);
        PrevTerm->setSuccessor(Idx, NewBB);
      }
    }
    DTU->applyUpdates({{DominatorTree::Delete, PrevBB, OldBB},
                       {DominatorTree::Insert, PrevBB, NewBB}});
  }

  /// Add new value mappings to the DefMap to keep track of all new definitions
  /// for a particular instruction. These will be used while updating SSA form.
  void updateDefMap(DefMap &NewDefs, ValueToValueMapTy &VMap) {
    SmallVector<std::pair<Instruction *, Instruction *>> NewDefsVector;
    NewDefsVector.reserve(VMap.size());

    for (auto Entry : VMap) {
      Instruction *Inst =
          dyn_cast<Instruction>(const_cast<Value *>(Entry.first));
      if (!Inst || !Entry.second || isa<BranchInst>(Inst) ||
          isa<SwitchInst>(Inst)) {
        continue;
      }

      Instruction *Cloned = dyn_cast<Instruction>(Entry.second);
      if (!Cloned)
        continue;

      NewDefsVector.push_back({Inst, Cloned});
    }

    // Sort the defs to get deterministic insertion order into NewDefs.
    sort(NewDefsVector, [](const auto &LHS, const auto &RHS) {
      if (LHS.first == RHS.first)
        return LHS.second->comesBefore(RHS.second);
      return LHS.first->comesBefore(RHS.first);
    });

    for (const auto &KV : NewDefsVector)
      NewDefs[KV.first].push_back(KV.second);
  }

  /// Update the last branch of a particular cloned path to point to the correct
  /// case successor.
  ///
  /// Note that this is an optional step and would have been done in later
  /// optimizations, but it makes the CFG significantly easier to work with.
  void updateLastSuccessor(const ThreadingPath &TPath,
                           DuplicateBlockMap &DuplicateMap,
                           DomTreeUpdater *DTU) {
    APInt NextState = TPath.getExitValue();
    BasicBlock *BB = TPath.getPath().back();
    BasicBlock *LastBlock = getClonedBB(BB, NextState, DuplicateMap);

    // Note multiple paths can end at the same block so check that it is not
    // updated yet
    if (!isa<SwitchInst>(LastBlock->getTerminator()))
      return;
    SwitchInst *Switch = cast<SwitchInst>(LastBlock->getTerminator());
    BasicBlock *NextCase = getNextCaseSuccessor(Switch, NextState);

    std::vector<DominatorTree::UpdateType> DTUpdates;
    SmallPtrSet<BasicBlock *, 4> SuccSet;
    for (BasicBlock *Succ : successors(LastBlock)) {
      if (Succ != NextCase && SuccSet.insert(Succ).second)
        DTUpdates.push_back({DominatorTree::Delete, LastBlock, Succ});
    }

    Switch->eraseFromParent();
    BranchInst::Create(NextCase, LastBlock);

    DTU->applyUpdates(DTUpdates);
  }

  /// After cloning blocks, some of the phi nodes have extra incoming values
  /// that are no longer used. This function removes them.
  void cleanPhiNodes(BasicBlock *BB) {
    // If BB is no longer reachable, remove any remaining phi nodes
    if (pred_empty(BB)) {
      for (PHINode &PN : make_early_inc_range(BB->phis())) {
        PN.replaceAllUsesWith(PoisonValue::get(PN.getType()));
        PN.eraseFromParent();
      }
      return;
    }

    // Remove any incoming values that come from an invalid predecessor
    for (PHINode &Phi : BB->phis())
      Phi.removeIncomingValueIf([&](unsigned Index) {
        BasicBlock *IncomingBB = Phi.getIncomingBlock(Index);
        return !isPredecessor(BB, IncomingBB);
      });
  }

  /// Checks if BB was already cloned for a particular next state value. If it
  /// was then it returns this cloned block, and otherwise null.
  BasicBlock *getClonedBB(BasicBlock *BB, const APInt &NextState,
                          DuplicateBlockMap &DuplicateMap) {
    CloneList ClonedBBs = DuplicateMap[BB];

    // Find an entry in the CloneList with this NextState. If it exists then
    // return the corresponding BB
    auto It = llvm::find_if(ClonedBBs, [NextState](const ClonedBlock &C) {
      return C.State == NextState;
    });
    return It != ClonedBBs.end() ? (*It).BB : nullptr;
  }

  /// Returns true if IncomingBB is a predecessor of BB.
  bool isPredecessor(BasicBlock *BB, BasicBlock *IncomingBB) {
    return llvm::is_contained(predecessors(BB), IncomingBB);
  }

  AllSwitchPaths *SwitchPaths;
  DomTreeUpdater *DTU;
  AssumptionCache *AC;
  TargetTransformInfo *TTI;
  OptimizationRemarkEmitter *ORE;
  SmallPtrSet<const Value *, 32> EphValues;
  std::vector<ThreadingPath> TPaths;
};
} // namespace

bool DFAJumpThreading::run(Function &F) {
  LLVM_DEBUG(dbgs() << "\nDFA Jump threading: " << F.getName() << "\n");

  if (F.hasOptSize()) {
    LLVM_DEBUG(dbgs() << "Skipping due to the 'minsize' attribute\n");
    return false;
  }

  if (ClViewCfgBefore)
    F.viewCFG();

  SmallVector<AllSwitchPaths, 2> ThreadableLoops;
  bool MadeChanges = false;
  LoopInfoBroken = false;

  for (BasicBlock &BB : F) {
    auto *SI = dyn_cast<SwitchInst>(BB.getTerminator());
    if (!SI)
      continue;

    LLVM_DEBUG(dbgs() << "\nCheck if SwitchInst in BB " << BB.getName()
                      << " is a candidate\n");
    MainSwitch Switch(SI, LI, ORE);

    if (!Switch.getInstr()) {
      LLVM_DEBUG(dbgs() << "\nSwitchInst in BB " << BB.getName() << " is not a "
                        << "candidate for jump threading\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "\nSwitchInst in BB " << BB.getName() << " is a "
                      << "candidate for jump threading\n");
    LLVM_DEBUG(SI->dump());

    unfoldSelectInstrs(Switch.getSelectInsts());
    if (!Switch.getSelectInsts().empty())
      MadeChanges = true;

    AllSwitchPaths SwitchPaths(&Switch, ORE, LI,
                               LI->getLoopFor(&BB)->getOutermostLoop());
    SwitchPaths.run();

    if (SwitchPaths.getNumThreadingPaths() > 0) {
      ThreadableLoops.push_back(SwitchPaths);

      // For the time being limit this optimization to occurring once in a
      // function since it can change the CFG significantly. This is not a
      // strict requirement but it can cause buggy behavior if there is an
      // overlap of blocks in different opportunities. There is a lot of room to
      // experiment with catching more opportunities here.
      // NOTE: To release this contraint, we must handle LoopInfo invalidation
      break;
    }
  }

#ifdef NDEBUG
  LI->verify(DTU->getDomTree());
#endif

  SmallPtrSet<const Value *, 32> EphValues;
  if (ThreadableLoops.size() > 0)
    CodeMetrics::collectEphemeralValues(&F, AC, EphValues);

  for (AllSwitchPaths SwitchPaths : ThreadableLoops) {
    TransformDFA Transform(&SwitchPaths, DTU, AC, TTI, ORE, EphValues);
    if (Transform.run())
      MadeChanges = LoopInfoBroken = true;
  }

  DTU->flush();

#ifdef EXPENSIVE_CHECKS
  verifyFunction(F, &dbgs());
#endif

  if (MadeChanges && VerifyDomInfo)
    assert(DTU->getDomTree().verify(DominatorTree::VerificationLevel::Full) &&
           "Failed to maintain validity of domtree!");

  return MadeChanges;
}

/// Integrate with the new Pass Manager
PreservedAnalyses DFAJumpThreadingPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  OptimizationRemarkEmitter ORE(&F);

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  DFAJumpThreading ThreadImpl(&AC, &DTU, &LI, &TTI, &ORE);
  if (!ThreadImpl.run(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  if (!ThreadImpl.LoopInfoBroken)
    PA.preserve<LoopAnalysis>();
  return PA;
}
