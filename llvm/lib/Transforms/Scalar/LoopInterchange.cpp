//===- LoopInterchange.cpp - Loop interchange pass------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This Pass handles loop interchange transform.
// This pass interchanges loops to provide a more cache-friendly memory access
// patterns.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

#define DEBUG_TYPE "loop-interchange"

static cl::opt<int> LoopInterchangeCostThreshold(
    "loop-interchange-threshold", cl::init(0), cl::Hidden,
    cl::desc("Interchange if you gain more than this number"));

namespace {

typedef SmallVector<Loop *, 8> LoopVector;

// TODO: Check if we can use a sparse matrix here.
typedef std::vector<std::vector<char>> CharMatrix;

// Maximum number of dependencies that can be handled in the dependency matrix.
static const unsigned MaxMemInstrCount = 100;

// Maximum loop depth supported.
static const unsigned MaxLoopNestDepth = 10;

struct LoopInterchange;

#ifdef DUMP_DEP_MATRICIES
void printDepMatrix(CharMatrix &DepMatrix) {
  for (auto I = DepMatrix.begin(), E = DepMatrix.end(); I != E; ++I) {
    std::vector<char> Vec = *I;
    for (auto II = Vec.begin(), EE = Vec.end(); II != EE; ++II)
      DEBUG(dbgs() << *II << " ");
    DEBUG(dbgs() << "\n");
  }
}
#endif

static bool populateDependencyMatrix(CharMatrix &DepMatrix, unsigned Level,
                                     Loop *L, DependenceInfo *DI) {
  typedef SmallVector<Value *, 16> ValueVector;
  ValueVector MemInstr;

  // For each block.
  for (Loop::block_iterator BB = L->block_begin(), BE = L->block_end();
       BB != BE; ++BB) {
    // Scan the BB and collect legal loads and stores.
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E;
         ++I) {
      if (!isa<Instruction>(I))
        return false;
      if (LoadInst *Ld = dyn_cast<LoadInst>(I)) {
        if (!Ld->isSimple())
          return false;
        MemInstr.push_back(&*I);
      } else if (StoreInst *St = dyn_cast<StoreInst>(I)) {
        if (!St->isSimple())
          return false;
        MemInstr.push_back(&*I);
      }
    }
  }

  DEBUG(dbgs() << "Found " << MemInstr.size()
               << " Loads and Stores to analyze\n");

  ValueVector::iterator I, IE, J, JE;

  for (I = MemInstr.begin(), IE = MemInstr.end(); I != IE; ++I) {
    for (J = I, JE = MemInstr.end(); J != JE; ++J) {
      std::vector<char> Dep;
      Instruction *Src = cast<Instruction>(*I);
      Instruction *Dst = cast<Instruction>(*J);
      if (Src == Dst)
        continue;
      if (isa<LoadInst>(Src) && isa<LoadInst>(Dst))
        continue;
      if (auto D = DI->depends(Src, Dst, true)) {
        DEBUG(dbgs() << "Found Dependency between Src and Dst\n"
                     << " Src:" << *Src << "\n Dst:" << *Dst << '\n');
        if (D->isFlow()) {
          // TODO: Handle Flow dependence.Check if it is sufficient to populate
          // the Dependence Matrix with the direction reversed.
          DEBUG(dbgs() << "Flow dependence not handled\n");
          return false;
        }
        if (D->isAnti()) {
          DEBUG(dbgs() << "Found Anti dependence\n");
          unsigned Levels = D->getLevels();
          char Direction;
          for (unsigned II = 1; II <= Levels; ++II) {
            const SCEV *Distance = D->getDistance(II);
            const SCEVConstant *SCEVConst =
                dyn_cast_or_null<SCEVConstant>(Distance);
            if (SCEVConst) {
              const ConstantInt *CI = SCEVConst->getValue();
              if (CI->isNegative())
                Direction = '<';
              else if (CI->isZero())
                Direction = '=';
              else
                Direction = '>';
              Dep.push_back(Direction);
            } else if (D->isScalar(II)) {
              Direction = 'S';
              Dep.push_back(Direction);
            } else {
              unsigned Dir = D->getDirection(II);
              if (Dir == Dependence::DVEntry::LT ||
                  Dir == Dependence::DVEntry::LE)
                Direction = '<';
              else if (Dir == Dependence::DVEntry::GT ||
                       Dir == Dependence::DVEntry::GE)
                Direction = '>';
              else if (Dir == Dependence::DVEntry::EQ)
                Direction = '=';
              else
                Direction = '*';
              Dep.push_back(Direction);
            }
          }
          while (Dep.size() != Level) {
            Dep.push_back('I');
          }

          DepMatrix.push_back(Dep);
          if (DepMatrix.size() > MaxMemInstrCount) {
            DEBUG(dbgs() << "Cannot handle more than " << MaxMemInstrCount
                         << " dependencies inside loop\n");
            return false;
          }
        }
      }
    }
  }

  // We don't have a DepMatrix to check legality return false.
  if (DepMatrix.size() == 0)
    return false;
  return true;
}

// A loop is moved from index 'from' to an index 'to'. Update the Dependence
// matrix by exchanging the two columns.
static void interChangeDependencies(CharMatrix &DepMatrix, unsigned FromIndx,
                                    unsigned ToIndx) {
  unsigned numRows = DepMatrix.size();
  for (unsigned i = 0; i < numRows; ++i) {
    char TmpVal = DepMatrix[i][ToIndx];
    DepMatrix[i][ToIndx] = DepMatrix[i][FromIndx];
    DepMatrix[i][FromIndx] = TmpVal;
  }
}

// Checks if outermost non '=','S'or'I' dependence in the dependence matrix is
// '>'
static bool isOuterMostDepPositive(CharMatrix &DepMatrix, unsigned Row,
                                   unsigned Column) {
  for (unsigned i = 0; i <= Column; ++i) {
    if (DepMatrix[Row][i] == '<')
      return false;
    if (DepMatrix[Row][i] == '>')
      return true;
  }
  // All dependencies were '=','S' or 'I'
  return false;
}

// Checks if no dependence exist in the dependency matrix in Row before Column.
static bool containsNoDependence(CharMatrix &DepMatrix, unsigned Row,
                                 unsigned Column) {
  for (unsigned i = 0; i < Column; ++i) {
    if (DepMatrix[Row][i] != '=' || DepMatrix[Row][i] != 'S' ||
        DepMatrix[Row][i] != 'I')
      return false;
  }
  return true;
}

static bool validDepInterchange(CharMatrix &DepMatrix, unsigned Row,
                                unsigned OuterLoopId, char InnerDep,
                                char OuterDep) {

  if (isOuterMostDepPositive(DepMatrix, Row, OuterLoopId))
    return false;

  if (InnerDep == OuterDep)
    return true;

  // It is legal to interchange if and only if after interchange no row has a
  // '>' direction as the leftmost non-'='.

  if (InnerDep == '=' || InnerDep == 'S' || InnerDep == 'I')
    return true;

  if (InnerDep == '<')
    return true;

  if (InnerDep == '>') {
    // If OuterLoopId represents outermost loop then interchanging will make the
    // 1st dependency as '>'
    if (OuterLoopId == 0)
      return false;

    // If all dependencies before OuterloopId are '=','S'or 'I'. Then
    // interchanging will result in this row having an outermost non '='
    // dependency of '>'
    if (!containsNoDependence(DepMatrix, Row, OuterLoopId))
      return true;
  }

  return false;
}

// Checks if it is legal to interchange 2 loops.
// [Theorem] A permutation of the loops in a perfect nest is legal if and only
// if the direction matrix, after the same permutation is applied to its
// columns, has no ">" direction as the leftmost non-"=" direction in any row.
static bool isLegalToInterChangeLoops(CharMatrix &DepMatrix,
                                      unsigned InnerLoopId,
                                      unsigned OuterLoopId) {

  unsigned NumRows = DepMatrix.size();
  // For each row check if it is valid to interchange.
  for (unsigned Row = 0; Row < NumRows; ++Row) {
    char InnerDep = DepMatrix[Row][InnerLoopId];
    char OuterDep = DepMatrix[Row][OuterLoopId];
    if (InnerDep == '*' || OuterDep == '*')
      return false;
    if (!validDepInterchange(DepMatrix, Row, OuterLoopId, InnerDep, OuterDep))
      return false;
  }
  return true;
}

static void populateWorklist(Loop &L, SmallVector<LoopVector, 8> &V) {

  DEBUG(dbgs() << "Calling populateWorklist on Func: "
               << L.getHeader()->getParent()->getName() << " Loop: %"
               << L.getHeader()->getName() << '\n');
  LoopVector LoopList;
  Loop *CurrentLoop = &L;
  const std::vector<Loop *> *Vec = &CurrentLoop->getSubLoops();
  while (!Vec->empty()) {
    // The current loop has multiple subloops in it hence it is not tightly
    // nested.
    // Discard all loops above it added into Worklist.
    if (Vec->size() != 1) {
      LoopList.clear();
      return;
    }
    LoopList.push_back(CurrentLoop);
    CurrentLoop = Vec->front();
    Vec = &CurrentLoop->getSubLoops();
  }
  LoopList.push_back(CurrentLoop);
  V.push_back(std::move(LoopList));
}

static PHINode *getInductionVariable(Loop *L, ScalarEvolution *SE) {
  PHINode *InnerIndexVar = L->getCanonicalInductionVariable();
  if (InnerIndexVar)
    return InnerIndexVar;
  if (L->getLoopLatch() == nullptr || L->getLoopPredecessor() == nullptr)
    return nullptr;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    PHINode *PhiVar = cast<PHINode>(I);
    Type *PhiTy = PhiVar->getType();
    if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
        !PhiTy->isPointerTy())
      return nullptr;
    const SCEVAddRecExpr *AddRec =
        dyn_cast<SCEVAddRecExpr>(SE->getSCEV(PhiVar));
    if (!AddRec || !AddRec->isAffine())
      continue;
    const SCEV *Step = AddRec->getStepRecurrence(*SE);
    if (!isa<SCEVConstant>(Step))
      continue;
    // Found the induction variable.
    // FIXME: Handle loops with more than one induction variable. Note that,
    // currently, legality makes sure we have only one induction variable.
    return PhiVar;
  }
  return nullptr;
}

/// LoopInterchangeLegality checks if it is legal to interchange the loop.
class LoopInterchangeLegality {
public:
  LoopInterchangeLegality(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                          LoopInfo *LI, DominatorTree *DT, bool PreserveLCSSA)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), LI(LI), DT(DT),
        PreserveLCSSA(PreserveLCSSA), InnerLoopHasReduction(false) {}

  /// Check if the loops can be interchanged.
  bool canInterchangeLoops(unsigned InnerLoopId, unsigned OuterLoopId,
                           CharMatrix &DepMatrix);
  /// Check if the loop structure is understood. We do not handle triangular
  /// loops for now.
  bool isLoopStructureUnderstood(PHINode *InnerInductionVar);

  bool currentLimitations();

  bool hasInnerLoopReduction() { return InnerLoopHasReduction; }

private:
  bool tightlyNested(Loop *Outer, Loop *Inner);
  bool containsUnsafeInstructionsInHeader(BasicBlock *BB);
  bool areAllUsesReductions(Instruction *Ins, Loop *L);
  bool containsUnsafeInstructionsInLatch(BasicBlock *BB);
  bool findInductionAndReductions(Loop *L,
                                  SmallVector<PHINode *, 8> &Inductions,
                                  SmallVector<PHINode *, 8> &Reductions);
  Loop *OuterLoop;
  Loop *InnerLoop;

  ScalarEvolution *SE;
  LoopInfo *LI;
  DominatorTree *DT;
  bool PreserveLCSSA;

  bool InnerLoopHasReduction;
};

/// LoopInterchangeProfitability checks if it is profitable to interchange the
/// loop.
class LoopInterchangeProfitability {
public:
  LoopInterchangeProfitability(Loop *Outer, Loop *Inner, ScalarEvolution *SE)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE) {}

  /// Check if the loop interchange is profitable.
  bool isProfitable(unsigned InnerLoopId, unsigned OuterLoopId,
                    CharMatrix &DepMatrix);

private:
  int getInstrOrderCost();

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;
};

/// LoopInterchangeTransform interchanges the loop.
class LoopInterchangeTransform {
public:
  LoopInterchangeTransform(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                           LoopInfo *LI, DominatorTree *DT,
                           BasicBlock *LoopNestExit,
                           bool InnerLoopContainsReductions)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), LI(LI), DT(DT),
        LoopExit(LoopNestExit),
        InnerLoopHasReduction(InnerLoopContainsReductions) {}

  /// Interchange OuterLoop and InnerLoop.
  bool transform();
  void restructureLoops(Loop *InnerLoop, Loop *OuterLoop);
  void removeChildLoop(Loop *OuterLoop, Loop *InnerLoop);

private:
  void splitInnerLoopLatch(Instruction *);
  void splitInnerLoopHeader();
  bool adjustLoopLinks();
  void adjustLoopPreheaders();
  bool adjustLoopBranches();
  void updateIncomingBlock(BasicBlock *CurrBlock, BasicBlock *OldPred,
                           BasicBlock *NewPred);

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;
  LoopInfo *LI;
  DominatorTree *DT;
  BasicBlock *LoopExit;
  bool InnerLoopHasReduction;
};

// Main LoopInterchange Pass.
struct LoopInterchange : public FunctionPass {
  static char ID;
  ScalarEvolution *SE;
  LoopInfo *LI;
  DependenceInfo *DI;
  DominatorTree *DT;
  bool PreserveLCSSA;
  LoopInterchange()
      : FunctionPass(ID), SE(nullptr), LI(nullptr), DI(nullptr), DT(nullptr) {
    initializeLoopInterchangePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DependenceAnalysisWrapperPass>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DI = &getAnalysis<DependenceAnalysisWrapperPass>().getDI();
    auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    DT = DTWP ? &DTWP->getDomTree() : nullptr;
    PreserveLCSSA = mustPreserveAnalysisID(LCSSAID);

    // Build up a worklist of loop pairs to analyze.
    SmallVector<LoopVector, 8> Worklist;

    for (Loop *L : *LI)
      populateWorklist(*L, Worklist);

    DEBUG(dbgs() << "Worklist size = " << Worklist.size() << "\n");
    bool Changed = true;
    while (!Worklist.empty()) {
      LoopVector LoopList = Worklist.pop_back_val();
      Changed = processLoopList(LoopList, F);
    }
    return Changed;
  }

  bool isComputableLoopNest(LoopVector LoopList) {
    for (Loop *L : LoopList) {
      const SCEV *ExitCountOuter = SE->getBackedgeTakenCount(L);
      if (ExitCountOuter == SE->getCouldNotCompute()) {
        DEBUG(dbgs() << "Couldn't compute backedge count\n");
        return false;
      }
      if (L->getNumBackEdges() != 1) {
        DEBUG(dbgs() << "NumBackEdges is not equal to 1\n");
        return false;
      }
      if (!L->getExitingBlock()) {
        DEBUG(dbgs() << "Loop doesn't have unique exit block\n");
        return false;
      }
    }
    return true;
  }

  unsigned selectLoopForInterchange(const LoopVector &LoopList) {
    // TODO: Add a better heuristic to select the loop to be interchanged based
    // on the dependence matrix. Currently we select the innermost loop.
    return LoopList.size() - 1;
  }

  bool processLoopList(LoopVector LoopList, Function &F) {

    bool Changed = false;
    unsigned LoopNestDepth = LoopList.size();
    if (LoopNestDepth < 2) {
      DEBUG(dbgs() << "Loop doesn't contain minimum nesting level.\n");
      return false;
    }
    if (LoopNestDepth > MaxLoopNestDepth) {
      DEBUG(dbgs() << "Cannot handle loops of depth greater than "
                   << MaxLoopNestDepth << "\n");
      return false;
    }
    if (!isComputableLoopNest(LoopList)) {
      DEBUG(dbgs() << "Not valid loop candidate for interchange\n");
      return false;
    }

    DEBUG(dbgs() << "Processing LoopList of size = " << LoopNestDepth << "\n");

    CharMatrix DependencyMatrix;
    Loop *OuterMostLoop = *(LoopList.begin());
    if (!populateDependencyMatrix(DependencyMatrix, LoopNestDepth,
                                  OuterMostLoop, DI)) {
      DEBUG(dbgs() << "Populating dependency matrix failed\n");
      return false;
    }
#ifdef DUMP_DEP_MATRICIES
    DEBUG(dbgs() << "Dependence before interchange\n");
    printDepMatrix(DependencyMatrix);
#endif

    BasicBlock *OuterMostLoopLatch = OuterMostLoop->getLoopLatch();
    BranchInst *OuterMostLoopLatchBI =
        dyn_cast<BranchInst>(OuterMostLoopLatch->getTerminator());
    if (!OuterMostLoopLatchBI)
      return false;

    // Since we currently do not handle LCSSA PHI's any failure in loop
    // condition will now branch to LoopNestExit.
    // TODO: This should be removed once we handle LCSSA PHI nodes.

    // Get the Outermost loop exit.
    BasicBlock *LoopNestExit;
    if (OuterMostLoopLatchBI->getSuccessor(0) == OuterMostLoop->getHeader())
      LoopNestExit = OuterMostLoopLatchBI->getSuccessor(1);
    else
      LoopNestExit = OuterMostLoopLatchBI->getSuccessor(0);

    if (isa<PHINode>(LoopNestExit->begin())) {
      DEBUG(dbgs() << "PHI Nodes in loop nest exit is not handled for now "
                      "since on failure all loops branch to loop nest exit.\n");
      return false;
    }

    unsigned SelecLoopId = selectLoopForInterchange(LoopList);
    // Move the selected loop outwards to the best possible position.
    for (unsigned i = SelecLoopId; i > 0; i--) {
      bool Interchanged =
          processLoop(LoopList, i, i - 1, LoopNestExit, DependencyMatrix);
      if (!Interchanged)
        return Changed;
      // Loops interchanged reflect the same in LoopList
      std::swap(LoopList[i - 1], LoopList[i]);

      // Update the DependencyMatrix
      interChangeDependencies(DependencyMatrix, i, i - 1);
      DT->recalculate(F);
#ifdef DUMP_DEP_MATRICIES
      DEBUG(dbgs() << "Dependence after interchange\n");
      printDepMatrix(DependencyMatrix);
#endif
      Changed |= Interchanged;
    }
    return Changed;
  }

  bool processLoop(LoopVector LoopList, unsigned InnerLoopId,
                   unsigned OuterLoopId, BasicBlock *LoopNestExit,
                   std::vector<std::vector<char>> &DependencyMatrix) {

    DEBUG(dbgs() << "Processing Inner Loop Id = " << InnerLoopId
                 << " and OuterLoopId = " << OuterLoopId << "\n");
    Loop *InnerLoop = LoopList[InnerLoopId];
    Loop *OuterLoop = LoopList[OuterLoopId];

    LoopInterchangeLegality LIL(OuterLoop, InnerLoop, SE, LI, DT,
                                PreserveLCSSA);
    if (!LIL.canInterchangeLoops(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      DEBUG(dbgs() << "Not interchanging Loops. Cannot prove legality\n");
      return false;
    }
    DEBUG(dbgs() << "Loops are legal to interchange\n");
    LoopInterchangeProfitability LIP(OuterLoop, InnerLoop, SE);
    if (!LIP.isProfitable(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      DEBUG(dbgs() << "Interchanging loops not profitable\n");
      return false;
    }

    LoopInterchangeTransform LIT(OuterLoop, InnerLoop, SE, LI, DT,
                                 LoopNestExit, LIL.hasInnerLoopReduction());
    LIT.transform();
    DEBUG(dbgs() << "Loops interchanged\n");
    return true;
  }
};

} // end of namespace
bool LoopInterchangeLegality::areAllUsesReductions(Instruction *Ins, Loop *L) {
  return none_of(Ins->users(), [=](User *U) -> bool {
    auto *UserIns = dyn_cast<PHINode>(U);
    RecurrenceDescriptor RD;
    return !UserIns || !RecurrenceDescriptor::isReductionPHI(UserIns, L, RD);
  });
}

bool LoopInterchangeLegality::containsUnsafeInstructionsInHeader(
    BasicBlock *BB) {
  for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
    // Load corresponding to reduction PHI's are safe while concluding if
    // tightly nested.
    if (LoadInst *L = dyn_cast<LoadInst>(I)) {
      if (!areAllUsesReductions(L, InnerLoop))
        return true;
    } else if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      return true;
  }
  return false;
}

bool LoopInterchangeLegality::containsUnsafeInstructionsInLatch(
    BasicBlock *BB) {
  for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
    // Stores corresponding to reductions are safe while concluding if tightly
    // nested.
    if (StoreInst *L = dyn_cast<StoreInst>(I)) {
      if (!isa<PHINode>(L->getOperand(0)))
        return true;
    } else if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      return true;
  }
  return false;
}

bool LoopInterchangeLegality::tightlyNested(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();

  DEBUG(dbgs() << "Checking if loops are tightly nested\n");

  // A perfectly nested loop will not have any branch in between the outer and
  // inner block i.e. outer header will branch to either inner preheader and
  // outerloop latch.
  BranchInst *OuterLoopHeaderBI =
      dyn_cast<BranchInst>(OuterLoopHeader->getTerminator());
  if (!OuterLoopHeaderBI)
    return false;

  for (unsigned i = 0, e = OuterLoopHeaderBI->getNumSuccessors(); i < e; ++i) {
    if (OuterLoopHeaderBI->getSuccessor(i) != InnerLoopPreHeader &&
        OuterLoopHeaderBI->getSuccessor(i) != OuterLoopLatch)
      return false;
  }

  DEBUG(dbgs() << "Checking instructions in Loop header and Loop latch\n");
  // We do not have any basic block in between now make sure the outer header
  // and outer loop latch doesn't contain any unsafe instructions.
  if (containsUnsafeInstructionsInHeader(OuterLoopHeader) ||
      containsUnsafeInstructionsInLatch(OuterLoopLatch))
    return false;

  DEBUG(dbgs() << "Loops are perfectly nested\n");
  // We have a perfect loop nest.
  return true;
}


bool LoopInterchangeLegality::isLoopStructureUnderstood(
    PHINode *InnerInduction) {

  unsigned Num = InnerInduction->getNumOperands();
  BasicBlock *InnerLoopPreheader = InnerLoop->getLoopPreheader();
  for (unsigned i = 0; i < Num; ++i) {
    Value *Val = InnerInduction->getOperand(i);
    if (isa<Constant>(Val))
      continue;
    Instruction *I = dyn_cast<Instruction>(Val);
    if (!I)
      return false;
    // TODO: Handle triangular loops.
    // e.g. for(int i=0;i<N;i++)
    //        for(int j=i;j<N;j++)
    unsigned IncomBlockIndx = PHINode::getIncomingValueNumForOperand(i);
    if (InnerInduction->getIncomingBlock(IncomBlockIndx) ==
            InnerLoopPreheader &&
        !OuterLoop->isLoopInvariant(I)) {
      return false;
    }
  }
  return true;
}

bool LoopInterchangeLegality::findInductionAndReductions(
    Loop *L, SmallVector<PHINode *, 8> &Inductions,
    SmallVector<PHINode *, 8> &Reductions) {
  if (!L->getLoopLatch() || !L->getLoopPredecessor())
    return false;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    RecurrenceDescriptor RD;
    InductionDescriptor ID;
    PHINode *PHI = cast<PHINode>(I);
    if (InductionDescriptor::isInductionPHI(PHI, L, SE, ID))
      Inductions.push_back(PHI);
    else if (RecurrenceDescriptor::isReductionPHI(PHI, L, RD))
      Reductions.push_back(PHI);
    else {
      DEBUG(
          dbgs() << "Failed to recognize PHI as an induction or reduction.\n");
      return false;
    }
  }
  return true;
}

static bool containsSafePHI(BasicBlock *Block, bool isOuterLoopExitBlock) {
  for (auto I = Block->begin(); isa<PHINode>(I); ++I) {
    PHINode *PHI = cast<PHINode>(I);
    // Reduction lcssa phi will have only 1 incoming block that from loop latch.
    if (PHI->getNumIncomingValues() > 1)
      return false;
    Instruction *Ins = dyn_cast<Instruction>(PHI->getIncomingValue(0));
    if (!Ins)
      return false;
    // Incoming value for lcssa phi's in outer loop exit can only be inner loop
    // exits lcssa phi else it would not be tightly nested.
    if (!isa<PHINode>(Ins) && isOuterLoopExitBlock)
      return false;
  }
  return true;
}

static BasicBlock *getLoopLatchExitBlock(BasicBlock *LatchBlock,
                                         BasicBlock *LoopHeader) {
  if (BranchInst *BI = dyn_cast<BranchInst>(LatchBlock->getTerminator())) {
    unsigned Num = BI->getNumSuccessors();
    assert(Num == 2);
    for (unsigned i = 0; i < Num; ++i) {
      if (BI->getSuccessor(i) == LoopHeader)
        continue;
      return BI->getSuccessor(i);
    }
  }
  return nullptr;
}

// This function indicates the current limitations in the transform as a result
// of which we do not proceed.
bool LoopInterchangeLegality::currentLimitations() {

  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();

  PHINode *InnerInductionVar;
  SmallVector<PHINode *, 8> Inductions;
  SmallVector<PHINode *, 8> Reductions;
  if (!findInductionAndReductions(InnerLoop, Inductions, Reductions))
    return true;

  // TODO: Currently we handle only loops with 1 induction variable.
  if (Inductions.size() != 1) {
    DEBUG(dbgs() << "We currently only support loops with 1 induction variable."
                 << "Failed to interchange due to current limitation\n");
    return true;
  }
  if (Reductions.size() > 0)
    InnerLoopHasReduction = true;

  InnerInductionVar = Inductions.pop_back_val();
  Reductions.clear();
  if (!findInductionAndReductions(OuterLoop, Inductions, Reductions))
    return true;

  // Outer loop cannot have reduction because then loops will not be tightly
  // nested.
  if (!Reductions.empty())
    return true;
  // TODO: Currently we handle only loops with 1 induction variable.
  if (Inductions.size() != 1)
    return true;

  // TODO: Triangular loops are not handled for now.
  if (!isLoopStructureUnderstood(InnerInductionVar)) {
    DEBUG(dbgs() << "Loop structure not understood by pass\n");
    return true;
  }

  // TODO: We only handle LCSSA PHI's corresponding to reduction for now.
  BasicBlock *LoopExitBlock =
      getLoopLatchExitBlock(OuterLoopLatch, OuterLoopHeader);
  if (!LoopExitBlock || !containsSafePHI(LoopExitBlock, true))
    return true;

  LoopExitBlock = getLoopLatchExitBlock(InnerLoopLatch, InnerLoopHeader);
  if (!LoopExitBlock || !containsSafePHI(LoopExitBlock, false))
    return true;

  // TODO: Current limitation: Since we split the inner loop latch at the point
  // were induction variable is incremented (induction.next); We cannot have
  // more than 1 user of induction.next since it would result in broken code
  // after split.
  // e.g.
  // for(i=0;i<N;i++) {
  //    for(j = 0;j<M;j++) {
  //      A[j+1][i+2] = A[j][i]+k;
  //  }
  // }
  Instruction *InnerIndexVarInc = nullptr;
  if (InnerInductionVar->getIncomingBlock(0) == InnerLoopPreHeader)
    InnerIndexVarInc =
        dyn_cast<Instruction>(InnerInductionVar->getIncomingValue(1));
  else
    InnerIndexVarInc =
        dyn_cast<Instruction>(InnerInductionVar->getIncomingValue(0));

  if (!InnerIndexVarInc)
    return true;

  // Since we split the inner loop latch on this induction variable. Make sure
  // we do not have any instruction between the induction variable and branch
  // instruction.

  bool FoundInduction = false;
  for (const Instruction &I : reverse(*InnerLoopLatch)) {
    if (isa<BranchInst>(I) || isa<CmpInst>(I) || isa<TruncInst>(I))
      continue;
    // We found an instruction. If this is not induction variable then it is not
    // safe to split this loop latch.
    if (!I.isIdenticalTo(InnerIndexVarInc))
      return true;

    FoundInduction = true;
    break;
  }
  // The loop latch ended and we didn't find the induction variable return as
  // current limitation.
  if (!FoundInduction)
    return true;

  return false;
}

bool LoopInterchangeLegality::canInterchangeLoops(unsigned InnerLoopId,
                                                  unsigned OuterLoopId,
                                                  CharMatrix &DepMatrix) {

  if (!isLegalToInterChangeLoops(DepMatrix, InnerLoopId, OuterLoopId)) {
    DEBUG(dbgs() << "Failed interchange InnerLoopId = " << InnerLoopId
                 << " and OuterLoopId = " << OuterLoopId
                 << " due to dependence\n");
    return false;
  }

  // Create unique Preheaders if we already do not have one.
  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();

  // Create  a unique outer preheader -
  // 1) If OuterLoop preheader is not present.
  // 2) If OuterLoop Preheader is same as OuterLoop Header
  // 3) If OuterLoop Preheader is same as Header of the previous loop.
  // 4) If OuterLoop Preheader is Entry node.
  if (!OuterLoopPreHeader || OuterLoopPreHeader == OuterLoop->getHeader() ||
      isa<PHINode>(OuterLoopPreHeader->begin()) ||
      !OuterLoopPreHeader->getUniquePredecessor()) {
    OuterLoopPreHeader =
        InsertPreheaderForLoop(OuterLoop, DT, LI, PreserveLCSSA);
  }

  if (!InnerLoopPreHeader || InnerLoopPreHeader == InnerLoop->getHeader() ||
      InnerLoopPreHeader == OuterLoop->getHeader()) {
    InnerLoopPreHeader =
        InsertPreheaderForLoop(InnerLoop, DT, LI, PreserveLCSSA);
  }

  // TODO: The loops could not be interchanged due to current limitations in the
  // transform module.
  if (currentLimitations()) {
    DEBUG(dbgs() << "Not legal because of current transform limitation\n");
    return false;
  }

  // Check if the loops are tightly nested.
  if (!tightlyNested(OuterLoop, InnerLoop)) {
    DEBUG(dbgs() << "Loops not tightly nested\n");
    return false;
  }

  return true;
}

int LoopInterchangeProfitability::getInstrOrderCost() {
  unsigned GoodOrder, BadOrder;
  BadOrder = GoodOrder = 0;
  for (auto BI = InnerLoop->block_begin(), BE = InnerLoop->block_end();
       BI != BE; ++BI) {
    for (Instruction &Ins : **BI) {
      if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&Ins)) {
        unsigned NumOp = GEP->getNumOperands();
        bool FoundInnerInduction = false;
        bool FoundOuterInduction = false;
        for (unsigned i = 0; i < NumOp; ++i) {
          const SCEV *OperandVal = SE->getSCEV(GEP->getOperand(i));
          const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(OperandVal);
          if (!AR)
            continue;

          // If we find the inner induction after an outer induction e.g.
          // for(int i=0;i<N;i++)
          //   for(int j=0;j<N;j++)
          //     A[i][j] = A[i-1][j-1]+k;
          // then it is a good order.
          if (AR->getLoop() == InnerLoop) {
            // We found an InnerLoop induction after OuterLoop induction. It is
            // a good order.
            FoundInnerInduction = true;
            if (FoundOuterInduction) {
              GoodOrder++;
              break;
            }
          }
          // If we find the outer induction after an inner induction e.g.
          // for(int i=0;i<N;i++)
          //   for(int j=0;j<N;j++)
          //     A[j][i] = A[j-1][i-1]+k;
          // then it is a bad order.
          if (AR->getLoop() == OuterLoop) {
            // We found an OuterLoop induction after InnerLoop induction. It is
            // a bad order.
            FoundOuterInduction = true;
            if (FoundInnerInduction) {
              BadOrder++;
              break;
            }
          }
        }
      }
    }
  }
  return GoodOrder - BadOrder;
}

static bool isProfitableForVectorization(unsigned InnerLoopId,
                                         unsigned OuterLoopId,
                                         CharMatrix &DepMatrix) {
  // TODO: Improve this heuristic to catch more cases.
  // If the inner loop is loop independent or doesn't carry any dependency it is
  // profitable to move this to outer position.
  unsigned Row = DepMatrix.size();
  for (unsigned i = 0; i < Row; ++i) {
    if (DepMatrix[i][InnerLoopId] != 'S' && DepMatrix[i][InnerLoopId] != 'I')
      return false;
    // TODO: We need to improve this heuristic.
    if (DepMatrix[i][OuterLoopId] != '=')
      return false;
  }
  // If outer loop has dependence and inner loop is loop independent then it is
  // profitable to interchange to enable parallelism.
  return true;
}

bool LoopInterchangeProfitability::isProfitable(unsigned InnerLoopId,
                                                unsigned OuterLoopId,
                                                CharMatrix &DepMatrix) {

  // TODO: Add better profitability checks.
  // e.g
  // 1) Construct dependency matrix and move the one with no loop carried dep
  //    inside to enable vectorization.

  // This is rough cost estimation algorithm. It counts the good and bad order
  // of induction variables in the instruction and allows reordering if number
  // of bad orders is more than good.
  int Cost = getInstrOrderCost();
  DEBUG(dbgs() << "Cost = " << Cost << "\n");
  if (Cost < -LoopInterchangeCostThreshold)
    return true;

  // It is not profitable as per current cache profitability model. But check if
  // we can move this loop outside to improve parallelism.
  bool ImprovesPar =
      isProfitableForVectorization(InnerLoopId, OuterLoopId, DepMatrix);
  return ImprovesPar;
}

void LoopInterchangeTransform::removeChildLoop(Loop *OuterLoop,
                                               Loop *InnerLoop) {
  for (Loop::iterator I = OuterLoop->begin(), E = OuterLoop->end(); I != E;
       ++I) {
    if (*I == InnerLoop) {
      OuterLoop->removeChildLoop(I);
      return;
    }
  }
  llvm_unreachable("Couldn't find loop");
}

void LoopInterchangeTransform::restructureLoops(Loop *InnerLoop,
                                                Loop *OuterLoop) {
  Loop *OuterLoopParent = OuterLoop->getParentLoop();
  if (OuterLoopParent) {
    // Remove the loop from its parent loop.
    removeChildLoop(OuterLoopParent, OuterLoop);
    removeChildLoop(OuterLoop, InnerLoop);
    OuterLoopParent->addChildLoop(InnerLoop);
  } else {
    removeChildLoop(OuterLoop, InnerLoop);
    LI->changeTopLevelLoop(OuterLoop, InnerLoop);
  }

  while (!InnerLoop->empty())
    OuterLoop->addChildLoop(InnerLoop->removeChildLoop(InnerLoop->begin()));

  InnerLoop->addChildLoop(OuterLoop);
}

bool LoopInterchangeTransform::transform() {
  bool Transformed = false;
  Instruction *InnerIndexVar;

  if (InnerLoop->getSubLoops().size() == 0) {
    BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
    DEBUG(dbgs() << "Calling Split Inner Loop\n");
    PHINode *InductionPHI = getInductionVariable(InnerLoop, SE);
    if (!InductionPHI) {
      DEBUG(dbgs() << "Failed to find the point to split loop latch \n");
      return false;
    }

    if (InductionPHI->getIncomingBlock(0) == InnerLoopPreHeader)
      InnerIndexVar = dyn_cast<Instruction>(InductionPHI->getIncomingValue(1));
    else
      InnerIndexVar = dyn_cast<Instruction>(InductionPHI->getIncomingValue(0));

    //
    // Split at the place were the induction variable is
    // incremented/decremented.
    // TODO: This splitting logic may not work always. Fix this.
    splitInnerLoopLatch(InnerIndexVar);
    DEBUG(dbgs() << "splitInnerLoopLatch done\n");

    // Splits the inner loops phi nodes out into a separate basic block.
    splitInnerLoopHeader();
    DEBUG(dbgs() << "splitInnerLoopHeader done\n");
  }

  Transformed |= adjustLoopLinks();
  if (!Transformed) {
    DEBUG(dbgs() << "adjustLoopLinks failed\n");
    return false;
  }

  restructureLoops(InnerLoop, OuterLoop);
  return true;
}

void LoopInterchangeTransform::splitInnerLoopLatch(Instruction *Inc) {
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock *InnerLoopLatchPred = InnerLoopLatch;
  InnerLoopLatch = SplitBlock(InnerLoopLatchPred, Inc, DT, LI);
}

void LoopInterchangeTransform::splitInnerLoopHeader() {

  // Split the inner loop header out. Here make sure that the reduction PHI's
  // stay in the innerloop body.
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  if (InnerLoopHasReduction) {
    // FIXME: Check if the induction PHI will always be the first PHI.
    BasicBlock *New = InnerLoopHeader->splitBasicBlock(
        ++(InnerLoopHeader->begin()), InnerLoopHeader->getName() + ".split");
    if (LI)
      if (Loop *L = LI->getLoopFor(InnerLoopHeader))
        L->addBasicBlockToLoop(New, *LI);

    // Adjust Reduction PHI's in the block.
    SmallVector<PHINode *, 8> PHIVec;
    for (auto I = New->begin(); isa<PHINode>(I); ++I) {
      PHINode *PHI = dyn_cast<PHINode>(I);
      Value *V = PHI->getIncomingValueForBlock(InnerLoopPreHeader);
      PHI->replaceAllUsesWith(V);
      PHIVec.push_back((PHI));
    }
    for (PHINode *P : PHIVec) {
      P->eraseFromParent();
    }
  } else {
    SplitBlock(InnerLoopHeader, InnerLoopHeader->getFirstNonPHI(), DT, LI);
  }

  DEBUG(dbgs() << "Output of splitInnerLoopHeader InnerLoopHeaderSucc & "
                  "InnerLoopHeader\n");
}

/// \brief Move all instructions except the terminator from FromBB right before
/// InsertBefore
static void moveBBContents(BasicBlock *FromBB, Instruction *InsertBefore) {
  auto &ToList = InsertBefore->getParent()->getInstList();
  auto &FromList = FromBB->getInstList();

  ToList.splice(InsertBefore->getIterator(), FromList, FromList.begin(),
                FromBB->getTerminator()->getIterator());
}

void LoopInterchangeTransform::updateIncomingBlock(BasicBlock *CurrBlock,
                                                   BasicBlock *OldPred,
                                                   BasicBlock *NewPred) {
  for (auto I = CurrBlock->begin(); isa<PHINode>(I); ++I) {
    PHINode *PHI = cast<PHINode>(I);
    unsigned Num = PHI->getNumIncomingValues();
    for (unsigned i = 0; i < Num; ++i) {
      if (PHI->getIncomingBlock(i) == OldPred)
        PHI->setIncomingBlock(i, NewPred);
    }
  }
}

bool LoopInterchangeTransform::adjustLoopBranches() {

  DEBUG(dbgs() << "adjustLoopBranches called\n");
  // Adjust the loop preheader
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();
  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopPredecessor = OuterLoopPreHeader->getUniquePredecessor();
  BasicBlock *InnerLoopLatchPredecessor =
      InnerLoopLatch->getUniquePredecessor();
  BasicBlock *InnerLoopLatchSuccessor;
  BasicBlock *OuterLoopLatchSuccessor;

  BranchInst *OuterLoopLatchBI =
      dyn_cast<BranchInst>(OuterLoopLatch->getTerminator());
  BranchInst *InnerLoopLatchBI =
      dyn_cast<BranchInst>(InnerLoopLatch->getTerminator());
  BranchInst *OuterLoopHeaderBI =
      dyn_cast<BranchInst>(OuterLoopHeader->getTerminator());
  BranchInst *InnerLoopHeaderBI =
      dyn_cast<BranchInst>(InnerLoopHeader->getTerminator());

  if (!OuterLoopPredecessor || !InnerLoopLatchPredecessor ||
      !OuterLoopLatchBI || !InnerLoopLatchBI || !OuterLoopHeaderBI ||
      !InnerLoopHeaderBI)
    return false;

  BranchInst *InnerLoopLatchPredecessorBI =
      dyn_cast<BranchInst>(InnerLoopLatchPredecessor->getTerminator());
  BranchInst *OuterLoopPredecessorBI =
      dyn_cast<BranchInst>(OuterLoopPredecessor->getTerminator());

  if (!OuterLoopPredecessorBI || !InnerLoopLatchPredecessorBI)
    return false;
  BasicBlock *InnerLoopHeaderSuccessor = InnerLoopHeader->getUniqueSuccessor();
  if (!InnerLoopHeaderSuccessor)
    return false;

  // Adjust Loop Preheader and headers

  unsigned NumSucc = OuterLoopPredecessorBI->getNumSuccessors();
  for (unsigned i = 0; i < NumSucc; ++i) {
    if (OuterLoopPredecessorBI->getSuccessor(i) == OuterLoopPreHeader)
      OuterLoopPredecessorBI->setSuccessor(i, InnerLoopPreHeader);
  }

  NumSucc = OuterLoopHeaderBI->getNumSuccessors();
  for (unsigned i = 0; i < NumSucc; ++i) {
    if (OuterLoopHeaderBI->getSuccessor(i) == OuterLoopLatch)
      OuterLoopHeaderBI->setSuccessor(i, LoopExit);
    else if (OuterLoopHeaderBI->getSuccessor(i) == InnerLoopPreHeader)
      OuterLoopHeaderBI->setSuccessor(i, InnerLoopHeaderSuccessor);
  }

  // Adjust reduction PHI's now that the incoming block has changed.
  updateIncomingBlock(InnerLoopHeaderSuccessor, InnerLoopHeader,
                      OuterLoopHeader);

  BranchInst::Create(OuterLoopPreHeader, InnerLoopHeaderBI);
  InnerLoopHeaderBI->eraseFromParent();

  // -------------Adjust loop latches-----------
  if (InnerLoopLatchBI->getSuccessor(0) == InnerLoopHeader)
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(1);
  else
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(0);

  NumSucc = InnerLoopLatchPredecessorBI->getNumSuccessors();
  for (unsigned i = 0; i < NumSucc; ++i) {
    if (InnerLoopLatchPredecessorBI->getSuccessor(i) == InnerLoopLatch)
      InnerLoopLatchPredecessorBI->setSuccessor(i, InnerLoopLatchSuccessor);
  }

  // Adjust PHI nodes in InnerLoopLatchSuccessor. Update all uses of PHI with
  // the value and remove this PHI node from inner loop.
  SmallVector<PHINode *, 8> LcssaVec;
  for (auto I = InnerLoopLatchSuccessor->begin(); isa<PHINode>(I); ++I) {
    PHINode *LcssaPhi = cast<PHINode>(I);
    LcssaVec.push_back(LcssaPhi);
  }
  for (PHINode *P : LcssaVec) {
    Value *Incoming = P->getIncomingValueForBlock(InnerLoopLatch);
    P->replaceAllUsesWith(Incoming);
    P->eraseFromParent();
  }

  if (OuterLoopLatchBI->getSuccessor(0) == OuterLoopHeader)
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(1);
  else
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(0);

  if (InnerLoopLatchBI->getSuccessor(1) == InnerLoopLatchSuccessor)
    InnerLoopLatchBI->setSuccessor(1, OuterLoopLatchSuccessor);
  else
    InnerLoopLatchBI->setSuccessor(0, OuterLoopLatchSuccessor);

  updateIncomingBlock(OuterLoopLatchSuccessor, OuterLoopLatch, InnerLoopLatch);

  if (OuterLoopLatchBI->getSuccessor(0) == OuterLoopLatchSuccessor) {
    OuterLoopLatchBI->setSuccessor(0, InnerLoopLatch);
  } else {
    OuterLoopLatchBI->setSuccessor(1, InnerLoopLatch);
  }

  return true;
}
void LoopInterchangeTransform::adjustLoopPreheaders() {

  // We have interchanged the preheaders so we need to interchange the data in
  // the preheader as well.
  // This is because the content of inner preheader was previously executed
  // inside the outer loop.
  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BranchInst *InnerTermBI =
      cast<BranchInst>(InnerLoopPreHeader->getTerminator());

  // These instructions should now be executed inside the loop.
  // Move instruction into a new block after outer header.
  moveBBContents(InnerLoopPreHeader, OuterLoopHeader->getTerminator());
  // These instructions were not executed previously in the loop so move them to
  // the older inner loop preheader.
  moveBBContents(OuterLoopPreHeader, InnerTermBI);
}

bool LoopInterchangeTransform::adjustLoopLinks() {

  // Adjust all branches in the inner and outer loop.
  bool Changed = adjustLoopBranches();
  if (Changed)
    adjustLoopPreheaders();
  return Changed;
}

char LoopInterchange::ID = 0;
INITIALIZE_PASS_BEGIN(LoopInterchange, "loop-interchange",
                      "Interchanges loops for cache reuse", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)

INITIALIZE_PASS_END(LoopInterchange, "loop-interchange",
                    "Interchanges loops for cache reuse", false, false)

Pass *llvm::createLoopInterchangePass() { return new LoopInterchange(); }
