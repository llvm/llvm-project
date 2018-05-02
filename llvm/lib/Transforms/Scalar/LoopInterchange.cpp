//===- LoopInterchange.cpp - Loop interchange pass-------------------------===//
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <cassert>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "loop-interchange"

STATISTIC(LoopsInterchanged, "Number of loops interchanged");

static cl::opt<int> LoopInterchangeCostThreshold(
    "loop-interchange-threshold", cl::init(0), cl::Hidden,
    cl::desc("Interchange if you gain more than this number"));

namespace {

using LoopVector = SmallVector<Loop *, 8>;

// TODO: Check if we can use a sparse matrix here.
using CharMatrix = std::vector<std::vector<char>>;

} // end anonymous namespace

// Maximum number of dependencies that can be handled in the dependency matrix.
static const unsigned MaxMemInstrCount = 100;

// Maximum loop depth supported.
static const unsigned MaxLoopNestDepth = 10;

#ifdef DUMP_DEP_MATRICIES
static void printDepMatrix(CharMatrix &DepMatrix) {
  for (auto &Row : DepMatrix) {
    for (auto D : Row)
      DEBUG(dbgs() << D << " ");
    DEBUG(dbgs() << "\n");
  }
}
#endif

static bool populateDependencyMatrix(CharMatrix &DepMatrix, unsigned Level,
                                     Loop *L, DependenceInfo *DI) {
  using ValueVector = SmallVector<Value *, 16>;

  ValueVector MemInstr;

  // For each block.
  for (BasicBlock *BB : L->blocks()) {
    // Scan the BB and collect legal loads and stores.
    for (Instruction &I : *BB) {
      if (!isa<Instruction>(I))
        return false;
      if (auto *Ld = dyn_cast<LoadInst>(&I)) {
        if (!Ld->isSimple())
          return false;
        MemInstr.push_back(&I);
      } else if (auto *St = dyn_cast<StoreInst>(&I)) {
        if (!St->isSimple())
          return false;
        MemInstr.push_back(&I);
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
      // Ignore Input dependencies.
      if (isa<LoadInst>(Src) && isa<LoadInst>(Dst))
        continue;
      // Track Output, Flow, and Anti dependencies.
      if (auto D = DI->depends(Src, Dst, true)) {
        assert(D->isOrdered() && "Expected an output, flow or anti dep.");
        DEBUG(StringRef DepType =
                  D->isFlow() ? "flow" : D->isAnti() ? "anti" : "output";
              dbgs() << "Found " << DepType
                     << " dependency between Src and Dst\n"
                     << " Src:" << *Src << "\n Dst:" << *Dst << '\n');
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
    if (DepMatrix[Row][i] != '=' && DepMatrix[Row][i] != 'S' &&
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

namespace {

/// LoopInterchangeLegality checks if it is legal to interchange the loop.
class LoopInterchangeLegality {
public:
  LoopInterchangeLegality(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                          LoopInfo *LI, DominatorTree *DT, bool PreserveLCSSA,
                          OptimizationRemarkEmitter *ORE)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), LI(LI), DT(DT),
        PreserveLCSSA(PreserveLCSSA), ORE(ORE) {}

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

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  bool InnerLoopHasReduction = false;
};

/// LoopInterchangeProfitability checks if it is profitable to interchange the
/// loop.
class LoopInterchangeProfitability {
public:
  LoopInterchangeProfitability(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                               OptimizationRemarkEmitter *ORE)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), ORE(ORE) {}

  /// Check if the loop interchange is profitable.
  bool isProfitable(unsigned InnerLoopId, unsigned OuterLoopId,
                    CharMatrix &DepMatrix);

private:
  int getInstrOrderCost();

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;
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
  void restructureLoops(Loop *NewInner, Loop *NewOuter,
                        BasicBlock *OrigInnerPreHeader,
                        BasicBlock *OrigOuterPreHeader);
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
  ScalarEvolution *SE = nullptr;
  LoopInfo *LI = nullptr;
  DependenceInfo *DI = nullptr;
  DominatorTree *DT = nullptr;
  bool PreserveLCSSA;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  LoopInterchange() : FunctionPass(ID) {
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
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();

    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DI = &getAnalysis<DependenceAnalysisWrapperPass>().getDI();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
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

    // Get the Outermost loop exit.
    BasicBlock *LoopNestExit = OuterMostLoop->getExitBlock();
    if (!LoopNestExit) {
      DEBUG(dbgs() << "OuterMostLoop needs an unique exit block");
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
                                PreserveLCSSA, ORE);
    if (!LIL.canInterchangeLoops(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      DEBUG(dbgs() << "Not interchanging loops. Cannot prove legality.\n");
      return false;
    }
    DEBUG(dbgs() << "Loops are legal to interchange\n");
    LoopInterchangeProfitability LIP(OuterLoop, InnerLoop, SE, ORE);
    if (!LIP.isProfitable(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      DEBUG(dbgs() << "Interchanging loops not profitable.\n");
      return false;
    }

    ORE->emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Interchanged",
                                InnerLoop->getStartLoc(),
                                InnerLoop->getHeader())
             << "Loop interchanged with enclosing loop.";
    });

    LoopInterchangeTransform LIT(OuterLoop, InnerLoop, SE, LI, DT,
                                 LoopNestExit, LIL.hasInnerLoopReduction());
    LIT.transform();
    DEBUG(dbgs() << "Loops interchanged.\n");
    LoopsInterchanged++;
    return true;
  }
};

} // end anonymous namespace

bool LoopInterchangeLegality::areAllUsesReductions(Instruction *Ins, Loop *L) {
  return llvm::none_of(Ins->users(), [=](User *U) -> bool {
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

  for (BasicBlock *Succ : OuterLoopHeaderBI->successors())
    if (Succ != InnerLoopPreHeader && Succ != OuterLoopLatch)
      return false;

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

// This function indicates the current limitations in the transform as a result
// of which we do not proceed.
bool LoopInterchangeLegality::currentLimitations() {
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();

  // transform currently expects the loop latches to also be the exiting
  // blocks.
  if (InnerLoop->getExitingBlock() != InnerLoopLatch ||
      OuterLoop->getExitingBlock() != OuterLoop->getLoopLatch() ||
      !isa<BranchInst>(InnerLoopLatch->getTerminator()) ||
      !isa<BranchInst>(OuterLoop->getLoopLatch()->getTerminator())) {
    DEBUG(dbgs() << "Loops where the latch is not the exiting block are not"
                 << " supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "ExitingNotLatch",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Loops where the latch is not the exiting block cannot be"
                " interchange currently.";
    });
    return true;
  }

  PHINode *InnerInductionVar;
  SmallVector<PHINode *, 8> Inductions;
  SmallVector<PHINode *, 8> Reductions;
  if (!findInductionAndReductions(InnerLoop, Inductions, Reductions)) {
    DEBUG(dbgs() << "Only inner loops with induction or reduction PHI nodes "
                 << "are supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedPHIInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Only inner loops with induction or reduction PHI nodes can be"
                " interchange currently.";
    });
    return true;
  }

  // TODO: Currently we handle only loops with 1 induction variable.
  if (Inductions.size() != 1) {
    DEBUG(dbgs() << "We currently only support loops with 1 induction variable."
                 << "Failed to interchange due to current limitation\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "MultiInductionInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Only inner loops with 1 induction variable can be "
                "interchanged currently.";
    });
    return true;
  }
  if (Reductions.size() > 0)
    InnerLoopHasReduction = true;

  InnerInductionVar = Inductions.pop_back_val();
  Reductions.clear();
  if (!findInductionAndReductions(OuterLoop, Inductions, Reductions)) {
    DEBUG(dbgs() << "Only outer loops with induction or reduction PHI nodes "
                 << "are supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedPHIOuter",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Only outer loops with induction or reduction PHI nodes can be"
                " interchanged currently.";
    });
    return true;
  }

  // Outer loop cannot have reduction because then loops will not be tightly
  // nested.
  if (!Reductions.empty()) {
    DEBUG(dbgs() << "Outer loops with reductions are not supported "
                 << "currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "ReductionsOuter",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Outer loops with reductions cannot be interchangeed "
                "currently.";
    });
    return true;
  }
  // TODO: Currently we handle only loops with 1 induction variable.
  if (Inductions.size() != 1) {
    DEBUG(dbgs() << "Loops with more than 1 induction variables are not "
                 << "supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "MultiIndutionOuter",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Only outer loops with 1 induction variable can be "
                "interchanged currently.";
    });
    return true;
  }

  // TODO: Triangular loops are not handled for now.
  if (!isLoopStructureUnderstood(InnerInductionVar)) {
    DEBUG(dbgs() << "Loop structure not understood by pass\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedStructureInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Inner loop structure not understood currently.";
    });
    return true;
  }

  // TODO: We only handle LCSSA PHI's corresponding to reduction for now.
  BasicBlock *InnerExit = InnerLoop->getExitBlock();
  if (!containsSafePHI(InnerExit, false)) {
    DEBUG(dbgs() << "Can only handle LCSSA PHIs in inner loops currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "NoLCSSAPHIOuterInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Only inner loops with LCSSA PHIs can be interchange "
                "currently.";
    });
    return true;
  }

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

  if (!InnerIndexVarInc) {
    DEBUG(dbgs() << "Did not find an instruction to increment the induction "
                 << "variable.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "NoIncrementInInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "The inner loop does not increment the induction variable.";
    });
    return true;
  }

  // Since we split the inner loop latch on this induction variable. Make sure
  // we do not have any instruction between the induction variable and branch
  // instruction.

  bool FoundInduction = false;
  for (const Instruction &I :
       llvm::reverse(InnerLoopLatch->instructionsWithoutDebug())) {
    if (isa<BranchInst>(I) || isa<CmpInst>(I) || isa<TruncInst>(I) ||
        isa<ZExtInst>(I))
      continue;

    // We found an instruction. If this is not induction variable then it is not
    // safe to split this loop latch.
    if (!I.isIdenticalTo(InnerIndexVarInc)) {
      DEBUG(dbgs() << "Found unsupported instructions between induction "
                   << "variable increment and branch.\n");
      ORE->emit([&]() {
        return OptimizationRemarkMissed(
                   DEBUG_TYPE, "UnsupportedInsBetweenInduction",
                   InnerLoop->getStartLoc(), InnerLoop->getHeader())
               << "Found unsupported instruction between induction variable "
                  "increment and branch.";
      });
      return true;
    }

    FoundInduction = true;
    break;
  }
  // The loop latch ended and we didn't find the induction variable return as
  // current limitation.
  if (!FoundInduction) {
    DEBUG(dbgs() << "Did not find the induction variable.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "NoIndutionVariable",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Did not find the induction variable.";
    });
    return true;
  }
  return false;
}

// We currently support LCSSA PHI nodes in the outer loop exit, if their
// incoming values do not come from the outer loop latch or if the
// outer loop latch has a single predecessor. In that case, the value will
// be available if both the inner and outer loop conditions are true, which
// will still be true after interchanging. If we have multiple predecessor,
// that may not be the case, e.g. because the outer loop latch may be executed
// if the inner loop is not executed.
static bool areLoopExitPHIsSupported(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *LoopNestExit = OuterLoop->getUniqueExitBlock();
  for (PHINode &PHI : LoopNestExit->phis()) {
    //  FIXME: We currently are not able to detect floating point reductions
    //         and have to use floating point PHIs as a proxy to prevent
    //         interchanging in the presence of floating point reductions.
    if (PHI.getType()->isFloatingPointTy())
      return false;
    for (unsigned i = 0; i < PHI.getNumIncomingValues(); i++) {
     Instruction *IncomingI = dyn_cast<Instruction>(PHI.getIncomingValue(i));
     if (!IncomingI || IncomingI->getParent() != OuterLoop->getLoopLatch())
       continue;

     // The incoming value is defined in the outer loop latch. Currently we
     // only support that in case the outer loop latch has a single predecessor.
     // This guarantees that the outer loop latch is executed if and only if
     // the inner loop is executed (because tightlyNested() guarantees that the
     // outer loop header only branches to the inner loop or the outer loop
     // latch).
     // FIXME: We could weaken this logic and allow multiple predecessors,
     //        if the values are produced outside the loop latch. We would need
     //        additional logic to update the PHI nodes in the exit block as
     //        well.
     if (OuterLoop->getLoopLatch()->getUniquePredecessor() == nullptr)
       return false;
    }
  }
  return true;
}

bool LoopInterchangeLegality::canInterchangeLoops(unsigned InnerLoopId,
                                                  unsigned OuterLoopId,
                                                  CharMatrix &DepMatrix) {
  if (!isLegalToInterChangeLoops(DepMatrix, InnerLoopId, OuterLoopId)) {
    DEBUG(dbgs() << "Failed interchange InnerLoopId = " << InnerLoopId
                 << " and OuterLoopId = " << OuterLoopId
                 << " due to dependence\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "Dependence",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Cannot interchange loops due to dependences.";
    });
    return false;
  }
  // Check if outer and inner loop contain legal instructions only.
  for (auto *BB : OuterLoop->blocks())
    for (Instruction &I : BB->instructionsWithoutDebug())
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        // readnone functions do not prevent interchanging.
        if (CI->doesNotReadMemory())
          continue;
        DEBUG(dbgs() << "Loops with call instructions cannot be interchanged "
                     << "safely.");
        ORE->emit([&]() {
          return OptimizationRemarkMissed(DEBUG_TYPE, "CallInst",
                                          CI->getDebugLoc(),
                                          CI->getParent())
                 << "Cannot interchange loops due to call instruction.";
        });

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
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "NotTightlyNested",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Cannot interchange loops because they are not tightly "
                "nested.";
    });
    return false;
  }

  if (!areLoopExitPHIsSupported(OuterLoop, InnerLoop)) {
    DEBUG(dbgs() << "Found unsupported PHI nodes in outer loop exit.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedExitPHI",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Found unsupported PHI node in loop exit.";
    });
    return false;
  }

  return true;
}

int LoopInterchangeProfitability::getInstrOrderCost() {
  unsigned GoodOrder, BadOrder;
  BadOrder = GoodOrder = 0;
  for (BasicBlock *BB : InnerLoop->blocks()) {
    for (Instruction &Ins : *BB) {
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
  for (auto &Row : DepMatrix) {
    if (Row[InnerLoopId] != 'S' && Row[InnerLoopId] != 'I')
      return false;
    // TODO: We need to improve this heuristic.
    if (Row[OuterLoopId] != '=')
      return false;
  }
  // If outer loop has dependence and inner loop is loop independent then it is
  // profitable to interchange to enable parallelism.
  // If there are no dependences, interchanging will not improve anything.
  return !DepMatrix.empty();
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
  if (isProfitableForVectorization(InnerLoopId, OuterLoopId, DepMatrix))
    return true;

  ORE->emit([&]() {
    return OptimizationRemarkMissed(DEBUG_TYPE, "InterchangeNotProfitable",
                                    InnerLoop->getStartLoc(),
                                    InnerLoop->getHeader())
           << "Interchanging loops is too costly (cost="
           << ore::NV("Cost", Cost) << ", threshold="
           << ore::NV("Threshold", LoopInterchangeCostThreshold)
           << ") and it does not improve parallelism.";
  });
  return false;
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

/// Update LoopInfo, after interchanging. NewInner and NewOuter refer to the
/// new inner and outer loop after interchanging: NewInner is the original
/// outer loop and NewOuter is the original inner loop.
///
/// Before interchanging, we have the following structure
/// Outer preheader
//  Outer header
//    Inner preheader
//    Inner header
//      Inner body
//      Inner latch
//   outer bbs
//   Outer latch
//
// After interchanging:
// Inner preheader
// Inner header
//   Outer preheader
//   Outer header
//     Inner body
//     outer bbs
//     Outer latch
//   Inner latch
void LoopInterchangeTransform::restructureLoops(
    Loop *NewInner, Loop *NewOuter, BasicBlock *OrigInnerPreHeader,
    BasicBlock *OrigOuterPreHeader) {
  Loop *OuterLoopParent = OuterLoop->getParentLoop();
  // The original inner loop preheader moves from the new inner loop to
  // the parent loop, if there is one.
  NewInner->removeBlockFromLoop(OrigInnerPreHeader);
  LI->changeLoopFor(OrigInnerPreHeader, OuterLoopParent);

  // Switch the loop levels.
  if (OuterLoopParent) {
    // Remove the loop from its parent loop.
    removeChildLoop(OuterLoopParent, NewInner);
    removeChildLoop(NewInner, NewOuter);
    OuterLoopParent->addChildLoop(NewOuter);
  } else {
    removeChildLoop(NewInner, NewOuter);
    LI->changeTopLevelLoop(NewInner, NewOuter);
  }
  while (!NewOuter->empty())
    NewInner->addChildLoop(NewOuter->removeChildLoop(NewOuter->begin()));
  NewOuter->addChildLoop(NewInner);

  // BBs from the original inner loop.
  SmallVector<BasicBlock *, 8> OrigInnerBBs(NewOuter->blocks());

  // Add BBs from the original outer loop to the original inner loop (excluding
  // BBs already in inner loop)
  for (BasicBlock *BB : NewInner->blocks())
    if (LI->getLoopFor(BB) == NewInner)
      NewOuter->addBlockEntry(BB);

  // Now remove inner loop header and latch from the new inner loop and move
  // other BBs (the loop body) to the new inner loop.
  BasicBlock *OuterHeader = NewOuter->getHeader();
  BasicBlock *OuterLatch = NewOuter->getLoopLatch();
  for (BasicBlock *BB : OrigInnerBBs) {
    // Nothing will change for BBs in child loops.
    if (LI->getLoopFor(BB) != NewOuter)
      continue;
    // Remove the new outer loop header and latch from the new inner loop.
    if (BB == OuterHeader || BB == OuterLatch)
      NewInner->removeBlockFromLoop(BB);
    else
      LI->changeLoopFor(BB, NewInner);
  }

  // The preheader of the original outer loop becomes part of the new
  // outer loop.
  NewOuter->addBlockEntry(OrigOuterPreHeader);
  LI->changeLoopFor(OrigOuterPreHeader, NewOuter);
}

bool LoopInterchangeTransform::transform() {
  bool Transformed = false;
  Instruction *InnerIndexVar;

  if (InnerLoop->getSubLoops().empty()) {
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

    // Ensure that InductionPHI is the first Phi node as required by
    // splitInnerLoopHeader
    if (&InductionPHI->getParent()->front() != InductionPHI)
      InductionPHI->moveBefore(&InductionPHI->getParent()->front());

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
  SplitBlock(InnerLoopHeader, InnerLoopHeader->getFirstNonPHI(), DT, LI);
  if (InnerLoopHasReduction) {
    // Adjust Reduction PHI's in the block. The induction PHI must be the first
    // PHI in InnerLoopHeader for this to work.
    SmallVector<PHINode *, 8> PHIVec;
    for (auto I = std::next(InnerLoopHeader->begin()); isa<PHINode>(I); ++I) {
      PHINode *PHI = dyn_cast<PHINode>(I);
      Value *V = PHI->getIncomingValueForBlock(InnerLoopPreHeader);
      PHI->replaceAllUsesWith(V);
      PHIVec.push_back((PHI));
    }
    for (PHINode *P : PHIVec) {
      P->eraseFromParent();
    }
  }

  DEBUG(dbgs() << "Output of splitInnerLoopHeader InnerLoopHeaderSucc & "
                  "InnerLoopHeader\n");
}

/// Move all instructions except the terminator from FromBB right before
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

/// Update BI to jump to NewBB instead of OldBB. Records updates to
/// the dominator tree in DTUpdates, if DT should be preserved.
static void updateSuccessor(BranchInst *BI, BasicBlock *OldBB,
                            BasicBlock *NewBB,
                            std::vector<DominatorTree::UpdateType> &DTUpdates) {
  assert(llvm::count_if(BI->successors(),
                        [OldBB](BasicBlock *BB) { return BB == OldBB; }) < 2 &&
         "BI must jump to OldBB at most once.");
  for (unsigned i = 0, e = BI->getNumSuccessors(); i < e; ++i) {
    if (BI->getSuccessor(i) == OldBB) {
      BI->setSuccessor(i, NewBB);

      DTUpdates.push_back(
          {DominatorTree::UpdateKind::Insert, BI->getParent(), NewBB});
      DTUpdates.push_back(
          {DominatorTree::UpdateKind::Delete, BI->getParent(), OldBB});
      break;
    }
  }
}

bool LoopInterchangeTransform::adjustLoopBranches() {
  DEBUG(dbgs() << "adjustLoopBranches called\n");
  std::vector<DominatorTree::UpdateType> DTUpdates;

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
  updateSuccessor(OuterLoopPredecessorBI, OuterLoopPreHeader,
                  InnerLoopPreHeader, DTUpdates);
  updateSuccessor(OuterLoopHeaderBI, OuterLoopLatch, LoopExit, DTUpdates);
  updateSuccessor(OuterLoopHeaderBI, InnerLoopPreHeader,
                  InnerLoopHeaderSuccessor, DTUpdates);

  // Adjust reduction PHI's now that the incoming block has changed.
  updateIncomingBlock(InnerLoopHeaderSuccessor, InnerLoopHeader,
                      OuterLoopHeader);

  updateSuccessor(InnerLoopHeaderBI, InnerLoopHeaderSuccessor,
                  OuterLoopPreHeader, DTUpdates);

  // -------------Adjust loop latches-----------
  if (InnerLoopLatchBI->getSuccessor(0) == InnerLoopHeader)
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(1);
  else
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(0);

  updateSuccessor(InnerLoopLatchPredecessorBI, InnerLoopLatch,
                  InnerLoopLatchSuccessor, DTUpdates);

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

  updateSuccessor(InnerLoopLatchBI, InnerLoopLatchSuccessor,
                  OuterLoopLatchSuccessor, DTUpdates);
  updateSuccessor(OuterLoopLatchBI, OuterLoopLatchSuccessor, InnerLoopLatch,
                  DTUpdates);

  updateIncomingBlock(OuterLoopLatchSuccessor, OuterLoopLatch, InnerLoopLatch);

  DT->applyUpdates(DTUpdates);
  restructureLoops(OuterLoop, InnerLoop, InnerLoopPreHeader,
                   OuterLoopPreHeader);

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
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)

INITIALIZE_PASS_END(LoopInterchange, "loop-interchange",
                    "Interchanges loops for cache reuse", false, false)

Pass *llvm::createLoopInterchangePass() { return new LoopInterchange(); }
