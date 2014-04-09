//===-- LoopReroll.cpp - Loop rerolling pass ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple loop reroller.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-reroll"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

STATISTIC(NumRerolledLoops, "Number of rerolled loops");

static cl::opt<unsigned>
MaxInc("max-reroll-increment", cl::init(2048), cl::Hidden,
  cl::desc("The maximum increment for loop rerolling"));

// This loop re-rolling transformation aims to transform loops like this:
//
// int foo(int a);
// void bar(int *x) {
//   for (int i = 0; i < 500; i += 3) {
//     foo(i);
//     foo(i+1);
//     foo(i+2);
//   }
// }
//
// into a loop like this:
//
// void bar(int *x) {
//   for (int i = 0; i < 500; ++i)
//     foo(i);
// }
//
// It does this by looking for loops that, besides the latch code, are composed
// of isomorphic DAGs of instructions, with each DAG rooted at some increment
// to the induction variable, and where each DAG is isomorphic to the DAG
// rooted at the induction variable (excepting the sub-DAGs which root the
// other induction-variable increments). In other words, we're looking for loop
// bodies of the form:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// f(%iv)
// %iv.1 = add %iv, 1                <-- a root increment
// f(%iv.1)
// %iv.2 = add %iv, 2                <-- a root increment
// f(%iv.2)
// %iv.scale_m_1 = add %iv, scale-1  <-- a root increment
// f(%iv.scale_m_1)
// ...
// %iv.next = add %iv, scale
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit
//
// where each f(i) is a set of instructions that, collectively, are a function
// only of i (and other loop-invariant values).
//
// As a special case, we can also reroll loops like this:
//
// int foo(int);
// void bar(int *x) {
//   for (int i = 0; i < 500; ++i) {
//     x[3*i] = foo(0);
//     x[3*i+1] = foo(0);
//     x[3*i+2] = foo(0);
//   }
// }
//
// into this:
//
// void bar(int *x) {
//   for (int i = 0; i < 1500; ++i)
//     x[i] = foo(0);
// }
//
// in which case, we're looking for inputs like this:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// %scaled.iv = mul %iv, scale
// f(%scaled.iv)
// %scaled.iv.1 = add %scaled.iv, 1
// f(%scaled.iv.1)
// %scaled.iv.2 = add %scaled.iv, 2
// f(%scaled.iv.2)
// %scaled.iv.scale_m_1 = add %scaled.iv, scale-1
// f(%scaled.iv.scale_m_1)
// ...
// %iv.next = add %iv, 1
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit

namespace {
  class LoopReroll : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopReroll() : LoopPass(ID) {
      initializeLoopRerollPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequired<DominatorTree>();
      AU.addPreserved<DominatorTree>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequired<TargetLibraryInfo>();
    }

protected:
    AliasAnalysis *AA;
    LoopInfo *LI;
    ScalarEvolution *SE;
    DataLayout *DL;
    TargetLibraryInfo *TLI;
    DominatorTree *DT;

    typedef SmallVector<Instruction *, 16> SmallInstructionVector;
    typedef SmallSet<Instruction *, 16>   SmallInstructionSet;

    // A chain of isomorphic instructions, indentified by a single-use PHI,
    // representing a reduction. Only the last value may be used outside the
    // loop.
    struct SimpleLoopReduction {
      SimpleLoopReduction(Instruction *P, Loop *L)
        : Valid(false), Instructions(1, P) {
        assert(isa<PHINode>(P) && "First reduction instruction must be a PHI");
        add(L);
      }

      bool valid() const {
        return Valid;
      }

      Instruction *getPHI() const {
        assert(Valid && "Using invalid reduction");
        return Instructions.front();
      }

      Instruction *getReducedValue() const {
        assert(Valid && "Using invalid reduction");
        return Instructions.back();
      }

      Instruction *get(size_t i) const {
        assert(Valid && "Using invalid reduction");
        return Instructions[i+1];
      }

      Instruction *operator [] (size_t i) const { return get(i); }

      // The size, ignoring the initial PHI.
      size_t size() const {
        assert(Valid && "Using invalid reduction");
        return Instructions.size()-1;
      }

      typedef SmallInstructionVector::iterator iterator;
      typedef SmallInstructionVector::const_iterator const_iterator;

      iterator begin() {
        assert(Valid && "Using invalid reduction");
        return llvm::next(Instructions.begin());
      }

      const_iterator begin() const {
        assert(Valid && "Using invalid reduction");
        return llvm::next(Instructions.begin());
      }

      iterator end() { return Instructions.end(); }
      const_iterator end() const { return Instructions.end(); }

    protected:
      bool Valid;
      SmallInstructionVector Instructions;

      void add(Loop *L);
    };

    // The set of all reductions, and state tracking of possible reductions
    // during loop instruction processing.
    struct ReductionTracker {
      typedef SmallVector<SimpleLoopReduction, 16> SmallReductionVector;

      // Add a new possible reduction.
      void addSLR(SimpleLoopReduction &SLR) {
        PossibleReds.push_back(SLR);
      }

      // Setup to track possible reductions corresponding to the provided
      // rerolling scale. Only reductions with a number of non-PHI instructions
      // that is divisible by the scale are considered. Three instructions sets
      // are filled in:
      //   - A set of all possible instructions in eligible reductions.
      //   - A set of all PHIs in eligible reductions
      //   - A set of all reduced values (last instructions) in eligible reductions.
      void restrictToScale(uint64_t Scale,
                           SmallInstructionSet &PossibleRedSet,
                           SmallInstructionSet &PossibleRedPHISet,
                           SmallInstructionSet &PossibleRedLastSet) {
        PossibleRedIdx.clear();
        PossibleRedIter.clear();
        Reds.clear();

        for (unsigned i = 0, e = PossibleReds.size(); i != e; ++i)
          if (PossibleReds[i].size() % Scale == 0) {
            PossibleRedLastSet.insert(PossibleReds[i].getReducedValue());
            PossibleRedPHISet.insert(PossibleReds[i].getPHI());
      
            PossibleRedSet.insert(PossibleReds[i].getPHI());
            PossibleRedIdx[PossibleReds[i].getPHI()] = i;
            for (SimpleLoopReduction::iterator J = PossibleReds[i].begin(),
                 JE = PossibleReds[i].end(); J != JE; ++J) {
              PossibleRedSet.insert(*J);
              PossibleRedIdx[*J] = i;
            }
          }
      }

      // The functions below are used while processing the loop instructions.

      // Are the two instructions both from reductions, and furthermore, from
      // the same reduction?
      bool isPairInSame(Instruction *J1, Instruction *J2) {
        DenseMap<Instruction *, int>::iterator J1I = PossibleRedIdx.find(J1);
        if (J1I != PossibleRedIdx.end()) {
          DenseMap<Instruction *, int>::iterator J2I = PossibleRedIdx.find(J2);
          if (J2I != PossibleRedIdx.end() && J1I->second == J2I->second)
            return true;
        }

        return false;
      }

      // The two provided instructions, the first from the base iteration, and
      // the second from iteration i, form a matched pair. If these are part of
      // a reduction, record that fact.
      void recordPair(Instruction *J1, Instruction *J2, unsigned i) {
        if (PossibleRedIdx.count(J1)) {
          assert(PossibleRedIdx.count(J2) &&
                 "Recording reduction vs. non-reduction instruction?");

          PossibleRedIter[J1] = 0;
          PossibleRedIter[J2] = i;

          int Idx = PossibleRedIdx[J1];
          assert(Idx == PossibleRedIdx[J2] &&
                 "Recording pair from different reductions?");
          Reds.insert(Idx);
        }
      }

      // The functions below can be called after we've finished processing all
      // instructions in the loop, and we know which reductions were selected.

      // Is the provided instruction the PHI of a reduction selected for
      // rerolling?
      bool isSelectedPHI(Instruction *J) {
        if (!isa<PHINode>(J))
          return false;

        for (DenseSet<int>::iterator RI = Reds.begin(), RIE = Reds.end();
             RI != RIE; ++RI) {
          int i = *RI;
          if (cast<Instruction>(J) == PossibleReds[i].getPHI())
            return true;
        }

        return false;
      }

      bool validateSelected();
      void replaceSelected();

    protected:
      // The vector of all possible reductions (for any scale).
      SmallReductionVector PossibleReds;

      DenseMap<Instruction *, int> PossibleRedIdx;
      DenseMap<Instruction *, int> PossibleRedIter;
      DenseSet<int> Reds;
    };

    void collectPossibleIVs(Loop *L, SmallInstructionVector &PossibleIVs);
    void collectPossibleReductions(Loop *L,
           ReductionTracker &Reductions);
    void collectInLoopUserSet(Loop *L,
           const SmallInstructionVector &Roots,
           const SmallInstructionSet &Exclude,
           const SmallInstructionSet &Final,
           DenseSet<Instruction *> &Users);
    void collectInLoopUserSet(Loop *L,
           Instruction * Root,
           const SmallInstructionSet &Exclude,
           const SmallInstructionSet &Final,
           DenseSet<Instruction *> &Users);
    bool findScaleFromMul(Instruction *RealIV, uint64_t &Scale,
                          Instruction *&IV,
                          SmallInstructionVector &LoopIncs);
    bool collectAllRoots(Loop *L, uint64_t Inc, uint64_t Scale, Instruction *IV,
                         SmallVector<SmallInstructionVector, 32> &Roots,
                         SmallInstructionSet &AllRoots,
                         SmallInstructionVector &LoopIncs);
    bool reroll(Instruction *IV, Loop *L, BasicBlock *Header, const SCEV *IterCount,
                ReductionTracker &Reductions);
  };
}

char LoopReroll::ID = 0;
INITIALIZE_PASS_BEGIN(LoopReroll, "loop-reroll", "Reroll loops", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(LoopReroll, "loop-reroll", "Reroll loops", false, false)

Pass *llvm::createLoopRerollPass() {
  return new LoopReroll;
}

// Returns true if the provided instruction is used outside the given loop.
// This operates like Instruction::isUsedOutsideOfBlock, but considers PHIs in
// non-loop blocks to be outside the loop.
static bool hasUsesOutsideLoop(Instruction *I, Loop *L) {
  for (Value::use_iterator UI = I->use_begin(),
       UIE = I->use_end(); UI != UIE; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    if (!L->contains(User))
      return true;
  }

  return false;
}

// Collect the list of loop induction variables with respect to which it might
// be possible to reroll the loop.
void LoopReroll::collectPossibleIVs(Loop *L,
                                    SmallInstructionVector &PossibleIVs) {
  BasicBlock *Header = L->getHeader();
  for (BasicBlock::iterator I = Header->begin(),
       IE = Header->getFirstInsertionPt(); I != IE; ++I) {
    if (!isa<PHINode>(I))
      continue;
    if (!I->getType()->isIntegerTy())
      continue;

    if (const SCEVAddRecExpr *PHISCEV =
        dyn_cast<SCEVAddRecExpr>(SE->getSCEV(I))) {
      if (PHISCEV->getLoop() != L)
        continue;
      if (!PHISCEV->isAffine())
        continue;
      if (const SCEVConstant *IncSCEV =
          dyn_cast<SCEVConstant>(PHISCEV->getStepRecurrence(*SE))) {
        if (!IncSCEV->getValue()->getValue().isStrictlyPositive())
          continue;
        if (IncSCEV->getValue()->uge(MaxInc))
          continue;

        DEBUG(dbgs() << "LRR: Possible IV: " << *I << " = " <<
              *PHISCEV << "\n");
        PossibleIVs.push_back(I);
      }
    }
  }
}

// Add the remainder of the reduction-variable chain to the instruction vector
// (the initial PHINode has already been added). If successful, the object is
// marked as valid.
void LoopReroll::SimpleLoopReduction::add(Loop *L) {
  assert(!Valid && "Cannot add to an already-valid chain");

  // The reduction variable must be a chain of single-use instructions
  // (including the PHI), except for the last value (which is used by the PHI
  // and also outside the loop).
  Instruction *C = Instructions.front();

  do {
    C = cast<Instruction>(*C->use_begin());
    if (C->hasOneUse()) {
      if (!C->isBinaryOp())
        return;

      if (!(isa<PHINode>(Instructions.back()) ||
            C->isSameOperationAs(Instructions.back())))
        return;

      Instructions.push_back(C);
    }
  } while (C->hasOneUse());

  if (Instructions.size() < 2 ||
      !C->isSameOperationAs(Instructions.back()) ||
      C->use_begin() == C->use_end())
    return;

  // C is now the (potential) last instruction in the reduction chain.
  for (Value::use_iterator UI = C->use_begin(), UIE = C->use_end();
       UI != UIE; ++UI) {
    // The only in-loop user can be the initial PHI.
    if (L->contains(cast<Instruction>(*UI)))
      if (cast<Instruction>(*UI ) != Instructions.front())
        return;
  }

  Instructions.push_back(C);
  Valid = true;
}

// Collect the vector of possible reduction variables.
void LoopReroll::collectPossibleReductions(Loop *L,
  ReductionTracker &Reductions) {
  BasicBlock *Header = L->getHeader();
  for (BasicBlock::iterator I = Header->begin(),
       IE = Header->getFirstInsertionPt(); I != IE; ++I) {
    if (!isa<PHINode>(I))
      continue;
    if (!I->getType()->isSingleValueType())
      continue;

    SimpleLoopReduction SLR(I, L);
    if (!SLR.valid())
      continue;

    DEBUG(dbgs() << "LRR: Possible reduction: " << *I << " (with " <<
          SLR.size() << " chained instructions)\n");
    Reductions.addSLR(SLR);
  }
}

// Collect the set of all users of the provided root instruction. This set of
// users contains not only the direct users of the root instruction, but also
// all users of those users, and so on. There are two exceptions:
//
//   1. Instructions in the set of excluded instructions are never added to the
//   use set (even if they are users). This is used, for example, to exclude
//   including root increments in the use set of the primary IV.
//
//   2. Instructions in the set of final instructions are added to the use set
//   if they are users, but their users are not added. This is used, for
//   example, to prevent a reduction update from forcing all later reduction
//   updates into the use set.
void LoopReroll::collectInLoopUserSet(Loop *L,
  Instruction *Root, const SmallInstructionSet &Exclude,
  const SmallInstructionSet &Final,
  DenseSet<Instruction *> &Users) {
  SmallInstructionVector Queue(1, Root);
  while (!Queue.empty()) {
    Instruction *I = Queue.pop_back_val();
    if (!Users.insert(I).second)
      continue;

    if (!Final.count(I))
      for (Value::use_iterator UI = I->use_begin(),
           UIE = I->use_end(); UI != UIE; ++UI) {
        Instruction *User = cast<Instruction>(*UI);
        if (PHINode *PN = dyn_cast<PHINode>(User)) {
          // Ignore "wrap-around" uses to PHIs of this loop's header.
          if (PN->getIncomingBlock(UI) == L->getHeader())
            continue;
        }
  
        if (L->contains(User) && !Exclude.count(User)) {
          Queue.push_back(User);
        }
      }

    // We also want to collect single-user "feeder" values.
    for (User::op_iterator OI = I->op_begin(),
         OIE = I->op_end(); OI != OIE; ++OI) {
      if (Instruction *Op = dyn_cast<Instruction>(*OI))
        if (Op->hasOneUse() && L->contains(Op) && !Exclude.count(Op) &&
            !Final.count(Op))
          Queue.push_back(Op);
    }
  }
}

// Collect all of the users of all of the provided root instructions (combined
// into a single set).
void LoopReroll::collectInLoopUserSet(Loop *L,
  const SmallInstructionVector &Roots,
  const SmallInstructionSet &Exclude,
  const SmallInstructionSet &Final,
  DenseSet<Instruction *> &Users) {
  for (SmallInstructionVector::const_iterator I = Roots.begin(),
       IE = Roots.end(); I != IE; ++I)
    collectInLoopUserSet(L, *I, Exclude, Final, Users);
}

static bool isSimpleLoadStore(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isSimple();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isSimple();
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I))
    return !MI->isVolatile();
  return false;
}

// Recognize loops that are setup like this:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// %scaled.iv = mul %iv, scale
// f(%scaled.iv)
// %scaled.iv.1 = add %scaled.iv, 1
// f(%scaled.iv.1)
// %scaled.iv.2 = add %scaled.iv, 2
// f(%scaled.iv.2)
// %scaled.iv.scale_m_1 = add %scaled.iv, scale-1
// f(%scaled.iv.scale_m_1)
// ...
// %iv.next = add %iv, 1
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit
//
// and, if found, set IV = %scaled.iv, and add %iv.next to LoopIncs.
bool LoopReroll::findScaleFromMul(Instruction *RealIV, uint64_t &Scale,
                                  Instruction *&IV,
                                  SmallInstructionVector &LoopIncs) {
  // This is a special case: here we're looking for all uses (except for
  // the increment) to be multiplied by a common factor. The increment must
  // be by one. This is to capture loops like:
  //   for (int i = 0; i < 500; ++i) {
  //     foo(3*i); foo(3*i+1); foo(3*i+2);
  //   }
  if (RealIV->getNumUses() != 2)
    return false;
  const SCEVAddRecExpr *RealIVSCEV = cast<SCEVAddRecExpr>(SE->getSCEV(RealIV));
  Instruction *User1 = cast<Instruction>(*RealIV->use_begin()),
              *User2 = cast<Instruction>(*llvm::next(RealIV->use_begin()));
  if (!SE->isSCEVable(User1->getType()) || !SE->isSCEVable(User2->getType()))
    return false;
  const SCEVAddRecExpr *User1SCEV =
                         dyn_cast<SCEVAddRecExpr>(SE->getSCEV(User1)),
                       *User2SCEV =
                         dyn_cast<SCEVAddRecExpr>(SE->getSCEV(User2));
  if (!User1SCEV || !User1SCEV->isAffine() ||
      !User2SCEV || !User2SCEV->isAffine())
    return false;

  // We assume below that User1 is the scale multiply and User2 is the
  // increment. If this can't be true, then swap them.
  if (User1SCEV == RealIVSCEV->getPostIncExpr(*SE)) {
    std::swap(User1, User2);
    std::swap(User1SCEV, User2SCEV);
  }

  if (User2SCEV != RealIVSCEV->getPostIncExpr(*SE))
    return false;
  assert(User2SCEV->getStepRecurrence(*SE)->isOne() &&
         "Invalid non-unit step for multiplicative scaling");
  LoopIncs.push_back(User2);

  if (const SCEVConstant *MulScale =
      dyn_cast<SCEVConstant>(User1SCEV->getStepRecurrence(*SE))) {
    // Make sure that both the start and step have the same multiplier.
    if (RealIVSCEV->getStart()->getType() != MulScale->getType())
      return false;
    if (SE->getMulExpr(RealIVSCEV->getStart(), MulScale) !=
        User1SCEV->getStart())
      return false;

    ConstantInt *MulScaleCI = MulScale->getValue();
    if (!MulScaleCI->uge(2) || MulScaleCI->uge(MaxInc))
      return false;
    Scale = MulScaleCI->getZExtValue();
    IV = User1;
  } else
    return false;

  DEBUG(dbgs() << "LRR: Found possible scaling " << *User1 << "\n");
  return true;
}

// Collect all root increments with respect to the provided induction variable
// (normally the PHI, but sometimes a multiply). A root increment is an
// instruction, normally an add, with a positive constant less than Scale. In a
// rerollable loop, each of these increments is the root of an instruction
// graph isomorphic to the others. Also, we collect the final induction
// increment (the increment equal to the Scale), and its users in LoopIncs.
bool LoopReroll::collectAllRoots(Loop *L, uint64_t Inc, uint64_t Scale,
                                 Instruction *IV,
                                 SmallVector<SmallInstructionVector, 32> &Roots,
                                 SmallInstructionSet &AllRoots,
                                 SmallInstructionVector &LoopIncs) {
  for (Value::use_iterator UI = IV->use_begin(),
       UIE = IV->use_end(); UI != UIE; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    if (!SE->isSCEVable(User->getType()))
      continue;
    if (User->getType() != IV->getType())
      continue;
    if (!L->contains(User))
      continue;
    if (hasUsesOutsideLoop(User, L))
      continue;

    if (const SCEVConstant *Diff = dyn_cast<SCEVConstant>(SE->getMinusSCEV(
          SE->getSCEV(User), SE->getSCEV(IV)))) {
      uint64_t Idx = Diff->getValue()->getValue().getZExtValue();
      if (Idx > 0 && Idx < Scale) {
        Roots[Idx-1].push_back(User);
        AllRoots.insert(User);
      } else if (Idx == Scale && Inc > 1) {
        LoopIncs.push_back(User);
      }
    }
  }

  if (Roots[0].empty())
    return false;
  bool AllSame = true;
  for (unsigned i = 1; i < Scale-1; ++i)
    if (Roots[i].size() != Roots[0].size()) {
      AllSame = false;
      break;
    }

  if (!AllSame)
    return false;

  return true;
}

// Validate the selected reductions. All iterations must have an isomorphic
// part of the reduction chain and, for non-associative reductions, the chain
// entries must appear in order.
bool LoopReroll::ReductionTracker::validateSelected() {
  // For a non-associative reduction, the chain entries must appear in order.
  for (DenseSet<int>::iterator RI = Reds.begin(), RIE = Reds.end();
       RI != RIE; ++RI) {
    int i = *RI;
    int PrevIter = 0, BaseCount = 0, Count = 0;
    for (SimpleLoopReduction::iterator J = PossibleReds[i].begin(),
         JE = PossibleReds[i].end(); J != JE; ++J) {
	// Note that all instructions in the chain must have been found because
	// all instructions in the function must have been assigned to some
	// iteration.
      int Iter = PossibleRedIter[*J];
      if (Iter != PrevIter && Iter != PrevIter + 1 &&
          !PossibleReds[i].getReducedValue()->isAssociative()) {
        DEBUG(dbgs() << "LRR: Out-of-order non-associative reduction: " <<
                        *J << "\n");
        return false;
      }

      if (Iter != PrevIter) {
        if (Count != BaseCount) {
          DEBUG(dbgs() << "LRR: Iteration " << PrevIter <<
                " reduction use count " << Count <<
                " is not equal to the base use count " <<
                BaseCount << "\n");
          return false;
        }

        Count = 0;
      }

      ++Count;
      if (Iter == 0)
        ++BaseCount;

      PrevIter = Iter;
    }
  }

  return true;
}

// For all selected reductions, remove all parts except those in the first
// iteration (and the PHI). Replace outside uses of the reduced value with uses
// of the first-iteration reduced value (in other words, reroll the selected
// reductions).
void LoopReroll::ReductionTracker::replaceSelected() {
  // Fixup reductions to refer to the last instruction associated with the
  // first iteration (not the last).
  for (DenseSet<int>::iterator RI = Reds.begin(), RIE = Reds.end();
       RI != RIE; ++RI) {
    int i = *RI;
    int j = 0;
    for (int e = PossibleReds[i].size(); j != e; ++j)
      if (PossibleRedIter[PossibleReds[i][j]] != 0) {
        --j;
        break;
      }

    // Replace users with the new end-of-chain value.
    SmallInstructionVector Users;
    for (Value::use_iterator UI =
           PossibleReds[i].getReducedValue()->use_begin(),
         UIE = PossibleReds[i].getReducedValue()->use_end(); UI != UIE; ++UI)
      Users.push_back(cast<Instruction>(*UI));

    for (SmallInstructionVector::iterator J = Users.begin(),
         JE = Users.end(); J != JE; ++J)
      (*J)->replaceUsesOfWith(PossibleReds[i].getReducedValue(),
                              PossibleReds[i][j]);
  }
}

// Reroll the provided loop with respect to the provided induction variable.
// Generally, we're looking for a loop like this:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// f(%iv)
// %iv.1 = add %iv, 1                <-- a root increment
// f(%iv.1)
// %iv.2 = add %iv, 2                <-- a root increment
// f(%iv.2)
// %iv.scale_m_1 = add %iv, scale-1  <-- a root increment
// f(%iv.scale_m_1)
// ...
// %iv.next = add %iv, scale
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit
//
// Notably, we do not require that f(%iv), f(%iv.1), etc. be isolated groups of
// instructions. In other words, the instructions in f(%iv), f(%iv.1), etc. can
// be intermixed with eachother. The restriction imposed by this algorithm is
// that the relative order of the isomorphic instructions in f(%iv), f(%iv.1),
// etc. be the same.
//
// First, we collect the use set of %iv, excluding the other increment roots.
// This gives us f(%iv). Then we iterate over the loop instructions (scale-1)
// times, having collected the use set of f(%iv.(i+1)), during which we:
//   - Ensure that the next unmatched instruction in f(%iv) is isomorphic to
//     the next unmatched instruction in f(%iv.(i+1)).
//   - Ensure that both matched instructions don't have any external users
//     (with the exception of last-in-chain reduction instructions).
//   - Track the (aliasing) write set, and other side effects, of all
//     instructions that belong to future iterations that come before the matched
//     instructions. If the matched instructions read from that write set, then
//     f(%iv) or f(%iv.(i+1)) has some dependency on instructions in
//     f(%iv.(j+1)) for some j > i, and we cannot reroll the loop. Similarly,
//     if any of these future instructions had side effects (could not be
//     speculatively executed), and so do the matched instructions, when we
//     cannot reorder those side-effect-producing instructions, and rerolling
//     fails.
//
// Finally, we make sure that all loop instructions are either loop increment
// roots, belong to simple latch code, parts of validated reductions, part of
// f(%iv) or part of some f(%iv.i). If all of that is true (and all reductions
// have been validated), then we reroll the loop.
bool LoopReroll::reroll(Instruction *IV, Loop *L, BasicBlock *Header,
                        const SCEV *IterCount,
                        ReductionTracker &Reductions) {
  const SCEVAddRecExpr *RealIVSCEV = cast<SCEVAddRecExpr>(SE->getSCEV(IV));
  uint64_t Inc = cast<SCEVConstant>(RealIVSCEV->getOperand(1))->
                   getValue()->getZExtValue();
  // The collection of loop increment instructions.
  SmallInstructionVector LoopIncs;
  uint64_t Scale = Inc;

  // The effective induction variable, IV, is normally also the real induction
  // variable. When we're dealing with a loop like:
  //   for (int i = 0; i < 500; ++i)
  //     x[3*i] = ...;
  //     x[3*i+1] = ...;
  //     x[3*i+2] = ...;
  // then the real IV is still i, but the effective IV is (3*i).
  Instruction *RealIV = IV;
  if (Inc == 1 && !findScaleFromMul(RealIV, Scale, IV, LoopIncs))
    return false;

  assert(Scale <= MaxInc && "Scale is too large");
  assert(Scale > 1 && "Scale must be at least 2");

  // The set of increment instructions for each increment value.
  SmallVector<SmallInstructionVector, 32> Roots(Scale-1);
  SmallInstructionSet AllRoots;
  if (!collectAllRoots(L, Inc, Scale, IV, Roots, AllRoots, LoopIncs))
    return false;

  DEBUG(dbgs() << "LRR: Found all root induction increments for: " <<
                  *RealIV << "\n");

  // An array of just the possible reductions for this scale factor. When we
  // collect the set of all users of some root instructions, these reduction
  // instructions are treated as 'final' (their uses are not considered).
  // This is important because we don't want the root use set to search down
  // the reduction chain.
  SmallInstructionSet PossibleRedSet;
  SmallInstructionSet PossibleRedLastSet, PossibleRedPHISet;
  Reductions.restrictToScale(Scale, PossibleRedSet, PossibleRedPHISet,
                             PossibleRedLastSet);

  // We now need to check for equivalence of the use graph of each root with
  // that of the primary induction variable (excluding the roots). Our goal
  // here is not to solve the full graph isomorphism problem, but rather to
  // catch common cases without a lot of work. As a result, we will assume
  // that the relative order of the instructions in each unrolled iteration
  // is the same (although we will not make an assumption about how the
  // different iterations are intermixed). Note that while the order must be
  // the same, the instructions may not be in the same basic block.
  SmallInstructionSet Exclude(AllRoots);
  Exclude.insert(LoopIncs.begin(), LoopIncs.end());

  DenseSet<Instruction *> BaseUseSet;
  collectInLoopUserSet(L, IV, Exclude, PossibleRedSet, BaseUseSet);

  DenseSet<Instruction *> AllRootUses;
  std::vector<DenseSet<Instruction *> > RootUseSets(Scale-1);

  bool MatchFailed = false;
  for (unsigned i = 0; i < Scale-1 && !MatchFailed; ++i) {
    DenseSet<Instruction *> &RootUseSet = RootUseSets[i];
    collectInLoopUserSet(L, Roots[i], SmallInstructionSet(),
                         PossibleRedSet, RootUseSet);

    DEBUG(dbgs() << "LRR: base use set size: " << BaseUseSet.size() <<
                    " vs. iteration increment " << (i+1) <<
                    " use set size: " << RootUseSet.size() << "\n");

    if (BaseUseSet.size() != RootUseSet.size()) {
      MatchFailed = true;
      break;
    }

    // In addition to regular aliasing information, we need to look for
    // instructions from later (future) iterations that have side effects
    // preventing us from reordering them past other instructions with side
    // effects.
    bool FutureSideEffects = false;
    AliasSetTracker AST(*AA);

    // The map between instructions in f(%iv.(i+1)) and f(%iv).
    DenseMap<Value *, Value *> BaseMap;

    assert(L->getNumBlocks() == 1 && "Cannot handle multi-block loops");
    for (BasicBlock::iterator J1 = Header->begin(), J2 = Header->begin(),
         JE = Header->end(); J1 != JE && !MatchFailed; ++J1) {
      if (cast<Instruction>(J1) == RealIV)
        continue;
      if (cast<Instruction>(J1) == IV)
        continue;
      if (!BaseUseSet.count(J1))
        continue;
      if (PossibleRedPHISet.count(J1)) // Skip reduction PHIs.
        continue;

      while (J2 != JE && (!RootUseSet.count(J2) ||
             std::find(Roots[i].begin(), Roots[i].end(), J2) !=
               Roots[i].end())) {
        // As we iterate through the instructions, instructions that don't
        // belong to previous iterations (or the base case), must belong to
        // future iterations. We want to track the alias set of writes from
        // previous iterations.
        if (!isa<PHINode>(J2) && !BaseUseSet.count(J2) &&
            !AllRootUses.count(J2)) {
          if (J2->mayWriteToMemory())
            AST.add(J2);

          // Note: This is specifically guarded by a check on isa<PHINode>,
          // which while a valid (somewhat arbitrary) micro-optimization, is
          // needed because otherwise isSafeToSpeculativelyExecute returns
          // false on PHI nodes.
          if (!isSimpleLoadStore(J2) && !isSafeToSpeculativelyExecute(J2, DL))
            FutureSideEffects = true; 
        }

        ++J2;
      }

      if (!J1->isSameOperationAs(J2)) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *J1 <<
                        " vs. " << *J2 << "\n");
        MatchFailed = true;
        break;
      }

      // Make sure that this instruction, which is in the use set of this
      // root instruction, does not also belong to the base set or the set of
      // some previous root instruction.
      if (BaseUseSet.count(J2) || AllRootUses.count(J2)) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *J1 <<
                        " vs. " << *J2 << " (prev. case overlap)\n");
        MatchFailed = true;
        break;
      }

      // Make sure that we don't alias with any instruction in the alias set
      // tracker. If we do, then we depend on a future iteration, and we
      // can't reroll.
      if (J2->mayReadFromMemory()) {
        for (AliasSetTracker::iterator K = AST.begin(), KE = AST.end();
             K != KE && !MatchFailed; ++K) {
          if (K->aliasesUnknownInst(J2, *AA)) {
            DEBUG(dbgs() << "LRR: iteration root match failed at " << *J1 <<
                            " vs. " << *J2 << " (depends on future store)\n");
            MatchFailed = true;
            break;
          }
        }
      }

      // If we've past an instruction from a future iteration that may have
      // side effects, and this instruction might also, then we can't reorder
      // them, and this matching fails. As an exception, we allow the alias
      // set tracker to handle regular (simple) load/store dependencies.
      if (FutureSideEffects &&
            ((!isSimpleLoadStore(J1) && !isSafeToSpeculativelyExecute(J1)) ||
             (!isSimpleLoadStore(J2) && !isSafeToSpeculativelyExecute(J2)))) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *J1 <<
                        " vs. " << *J2 <<
                        " (side effects prevent reordering)\n");
        MatchFailed = true;
        break;
      }

      // For instructions that are part of a reduction, if the operation is
      // associative, then don't bother matching the operands (because we
      // already know that the instructions are isomorphic, and the order
      // within the iteration does not matter). For non-associative reductions,
      // we do need to match the operands, because we need to reject
      // out-of-order instructions within an iteration!
      // For example (assume floating-point addition), we need to reject this:
      //   x += a[i]; x += b[i];
      //   x += a[i+1]; x += b[i+1];
      //   x += b[i+2]; x += a[i+2];
      bool InReduction = Reductions.isPairInSame(J1, J2);

      if (!(InReduction && J1->isAssociative())) {
        bool Swapped = false, SomeOpMatched = false;;
        for (unsigned j = 0; j < J1->getNumOperands() && !MatchFailed; ++j) {
          Value *Op2 = J2->getOperand(j);

	  // If this is part of a reduction (and the operation is not
	  // associatve), then we match all operands, but not those that are
	  // part of the reduction.
          if (InReduction)
            if (Instruction *Op2I = dyn_cast<Instruction>(Op2))
              if (Reductions.isPairInSame(J2, Op2I))
                continue;

          DenseMap<Value *, Value *>::iterator BMI = BaseMap.find(Op2);
          if (BMI != BaseMap.end())
            Op2 = BMI->second;
          else if (std::find(Roots[i].begin(), Roots[i].end(),
                             (Instruction*) Op2) != Roots[i].end())
            Op2 = IV;

          if (J1->getOperand(Swapped ? unsigned(!j) : j) != Op2) {
	    // If we've not already decided to swap the matched operands, and
	    // we've not already matched our first operand (note that we could
	    // have skipped matching the first operand because it is part of a
	    // reduction above), and the instruction is commutative, then try
	    // the swapped match.
            if (!Swapped && J1->isCommutative() && !SomeOpMatched &&
                J1->getOperand(!j) == Op2) {
              Swapped = true;
            } else {
              DEBUG(dbgs() << "LRR: iteration root match failed at " << *J1 <<
                              " vs. " << *J2 << " (operand " << j << ")\n");
              MatchFailed = true;
              break;
            }
          }

          SomeOpMatched = true;
        }
      }

      if ((!PossibleRedLastSet.count(J1) && hasUsesOutsideLoop(J1, L)) ||
          (!PossibleRedLastSet.count(J2) && hasUsesOutsideLoop(J2, L))) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *J1 <<
                        " vs. " << *J2 << " (uses outside loop)\n");
        MatchFailed = true;
        break;
      }

      if (!MatchFailed)
        BaseMap.insert(std::pair<Value *, Value *>(J2, J1));

      AllRootUses.insert(J2);
      Reductions.recordPair(J1, J2, i+1);

      ++J2;
    }
  }

  if (MatchFailed)
    return false;

  DEBUG(dbgs() << "LRR: Matched all iteration increments for " <<
                  *RealIV << "\n");

  DenseSet<Instruction *> LoopIncUseSet;
  collectInLoopUserSet(L, LoopIncs, SmallInstructionSet(),
                       SmallInstructionSet(), LoopIncUseSet);
  DEBUG(dbgs() << "LRR: Loop increment set size: " <<
                  LoopIncUseSet.size() << "\n");

  // Make sure that all instructions in the loop have been included in some
  // use set.
  for (BasicBlock::iterator J = Header->begin(), JE = Header->end();
       J != JE; ++J) {
    if (isa<DbgInfoIntrinsic>(J))
      continue;
    if (cast<Instruction>(J) == RealIV)
      continue;
    if (cast<Instruction>(J) == IV)
      continue;
    if (BaseUseSet.count(J) || AllRootUses.count(J) ||
        (LoopIncUseSet.count(J) && (J->isTerminator() ||
                                    isSafeToSpeculativelyExecute(J, DL))))
      continue;

    if (AllRoots.count(J))
      continue;

    if (Reductions.isSelectedPHI(J))
      continue;

    DEBUG(dbgs() << "LRR: aborting reroll based on " << *RealIV <<
                    " unprocessed instruction found: " << *J << "\n");
    MatchFailed = true;
    break;
  }

  if (MatchFailed)
    return false;

  DEBUG(dbgs() << "LRR: all instructions processed from " <<
                  *RealIV << "\n");

  if (!Reductions.validateSelected())
    return false;

  // At this point, we've validated the rerolling, and we're committed to
  // making changes!

  Reductions.replaceSelected();

  // Remove instructions associated with non-base iterations.
  for (BasicBlock::reverse_iterator J = Header->rbegin();
       J != Header->rend();) {
    if (AllRootUses.count(&*J)) {
      Instruction *D = &*J;
      DEBUG(dbgs() << "LRR: removing: " << *D << "\n");
      D->eraseFromParent();
      continue;
    }

    ++J; 
  }

  // Insert the new induction variable.
  const SCEV *Start = RealIVSCEV->getStart();
  if (Inc == 1)
    Start = SE->getMulExpr(Start,
                           SE->getConstant(Start->getType(), Scale));
  const SCEVAddRecExpr *H =
    cast<SCEVAddRecExpr>(SE->getAddRecExpr(Start,
                           SE->getConstant(RealIVSCEV->getType(), 1),
                           L, SCEV::FlagAnyWrap));
  { // Limit the lifetime of SCEVExpander.
    SCEVExpander Expander(*SE, "reroll");
    Value *NewIV = Expander.expandCodeFor(H, IV->getType(), Header->begin());

    for (DenseSet<Instruction *>::iterator J = BaseUseSet.begin(),
         JE = BaseUseSet.end(); J != JE; ++J)
      (*J)->replaceUsesOfWith(IV, NewIV);

    if (BranchInst *BI = dyn_cast<BranchInst>(Header->getTerminator())) {
      if (LoopIncUseSet.count(BI)) {
        const SCEV *ICSCEV = RealIVSCEV->evaluateAtIteration(IterCount, *SE);
        if (Inc == 1)
          ICSCEV =
            SE->getMulExpr(ICSCEV, SE->getConstant(ICSCEV->getType(), Scale));
        // Iteration count SCEV minus 1
        const SCEV *ICMinus1SCEV =
          SE->getMinusSCEV(ICSCEV, SE->getConstant(ICSCEV->getType(), 1));

        Value *ICMinus1; // Iteration count minus 1
        if (isa<SCEVConstant>(ICMinus1SCEV)) {
          ICMinus1 = Expander.expandCodeFor(ICMinus1SCEV, NewIV->getType(), BI);
        } else {
          BasicBlock *Preheader = L->getLoopPreheader();
          if (!Preheader)
            Preheader = InsertPreheaderForLoop(L, this);

          ICMinus1 = Expander.expandCodeFor(ICMinus1SCEV, NewIV->getType(),
                                            Preheader->getTerminator());
        }
 
        Value *Cond = new ICmpInst(BI, CmpInst::ICMP_EQ, NewIV, ICMinus1,
                                   "exitcond");
        BI->setCondition(Cond);

        if (BI->getSuccessor(1) != Header)
          BI->swapSuccessors();
      }
    }
  }

  SimplifyInstructionsInBlock(Header, DL, TLI);
  DeleteDeadPHIs(Header, TLI);
  ++NumRerolledLoops;
  return true;
}

bool LoopReroll::runOnLoop(Loop *L, LPPassManager &LPM) {
  AA = &getAnalysis<AliasAnalysis>();
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();
  TLI = &getAnalysis<TargetLibraryInfo>();
  DL = getAnalysisIfAvailable<DataLayout>();
  DT = &getAnalysis<DominatorTree>();

  BasicBlock *Header = L->getHeader();
  DEBUG(dbgs() << "LRR: F[" << Header->getParent()->getName() <<
        "] Loop %" << Header->getName() << " (" <<
        L->getNumBlocks() << " block(s))\n");

  bool Changed = false;

  // For now, we'll handle only single BB loops.
  if (L->getNumBlocks() > 1)
    return Changed;

  if (!SE->hasLoopInvariantBackedgeTakenCount(L))
    return Changed;

  const SCEV *LIBETC = SE->getBackedgeTakenCount(L);
  const SCEV *IterCount =
    SE->getAddExpr(LIBETC, SE->getConstant(LIBETC->getType(), 1));
  DEBUG(dbgs() << "LRR: iteration count = " << *IterCount << "\n");

  // First, we need to find the induction variable with respect to which we can
  // reroll (there may be several possible options).
  SmallInstructionVector PossibleIVs;
  collectPossibleIVs(L, PossibleIVs);

  if (PossibleIVs.empty()) {
    DEBUG(dbgs() << "LRR: No possible IVs found\n");
    return Changed;
  }

  ReductionTracker Reductions;
  collectPossibleReductions(L, Reductions);

  // For each possible IV, collect the associated possible set of 'root' nodes
  // (i+1, i+2, etc.).
  for (SmallInstructionVector::iterator I = PossibleIVs.begin(),
       IE = PossibleIVs.end(); I != IE; ++I)
    if (reroll(*I, L, Header, IterCount, Reductions)) {
      Changed = true;
      break;
    }

  return Changed;
}

