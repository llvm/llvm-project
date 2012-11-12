//===- IVUsers.cpp - Induction Variable Users -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bookkeeping for "interesting" users of expressions
// computed from induction variables.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "iv-users"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/DataLayout.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

char IVUsers::ID = 0;
INITIALIZE_PASS_BEGIN(IVUsers, "iv-users",
                      "Induction Variable Users", false, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(IVUsers, "iv-users",
                      "Induction Variable Users", false, true)

Pass *llvm::createIVUsersPass() {
  return new IVUsers();
}

/// isInteresting - Test whether the given expression is "interesting" when
/// used by the given expression, within the context of analyzing the
/// given loop.
static bool isInteresting(const SCEV *S, const Instruction *I, const Loop *L,
                          ScalarEvolution *SE, LoopInfo *LI) {
  // An addrec is interesting if it's affine or if it has an interesting start.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // Keep things simple. Don't touch loop-variant strides unless they're
    // only used outside the loop and we can simplify them.
    if (AR->getLoop() == L)
      return AR->isAffine() ||
             (!L->contains(I) &&
              SE->getSCEVAtScope(AR, LI->getLoopFor(I->getParent())) != AR);
    // Otherwise recurse to see if the start value is interesting, and that
    // the step value is not interesting, since we don't yet know how to
    // do effective SCEV expansions for addrecs with interesting steps.
    return isInteresting(AR->getStart(), I, L, SE, LI) &&
          !isInteresting(AR->getStepRecurrence(*SE), I, L, SE, LI);
  }

  // An add is interesting if exactly one of its operands is interesting.
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    bool AnyInterestingYet = false;
    for (SCEVAddExpr::op_iterator OI = Add->op_begin(), OE = Add->op_end();
         OI != OE; ++OI)
      if (isInteresting(*OI, I, L, SE, LI)) {
        if (AnyInterestingYet)
          return false;
        AnyInterestingYet = true;
      }
    return AnyInterestingYet;
  }

  // Nothing else is interesting here.
  return false;
}

/// Return true if all loop headers that dominate this block are in simplified
/// form.
static bool isSimplifiedLoopNest(BasicBlock *BB, const DominatorTree *DT,
                                 const LoopInfo *LI,
                                 SmallPtrSet<Loop*,16> &SimpleLoopNests) {
  Loop *NearestLoop = 0;
  for (DomTreeNode *Rung = DT->getNode(BB);
       Rung; Rung = Rung->getIDom()) {
    BasicBlock *DomBB = Rung->getBlock();
    Loop *DomLoop = LI->getLoopFor(DomBB);
    if (DomLoop && DomLoop->getHeader() == DomBB) {
      // If the domtree walk reaches a loop with no preheader, return false.
      if (!DomLoop->isLoopSimplifyForm())
        return false;
      // If we have already checked this loop nest, stop checking.
      if (SimpleLoopNests.count(DomLoop))
        break;
      // If we have not already checked this loop nest, remember the loop
      // header nearest to BB. The nearest loop may not contain BB.
      if (!NearestLoop)
        NearestLoop = DomLoop;
    }
  }
  if (NearestLoop)
    SimpleLoopNests.insert(NearestLoop);
  return true;
}

/// AddUsersImpl - Inspect the specified instruction.  If it is a
/// reducible SCEV, recursively add its users to the IVUsesByStride set and
/// return true.  Otherwise, return false.
bool IVUsers::AddUsersImpl(Instruction *I,
                           SmallPtrSet<Loop*,16> &SimpleLoopNests) {
  // Add this IV user to the Processed set before returning false to ensure that
  // all IV users are members of the set. See IVUsers::isIVUserOrOperand.
  if (!Processed.insert(I))
    return true;    // Instruction already handled.

  if (!SE->isSCEVable(I->getType()))
    return false;   // Void and FP expressions cannot be reduced.

  // IVUsers is used by LSR which assumes that all SCEV expressions are safe to
  // pass to SCEVExpander. Expressions are not safe to expand if they represent
  // operations that are not safe to speculate, namely integer division.
  if (!isa<PHINode>(I) && !isSafeToSpeculativelyExecute(I, TD))
    return false;

  // LSR is not APInt clean, do not touch integers bigger than 64-bits.
  // Also avoid creating IVs of non-native types. For example, we don't want a
  // 64-bit IV in 32-bit code just because the loop has one 64-bit cast.
  uint64_t Width = SE->getTypeSizeInBits(I->getType());
  if (Width > 64 || (TD && !TD->isLegalInteger(Width)))
    return false;

  // Get the symbolic expression for this instruction.
  const SCEV *ISE = SE->getSCEV(I);

  // If we've come to an uninteresting expression, stop the traversal and
  // call this a user.
  if (!isInteresting(ISE, I, L, SE, LI))
    return false;

  SmallPtrSet<Instruction *, 4> UniqueUsers;
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    if (!UniqueUsers.insert(User))
      continue;

    // Do not infinitely recurse on PHI nodes.
    if (isa<PHINode>(User) && Processed.count(User))
      continue;

    // Only consider IVUsers that are dominated by simplified loop
    // headers. Otherwise, SCEVExpander will crash.
    BasicBlock *UseBB = User->getParent();
    // A phi's use is live out of its predecessor block.
    if (PHINode *PHI = dyn_cast<PHINode>(User)) {
      unsigned OperandNo = UI.getOperandNo();
      unsigned ValNo = PHINode::getIncomingValueNumForOperand(OperandNo);
      UseBB = PHI->getIncomingBlock(ValNo);
    }
    if (!isSimplifiedLoopNest(UseBB, DT, LI, SimpleLoopNests))
      return false;

    // Descend recursively, but not into PHI nodes outside the current loop.
    // It's important to see the entire expression outside the loop to get
    // choices that depend on addressing mode use right, although we won't
    // consider references outside the loop in all cases.
    // If User is already in Processed, we don't want to recurse into it again,
    // but do want to record a second reference in the same instruction.
    bool AddUserToIVUsers = false;
    if (LI->getLoopFor(User->getParent()) != L) {
      if (isa<PHINode>(User) || Processed.count(User) ||
          !AddUsersImpl(User, SimpleLoopNests)) {
        DEBUG(dbgs() << "FOUND USER in other loop: " << *User << '\n'
                     << "   OF SCEV: " << *ISE << '\n');
        AddUserToIVUsers = true;
      }
    } else if (Processed.count(User) || !AddUsersImpl(User, SimpleLoopNests)) {
      DEBUG(dbgs() << "FOUND USER: " << *User << '\n'
                   << "   OF SCEV: " << *ISE << '\n');
      AddUserToIVUsers = true;
    }

    if (AddUserToIVUsers) {
      // Okay, we found a user that we cannot reduce.
      IVUses.push_back(new IVStrideUse(this, User, I));
      IVStrideUse &NewUse = IVUses.back();
      // Autodetect the post-inc loop set, populating NewUse.PostIncLoops.
      // The regular return value here is discarded; instead of recording
      // it, we just recompute it when we need it.
      ISE = TransformForPostIncUse(NormalizeAutodetect,
                                   ISE, User, I,
                                   NewUse.PostIncLoops,
                                   *SE, *DT);
      DEBUG(if (SE->getSCEV(I) != ISE)
              dbgs() << "   NORMALIZED TO: " << *ISE << '\n');
    }
  }
  return true;
}

bool IVUsers::AddUsersIfInteresting(Instruction *I) {
  // SCEVExpander can only handle users that are dominated by simplified loop
  // entries. Keep track of all loops that are only dominated by other simple
  // loops so we don't traverse the domtree for each user.
  SmallPtrSet<Loop*,16> SimpleLoopNests;

  return AddUsersImpl(I, SimpleLoopNests);
}

IVStrideUse &IVUsers::AddUser(Instruction *User, Value *Operand) {
  IVUses.push_back(new IVStrideUse(this, User, Operand));
  return IVUses.back();
}

IVUsers::IVUsers()
    : LoopPass(ID) {
  initializeIVUsersPass(*PassRegistry::getPassRegistry());
}

void IVUsers::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();
  AU.addRequired<DominatorTree>();
  AU.addRequired<ScalarEvolution>();
  AU.setPreservesAll();
}

bool IVUsers::runOnLoop(Loop *l, LPPassManager &LPM) {

  L = l;
  LI = &getAnalysis<LoopInfo>();
  DT = &getAnalysis<DominatorTree>();
  SE = &getAnalysis<ScalarEvolution>();
  TD = getAnalysisIfAvailable<DataLayout>();

  // Find all uses of induction variables in this loop, and categorize
  // them by stride.  Start by finding all of the PHI nodes in the header for
  // this loop.  If they are induction variables, inspect their uses.
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I)
    (void)AddUsersIfInteresting(I);

  return false;
}

void IVUsers::print(raw_ostream &OS, const Module *M) const {
  OS << "IV Users for loop ";
  WriteAsOperand(OS, L->getHeader(), false);
  if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
    OS << " with backedge-taken count "
       << *SE->getBackedgeTakenCount(L);
  }
  OS << ":\n";

  for (ilist<IVStrideUse>::const_iterator UI = IVUses.begin(),
       E = IVUses.end(); UI != E; ++UI) {
    OS << "  ";
    WriteAsOperand(OS, UI->getOperandValToReplace(), false);
    OS << " = " << *getReplacementExpr(*UI);
    for (PostIncLoopSet::const_iterator
         I = UI->PostIncLoops.begin(),
         E = UI->PostIncLoops.end(); I != E; ++I) {
      OS << " (post-inc with loop ";
      WriteAsOperand(OS, (*I)->getHeader(), false);
      OS << ")";
    }
    OS << " in  ";
    UI->getUser()->print(OS);
    OS << '\n';
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void IVUsers::dump() const {
  print(dbgs());
}
#endif

void IVUsers::releaseMemory() {
  Processed.clear();
  IVUses.clear();
}

/// getReplacementExpr - Return a SCEV expression which computes the
/// value of the OperandValToReplace.
const SCEV *IVUsers::getReplacementExpr(const IVStrideUse &IU) const {
  return SE->getSCEV(IU.getOperandValToReplace());
}

/// getExpr - Return the expression for the use.
const SCEV *IVUsers::getExpr(const IVStrideUse &IU) const {
  return
    TransformForPostIncUse(Normalize, getReplacementExpr(IU),
                           IU.getUser(), IU.getOperandValToReplace(),
                           const_cast<PostIncLoopSet &>(IU.getPostIncLoops()),
                           *SE, *DT);
}

static const SCEVAddRecExpr *findAddRecForLoop(const SCEV *S, const Loop *L) {
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (AR->getLoop() == L)
      return AR;
    return findAddRecForLoop(AR->getStart(), L);
  }

  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (SCEVAddExpr::op_iterator I = Add->op_begin(), E = Add->op_end();
         I != E; ++I)
      if (const SCEVAddRecExpr *AR = findAddRecForLoop(*I, L))
        return AR;
    return 0;
  }

  return 0;
}

const SCEV *IVUsers::getStride(const IVStrideUse &IU, const Loop *L) const {
  if (const SCEVAddRecExpr *AR = findAddRecForLoop(getExpr(IU), L))
    return AR->getStepRecurrence(*SE);
  return 0;
}

void IVStrideUse::transformToPostInc(const Loop *L) {
  PostIncLoops.insert(L);
}

void IVStrideUse::deleted() {
  // Remove this user from the list.
  Parent->Processed.erase(this->getUser());
  Parent->IVUses.erase(this);
  // this now dangles!
}
