//===- TailRecursionElimination.cpp - Eliminate Tail Calls ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file transforms calls of the current function (self recursion) followed
// by a return instruction with a branch to the entry of the function, creating
// a loop.  This pass also implements the following extensions to the basic
// algorithm:
//
//  1. Trivial instructions between the call and return do not prevent the
//     transformation from taking place, though currently the analysis cannot
//     support moving any really useful instructions (only dead ones).
//  2. This pass transforms functions that are prevented from being tail
//     recursive by an associative and commutative expression to use an
//     accumulator variable, thus compiling the typical naive factorial or
//     'fib' implementation into efficient code.
//  3. TRE is performed if the function returns void, if the return
//     returns the result returned by the call, or if the function returns a
//     run-time constant on all exits from the function.  It is possible, though
//     unlikely, that the return returns something else (like constant 0), and
//     can still be TRE'd.  It can be TRE'd if ALL OTHER return instructions in
//     the function return the exact same value.
//  4. If it can prove that callees do not access their caller stack frame,
//     they are marked as eligible for tail call elimination (by the code
//     generator).
//
// There are several improvements that could be made:
//
//  1. If the function has any alloca instructions, these instructions will be
//     moved out of the entry block of the function, causing them to be
//     evaluated each time through the tail recursion.  Safely keeping allocas
//     in the entry block requires analysis to proves that the tail-called
//     function does not read or write the stack object.
//  2. Tail recursion is only performed if the call immediately precedes the
//     return instruction.  It's possible that there could be a jump between
//     the call and the return.
//  3. There can be intervening operations between the call and the return that
//     prevent the TRE from occurring.  For example, there could be GEP's and
//     stores to memory that will not be read or written by the call.  This
//     requires some substantial analysis (such as with DSA) to prove safe to
//     move ahead of the call, but doing so could allow many more TREs to be
//     performed, for example in TreeAdd/TreeAlloc from the treeadd benchmark.
//  4. The algorithm we use to detect if callees access their caller stack
//     frames is very primitive.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tailcallelim"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumEliminated, "Number of tail calls removed");
STATISTIC(NumRetDuped,   "Number of return duplicated");
STATISTIC(NumAccumAdded, "Number of accumulators introduced");

namespace {
  struct TailCallElim : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    TailCallElim() : FunctionPass(ID) {
      initializeTailCallElimPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);

  private:
    CallInst *FindTRECandidate(Instruction *I,
                               bool CannotTailCallElimCallsMarkedTail);
    bool EliminateRecursiveTailCall(CallInst *CI, ReturnInst *Ret,
                                    BasicBlock *&OldEntry,
                                    bool &TailCallsAreMarkedTail,
                                    SmallVector<PHINode*, 8> &ArgumentPHIs,
                                    bool CannotTailCallElimCallsMarkedTail);
    bool FoldReturnAndProcessPred(BasicBlock *BB,
                                  ReturnInst *Ret, BasicBlock *&OldEntry,
                                  bool &TailCallsAreMarkedTail,
                                  SmallVector<PHINode*, 8> &ArgumentPHIs,
                                  bool CannotTailCallElimCallsMarkedTail);
    bool ProcessReturningBlock(ReturnInst *RI, BasicBlock *&OldEntry,
                               bool &TailCallsAreMarkedTail,
                               SmallVector<PHINode*, 8> &ArgumentPHIs,
                               bool CannotTailCallElimCallsMarkedTail);
    bool CanMoveAboveCall(Instruction *I, CallInst *CI);
    Value *CanTransformAccumulatorRecursion(Instruction *I, CallInst *CI);
  };
}

char TailCallElim::ID = 0;
INITIALIZE_PASS(TailCallElim, "tailcallelim",
                "Tail Call Elimination", false, false)

// Public interface to the TailCallElimination pass
FunctionPass *llvm::createTailCallEliminationPass() {
  return new TailCallElim();
}

/// AllocaMightEscapeToCalls - Return true if this alloca may be accessed by
/// callees of this function.  We only do very simple analysis right now, this
/// could be expanded in the future to use mod/ref information for particular
/// call sites if desired.
static bool AllocaMightEscapeToCalls(AllocaInst *AI) {
  // FIXME: do simple 'address taken' analysis.
  return true;
}

/// CheckForEscapingAllocas - Scan the specified basic block for alloca
/// instructions.  If it contains any that might be accessed by calls, return
/// true.
static bool CheckForEscapingAllocas(BasicBlock *BB,
                                    bool &CannotTCETailMarkedCall) {
  bool RetVal = false;
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
      RetVal |= AllocaMightEscapeToCalls(AI);

      // If this alloca is in the body of the function, or if it is a variable
      // sized allocation, we cannot tail call eliminate calls marked 'tail'
      // with this mechanism.
      if (BB != &BB->getParent()->getEntryBlock() ||
          !isa<ConstantInt>(AI->getArraySize()))
        CannotTCETailMarkedCall = true;
    }
  return RetVal;
}

bool TailCallElim::runOnFunction(Function &F) {
  // If this function is a varargs function, we won't be able to PHI the args
  // right, so don't even try to convert it...
  if (F.getFunctionType()->isVarArg()) return false;

  BasicBlock *OldEntry = 0;
  bool TailCallsAreMarkedTail = false;
  SmallVector<PHINode*, 8> ArgumentPHIs;
  bool MadeChange = false;
  bool FunctionContainsEscapingAllocas = false;

  // CannotTCETailMarkedCall - If true, we cannot perform TCE on tail calls
  // marked with the 'tail' attribute, because doing so would cause the stack
  // size to increase (real TCE would deallocate variable sized allocas, TCE
  // doesn't).
  bool CannotTCETailMarkedCall = false;

  // Loop over the function, looking for any returning blocks, and keeping track
  // of whether this function has any non-trivially used allocas.
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (FunctionContainsEscapingAllocas && CannotTCETailMarkedCall)
      break;

    FunctionContainsEscapingAllocas |=
      CheckForEscapingAllocas(BB, CannotTCETailMarkedCall);
  }

  /// FIXME: The code generator produces really bad code when an 'escaping
  /// alloca' is changed from being a static alloca to being a dynamic alloca.
  /// Until this is resolved, disable this transformation if that would ever
  /// happen.  This bug is PR962.
  if (FunctionContainsEscapingAllocas)
    return false;

  // Second pass, change any tail calls to loops.
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
      bool Change = ProcessReturningBlock(Ret, OldEntry, TailCallsAreMarkedTail,
                                          ArgumentPHIs,CannotTCETailMarkedCall);
      if (!Change && BB->getFirstNonPHIOrDbg() == Ret)
        Change = FoldReturnAndProcessPred(BB, Ret, OldEntry,
                                          TailCallsAreMarkedTail, ArgumentPHIs,
                                          CannotTCETailMarkedCall);
      MadeChange |= Change;
    }
  }

  // If we eliminated any tail recursions, it's possible that we inserted some
  // silly PHI nodes which just merge an initial value (the incoming operand)
  // with themselves.  Check to see if we did and clean up our mess if so.  This
  // occurs when a function passes an argument straight through to its tail
  // call.
  if (!ArgumentPHIs.empty()) {
    for (unsigned i = 0, e = ArgumentPHIs.size(); i != e; ++i) {
      PHINode *PN = ArgumentPHIs[i];

      // If the PHI Node is a dynamic constant, replace it with the value it is.
      if (Value *PNV = SimplifyInstruction(PN)) {
        PN->replaceAllUsesWith(PNV);
        PN->eraseFromParent();
      }
    }
  }

  // Finally, if this function contains no non-escaping allocas, or calls
  // setjmp, mark all calls in the function as eligible for tail calls
  //(there is no stack memory for them to access).
  if (!FunctionContainsEscapingAllocas && !F.callsFunctionThatReturnsTwice())
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        if (CallInst *CI = dyn_cast<CallInst>(I)) {
          CI->setTailCall();
          MadeChange = true;
        }

  return MadeChange;
}


/// CanMoveAboveCall - Return true if it is safe to move the specified
/// instruction from after the call to before the call, assuming that all
/// instructions between the call and this instruction are movable.
///
bool TailCallElim::CanMoveAboveCall(Instruction *I, CallInst *CI) {
  // FIXME: We can move load/store/call/free instructions above the call if the
  // call does not mod/ref the memory location being processed.
  if (I->mayHaveSideEffects())  // This also handles volatile loads.
    return false;

  if (LoadInst *L = dyn_cast<LoadInst>(I)) {
    // Loads may always be moved above calls without side effects.
    if (CI->mayHaveSideEffects()) {
      // Non-volatile loads may be moved above a call with side effects if it
      // does not write to memory and the load provably won't trap.
      // FIXME: Writes to memory only matter if they may alias the pointer
      // being loaded from.
      if (CI->mayWriteToMemory() ||
          !isSafeToLoadUnconditionally(L->getPointerOperand(), L,
                                       L->getAlignment()))
        return false;
    }
  }

  // Otherwise, if this is a side-effect free instruction, check to make sure
  // that it does not use the return value of the call.  If it doesn't use the
  // return value of the call, it must only use things that are defined before
  // the call, or movable instructions between the call and the instruction
  // itself.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (I->getOperand(i) == CI)
      return false;
  return true;
}

// isDynamicConstant - Return true if the specified value is the same when the
// return would exit as it was when the initial iteration of the recursive
// function was executed.
//
// We currently handle static constants and arguments that are not modified as
// part of the recursion.
//
static bool isDynamicConstant(Value *V, CallInst *CI, ReturnInst *RI) {
  if (isa<Constant>(V)) return true; // Static constants are always dyn consts

  // Check to see if this is an immutable argument, if so, the value
  // will be available to initialize the accumulator.
  if (Argument *Arg = dyn_cast<Argument>(V)) {
    // Figure out which argument number this is...
    unsigned ArgNo = 0;
    Function *F = CI->getParent()->getParent();
    for (Function::arg_iterator AI = F->arg_begin(); &*AI != Arg; ++AI)
      ++ArgNo;

    // If we are passing this argument into call as the corresponding
    // argument operand, then the argument is dynamically constant.
    // Otherwise, we cannot transform this function safely.
    if (CI->getArgOperand(ArgNo) == Arg)
      return true;
  }

  // Switch cases are always constant integers. If the value is being switched
  // on and the return is only reachable from one of its cases, it's
  // effectively constant.
  if (BasicBlock *UniquePred = RI->getParent()->getUniquePredecessor())
    if (SwitchInst *SI = dyn_cast<SwitchInst>(UniquePred->getTerminator()))
      if (SI->getCondition() == V)
        return SI->getDefaultDest() != RI->getParent();

  // Not a constant or immutable argument, we can't safely transform.
  return false;
}

// getCommonReturnValue - Check to see if the function containing the specified
// tail call consistently returns the same runtime-constant value at all exit
// points except for IgnoreRI.  If so, return the returned value.
//
static Value *getCommonReturnValue(ReturnInst *IgnoreRI, CallInst *CI) {
  Function *F = CI->getParent()->getParent();
  Value *ReturnedValue = 0;

  for (Function::iterator BBI = F->begin(), E = F->end(); BBI != E; ++BBI) {
    ReturnInst *RI = dyn_cast<ReturnInst>(BBI->getTerminator());
    if (RI == 0 || RI == IgnoreRI) continue;

    // We can only perform this transformation if the value returned is
    // evaluatable at the start of the initial invocation of the function,
    // instead of at the end of the evaluation.
    //
    Value *RetOp = RI->getOperand(0);
    if (!isDynamicConstant(RetOp, CI, RI))
      return 0;

    if (ReturnedValue && RetOp != ReturnedValue)
      return 0;     // Cannot transform if differing values are returned.
    ReturnedValue = RetOp;
  }
  return ReturnedValue;
}

/// CanTransformAccumulatorRecursion - If the specified instruction can be
/// transformed using accumulator recursion elimination, return the constant
/// which is the start of the accumulator value.  Otherwise return null.
///
Value *TailCallElim::CanTransformAccumulatorRecursion(Instruction *I,
                                                      CallInst *CI) {
  if (!I->isAssociative() || !I->isCommutative()) return 0;
  assert(I->getNumOperands() == 2 &&
         "Associative/commutative operations should have 2 args!");

  // Exactly one operand should be the result of the call instruction.
  if ((I->getOperand(0) == CI && I->getOperand(1) == CI) ||
      (I->getOperand(0) != CI && I->getOperand(1) != CI))
    return 0;

  // The only user of this instruction we allow is a single return instruction.
  if (!I->hasOneUse() || !isa<ReturnInst>(I->use_back()))
    return 0;

  // Ok, now we have to check all of the other return instructions in this
  // function.  If they return non-constants or differing values, then we cannot
  // transform the function safely.
  return getCommonReturnValue(cast<ReturnInst>(I->use_back()), CI);
}

static Instruction *FirstNonDbg(BasicBlock::iterator I) {
  while (isa<DbgInfoIntrinsic>(I))
    ++I;
  return &*I;
}

CallInst*
TailCallElim::FindTRECandidate(Instruction *TI,
                               bool CannotTailCallElimCallsMarkedTail) {
  BasicBlock *BB = TI->getParent();
  Function *F = BB->getParent();

  if (&BB->front() == TI) // Make sure there is something before the terminator.
    return 0;

  // Scan backwards from the return, checking to see if there is a tail call in
  // this block.  If so, set CI to it.
  CallInst *CI = 0;
  BasicBlock::iterator BBI = TI;
  while (true) {
    CI = dyn_cast<CallInst>(BBI);
    if (CI && CI->getCalledFunction() == F)
      break;

    if (BBI == BB->begin())
      return 0;          // Didn't find a potential tail call.
    --BBI;
  }

  // If this call is marked as a tail call, and if there are dynamic allocas in
  // the function, we cannot perform this optimization.
  if (CI->isTailCall() && CannotTailCallElimCallsMarkedTail)
    return 0;

  // As a special case, detect code like this:
  //   double fabs(double f) { return __builtin_fabs(f); } // a 'fabs' call
  // and disable this xform in this case, because the code generator will
  // lower the call to fabs into inline code.
  if (BB == &F->getEntryBlock() &&
      FirstNonDbg(BB->front()) == CI &&
      FirstNonDbg(llvm::next(BB->begin())) == TI &&
      callIsSmall(CI)) {
    // A single-block function with just a call and a return. Check that
    // the arguments match.
    CallSite::arg_iterator I = CallSite(CI).arg_begin(),
                           E = CallSite(CI).arg_end();
    Function::arg_iterator FI = F->arg_begin(),
                           FE = F->arg_end();
    for (; I != E && FI != FE; ++I, ++FI)
      if (*I != &*FI) break;
    if (I == E && FI == FE)
      return 0;
  }

  return CI;
}

bool TailCallElim::EliminateRecursiveTailCall(CallInst *CI, ReturnInst *Ret,
                                       BasicBlock *&OldEntry,
                                       bool &TailCallsAreMarkedTail,
                                       SmallVector<PHINode*, 8> &ArgumentPHIs,
                                       bool CannotTailCallElimCallsMarkedTail) {
  // If we are introducing accumulator recursion to eliminate operations after
  // the call instruction that are both associative and commutative, the initial
  // value for the accumulator is placed in this variable.  If this value is set
  // then we actually perform accumulator recursion elimination instead of
  // simple tail recursion elimination.  If the operation is an LLVM instruction
  // (eg: "add") then it is recorded in AccumulatorRecursionInstr.  If not, then
  // we are handling the case when the return instruction returns a constant C
  // which is different to the constant returned by other return instructions
  // (which is recorded in AccumulatorRecursionEliminationInitVal).  This is a
  // special case of accumulator recursion, the operation being "return C".
  Value *AccumulatorRecursionEliminationInitVal = 0;
  Instruction *AccumulatorRecursionInstr = 0;

  // Ok, we found a potential tail call.  We can currently only transform the
  // tail call if all of the instructions between the call and the return are
  // movable to above the call itself, leaving the call next to the return.
  // Check that this is the case now.
  BasicBlock::iterator BBI = CI;
  for (++BBI; &*BBI != Ret; ++BBI) {
    if (CanMoveAboveCall(BBI, CI)) continue;

    // If we can't move the instruction above the call, it might be because it
    // is an associative and commutative operation that could be transformed
    // using accumulator recursion elimination.  Check to see if this is the
    // case, and if so, remember the initial accumulator value for later.
    if ((AccumulatorRecursionEliminationInitVal =
                           CanTransformAccumulatorRecursion(BBI, CI))) {
      // Yes, this is accumulator recursion.  Remember which instruction
      // accumulates.
      AccumulatorRecursionInstr = BBI;
    } else {
      return false;   // Otherwise, we cannot eliminate the tail recursion!
    }
  }

  // We can only transform call/return pairs that either ignore the return value
  // of the call and return void, ignore the value of the call and return a
  // constant, return the value returned by the tail call, or that are being
  // accumulator recursion variable eliminated.
  if (Ret->getNumOperands() == 1 && Ret->getReturnValue() != CI &&
      !isa<UndefValue>(Ret->getReturnValue()) &&
      AccumulatorRecursionEliminationInitVal == 0 &&
      !getCommonReturnValue(0, CI)) {
    // One case remains that we are able to handle: the current return
    // instruction returns a constant, and all other return instructions
    // return a different constant.
    if (!isDynamicConstant(Ret->getReturnValue(), CI, Ret))
      return false; // Current return instruction does not return a constant.
    // Check that all other return instructions return a common constant.  If
    // so, record it in AccumulatorRecursionEliminationInitVal.
    AccumulatorRecursionEliminationInitVal = getCommonReturnValue(Ret, CI);
    if (!AccumulatorRecursionEliminationInitVal)
      return false;
  }

  BasicBlock *BB = Ret->getParent();
  Function *F = BB->getParent();

  // OK! We can transform this tail call.  If this is the first one found,
  // create the new entry block, allowing us to branch back to the old entry.
  if (OldEntry == 0) {
    OldEntry = &F->getEntryBlock();
    BasicBlock *NewEntry = BasicBlock::Create(F->getContext(), "", F, OldEntry);
    NewEntry->takeName(OldEntry);
    OldEntry->setName("tailrecurse");
    BranchInst::Create(OldEntry, NewEntry);

    // If this tail call is marked 'tail' and if there are any allocas in the
    // entry block, move them up to the new entry block.
    TailCallsAreMarkedTail = CI->isTailCall();
    if (TailCallsAreMarkedTail)
      // Move all fixed sized allocas from OldEntry to NewEntry.
      for (BasicBlock::iterator OEBI = OldEntry->begin(), E = OldEntry->end(),
             NEBI = NewEntry->begin(); OEBI != E; )
        if (AllocaInst *AI = dyn_cast<AllocaInst>(OEBI++))
          if (isa<ConstantInt>(AI->getArraySize()))
            AI->moveBefore(NEBI);

    // Now that we have created a new block, which jumps to the entry
    // block, insert a PHI node for each argument of the function.
    // For now, we initialize each PHI to only have the real arguments
    // which are passed in.
    Instruction *InsertPos = OldEntry->begin();
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I) {
      PHINode *PN = PHINode::Create(I->getType(), 2,
                                    I->getName() + ".tr", InsertPos);
      I->replaceAllUsesWith(PN); // Everyone use the PHI node now!
      PN->addIncoming(I, NewEntry);
      ArgumentPHIs.push_back(PN);
    }
  }

  // If this function has self recursive calls in the tail position where some
  // are marked tail and some are not, only transform one flavor or another.  We
  // have to choose whether we move allocas in the entry block to the new entry
  // block or not, so we can't make a good choice for both.  NOTE: We could do
  // slightly better here in the case that the function has no entry block
  // allocas.
  if (TailCallsAreMarkedTail && !CI->isTailCall())
    return false;

  // Ok, now that we know we have a pseudo-entry block WITH all of the
  // required PHI nodes, add entries into the PHI node for the actual
  // parameters passed into the tail-recursive call.
  for (unsigned i = 0, e = CI->getNumArgOperands(); i != e; ++i)
    ArgumentPHIs[i]->addIncoming(CI->getArgOperand(i), BB);

  // If we are introducing an accumulator variable to eliminate the recursion,
  // do so now.  Note that we _know_ that no subsequent tail recursion
  // eliminations will happen on this function because of the way the
  // accumulator recursion predicate is set up.
  //
  if (AccumulatorRecursionEliminationInitVal) {
    Instruction *AccRecInstr = AccumulatorRecursionInstr;
    // Start by inserting a new PHI node for the accumulator.
    pred_iterator PB = pred_begin(OldEntry), PE = pred_end(OldEntry);
    PHINode *AccPN =
      PHINode::Create(AccumulatorRecursionEliminationInitVal->getType(),
                      std::distance(PB, PE) + 1,
                      "accumulator.tr", OldEntry->begin());

    // Loop over all of the predecessors of the tail recursion block.  For the
    // real entry into the function we seed the PHI with the initial value,
    // computed earlier.  For any other existing branches to this block (due to
    // other tail recursions eliminated) the accumulator is not modified.
    // Because we haven't added the branch in the current block to OldEntry yet,
    // it will not show up as a predecessor.
    for (pred_iterator PI = PB; PI != PE; ++PI) {
      BasicBlock *P = *PI;
      if (P == &F->getEntryBlock())
        AccPN->addIncoming(AccumulatorRecursionEliminationInitVal, P);
      else
        AccPN->addIncoming(AccPN, P);
    }

    if (AccRecInstr) {
      // Add an incoming argument for the current block, which is computed by
      // our associative and commutative accumulator instruction.
      AccPN->addIncoming(AccRecInstr, BB);

      // Next, rewrite the accumulator recursion instruction so that it does not
      // use the result of the call anymore, instead, use the PHI node we just
      // inserted.
      AccRecInstr->setOperand(AccRecInstr->getOperand(0) != CI, AccPN);
    } else {
      // Add an incoming argument for the current block, which is just the
      // constant returned by the current return instruction.
      AccPN->addIncoming(Ret->getReturnValue(), BB);
    }

    // Finally, rewrite any return instructions in the program to return the PHI
    // node instead of the "initval" that they do currently.  This loop will
    // actually rewrite the return value we are destroying, but that's ok.
    for (Function::iterator BBI = F->begin(), E = F->end(); BBI != E; ++BBI)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BBI->getTerminator()))
        RI->setOperand(0, AccPN);
    ++NumAccumAdded;
  }

  // Now that all of the PHI nodes are in place, remove the call and
  // ret instructions, replacing them with an unconditional branch.
  BranchInst *NewBI = BranchInst::Create(OldEntry, Ret);
  NewBI->setDebugLoc(CI->getDebugLoc());

  BB->getInstList().erase(Ret);  // Remove return.
  BB->getInstList().erase(CI);   // Remove call.
  ++NumEliminated;
  return true;
}

bool TailCallElim::FoldReturnAndProcessPred(BasicBlock *BB,
                                       ReturnInst *Ret, BasicBlock *&OldEntry,
                                       bool &TailCallsAreMarkedTail,
                                       SmallVector<PHINode*, 8> &ArgumentPHIs,
                                       bool CannotTailCallElimCallsMarkedTail) {
  bool Change = false;

  // If the return block contains nothing but the return and PHI's,
  // there might be an opportunity to duplicate the return in its
  // predecessors and perform TRC there. Look for predecessors that end
  // in unconditional branch and recursive call(s).
  SmallVector<BranchInst*, 8> UncondBranchPreds;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *Pred = *PI;
    TerminatorInst *PTI = Pred->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(PTI))
      if (BI->isUnconditional())
        UncondBranchPreds.push_back(BI);
  }

  while (!UncondBranchPreds.empty()) {
    BranchInst *BI = UncondBranchPreds.pop_back_val();
    BasicBlock *Pred = BI->getParent();
    if (CallInst *CI = FindTRECandidate(BI, CannotTailCallElimCallsMarkedTail)){
      DEBUG(dbgs() << "FOLDING: " << *BB
            << "INTO UNCOND BRANCH PRED: " << *Pred);
      EliminateRecursiveTailCall(CI, FoldReturnIntoUncondBranch(Ret, BB, Pred),
                                 OldEntry, TailCallsAreMarkedTail, ArgumentPHIs,
                                 CannotTailCallElimCallsMarkedTail);
      ++NumRetDuped;
      Change = true;
    }
  }

  return Change;
}

bool TailCallElim::ProcessReturningBlock(ReturnInst *Ret, BasicBlock *&OldEntry,
                                         bool &TailCallsAreMarkedTail,
                                         SmallVector<PHINode*, 8> &ArgumentPHIs,
                                       bool CannotTailCallElimCallsMarkedTail) {
  CallInst *CI = FindTRECandidate(Ret, CannotTailCallElimCallsMarkedTail);
  if (!CI)
    return false;

  return EliminateRecursiveTailCall(CI, Ret, OldEntry, TailCallsAreMarkedTail,
                                    ArgumentPHIs,
                                    CannotTailCallElimCallsMarkedTail);
}
