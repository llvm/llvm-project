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
//     recursive by an associative expression to use an accumulator variable,
//     thus compiling the typical naive factorial or 'fib' implementation into
//     efficient code.
//  3. TRE is performed if the function returns void, if the return
//     returns the result returned by the call, or if the function returns a
//     run-time constant on all exits from the function.  It is possible, though
//     unlikely, that the return returns something else (like constant 0), and
//     can still be TRE'd.  It can be TRE'd if ALL OTHER return instructions in
//     the function return the exact same value.
//  4. If it can prove that callees do not access theier caller stack frame,
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
//  2. Tail recursion is only performed if the call immediately preceeds the
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
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumEliminated, "Number of tail calls removed");
STATISTIC(NumAccumAdded, "Number of accumulators introduced");

namespace {
  struct VISIBILITY_HIDDEN TailCallElim : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    TailCallElim() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F);

  private:
    bool ProcessReturningBlock(ReturnInst *RI, BasicBlock *&OldEntry,
                               bool &TailCallsAreMarkedTail,
                               std::vector<PHINode*> &ArgumentPHIs,
                               bool CannotTailCallElimCallsMarkedTail);
    bool CanMoveAboveCall(Instruction *I, CallInst *CI);
    Value *CanTransformAccumulatorRecursion(Instruction *I, CallInst *CI);
  };
}

char TailCallElim::ID = 0;
static RegisterPass<TailCallElim> X("tailcallelim", "Tail Call Elimination");

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

/// FunctionContainsAllocas - Scan the specified basic block for alloca
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
  std::vector<PHINode*> ArgumentPHIs;
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
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator()))
      MadeChange |= ProcessReturningBlock(Ret, OldEntry, TailCallsAreMarkedTail,
                                          ArgumentPHIs,CannotTCETailMarkedCall);

  // If we eliminated any tail recursions, it's possible that we inserted some
  // silly PHI nodes which just merge an initial value (the incoming operand)
  // with themselves.  Check to see if we did and clean up our mess if so.  This
  // occurs when a function passes an argument straight through to its tail
  // call.
  if (!ArgumentPHIs.empty()) {
    for (unsigned i = 0, e = ArgumentPHIs.size(); i != e; ++i) {
      PHINode *PN = ArgumentPHIs[i];

      // If the PHI Node is a dynamic constant, replace it with the value it is.
      if (Value *PNV = PN->hasConstantValue()) {
        PN->replaceAllUsesWith(PNV);
        PN->eraseFromParent();
      }
    }
  }

  // Finally, if this function contains no non-escaping allocas, mark all calls
  // in the function as eligible for tail calls (there is no stack memory for
  // them to access).
  if (!FunctionContainsEscapingAllocas)
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
  if (I->mayWriteToMemory() || isa<LoadInst>(I))
    return false;

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
static bool isDynamicConstant(Value *V, CallInst *CI) {
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
    if (CI->getOperand(ArgNo+1) == Arg)
      return true;
  }
  // Not a constant or immutable argument, we can't safely transform.
  return false;
}

// getCommonReturnValue - Check to see if the function containing the specified
// return instruction and tail call consistently returns the same
// runtime-constant value at all exit points.  If so, return the returned value.
//
static Value *getCommonReturnValue(ReturnInst *TheRI, CallInst *CI) {
  Function *F = TheRI->getParent()->getParent();
  Value *ReturnedValue = 0;

  // TODO: Handle multiple value ret instructions;
  if (isa<StructType>(F->getReturnType()))
      return 0;

  for (Function::iterator BBI = F->begin(), E = F->end(); BBI != E; ++BBI)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BBI->getTerminator()))
      if (RI != TheRI) {
        Value *RetOp = RI->getOperand(0);

        // We can only perform this transformation if the value returned is
        // evaluatable at the start of the initial invocation of the function,
        // instead of at the end of the evaluation.
        //
        if (!isDynamicConstant(RetOp, CI))
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
  if (!I->isAssociative()) return 0;
  assert(I->getNumOperands() == 2 &&
         "Associative operations should have 2 args!");

  // Exactly one operand should be the result of the call instruction...
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

bool TailCallElim::ProcessReturningBlock(ReturnInst *Ret, BasicBlock *&OldEntry,
                                         bool &TailCallsAreMarkedTail,
                                         std::vector<PHINode*> &ArgumentPHIs,
                                       bool CannotTailCallElimCallsMarkedTail) {
  BasicBlock *BB = Ret->getParent();
  Function *F = BB->getParent();

  if (&BB->front() == Ret) // Make sure there is something before the ret...
    return false;
  
  // If the return is in the entry block, then making this transformation would
  // turn infinite recursion into an infinite loop.  This transformation is ok
  // in theory, but breaks some code like:
  //   double fabs(double f) { return __builtin_fabs(f); } // a 'fabs' call
  // disable this xform in this case, because the code generator will lower the
  // call to fabs into inline code.
  if (BB == &F->getEntryBlock())
    return false;

  // Scan backwards from the return, checking to see if there is a tail call in
  // this block.  If so, set CI to it.
  CallInst *CI;
  BasicBlock::iterator BBI = Ret;
  while (1) {
    CI = dyn_cast<CallInst>(BBI);
    if (CI && CI->getCalledFunction() == F)
      break;

    if (BBI == BB->begin())
      return false;          // Didn't find a potential tail call.
    --BBI;
  }

  // If this call is marked as a tail call, and if there are dynamic allocas in
  // the function, we cannot perform this optimization.
  if (CI->isTailCall() && CannotTailCallElimCallsMarkedTail)
    return false;

  // If we are introducing accumulator recursion to eliminate associative
  // operations after the call instruction, this variable contains the initial
  // value for the accumulator.  If this value is set, we actually perform
  // accumulator recursion elimination instead of simple tail recursion
  // elimination.
  Value *AccumulatorRecursionEliminationInitVal = 0;
  Instruction *AccumulatorRecursionInstr = 0;

  // Ok, we found a potential tail call.  We can currently only transform the
  // tail call if all of the instructions between the call and the return are
  // movable to above the call itself, leaving the call next to the return.
  // Check that this is the case now.
  for (BBI = CI, ++BBI; &*BBI != Ret; ++BBI)
    if (!CanMoveAboveCall(BBI, CI)) {
      // If we can't move the instruction above the call, it might be because it
      // is an associative operation that could be tranformed using accumulator
      // recursion elimination.  Check to see if this is the case, and if so,
      // remember the initial accumulator value for later.
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
      !getCommonReturnValue(Ret, CI))
    return false;

  // OK! We can transform this tail call.  If this is the first one found,
  // create the new entry block, allowing us to branch back to the old entry.
  if (OldEntry == 0) {
    OldEntry = &F->getEntryBlock();
    BasicBlock *NewEntry = BasicBlock::Create("", F, OldEntry);
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
      PHINode *PN = PHINode::Create(I->getType(),
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
  for (unsigned i = 0, e = CI->getNumOperands()-1; i != e; ++i)
    ArgumentPHIs[i]->addIncoming(CI->getOperand(i+1), BB);

  // If we are introducing an accumulator variable to eliminate the recursion,
  // do so now.  Note that we _know_ that no subsequent tail recursion
  // eliminations will happen on this function because of the way the
  // accumulator recursion predicate is set up.
  //
  if (AccumulatorRecursionEliminationInitVal) {
    Instruction *AccRecInstr = AccumulatorRecursionInstr;
    // Start by inserting a new PHI node for the accumulator.
    PHINode *AccPN = PHINode::Create(AccRecInstr->getType(), "accumulator.tr",
                                     OldEntry->begin());

    // Loop over all of the predecessors of the tail recursion block.  For the
    // real entry into the function we seed the PHI with the initial value,
    // computed earlier.  For any other existing branches to this block (due to
    // other tail recursions eliminated) the accumulator is not modified.
    // Because we haven't added the branch in the current block to OldEntry yet,
    // it will not show up as a predecessor.
    for (pred_iterator PI = pred_begin(OldEntry), PE = pred_end(OldEntry);
         PI != PE; ++PI) {
      if (*PI == &F->getEntryBlock())
        AccPN->addIncoming(AccumulatorRecursionEliminationInitVal, *PI);
      else
        AccPN->addIncoming(AccPN, *PI);
    }

    // Add an incoming argument for the current block, which is computed by our
    // associative accumulator instruction.
    AccPN->addIncoming(AccRecInstr, BB);

    // Next, rewrite the accumulator recursion instruction so that it does not
    // use the result of the call anymore, instead, use the PHI node we just
    // inserted.
    AccRecInstr->setOperand(AccRecInstr->getOperand(0) != CI, AccPN);

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
  BranchInst::Create(OldEntry, Ret);
  BB->getInstList().erase(Ret);  // Remove return.
  BB->getInstList().erase(CI);   // Remove call.
  ++NumEliminated;
  return true;
}
