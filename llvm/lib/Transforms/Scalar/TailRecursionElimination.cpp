//===- TailRecursionElimination.cpp - Eliminate Tail Calls ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;

#define DEBUG_TYPE "tailcallelim"

STATISTIC(NumEliminated, "Number of tail calls removed");
STATISTIC(NumRetDuped,   "Number of return duplicated");
STATISTIC(NumAccumAdded, "Number of accumulators introduced");

/// Scan the specified function for alloca instructions.
/// If it contains any dynamic allocas, returns false.
static bool canTRE(Function &F) {
  // Because of PR962, we don't TRE dynamic allocas.
  return llvm::all_of(instructions(F), [](Instruction &I) {
    auto *AI = dyn_cast<AllocaInst>(&I);
    return !AI || AI->isStaticAlloca();
  });
}

namespace {
struct AllocaDerivedValueTracker {
  // Start at a root value and walk its use-def chain to mark calls that use the
  // value or a derived value in AllocaUsers, and places where it may escape in
  // EscapePoints.
  void walk(Value *Root) {
    SmallVector<Use *, 32> Worklist;
    SmallPtrSet<Use *, 32> Visited;

    auto AddUsesToWorklist = [&](Value *V) {
      for (auto &U : V->uses()) {
        if (!Visited.insert(&U).second)
          continue;
        Worklist.push_back(&U);
      }
    };

    AddUsesToWorklist(Root);

    while (!Worklist.empty()) {
      Use *U = Worklist.pop_back_val();
      Instruction *I = cast<Instruction>(U->getUser());

      switch (I->getOpcode()) {
      case Instruction::Call:
      case Instruction::Invoke: {
        auto &CB = cast<CallBase>(*I);
        // If the alloca-derived argument is passed byval it is not an escape
        // point, or a use of an alloca. Calling with byval copies the contents
        // of the alloca into argument registers or stack slots, which exist
        // beyond the lifetime of the current frame.
        if (CB.isArgOperand(U) && CB.isByValArgument(CB.getArgOperandNo(U)))
          continue;
        bool IsNocapture =
            CB.isDataOperand(U) && CB.doesNotCapture(CB.getDataOperandNo(U));
        callUsesLocalStack(CB, IsNocapture);
        if (IsNocapture) {
          // If the alloca-derived argument is passed in as nocapture, then it
          // can't propagate to the call's return. That would be capturing.
          continue;
        }
        break;
      }
      case Instruction::Load: {
        // The result of a load is not alloca-derived (unless an alloca has
        // otherwise escaped, but this is a local analysis).
        continue;
      }
      case Instruction::Store: {
        if (U->getOperandNo() == 0)
          EscapePoints.insert(I);
        continue;  // Stores have no users to analyze.
      }
      case Instruction::BitCast:
      case Instruction::GetElementPtr:
      case Instruction::PHI:
      case Instruction::Select:
      case Instruction::AddrSpaceCast:
        break;
      default:
        EscapePoints.insert(I);
        break;
      }

      AddUsesToWorklist(I);
    }
  }

  void callUsesLocalStack(CallBase &CB, bool IsNocapture) {
    // Add it to the list of alloca users.
    AllocaUsers.insert(&CB);

    // If it's nocapture then it can't capture this alloca.
    if (IsNocapture)
      return;

    // If it can write to memory, it can leak the alloca value.
    if (!CB.onlyReadsMemory())
      EscapePoints.insert(&CB);
  }

  SmallPtrSet<Instruction *, 32> AllocaUsers;
  SmallPtrSet<Instruction *, 32> EscapePoints;
};
}

static bool markTails(Function &F, bool &AllCallsAreTailCalls,
                      OptimizationRemarkEmitter *ORE) {
  if (F.callsFunctionThatReturnsTwice())
    return false;
  AllCallsAreTailCalls = true;

  // The local stack holds all alloca instructions and all byval arguments.
  AllocaDerivedValueTracker Tracker;
  for (Argument &Arg : F.args()) {
    if (Arg.hasByValAttr())
      Tracker.walk(&Arg);
  }
  for (auto &BB : F) {
    for (auto &I : BB)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(&I))
        Tracker.walk(AI);
  }

  bool Modified = false;

  // Track whether a block is reachable after an alloca has escaped. Blocks that
  // contain the escaping instruction will be marked as being visited without an
  // escaped alloca, since that is how the block began.
  enum VisitType {
    UNVISITED,
    UNESCAPED,
    ESCAPED
  };
  DenseMap<BasicBlock *, VisitType> Visited;

  // We propagate the fact that an alloca has escaped from block to successor.
  // Visit the blocks that are propagating the escapedness first. To do this, we
  // maintain two worklists.
  SmallVector<BasicBlock *, 32> WorklistUnescaped, WorklistEscaped;

  // We may enter a block and visit it thinking that no alloca has escaped yet,
  // then see an escape point and go back around a loop edge and come back to
  // the same block twice. Because of this, we defer setting tail on calls when
  // we first encounter them in a block. Every entry in this list does not
  // statically use an alloca via use-def chain analysis, but may find an alloca
  // through other means if the block turns out to be reachable after an escape
  // point.
  SmallVector<CallInst *, 32> DeferredTails;

  BasicBlock *BB = &F.getEntryBlock();
  VisitType Escaped = UNESCAPED;
  do {
    for (auto &I : *BB) {
      if (Tracker.EscapePoints.count(&I))
        Escaped = ESCAPED;

      CallInst *CI = dyn_cast<CallInst>(&I);
      if (!CI || CI->isTailCall() || isa<DbgInfoIntrinsic>(&I))
        continue;

      bool IsNoTail = CI->isNoTailCall() || CI->hasOperandBundles();

      if (!IsNoTail && CI->doesNotAccessMemory()) {
        // A call to a readnone function whose arguments are all things computed
        // outside this function can be marked tail. Even if you stored the
        // alloca address into a global, a readnone function can't load the
        // global anyhow.
        //
        // Note that this runs whether we know an alloca has escaped or not. If
        // it has, then we can't trust Tracker.AllocaUsers to be accurate.
        bool SafeToTail = true;
        for (auto &Arg : CI->arg_operands()) {
          if (isa<Constant>(Arg.getUser()))
            continue;
          if (Argument *A = dyn_cast<Argument>(Arg.getUser()))
            if (!A->hasByValAttr())
              continue;
          SafeToTail = false;
          break;
        }
        if (SafeToTail) {
          using namespace ore;
          ORE->emit([&]() {
            return OptimizationRemark(DEBUG_TYPE, "tailcall-readnone", CI)
                   << "marked as tail call candidate (readnone)";
          });
          CI->setTailCall();
          Modified = true;
          continue;
        }
      }

      if (!IsNoTail && Escaped == UNESCAPED && !Tracker.AllocaUsers.count(CI)) {
        DeferredTails.push_back(CI);
      } else {
        AllCallsAreTailCalls = false;
      }
    }

    for (auto *SuccBB : make_range(succ_begin(BB), succ_end(BB))) {
      auto &State = Visited[SuccBB];
      if (State < Escaped) {
        State = Escaped;
        if (State == ESCAPED)
          WorklistEscaped.push_back(SuccBB);
        else
          WorklistUnescaped.push_back(SuccBB);
      }
    }

    if (!WorklistEscaped.empty()) {
      BB = WorklistEscaped.pop_back_val();
      Escaped = ESCAPED;
    } else {
      BB = nullptr;
      while (!WorklistUnescaped.empty()) {
        auto *NextBB = WorklistUnescaped.pop_back_val();
        if (Visited[NextBB] == UNESCAPED) {
          BB = NextBB;
          Escaped = UNESCAPED;
          break;
        }
      }
    }
  } while (BB);

  for (CallInst *CI : DeferredTails) {
    if (Visited[CI->getParent()] != ESCAPED) {
      // If the escape point was part way through the block, calls after the
      // escape point wouldn't have been put into DeferredTails.
      LLVM_DEBUG(dbgs() << "Marked as tail call candidate: " << *CI << "\n");
      CI->setTailCall();
      Modified = true;
    } else {
      AllCallsAreTailCalls = false;
    }
  }

  return Modified;
}

/// Return true if it is safe to move the specified
/// instruction from after the call to before the call, assuming that all
/// instructions between the call and this instruction are movable.
///
static bool canMoveAboveCall(Instruction *I, CallInst *CI, AliasAnalysis *AA) {
  // FIXME: We can move load/store/call/free instructions above the call if the
  // call does not mod/ref the memory location being processed.
  if (I->mayHaveSideEffects())  // This also handles volatile loads.
    return false;

  if (LoadInst *L = dyn_cast<LoadInst>(I)) {
    // Loads may always be moved above calls without side effects.
    if (CI->mayHaveSideEffects()) {
      // Non-volatile loads may be moved above a call with side effects if it
      // does not write to memory and the load provably won't trap.
      // Writes to memory only matter if they may alias the pointer
      // being loaded from.
      const DataLayout &DL = L->getModule()->getDataLayout();
      if (isModSet(AA->getModRefInfo(CI, MemoryLocation::get(L))) ||
          !isSafeToLoadUnconditionally(L->getPointerOperand(), L->getType(),
                                       L->getAlign(), DL, L))
        return false;
    }
  }

  // Otherwise, if this is a side-effect free instruction, check to make sure
  // that it does not use the return value of the call.  If it doesn't use the
  // return value of the call, it must only use things that are defined before
  // the call, or movable instructions between the call and the instruction
  // itself.
  return !is_contained(I->operands(), CI);
}

/// Return true if the specified value is the same when the return would exit
/// as it was when the initial iteration of the recursive function was executed.
///
/// We currently handle static constants and arguments that are not modified as
/// part of the recursion.
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

/// Check to see if the function containing the specified tail call consistently
/// returns the same runtime-constant value at all exit points except for
/// IgnoreRI. If so, return the returned value.
static Value *getCommonReturnValue(ReturnInst *IgnoreRI, CallInst *CI) {
  Function *F = CI->getParent()->getParent();
  Value *ReturnedValue = nullptr;

  for (BasicBlock &BBI : *F) {
    ReturnInst *RI = dyn_cast<ReturnInst>(BBI.getTerminator());
    if (RI == nullptr || RI == IgnoreRI) continue;

    // We can only perform this transformation if the value returned is
    // evaluatable at the start of the initial invocation of the function,
    // instead of at the end of the evaluation.
    //
    Value *RetOp = RI->getOperand(0);
    if (!isDynamicConstant(RetOp, CI, RI))
      return nullptr;

    if (ReturnedValue && RetOp != ReturnedValue)
      return nullptr;     // Cannot transform if differing values are returned.
    ReturnedValue = RetOp;
  }
  return ReturnedValue;
}

/// If the specified instruction can be transformed using accumulator recursion
/// elimination, return the constant which is the start of the accumulator
/// value.  Otherwise return null.
static Value *canTransformAccumulatorRecursion(Instruction *I, CallInst *CI) {
  if (!I->isAssociative() || !I->isCommutative()) return nullptr;
  assert(I->getNumOperands() == 2 &&
         "Associative/commutative operations should have 2 args!");

  // Exactly one operand should be the result of the call instruction.
  if ((I->getOperand(0) == CI && I->getOperand(1) == CI) ||
      (I->getOperand(0) != CI && I->getOperand(1) != CI))
    return nullptr;

  // The only user of this instruction we allow is a single return instruction.
  if (!I->hasOneUse() || !isa<ReturnInst>(I->user_back()))
    return nullptr;

  // Ok, now we have to check all of the other return instructions in this
  // function.  If they return non-constants or differing values, then we cannot
  // transform the function safely.
  return getCommonReturnValue(cast<ReturnInst>(I->user_back()), CI);
}

static Instruction *firstNonDbg(BasicBlock::iterator I) {
  while (isa<DbgInfoIntrinsic>(I))
    ++I;
  return &*I;
}

namespace {
class TailRecursionEliminator {
  Function &F;
  const TargetTransformInfo *TTI;
  AliasAnalysis *AA;
  OptimizationRemarkEmitter *ORE;
  DomTreeUpdater &DTU;

  // The below are shared state we want to have available when eliminating any
  // calls in the function. There values should be populated by
  // createTailRecurseLoopHeader the first time we find a call we can eliminate.
  BasicBlock *HeaderBB = nullptr;
  SmallVector<PHINode *, 8> ArgumentPHIs;
  bool RemovableCallsMustBeMarkedTail = false;

  // PHI node to store our return value.
  PHINode *RetPN = nullptr;

  // i1 PHI node to track if we have a valid return value stored in RetPN.
  PHINode *RetKnownPN = nullptr;

  // Vector of select instructions we insereted. These selects use RetKnownPN
  // to either propagate RetPN or select a new return value.
  SmallVector<SelectInst *, 8> RetSelects;

  TailRecursionEliminator(Function &F, const TargetTransformInfo *TTI,
                          AliasAnalysis *AA, OptimizationRemarkEmitter *ORE,
                          DomTreeUpdater &DTU)
      : F(F), TTI(TTI), AA(AA), ORE(ORE), DTU(DTU) {}

  CallInst *findTRECandidate(Instruction *TI,
                             bool CannotTailCallElimCallsMarkedTail);

  void createTailRecurseLoopHeader(CallInst *CI);

  PHINode *insertAccumulator(Value *AccumulatorRecursionEliminationInitVal);

  bool eliminateCall(CallInst *CI);

  bool foldReturnAndProcessPred(ReturnInst *Ret,
                                bool CannotTailCallElimCallsMarkedTail);

  bool processReturningBlock(ReturnInst *Ret,
                             bool CannotTailCallElimCallsMarkedTail);

  void cleanupAndFinalize();

public:
  static bool eliminate(Function &F, const TargetTransformInfo *TTI,
                        AliasAnalysis *AA, OptimizationRemarkEmitter *ORE,
                        DomTreeUpdater &DTU);
};
} // namespace

CallInst *TailRecursionEliminator::findTRECandidate(
    Instruction *TI, bool CannotTailCallElimCallsMarkedTail) {
  BasicBlock *BB = TI->getParent();

  if (&BB->front() == TI) // Make sure there is something before the terminator.
    return nullptr;

  // Scan backwards from the return, checking to see if there is a tail call in
  // this block.  If so, set CI to it.
  CallInst *CI = nullptr;
  BasicBlock::iterator BBI(TI);
  while (true) {
    CI = dyn_cast<CallInst>(BBI);
    if (CI && CI->getCalledFunction() == &F)
      break;

    if (BBI == BB->begin())
      return nullptr;          // Didn't find a potential tail call.
    --BBI;
  }

  // If this call is marked as a tail call, and if there are dynamic allocas in
  // the function, we cannot perform this optimization.
  if (CI->isTailCall() && CannotTailCallElimCallsMarkedTail)
    return nullptr;

  // As a special case, detect code like this:
  //   double fabs(double f) { return __builtin_fabs(f); } // a 'fabs' call
  // and disable this xform in this case, because the code generator will
  // lower the call to fabs into inline code.
  if (BB == &F.getEntryBlock() &&
      firstNonDbg(BB->front().getIterator()) == CI &&
      firstNonDbg(std::next(BB->begin())) == TI && CI->getCalledFunction() &&
      !TTI->isLoweredToCall(CI->getCalledFunction())) {
    // A single-block function with just a call and a return. Check that
    // the arguments match.
    auto I = CI->arg_begin(), E = CI->arg_end();
    Function::arg_iterator FI = F.arg_begin(), FE = F.arg_end();
    for (; I != E && FI != FE; ++I, ++FI)
      if (*I != &*FI) break;
    if (I == E && FI == FE)
      return nullptr;
  }

  return CI;
}

void TailRecursionEliminator::createTailRecurseLoopHeader(CallInst *CI) {
  HeaderBB = &F.getEntryBlock();
  BasicBlock *NewEntry = BasicBlock::Create(F.getContext(), "", &F, HeaderBB);
  NewEntry->takeName(HeaderBB);
  HeaderBB->setName("tailrecurse");
  BranchInst *BI = BranchInst::Create(HeaderBB, NewEntry);
  BI->setDebugLoc(CI->getDebugLoc());

  // If this function has self recursive calls in the tail position where some
  // are marked tail and some are not, only transform one flavor or another.
  // We have to choose whether we move allocas in the entry block to the new
  // entry block or not, so we can't make a good choice for both. We make this
  // decision here based on whether the first call we found to remove is
  // marked tail.
  // NOTE: We could do slightly better here in the case that the function has
  // no entry block allocas.
  RemovableCallsMustBeMarkedTail = CI->isTailCall();

  // If this tail call is marked 'tail' and if there are any allocas in the
  // entry block, move them up to the new entry block.
  if (RemovableCallsMustBeMarkedTail)
    // Move all fixed sized allocas from HeaderBB to NewEntry.
    for (BasicBlock::iterator OEBI = HeaderBB->begin(), E = HeaderBB->end(),
                              NEBI = NewEntry->begin();
         OEBI != E;)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(OEBI++))
        if (isa<ConstantInt>(AI->getArraySize()))
          AI->moveBefore(&*NEBI);

  // Now that we have created a new block, which jumps to the entry
  // block, insert a PHI node for each argument of the function.
  // For now, we initialize each PHI to only have the real arguments
  // which are passed in.
  Instruction *InsertPos = &HeaderBB->front();
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    PHINode *PN =
        PHINode::Create(I->getType(), 2, I->getName() + ".tr", InsertPos);
    I->replaceAllUsesWith(PN); // Everyone use the PHI node now!
    PN->addIncoming(&*I, NewEntry);
    ArgumentPHIs.push_back(PN);
  }

  // If the function doen't return void, create the RetPN and RetKnownPN PHI
  // nodes to track our return value. We initialize RetPN with undef and
  // RetKnownPN with false since we can't know our return value at function
  // entry.
  Type *RetType = F.getReturnType();
  if (!RetType->isVoidTy()) {
    Type *BoolType = Type::getInt1Ty(F.getContext());
    RetPN = PHINode::Create(RetType, 2, "ret.tr", InsertPos);
    RetKnownPN = PHINode::Create(BoolType, 2, "ret.known.tr", InsertPos);

    RetPN->addIncoming(UndefValue::get(RetType), NewEntry);
    RetKnownPN->addIncoming(ConstantInt::getFalse(BoolType), NewEntry);
  }

  // The entry block was changed from HeaderBB to NewEntry.
  // The forward DominatorTree needs to be recalculated when the EntryBB is
  // changed. In this corner-case we recalculate the entire tree.
  DTU.recalculate(*NewEntry->getParent());
}

PHINode *TailRecursionEliminator::insertAccumulator(
    Value *AccumulatorRecursionEliminationInitVal) {
  // Start by inserting a new PHI node for the accumulator.
  pred_iterator PB = pred_begin(HeaderBB), PE = pred_end(HeaderBB);
  PHINode *AccPN = PHINode::Create(
      AccumulatorRecursionEliminationInitVal->getType(),
      std::distance(PB, PE) + 1, "accumulator.tr", &HeaderBB->front());

  // Loop over all of the predecessors of the tail recursion block.  For the
  // real entry into the function we seed the PHI with the initial value,
  // computed earlier.  For any other existing branches to this block (due to
  // other tail recursions eliminated) the accumulator is not modified.
  // Because we haven't added the branch in the current block to HeaderBB yet,
  // it will not show up as a predecessor.
  for (pred_iterator PI = PB; PI != PE; ++PI) {
    BasicBlock *P = *PI;
    if (P == &F.getEntryBlock())
      AccPN->addIncoming(AccumulatorRecursionEliminationInitVal, P);
    else
      AccPN->addIncoming(AccPN, P);
  }

  return AccPN;
}

bool TailRecursionEliminator::eliminateCall(CallInst *CI) {
  ReturnInst *Ret = cast<ReturnInst>(CI->getParent()->getTerminator());

  // If we are introducing accumulator recursion to eliminate operations after
  // the call instruction that are both associative and commutative, the initial
  // value for the accumulator is placed in this variable.  If this value is set
  // then we actually perform accumulator recursion elimination instead of
  // simple tail recursion elimination.  If the operation is an LLVM instruction
  // (eg: "add") then it is recorded in AccumulatorRecursionInstr.
  Value *AccumulatorRecursionEliminationInitVal = nullptr;
  Instruction *AccumulatorRecursionInstr = nullptr;

  // Ok, we found a potential tail call.  We can currently only transform the
  // tail call if all of the instructions between the call and the return are
  // movable to above the call itself, leaving the call next to the return.
  // Check that this is the case now.
  BasicBlock::iterator BBI(CI);
  for (++BBI; &*BBI != Ret; ++BBI) {
    if (canMoveAboveCall(&*BBI, CI, AA))
      continue;

    // If we can't move the instruction above the call, it might be because it
    // is an associative and commutative operation that could be transformed
    // using accumulator recursion elimination.  Check to see if this is the
    // case, and if so, remember the initial accumulator value for later.
    if ((AccumulatorRecursionEliminationInitVal =
             canTransformAccumulatorRecursion(&*BBI, CI))) {
      // Yes, this is accumulator recursion.  Remember which instruction
      // accumulates.
      AccumulatorRecursionInstr = &*BBI;
    } else {
      return false;   // Otherwise, we cannot eliminate the tail recursion!
    }
  }

  BasicBlock *BB = Ret->getParent();

  using namespace ore;
  ORE->emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "tailcall-recursion", CI)
           << "transforming tail recursion into loop";
  });

  // OK! We can transform this tail call.  If this is the first one found,
  // create the new entry block, allowing us to branch back to the old entry.
  if (!HeaderBB)
    createTailRecurseLoopHeader(CI);

  if (RemovableCallsMustBeMarkedTail && !CI->isTailCall())
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
    PHINode *AccPN = insertAccumulator(AccumulatorRecursionEliminationInitVal);

    Instruction *AccRecInstr = AccumulatorRecursionInstr;

    // Add an incoming argument for the current block, which is computed by
    // our associative and commutative accumulator instruction.
    AccPN->addIncoming(AccRecInstr, BB);

    // Next, rewrite the accumulator recursion instruction so that it does not
    // use the result of the call anymore, instead, use the PHI node we just
    // inserted.
    AccRecInstr->setOperand(AccRecInstr->getOperand(0) != CI, AccPN);

    // Finally, rewrite any return instructions in the program to return the PHI
    // node instead of the "initval" that they do currently.  This loop will
    // actually rewrite the return value we are destroying, but that's ok.
    for (BasicBlock &BBI : F)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BBI.getTerminator()))
        RI->setOperand(0, AccPN);
    ++NumAccumAdded;
  }

  // Update our return value tracking
  if (RetPN) {
    if (Ret->getReturnValue() == CI || AccumulatorRecursionEliminationInitVal) {
      // Defer selecting a return value
      RetPN->addIncoming(RetPN, BB);
      RetKnownPN->addIncoming(RetKnownPN, BB);
    } else {
      // We found a return value we want to use, insert a select instruction to
      // select it if we don't already know what our return value will be and
      // store the result in our return value PHI node.
      SelectInst *SI = SelectInst::Create(
          RetKnownPN, RetPN, Ret->getReturnValue(), "current.ret.tr", Ret);
      RetSelects.push_back(SI);

      RetPN->addIncoming(SI, BB);
      RetKnownPN->addIncoming(ConstantInt::getTrue(RetKnownPN->getType()), BB);
    }
  }

  // Now that all of the PHI nodes are in place, remove the call and
  // ret instructions, replacing them with an unconditional branch.
  BranchInst *NewBI = BranchInst::Create(HeaderBB, Ret);
  NewBI->setDebugLoc(CI->getDebugLoc());

  BB->getInstList().erase(Ret);  // Remove return.
  BB->getInstList().erase(CI);   // Remove call.
  DTU.applyUpdates({{DominatorTree::Insert, BB, HeaderBB}});
  ++NumEliminated;
  return true;
}

bool TailRecursionEliminator::foldReturnAndProcessPred(
    ReturnInst *Ret, bool CannotTailCallElimCallsMarkedTail) {
  BasicBlock *BB = Ret->getParent();

  bool Change = false;

  // Make sure this block is a trivial return block.
  assert(BB->getFirstNonPHIOrDbg() == Ret &&
         "Trying to fold non-trivial return block");

  // If the return block contains nothing but the return and PHI's,
  // there might be an opportunity to duplicate the return in its
  // predecessors and perform TRE there. Look for predecessors that end
  // in unconditional branch and recursive call(s).
  SmallVector<BranchInst*, 8> UncondBranchPreds;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *Pred = *PI;
    Instruction *PTI = Pred->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(PTI))
      if (BI->isUnconditional())
        UncondBranchPreds.push_back(BI);
  }

  while (!UncondBranchPreds.empty()) {
    BranchInst *BI = UncondBranchPreds.pop_back_val();
    BasicBlock *Pred = BI->getParent();
    if (CallInst *CI =
            findTRECandidate(BI, CannotTailCallElimCallsMarkedTail)) {
      LLVM_DEBUG(dbgs() << "FOLDING: " << *BB
                        << "INTO UNCOND BRANCH PRED: " << *Pred);
      FoldReturnIntoUncondBranch(Ret, BB, Pred, &DTU);

      // Cleanup: if all predecessors of BB have been eliminated by
      // FoldReturnIntoUncondBranch, delete it.  It is important to empty it,
      // because the ret instruction in there is still using a value which
      // eliminateRecursiveTailCall will attempt to remove.
      if (!BB->hasAddressTaken() && pred_begin(BB) == pred_end(BB))
        DTU.deleteBB(BB);

      eliminateCall(CI);
      ++NumRetDuped;
      Change = true;
    }
  }

  return Change;
}

bool TailRecursionEliminator::processReturningBlock(
    ReturnInst *Ret, bool CannotTailCallElimCallsMarkedTail) {
  CallInst *CI = findTRECandidate(Ret, CannotTailCallElimCallsMarkedTail);
  if (!CI)
    return false;

  return eliminateCall(CI);
}

void TailRecursionEliminator::cleanupAndFinalize() {
  // If we eliminated any tail recursions, it's possible that we inserted some
  // silly PHI nodes which just merge an initial value (the incoming operand)
  // with themselves.  Check to see if we did and clean up our mess if so.  This
  // occurs when a function passes an argument straight through to its tail
  // call.
  for (PHINode *PN : ArgumentPHIs) {
    // If the PHI Node is a dynamic constant, replace it with the value it is.
    if (Value *PNV = SimplifyInstruction(PN, F.getParent()->getDataLayout())) {
      PN->replaceAllUsesWith(PNV);
      PN->eraseFromParent();
    }
  }

  if (RetPN) {
    if (RetSelects.empty()) {
      // If we didn't insert any select instructions, then we know we didn't
      // store a return value and we can remove the PHI nodes we inserted.
      RetPN->dropAllReferences();
      RetPN->eraseFromParent();

      RetKnownPN->dropAllReferences();
      RetKnownPN->eraseFromParent();
    } else {
      // We need to insert a select instruction before any return left in the
      // function to select our stored return value if we have one.
      for (BasicBlock &BB : F) {
        ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator());
        if (!RI)
          continue;

        SelectInst *SI = SelectInst::Create(
            RetKnownPN, RetPN, RI->getOperand(0), "current.ret.tr", RI);
        RI->setOperand(0, SI);
      }
    }
  }
}

bool TailRecursionEliminator::eliminate(Function &F,
                                        const TargetTransformInfo *TTI,
                                        AliasAnalysis *AA,
                                        OptimizationRemarkEmitter *ORE,
                                        DomTreeUpdater &DTU) {
  if (F.getFnAttribute("disable-tail-calls").getValueAsString() == "true")
    return false;

  bool MadeChange = false;
  bool AllCallsAreTailCalls = false;
  MadeChange |= markTails(F, AllCallsAreTailCalls, ORE);
  if (!AllCallsAreTailCalls)
    return MadeChange;

  // If this function is a varargs function, we won't be able to PHI the args
  // right, so don't even try to convert it...
  if (F.getFunctionType()->isVarArg())
    return false;

  // If false, we cannot perform TRE on tail calls marked with the 'tail'
  // attribute, because doing so would cause the stack size to increase (real
  // TRE would deallocate variable sized allocas, TRE doesn't).
  bool CanTRETailMarkedCall = canTRE(F);

  TailRecursionEliminator TRE(F, TTI, AA, ORE, DTU);

  // Change any tail recursive calls to loops.
  //
  // FIXME: The code generator produces really bad code when an 'escaping
  // alloca' is changed from being a static alloca to being a dynamic alloca.
  // Until this is resolved, disable this transformation if that would ever
  // happen.  This bug is PR962.
  for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E; /*in loop*/) {
    BasicBlock *BB = &*BBI++; // foldReturnAndProcessPred may delete BB.
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
      bool Change = TRE.processReturningBlock(Ret, !CanTRETailMarkedCall);
      if (!Change && BB->getFirstNonPHIOrDbg() == Ret)
        Change = TRE.foldReturnAndProcessPred(Ret, !CanTRETailMarkedCall);
      MadeChange |= Change;
    }
  }

  TRE.cleanupAndFinalize();

  return MadeChange;
}

namespace {
struct TailCallElim : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  TailCallElim() : FunctionPass(ID) {
    initializeTailCallElimPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<PostDominatorTreeWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    auto *DT = DTWP ? &DTWP->getDomTree() : nullptr;
    auto *PDTWP = getAnalysisIfAvailable<PostDominatorTreeWrapperPass>();
    auto *PDT = PDTWP ? &PDTWP->getPostDomTree() : nullptr;
    // There is no noticable performance difference here between Lazy and Eager
    // UpdateStrategy based on some test results. It is feasible to switch the
    // UpdateStrategy to Lazy if we find it profitable later.
    DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);

    return TailRecursionEliminator::eliminate(
        F, &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F),
        &getAnalysis<AAResultsWrapperPass>().getAAResults(),
        &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE(), DTU);
  }
};
}

char TailCallElim::ID = 0;
INITIALIZE_PASS_BEGIN(TailCallElim, "tailcallelim", "Tail Call Elimination",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(TailCallElim, "tailcallelim", "Tail Call Elimination",
                    false, false)

// Public interface to the TailCallElimination pass
FunctionPass *llvm::createTailCallEliminationPass() {
  return new TailCallElim();
}

PreservedAnalyses TailCallElimPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {

  TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  auto *DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  auto *PDT = AM.getCachedResult<PostDominatorTreeAnalysis>(F);
  // There is no noticable performance difference here between Lazy and Eager
  // UpdateStrategy based on some test results. It is feasible to switch the
  // UpdateStrategy to Lazy if we find it profitable later.
  DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);
  bool Changed = TailRecursionEliminator::eliminate(F, &TTI, &AA, &ORE, DTU);

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  return PA;
}
