//===- LowerInvoke.cpp - Eliminate Invoke & Unwind instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation is designed for use by code generators which do not yet
// support stack unwinding.  This pass supports two models of exception handling
// lowering, the 'cheap' support and the 'expensive' support.
//
// 'Cheap' exception handling support gives the program the ability to execute
// any program which does not "throw an exception", by turning 'invoke'
// instructions into calls and by turning 'unwind' instructions into calls to
// abort().  If the program does dynamically use the unwind instruction, the
// program will print a message then abort.
//
// 'Expensive' exception handling support gives the full exception handling
// support to the program at the cost of making the 'invoke' instruction
// really expensive.  It basically inserts setjmp/longjmp calls to emulate the
// exception handling as necessary.
//
// Because the 'expensive' support slows down programs a lot, and EH is only
// used for a subset of the programs, it must be specifically enabled by an
// option.
//
// Note that after this pass runs the CFG is not entirely accurate (exceptional
// control flow edges are not correct anymore) so only very simple things should
// be done after the lowerinvoke pass has run (like generation of native code).
// This should not be used as a general purpose "my LLVM-to-LLVM pass doesn't
// support the invoke instruction yet" lowering pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lowerinvoke"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetLowering.h"
#include <csetjmp>
#include <set>
using namespace llvm;

STATISTIC(NumInvokes, "Number of invokes replaced");
STATISTIC(NumSpilled, "Number of registers live across unwind edges");

static cl::opt<bool> ExpensiveEHSupport("enable-correct-eh-support",
 cl::desc("Make the -lowerinvoke pass insert expensive, but correct, EH code"));

namespace {
  class LowerInvoke : public FunctionPass {
    // Used for both models.
    Constant *AbortFn;

    // Used for expensive EH support.
    StructType *JBLinkTy;
    GlobalVariable *JBListHead;
    Constant *SetJmpFn, *LongJmpFn, *StackSaveFn, *StackRestoreFn;
    bool useExpensiveEHSupport;

    // We peek in TLI to grab the target's jmp_buf size and alignment
    const TargetLowering *TLI;

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit LowerInvoke(const TargetLowering *tli = NULL,
                         bool useExpensiveEHSupport = ExpensiveEHSupport)
      : FunctionPass(ID), useExpensiveEHSupport(useExpensiveEHSupport),
        TLI(tli) {
      initializeLowerInvokePass(*PassRegistry::getPassRegistry());
    }
    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // This is a cluster of orthogonal Transforms
      AU.addPreserved("mem2reg");
      AU.addPreservedID(LowerSwitchID);
    }

  private:
    bool insertCheapEHSupport(Function &F);
    void splitLiveRangesLiveAcrossInvokes(SmallVectorImpl<InvokeInst*>&Invokes);
    void rewriteExpensiveInvoke(InvokeInst *II, unsigned InvokeNo,
                                AllocaInst *InvokeNum, AllocaInst *StackPtr,
                                SwitchInst *CatchSwitch);
    bool insertExpensiveEHSupport(Function &F);
  };
}

char LowerInvoke::ID = 0;
INITIALIZE_PASS(LowerInvoke, "lowerinvoke",
                "Lower invoke and unwind, for unwindless code generators",
                false, false)

char &llvm::LowerInvokePassID = LowerInvoke::ID;

// Public Interface To the LowerInvoke pass.
FunctionPass *llvm::createLowerInvokePass(const TargetLowering *TLI) {
  return new LowerInvoke(TLI, ExpensiveEHSupport);
}
FunctionPass *llvm::createLowerInvokePass(const TargetLowering *TLI,
                                          bool useExpensiveEHSupport) {
  return new LowerInvoke(TLI, useExpensiveEHSupport);
}

// doInitialization - Make sure that there is a prototype for abort in the
// current module.
bool LowerInvoke::doInitialization(Module &M) {
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  if (useExpensiveEHSupport) {
    // Insert a type for the linked list of jump buffers.
    unsigned JBSize = TLI ? TLI->getJumpBufSize() : 0;
    JBSize = JBSize ? JBSize : 200;
    Type *JmpBufTy = ArrayType::get(VoidPtrTy, JBSize);

    JBLinkTy = StructType::create(M.getContext(), "llvm.sjljeh.jmpbufty");
    Type *Elts[] = { JmpBufTy, PointerType::getUnqual(JBLinkTy) };
    JBLinkTy->setBody(Elts);

    Type *PtrJBList = PointerType::getUnqual(JBLinkTy);

    // Now that we've done that, insert the jmpbuf list head global, unless it
    // already exists.
    if (!(JBListHead = M.getGlobalVariable("llvm.sjljeh.jblist", PtrJBList))) {
      JBListHead = new GlobalVariable(M, PtrJBList, false,
                                      GlobalValue::LinkOnceAnyLinkage,
                                      Constant::getNullValue(PtrJBList),
                                      "llvm.sjljeh.jblist");
    }

// VisualStudio defines setjmp as _setjmp
#if defined(_MSC_VER) && defined(setjmp) && \
                         !defined(setjmp_undefined_for_msvc)
#  pragma push_macro("setjmp")
#  undef setjmp
#  define setjmp_undefined_for_msvc
#endif

    SetJmpFn = Intrinsic::getDeclaration(&M, Intrinsic::setjmp);

#if defined(_MSC_VER) && defined(setjmp_undefined_for_msvc)
   // let's return it to _setjmp state
#  pragma pop_macro("setjmp")
#  undef setjmp_undefined_for_msvc
#endif

    LongJmpFn = Intrinsic::getDeclaration(&M, Intrinsic::longjmp);
    StackSaveFn = Intrinsic::getDeclaration(&M, Intrinsic::stacksave);
    StackRestoreFn = Intrinsic::getDeclaration(&M, Intrinsic::stackrestore);
  }

  // We need the 'write' and 'abort' functions for both models.
  AbortFn = M.getOrInsertFunction("abort", Type::getVoidTy(M.getContext()),
                                  (Type *)0);
  return true;
}

bool LowerInvoke::insertCheapEHSupport(Function &F) {
  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      SmallVector<Value*,16> CallArgs(II->op_begin(), II->op_end() - 3);
      // Insert a normal call instruction...
      CallInst *NewCall = CallInst::Create(II->getCalledValue(),
                                           CallArgs, "", II);
      NewCall->takeName(II);
      NewCall->setCallingConv(II->getCallingConv());
      NewCall->setAttributes(II->getAttributes());
      NewCall->setDebugLoc(II->getDebugLoc());
      II->replaceAllUsesWith(NewCall);

      // Insert an unconditional branch to the normal destination.
      BranchInst::Create(II->getNormalDest(), II);

      // Remove any PHI node entries from the exception destination.
      II->getUnwindDest()->removePredecessor(BB);

      // Remove the invoke instruction now.
      BB->getInstList().erase(II);

      ++NumInvokes; Changed = true;
    }
  return Changed;
}

/// rewriteExpensiveInvoke - Insert code and hack the function to replace the
/// specified invoke instruction with a call.
void LowerInvoke::rewriteExpensiveInvoke(InvokeInst *II, unsigned InvokeNo,
                                         AllocaInst *InvokeNum,
                                         AllocaInst *StackPtr,
                                         SwitchInst *CatchSwitch) {
  ConstantInt *InvokeNoC = ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                            InvokeNo);

  // If the unwind edge has phi nodes, split the edge.
  if (isa<PHINode>(II->getUnwindDest()->begin())) {
    SplitCriticalEdge(II, 1, this);

    // If there are any phi nodes left, they must have a single predecessor.
    while (PHINode *PN = dyn_cast<PHINode>(II->getUnwindDest()->begin())) {
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
      PN->eraseFromParent();
    }
  }

  // Insert a store of the invoke num before the invoke and store zero into the
  // location afterward.
  new StoreInst(InvokeNoC, InvokeNum, true, II);  // volatile
  
  // Insert a store of the stack ptr before the invoke, so we can restore it
  // later in the exception case.
  CallInst* StackSaveRet = CallInst::Create(StackSaveFn, "ssret", II);
  new StoreInst(StackSaveRet, StackPtr, true, II); // volatile

  BasicBlock::iterator NI = II->getNormalDest()->getFirstInsertionPt();
  // nonvolatile.
  new StoreInst(Constant::getNullValue(Type::getInt32Ty(II->getContext())), 
                InvokeNum, false, NI);

  Instruction* StackPtrLoad =
    new LoadInst(StackPtr, "stackptr.restore", true,
                 II->getUnwindDest()->getFirstInsertionPt());
  CallInst::Create(StackRestoreFn, StackPtrLoad, "")->insertAfter(StackPtrLoad);
    
  // Add a switch case to our unwind block.
  CatchSwitch->addCase(InvokeNoC, II->getUnwindDest());

  // Insert a normal call instruction.
  SmallVector<Value*,16> CallArgs(II->op_begin(), II->op_end() - 3);
  CallInst *NewCall = CallInst::Create(II->getCalledValue(),
                                       CallArgs, "", II);
  NewCall->takeName(II);
  NewCall->setCallingConv(II->getCallingConv());
  NewCall->setAttributes(II->getAttributes());
  NewCall->setDebugLoc(II->getDebugLoc());
  II->replaceAllUsesWith(NewCall);

  // Replace the invoke with an uncond branch.
  BranchInst::Create(II->getNormalDest(), NewCall->getParent());
  II->eraseFromParent();
}

/// MarkBlocksLiveIn - Insert BB and all of its predescessors into LiveBBs until
/// we reach blocks we've already seen.
static void MarkBlocksLiveIn(BasicBlock *BB, std::set<BasicBlock*> &LiveBBs) {
  if (!LiveBBs.insert(BB).second) return; // already been here.

  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    MarkBlocksLiveIn(*PI, LiveBBs);
}

// First thing we need to do is scan the whole function for values that are
// live across unwind edges.  Each value that is live across an unwind edge
// we spill into a stack location, guaranteeing that there is nothing live
// across the unwind edge.  This process also splits all critical edges
// coming out of invoke's.
void LowerInvoke::
splitLiveRangesLiveAcrossInvokes(SmallVectorImpl<InvokeInst*> &Invokes) {
  // First step, split all critical edges from invoke instructions.
  for (unsigned i = 0, e = Invokes.size(); i != e; ++i) {
    InvokeInst *II = Invokes[i];
    SplitCriticalEdge(II, 0, this);
    SplitCriticalEdge(II, 1, this);
    assert(!isa<PHINode>(II->getNormalDest()) &&
           !isa<PHINode>(II->getUnwindDest()) &&
           "critical edge splitting left single entry phi nodes?");
  }

  Function *F = Invokes.back()->getParent()->getParent();

  // To avoid having to handle incoming arguments specially, we lower each arg
  // to a copy instruction in the entry block.  This ensures that the argument
  // value itself cannot be live across the entry block.
  BasicBlock::iterator AfterAllocaInsertPt = F->begin()->begin();
  while (isa<AllocaInst>(AfterAllocaInsertPt) &&
        isa<ConstantInt>(cast<AllocaInst>(AfterAllocaInsertPt)->getArraySize()))
    ++AfterAllocaInsertPt;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
       AI != E; ++AI) {
    Type *Ty = AI->getType();
    // Aggregate types can't be cast, but are legal argument types, so we have
    // to handle them differently. We use an extract/insert pair as a
    // lightweight method to achieve the same goal.
    if (isa<StructType>(Ty) || isa<ArrayType>(Ty) || isa<VectorType>(Ty)) {
      Instruction *EI = ExtractValueInst::Create(AI, 0, "",AfterAllocaInsertPt);
      Instruction *NI = InsertValueInst::Create(AI, EI, 0);
      NI->insertAfter(EI);
      AI->replaceAllUsesWith(NI);
      // Set the operand of the instructions back to the AllocaInst.
      EI->setOperand(0, AI);
      NI->setOperand(0, AI);
    } else {
      // This is always a no-op cast because we're casting AI to AI->getType()
      // so src and destination types are identical. BitCast is the only
      // possibility.
      CastInst *NC = new BitCastInst(
        AI, AI->getType(), AI->getName()+".tmp", AfterAllocaInsertPt);
      AI->replaceAllUsesWith(NC);
      // Set the operand of the cast instruction back to the AllocaInst.
      // Normally it's forbidden to replace a CastInst's operand because it
      // could cause the opcode to reflect an illegal conversion. However,
      // we're replacing it here with the same value it was constructed with.
      // We do this because the above replaceAllUsesWith() clobbered the
      // operand, but we want this one to remain.
      NC->setOperand(0, AI);
    }
  }

  // Finally, scan the code looking for instructions with bad live ranges.
  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ++II) {
      // Ignore obvious cases we don't have to handle.  In particular, most
      // instructions either have no uses or only have a single use inside the
      // current block.  Ignore them quickly.
      Instruction *Inst = II;
      if (Inst->use_empty()) continue;
      if (Inst->hasOneUse() &&
          cast<Instruction>(Inst->use_back())->getParent() == BB &&
          !isa<PHINode>(Inst->use_back())) continue;

      // If this is an alloca in the entry block, it's not a real register
      // value.
      if (AllocaInst *AI = dyn_cast<AllocaInst>(Inst))
        if (isa<ConstantInt>(AI->getArraySize()) && BB == F->begin())
          continue;

      // Avoid iterator invalidation by copying users to a temporary vector.
      SmallVector<Instruction*,16> Users;
      for (Value::use_iterator UI = Inst->use_begin(), E = Inst->use_end();
           UI != E; ++UI) {
        Instruction *User = cast<Instruction>(*UI);
        if (User->getParent() != BB || isa<PHINode>(User))
          Users.push_back(User);
      }

      // Scan all of the uses and see if the live range is live across an unwind
      // edge.  If we find a use live across an invoke edge, create an alloca
      // and spill the value.
      std::set<InvokeInst*> InvokesWithStoreInserted;

      // Find all of the blocks that this value is live in.
      std::set<BasicBlock*> LiveBBs;
      LiveBBs.insert(Inst->getParent());
      while (!Users.empty()) {
        Instruction *U = Users.back();
        Users.pop_back();

        if (!isa<PHINode>(U)) {
          MarkBlocksLiveIn(U->getParent(), LiveBBs);
        } else {
          // Uses for a PHI node occur in their predecessor block.
          PHINode *PN = cast<PHINode>(U);
          for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
            if (PN->getIncomingValue(i) == Inst)
              MarkBlocksLiveIn(PN->getIncomingBlock(i), LiveBBs);
        }
      }

      // Now that we know all of the blocks that this thing is live in, see if
      // it includes any of the unwind locations.
      bool NeedsSpill = false;
      for (unsigned i = 0, e = Invokes.size(); i != e; ++i) {
        BasicBlock *UnwindBlock = Invokes[i]->getUnwindDest();
        if (UnwindBlock != BB && LiveBBs.count(UnwindBlock)) {
          NeedsSpill = true;
        }
      }

      // If we decided we need a spill, do it.
      if (NeedsSpill) {
        ++NumSpilled;
        DemoteRegToStack(*Inst, true);
      }
    }
}

bool LowerInvoke::insertExpensiveEHSupport(Function &F) {
  SmallVector<ReturnInst*,16> Returns;
  SmallVector<InvokeInst*,16> Invokes;
  UnreachableInst* UnreachablePlaceholder = 0;

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
      // Remember all return instructions in case we insert an invoke into this
      // function.
      Returns.push_back(RI);
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      Invokes.push_back(II);
    }

  if (Invokes.empty()) return false;

  NumInvokes += Invokes.size();

  // TODO: This is not an optimal way to do this.  In particular, this always
  // inserts setjmp calls into the entries of functions with invoke instructions
  // even though there are possibly paths through the function that do not
  // execute any invokes.  In particular, for functions with early exits, e.g.
  // the 'addMove' method in hexxagon, it would be nice to not have to do the
  // setjmp stuff on the early exit path.  This requires a bit of dataflow, but
  // would not be too hard to do.

  // If we have an invoke instruction, insert a setjmp that dominates all
  // invokes.  After the setjmp, use a cond branch that goes to the original
  // code path on zero, and to a designated 'catch' block of nonzero.
  Value *OldJmpBufPtr = 0;
  if (!Invokes.empty()) {
    // First thing we need to do is scan the whole function for values that are
    // live across unwind edges.  Each value that is live across an unwind edge
    // we spill into a stack location, guaranteeing that there is nothing live
    // across the unwind edge.  This process also splits all critical edges
    // coming out of invoke's.
    splitLiveRangesLiveAcrossInvokes(Invokes);

    BasicBlock *EntryBB = F.begin();

    // Create an alloca for the incoming jump buffer ptr and the new jump buffer
    // that needs to be restored on all exits from the function.  This is an
    // alloca because the value needs to be live across invokes.
    unsigned Align = TLI ? TLI->getJumpBufAlignment() : 0;
    AllocaInst *JmpBuf =
      new AllocaInst(JBLinkTy, 0, Align,
                     "jblink", F.begin()->begin());

    Value *Idx[] = { Constant::getNullValue(Type::getInt32Ty(F.getContext())),
                     ConstantInt::get(Type::getInt32Ty(F.getContext()), 1) };
    OldJmpBufPtr = GetElementPtrInst::Create(JmpBuf, Idx, "OldBuf",
                                             EntryBB->getTerminator());

    // Copy the JBListHead to the alloca.
    Value *OldBuf = new LoadInst(JBListHead, "oldjmpbufptr", true,
                                 EntryBB->getTerminator());
    new StoreInst(OldBuf, OldJmpBufPtr, true, EntryBB->getTerminator());

    // Add the new jumpbuf to the list.
    new StoreInst(JmpBuf, JBListHead, true, EntryBB->getTerminator());

    // Create the catch block.  The catch block is basically a big switch
    // statement that goes to all of the invoke catch blocks.
    BasicBlock *CatchBB =
            BasicBlock::Create(F.getContext(), "setjmp.catch", &F);

    // Create an alloca which keeps track of the stack pointer before every
    // invoke, this allows us to properly restore the stack pointer after
    // long jumping.
    AllocaInst *StackPtr = new AllocaInst(Type::getInt8PtrTy(F.getContext()), 0,
                                          "stackptr", EntryBB->begin());

    // Create an alloca which keeps track of which invoke is currently
    // executing.  For normal calls it contains zero.
    AllocaInst *InvokeNum = new AllocaInst(Type::getInt32Ty(F.getContext()), 0,
                                           "invokenum",EntryBB->begin());
    new StoreInst(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0), 
                  InvokeNum, true, EntryBB->getTerminator());

    // Insert a load in the Catch block, and a switch on its value.  By default,
    // we go to a block that just does an unwind (which is the correct action
    // for a standard call). We insert an unreachable instruction here and
    // modify the block to jump to the correct unwinding pad later.
    BasicBlock *UnwindBB = BasicBlock::Create(F.getContext(), "unwindbb", &F);
    UnreachablePlaceholder = new UnreachableInst(F.getContext(), UnwindBB);

    Value *CatchLoad = new LoadInst(InvokeNum, "invoke.num", true, CatchBB);
    SwitchInst *CatchSwitch =
      SwitchInst::Create(CatchLoad, UnwindBB, Invokes.size(), CatchBB);

    // Now that things are set up, insert the setjmp call itself.

    // Split the entry block to insert the conditional branch for the setjmp.
    BasicBlock *ContBlock = EntryBB->splitBasicBlock(EntryBB->getTerminator(),
                                                     "setjmp.cont");

    Idx[1] = ConstantInt::get(Type::getInt32Ty(F.getContext()), 0);
    Value *JmpBufPtr = GetElementPtrInst::Create(JmpBuf, Idx, "TheJmpBuf",
                                                 EntryBB->getTerminator());
    JmpBufPtr = new BitCastInst(JmpBufPtr,
                        Type::getInt8PtrTy(F.getContext()),
                                "tmp", EntryBB->getTerminator());
    Value *SJRet = CallInst::Create(SetJmpFn, JmpBufPtr, "sjret",
                                    EntryBB->getTerminator());

    // Compare the return value to zero.
    Value *IsNormal = new ICmpInst(EntryBB->getTerminator(),
                                   ICmpInst::ICMP_EQ, SJRet,
                                   Constant::getNullValue(SJRet->getType()),
                                   "notunwind");
    // Nuke the uncond branch.
    EntryBB->getTerminator()->eraseFromParent();

    // Put in a new condbranch in its place.
    BranchInst::Create(ContBlock, CatchBB, IsNormal, EntryBB);

    // At this point, we are all set up, rewrite each invoke instruction.
    for (unsigned i = 0, e = Invokes.size(); i != e; ++i)
      rewriteExpensiveInvoke(Invokes[i], i+1, InvokeNum, StackPtr, CatchSwitch);
  }

  // We know that there is at least one unwind.

  // Create three new blocks, the block to load the jmpbuf ptr and compare
  // against null, the block to do the longjmp, and the error block for if it
  // is null.  Add them at the end of the function because they are not hot.
  BasicBlock *UnwindHandler = BasicBlock::Create(F.getContext(),
                                                "dounwind", &F);
  BasicBlock *UnwindBlock = BasicBlock::Create(F.getContext(), "unwind", &F);
  BasicBlock *TermBlock = BasicBlock::Create(F.getContext(), "unwinderror", &F);

  // If this function contains an invoke, restore the old jumpbuf ptr.
  Value *BufPtr;
  if (OldJmpBufPtr) {
    // Before the return, insert a copy from the saved value to the new value.
    BufPtr = new LoadInst(OldJmpBufPtr, "oldjmpbufptr", UnwindHandler);
    new StoreInst(BufPtr, JBListHead, UnwindHandler);
  } else {
    BufPtr = new LoadInst(JBListHead, "ehlist", UnwindHandler);
  }

  // Load the JBList, if it's null, then there was no catch!
  Value *NotNull = new ICmpInst(*UnwindHandler, ICmpInst::ICMP_NE, BufPtr,
                                Constant::getNullValue(BufPtr->getType()),
                                "notnull");
  BranchInst::Create(UnwindBlock, TermBlock, NotNull, UnwindHandler);

  // Create the block to do the longjmp.
  // Get a pointer to the jmpbuf and longjmp.
  Value *Idx[] = { Constant::getNullValue(Type::getInt32Ty(F.getContext())),
                   ConstantInt::get(Type::getInt32Ty(F.getContext()), 0) };
  Idx[0] = GetElementPtrInst::Create(BufPtr, Idx, "JmpBuf", UnwindBlock);
  Idx[0] = new BitCastInst(Idx[0],
             Type::getInt8PtrTy(F.getContext()),
                           "tmp", UnwindBlock);
  Idx[1] = ConstantInt::get(Type::getInt32Ty(F.getContext()), 1);
  CallInst::Create(LongJmpFn, Idx, "", UnwindBlock);
  new UnreachableInst(F.getContext(), UnwindBlock);

  // Set up the term block ("throw without a catch").
  new UnreachableInst(F.getContext(), TermBlock);

  // Insert a call to abort()
  CallInst::Create(AbortFn, "",
                   TermBlock->getTerminator())->setTailCall();

  // Replace the inserted unreachable with a branch to the unwind handler.
  if (UnreachablePlaceholder) {
    BranchInst::Create(UnwindHandler, UnreachablePlaceholder);
    UnreachablePlaceholder->eraseFromParent();
  }

  // Finally, for any returns from this function, if this function contains an
  // invoke, restore the old jmpbuf pointer to its input value.
  if (OldJmpBufPtr) {
    for (unsigned i = 0, e = Returns.size(); i != e; ++i) {
      ReturnInst *R = Returns[i];

      // Before the return, insert a copy from the saved value to the new value.
      Value *OldBuf = new LoadInst(OldJmpBufPtr, "oldjmpbufptr", true, R);
      new StoreInst(OldBuf, JBListHead, true, R);
    }
  }

  return true;
}

bool LowerInvoke::runOnFunction(Function &F) {
  if (useExpensiveEHSupport)
    return insertExpensiveEHSupport(F);
  else
    return insertCheapEHSupport(F);
}
