//===-- StackProtector.cpp - Stack Protector Insertion --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass inserts stack protectors into functions which need them. A variable
// with a random value in it is stored onto the stack before the local variables
// are allocated. Upon exiting the block, the stored value is checked. If it's
// changed, then there was some sort of violation and the program aborts.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "stack-protector"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Attributes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

namespace {
  class StackProtector : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// target type sizes.
    const TargetLowering *TLI;

    Function *F;
    Module *M;

    DominatorTree *DT;

    /// InsertStackProtectors - Insert code into the prologue and epilogue of
    /// the function.
    ///
    ///  - The prologue code loads and stores the stack guard onto the stack.
    ///  - The epilogue checks the value stored in the prologue against the
    ///    original value. It calls __stack_chk_fail if they differ.
    bool InsertStackProtectors();

    /// CreateFailBB - Create a basic block to jump to when the stack protector
    /// check fails.
    BasicBlock *CreateFailBB();

    /// ContainsProtectableArray - Check whether the type either is an array or
    /// contains an array of sufficient size so that we need stack protectors
    /// for it.
    bool ContainsProtectableArray(Type *Ty, bool InStruct = false) const;

    /// RequiresStackProtector - Check whether or not this function needs a
    /// stack protector based upon the stack protector level.
    bool RequiresStackProtector() const;
  public:
    static char ID;             // Pass identification, replacement for typeid.
    StackProtector() : FunctionPass(ID), TLI(0) {
      initializeStackProtectorPass(*PassRegistry::getPassRegistry());
    }
    StackProtector(const TargetLowering *tli)
      : FunctionPass(ID), TLI(tli) {
      initializeStackProtectorPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<DominatorTree>();
    }

    virtual bool runOnFunction(Function &Fn);
  };
} // end anonymous namespace

char StackProtector::ID = 0;
INITIALIZE_PASS(StackProtector, "stack-protector",
                "Insert stack protectors", false, false)

FunctionPass *llvm::createStackProtectorPass(const TargetLowering *tli) {
  return new StackProtector(tli);
}

bool StackProtector::runOnFunction(Function &Fn) {
  F = &Fn;
  M = F->getParent();
  DT = getAnalysisIfAvailable<DominatorTree>();

  if (!RequiresStackProtector()) return false;

  return InsertStackProtectors();
}

/// ContainsProtectableArray - Check whether the type either is an array or
/// contains a char array of sufficient size so that we need stack protectors
/// for it.
bool StackProtector::ContainsProtectableArray(Type *Ty, bool InStruct) const {
  if (!Ty) return false;
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    const TargetMachine &TM = TLI->getTargetMachine();
    if (!AT->getElementType()->isIntegerTy(8)) {
      Triple Trip(TM.getTargetTriple());

      // If we're on a non-Darwin platform or we're inside of a structure, don't
      // add stack protectors unless the array is a character array.
      if (InStruct || !Trip.isOSDarwin())
          return false;
    }

    // If an array has more than SSPBufferSize bytes of allocated space, then we
    // emit stack protectors.
    if (TM.Options.SSPBufferSize <= TLI->getDataLayout()->getTypeAllocSize(AT))
      return true;
  }

  const StructType *ST = dyn_cast<StructType>(Ty);
  if (!ST) return false;

  for (StructType::element_iterator I = ST->element_begin(),
         E = ST->element_end(); I != E; ++I)
    if (ContainsProtectableArray(*I, true))
      return true;

  return false;
}

/// RequiresStackProtector - Check whether or not this function needs a stack
/// protector based upon the stack protector level. The heuristic we use is to
/// add a guard variable to functions that call alloca, and functions with
/// buffers larger than SSPBufferSize bytes.
bool StackProtector::RequiresStackProtector() const {
  if (F->getFnAttributes().hasAttribute(Attributes::StackProtectReq))
    return true;

  if (!F->getFnAttributes().hasAttribute(Attributes::StackProtect))
    return false;

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    BasicBlock *BB = I;

    for (BasicBlock::iterator
           II = BB->begin(), IE = BB->end(); II != IE; ++II)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
        if (AI->isArrayAllocation())
          // This is a call to alloca with a variable size. Emit stack
          // protectors.
          return true;

        if (ContainsProtectableArray(AI->getAllocatedType()))
          return true;
      }
  }

  return false;
}

/// InsertStackProtectors - Insert code into the prologue and epilogue of the
/// function.
///
///  - The prologue code loads and stores the stack guard onto the stack.
///  - The epilogue checks the value stored in the prologue against the original
///    value. It calls __stack_chk_fail if they differ.
bool StackProtector::InsertStackProtectors() {
  BasicBlock *FailBB = 0;       // The basic block to jump to if check fails.
  BasicBlock *FailBBDom = 0;    // FailBB's dominator.
  AllocaInst *AI = 0;           // Place on stack that stores the stack guard.
  Value *StackGuardVar = 0;  // The stack guard variable.

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ) {
    BasicBlock *BB = I++;
    ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator());
    if (!RI) continue;

    if (!FailBB) {
      // Insert code into the entry block that stores the __stack_chk_guard
      // variable onto the stack:
      //
      //   entry:
      //     StackGuardSlot = alloca i8*
      //     StackGuard = load __stack_chk_guard
      //     call void @llvm.stackprotect.create(StackGuard, StackGuardSlot)
      //
      PointerType *PtrTy = Type::getInt8PtrTy(RI->getContext());
      unsigned AddressSpace, Offset;
      if (TLI->getStackCookieLocation(AddressSpace, Offset)) {
        Constant *OffsetVal =
          ConstantInt::get(Type::getInt32Ty(RI->getContext()), Offset);

        StackGuardVar = ConstantExpr::getIntToPtr(OffsetVal,
                                      PointerType::get(PtrTy, AddressSpace));
      } else {
        StackGuardVar = M->getOrInsertGlobal("__stack_chk_guard", PtrTy);
      }

      BasicBlock &Entry = F->getEntryBlock();
      Instruction *InsPt = &Entry.front();

      AI = new AllocaInst(PtrTy, "StackGuardSlot", InsPt);
      LoadInst *LI = new LoadInst(StackGuardVar, "StackGuard", false, InsPt);

      Value *Args[] = { LI, AI };
      CallInst::
        Create(Intrinsic::getDeclaration(M, Intrinsic::stackprotector),
               Args, "", InsPt);

      // Create the basic block to jump to when the guard check fails.
      FailBB = CreateFailBB();
    }

    // For each block with a return instruction, convert this:
    //
    //   return:
    //     ...
    //     ret ...
    //
    // into this:
    //
    //   return:
    //     ...
    //     %1 = load __stack_chk_guard
    //     %2 = load StackGuardSlot
    //     %3 = cmp i1 %1, %2
    //     br i1 %3, label %SP_return, label %CallStackCheckFailBlk
    //
    //   SP_return:
    //     ret ...
    //
    //   CallStackCheckFailBlk:
    //     call void @__stack_chk_fail()
    //     unreachable

    // Split the basic block before the return instruction.
    BasicBlock *NewBB = BB->splitBasicBlock(RI, "SP_return");

    if (DT && DT->isReachableFromEntry(BB)) {
      DT->addNewBlock(NewBB, BB);
      FailBBDom = FailBBDom ? DT->findNearestCommonDominator(FailBBDom, BB) :BB;
    }

    // Remove default branch instruction to the new BB.
    BB->getTerminator()->eraseFromParent();

    // Move the newly created basic block to the point right after the old basic
    // block so that it's in the "fall through" position.
    NewBB->moveAfter(BB);

    // Generate the stack protector instructions in the old basic block.
    LoadInst *LI1 = new LoadInst(StackGuardVar, "", false, BB);
    LoadInst *LI2 = new LoadInst(AI, "", true, BB);
    ICmpInst *Cmp = new ICmpInst(*BB, CmpInst::ICMP_EQ, LI1, LI2, "");
    BranchInst::Create(NewBB, FailBB, Cmp, BB);
  }

  // Return if we didn't modify any basic blocks. I.e., there are no return
  // statements in the function.
  if (!FailBB) return false;

  if (DT && FailBBDom)
    DT->addNewBlock(FailBB, FailBBDom);

  return true;
}

/// CreateFailBB - Create a basic block to jump to when the stack protector
/// check fails.
BasicBlock *StackProtector::CreateFailBB() {
  BasicBlock *FailBB = BasicBlock::Create(F->getContext(),
                                          "CallStackCheckFailBlk", F);
  Constant *StackChkFail =
    M->getOrInsertFunction("__stack_chk_fail",
                           Type::getVoidTy(F->getContext()), NULL);
  CallInst::Create(StackChkFail, "", FailBB);
  new UnreachableInst(F->getContext(), FailBB);
  return FailBB;
}
