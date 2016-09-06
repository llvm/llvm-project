//===- CoroEarly.cpp - Coroutine Early Function Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass lowers coroutine intrinsics that hide the details of the exact
// calling convention for coroutine resume and destroy functions and details of
// the structure of the coroutine frame.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "coro-early"

namespace {
// Created on demand if CoroEarly pass has work to do.
class Lowerer : public coro::LowererBase {
  void lowerResumeOrDestroy(CallSite CS, CoroSubFnInst::ResumeKind);

public:
  Lowerer(Module &M) : LowererBase(M) {}
  bool lowerEarlyIntrinsics(Function &F);
};
}

// Replace a direct call to coro.resume or coro.destroy with an indirect call to
// an address returned by coro.subfn.addr intrinsic. This is done so that
// CGPassManager recognizes devirtualization when CoroElide pass replaces a call
// to coro.subfn.addr with an appropriate function address.
void Lowerer::lowerResumeOrDestroy(CallSite CS,
                                   CoroSubFnInst::ResumeKind Index) {
  Value *ResumeAddr =
      makeSubFnCall(CS.getArgOperand(0), Index, CS.getInstruction());
  CS.setCalledFunction(ResumeAddr);
  CS.setCallingConv(CallingConv::Fast);
}

// Prior to CoroSplit, calls to coro.begin needs to be marked as NoDuplicate,
// as CoroSplit assumes there is exactly one coro.begin. After CoroSplit,
// NoDuplicate attribute will be removed from coro.begin otherwise, it will
// interfere with inlining.
static void setCannotDuplicate(CoroIdInst *CoroId) {
  for (User *U : CoroId->users())
    if (auto *CB = dyn_cast<CoroBeginInst>(U))
      CB->setCannotDuplicate();
}

bool Lowerer::lowerEarlyIntrinsics(Function &F) {
  bool Changed = false;
  for (auto IB = inst_begin(F), IE = inst_end(F); IB != IE;) {
    Instruction &I = *IB++;
    if (auto CS = CallSite(&I)) {
      switch (CS.getIntrinsicID()) {
      default:
        continue;
      case Intrinsic::coro_suspend:
        // Make sure that final suspend point is not duplicated as CoroSplit
        // pass expects that there is at most one final suspend point.
        if (cast<CoroSuspendInst>(&I)->isFinal())
          CS.setCannotDuplicate();
        break;
      case Intrinsic::coro_end:
        // Make sure that fallthrough coro.end is not duplicated as CoroSplit
        // pass expects that there is at most one fallthrough coro.end.
        if (cast<CoroEndInst>(&I)->isFallthrough())
          CS.setCannotDuplicate();
        break;
      case Intrinsic::coro_id:
        // Mark a function that comes out of the frontend that has a coro.id
        // with a coroutine attribute.
        if (auto *CII = cast<CoroIdInst>(&I)) {
          if (CII->getInfo().isPreSplit()) {
            F.addFnAttr(CORO_PRESPLIT_ATTR, UNPREPARED_FOR_SPLIT);
            setCannotDuplicate(CII);
          }
        }
        break;
      case Intrinsic::coro_resume:
        lowerResumeOrDestroy(CS, CoroSubFnInst::ResumeIndex);
        break;
      case Intrinsic::coro_destroy:
        lowerResumeOrDestroy(CS, CoroSubFnInst::DestroyIndex);
        break;
      }
      Changed = true;
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

struct CoroEarly : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid.
  CoroEarly() : FunctionPass(ID) {}

  std::unique_ptr<Lowerer> L;

  // This pass has work to do only if we find intrinsics we are going to lower
  // in the module.
  bool doInitialization(Module &M) override {
    if (coro::declaresIntrinsics(M, {"llvm.coro.begin", "llvm.coro.resume",
                                     "llvm.coro.destroy", "llvm.coro.suspend",
                                     "llvm.coro.end"}))
      L = llvm::make_unique<Lowerer>(M);
    return false;
  }

  bool runOnFunction(Function &F) override {
    if (!L)
      return false;

    return L->lowerEarlyIntrinsics(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};
}

char CoroEarly::ID = 0;
INITIALIZE_PASS(CoroEarly, "coro-early", "Lower early coroutine intrinsics",
                false, false)

Pass *llvm::createCoroEarlyPass() { return new CoroEarly(); }
