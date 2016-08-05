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
  static std::unique_ptr<Lowerer> createIfNeeded(Module &M);
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

bool Lowerer::lowerEarlyIntrinsics(Function &F) {
  bool Changed = false;
  for (auto IB = inst_begin(F), IE = inst_end(F); IB != IE;) {
    Instruction &I = *IB++;
    if (auto CS = CallSite(&I)) {
      switch (CS.getIntrinsicID()) {
      default:
        continue;
      case Intrinsic::coro_resume:
        lowerResumeOrDestroy(CS, CoroSubFnInst::ResumeIndex);
        break;
      case Intrinsic::coro_destroy:
        lowerResumeOrDestroy(CS, CoroSubFnInst::DestroyIndex);
        break;
      }
      Changed = true;
      continue;
    }
  }
  return Changed;
}

// This pass has work to do only if we find intrinsics we are going to lower in
// the module.
std::unique_ptr<Lowerer> Lowerer::createIfNeeded(Module &M) {
  if (declaresIntrinsics(M, {"llvm.coro.resume", "llvm.coro.destroy"}))
    return llvm::make_unique<Lowerer>(M);

  return {};
}

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

struct CoroEarly : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid.
  CoroEarly() : FunctionPass(ID) {}

  std::unique_ptr<Lowerer> L;

  bool doInitialization(Module &M) override {
    L = Lowerer::createIfNeeded(M);
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
