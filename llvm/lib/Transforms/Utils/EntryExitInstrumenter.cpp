//===- EntryExitInstrumenter.cpp - Function Entry/Exit Instrumentation ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

static void insertCall(Function &CurFn, StringRef Func,
                       Instruction *InsertionPt) {
  Module &M = *InsertionPt->getParent()->getParent()->getParent();
  LLVMContext &C = InsertionPt->getParent()->getContext();

  if (Func == "mcount" ||
      Func == ".mcount" ||
      Func == "\01__gnu_mcount_nc" ||
      Func == "\01_mcount" ||
      Func == "\01mcount" ||
      Func == "__mcount" ||
      Func == "_mcount") {
    Constant *Fn = M.getOrInsertFunction(Func, Type::getVoidTy(C));
    CallInst::Create(Fn, "", InsertionPt);
    return;
  }

  if (Func == "__cyg_profile_func_enter" || Func == "__cyg_profile_func_exit") {
    Type *ArgTypes[] = {Type::getInt8PtrTy(C), Type::getInt8PtrTy(C)};

    Constant *Fn = M.getOrInsertFunction(
        Func, FunctionType::get(Type::getVoidTy(C), ArgTypes, false));

    Instruction *RetAddr = CallInst::Create(
        Intrinsic::getDeclaration(&M, Intrinsic::returnaddress),
        ArrayRef<Value *>(ConstantInt::get(Type::getInt32Ty(C), 0)), "",
        InsertionPt);

    Value *Args[] = {ConstantExpr::getBitCast(&CurFn, Type::getInt8PtrTy(C)),
                     RetAddr};

    CallInst::Create(Fn, ArrayRef<Value *>(Args), "", InsertionPt);
    return;
  }

  // We only know how to call a fixed set of instrumentation functions, because
  // they all expect different arguments, etc.
  report_fatal_error(Twine("Unknown instrumentation function: '") + Func + "'");
}

static bool runOnFunction(Function &F, bool PostInlining) {
  StringRef EntryAttr = PostInlining ? "instrument-function-entry-inlined"
                                     : "instrument-function-entry";

  StringRef ExitAttr = PostInlining ? "instrument-function-exit-inlined"
                                    : "instrument-function-exit";

  StringRef EntryFunc = F.getFnAttribute(EntryAttr).getValueAsString();
  StringRef ExitFunc = F.getFnAttribute(ExitAttr).getValueAsString();

  bool Changed = false;

  // If the attribute is specified, insert instrumentation and then "consume"
  // the attribute so that it's not inserted again if the pass should happen to
  // run later for some reason.

  if (!EntryFunc.empty()) {
    insertCall(F, EntryFunc, &*F.begin()->getFirstInsertionPt());
    Changed = true;
    F.removeAttribute(AttributeList::FunctionIndex, EntryAttr);
  }

  if (!ExitFunc.empty()) {
    for (BasicBlock &BB : F) {
      TerminatorInst *T = BB.getTerminator();
      if (isa<ReturnInst>(T)) {
        insertCall(F, ExitFunc, T);
        Changed = true;
      }
    }
    F.removeAttribute(AttributeList::FunctionIndex, ExitAttr);
  }

  return Changed;
}

namespace {
struct EntryExitInstrumenter : public FunctionPass {
  static char ID;
  EntryExitInstrumenter() : FunctionPass(ID) {
    initializeEntryExitInstrumenterPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
  bool runOnFunction(Function &F) override { return ::runOnFunction(F, false); }
};
char EntryExitInstrumenter::ID = 0;

struct PostInlineEntryExitInstrumenter : public FunctionPass {
  static char ID;
  PostInlineEntryExitInstrumenter() : FunctionPass(ID) {
    initializePostInlineEntryExitInstrumenterPass(
        *PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
  bool runOnFunction(Function &F) override { return ::runOnFunction(F, true); }
};
char PostInlineEntryExitInstrumenter::ID = 0;
}

INITIALIZE_PASS(
    EntryExitInstrumenter, "ee-instrument",
    "Instrument function entry/exit with calls to e.g. mcount() (pre inlining)",
    false, false);

INITIALIZE_PASS(PostInlineEntryExitInstrumenter, "post-inline-ee-instrument",
                "Instrument function entry/exit with calls to e.g. mcount() "
                "(post inlining)",
                false, false);

FunctionPass *llvm::createEntryExitInstrumenterPass() {
  return new EntryExitInstrumenter();
}

FunctionPass *llvm::createPostInlineEntryExitInstrumenterPass() {
  return new PostInlineEntryExitInstrumenter();
}

PreservedAnalyses
llvm::EntryExitInstrumenterPass::run(Function &F, FunctionAnalysisManager &AM) {
  runOnFunction(F, PostInlining);
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
