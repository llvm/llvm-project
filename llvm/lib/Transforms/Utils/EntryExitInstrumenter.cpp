//===- EntryExitInstrumenter.cpp - Function Entry/Exit Instrumentation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;

static void insertCall(Function &CurFn, StringRef Func,
                       Instruction *InsertionPt, DebugLoc DL) {
  Module &M = *InsertionPt->getParent()->getParent()->getParent();
  LLVMContext &C = InsertionPt->getParent()->getContext();

  if (Func == "mcount" ||
      Func == ".mcount" ||
      Func == "llvm.arm.gnu.eabi.mcount" ||
      Func == "\01_mcount" ||
      Func == "\01mcount" ||
      Func == "__mcount" ||
      Func == "_mcount" ||
      Func == "__cyg_profile_func_enter_bare") {
    FunctionCallee Fn = M.getOrInsertFunction(Func, Type::getVoidTy(C));
    CallInst *Call = CallInst::Create(Fn, "", InsertionPt);
    Call->setDebugLoc(DL);
    return;
  }

  if (Func == "__cyg_profile_func_enter" || Func == "__cyg_profile_func_exit") {
    Type *ArgTypes[] = {Type::getInt8PtrTy(C), Type::getInt8PtrTy(C)};

    FunctionCallee Fn = M.getOrInsertFunction(
        Func, FunctionType::get(Type::getVoidTy(C), ArgTypes, false));

    Instruction *RetAddr = CallInst::Create(
        Intrinsic::getDeclaration(&M, Intrinsic::returnaddress),
        ArrayRef<Value *>(ConstantInt::get(Type::getInt32Ty(C), 0)), "",
        InsertionPt);
    RetAddr->setDebugLoc(DL);

    Value *Args[] = {ConstantExpr::getBitCast(&CurFn, Type::getInt8PtrTy(C)),
                     RetAddr};

    CallInst *Call =
        CallInst::Create(Fn, ArrayRef<Value *>(Args), "", InsertionPt);
    Call->setDebugLoc(DL);
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
    DebugLoc DL;
    if (auto SP = F.getSubprogram())
      DL = DILocation::get(SP->getContext(), SP->getScopeLine(), 0, SP);

    insertCall(F, EntryFunc, &*F.begin()->getFirstInsertionPt(), DL);
    Changed = true;
    F.removeFnAttr(EntryAttr);
  }

  if (!ExitFunc.empty()) {
    for (BasicBlock &BB : F) {
      Instruction *T = BB.getTerminator();
      if (!isa<ReturnInst>(T))
        continue;

      // If T is preceded by a musttail call, that's the real terminator.
      if (CallInst *CI = BB.getTerminatingMustTailCall())
        T = CI;

      DebugLoc DL;
      if (DebugLoc TerminatorDL = T->getDebugLoc())
        DL = TerminatorDL;
      else if (auto SP = F.getSubprogram())
        DL = DILocation::get(SP->getContext(), 0, 0, SP);

      insertCall(F, ExitFunc, T, DL);
      Changed = true;
    }
    F.removeFnAttr(ExitAttr);
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
    AU.addPreserved<DominatorTreeWrapperPass>();
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
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
  bool runOnFunction(Function &F) override { return ::runOnFunction(F, true); }
};
char PostInlineEntryExitInstrumenter::ID = 0;
}

INITIALIZE_PASS_BEGIN(
    EntryExitInstrumenter, "ee-instrument",
    "Instrument function entry/exit with calls to e.g. mcount() (pre inlining)",
    false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(
    EntryExitInstrumenter, "ee-instrument",
    "Instrument function entry/exit with calls to e.g. mcount() (pre inlining)",
    false, false)

INITIALIZE_PASS_BEGIN(
    PostInlineEntryExitInstrumenter, "post-inline-ee-instrument",
    "Instrument function entry/exit with calls to e.g. mcount() "
    "(post inlining)",
    false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(
    PostInlineEntryExitInstrumenter, "post-inline-ee-instrument",
    "Instrument function entry/exit with calls to e.g. mcount() "
    "(post inlining)",
    false, false)

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

void llvm::EntryExitInstrumenterPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<llvm::EntryExitInstrumenterPass> *>(this)
      ->printPipeline(OS, MapClassName2PassName);
  OS << "<";
  if (PostInlining)
    OS << "post-inline";
  OS << ">";
}
