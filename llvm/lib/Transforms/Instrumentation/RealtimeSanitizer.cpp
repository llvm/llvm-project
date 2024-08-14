//===- RealtimeSanitizer.cpp - RealtimeSanitizer instrumentation *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the RealtimeSanitizer, an LLVM transformation for
// detecting and reporting realtime safety violations.
//
// See also: llvm-project/compiler-rt/lib/rtsan/
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Analysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include "llvm/Transforms/Instrumentation/RealtimeSanitizer.h"

using namespace llvm;

static void insertCallBeforeInstruction(Function &Fn, Instruction &Instruction,
                                        const char *FunctionName) {
  LLVMContext &Context = Fn.getContext();
  FunctionType *FuncType = FunctionType::get(Type::getVoidTy(Context), false);
  FunctionCallee Func =
      Fn.getParent()->getOrInsertFunction(FunctionName, FuncType);
  IRBuilder<> Builder{&Instruction};
  Builder.CreateCall(Func, {});
}

static void insertCallAtFunctionEntryPoint(Function &Fn,
                                           const char *InsertFnName) {

  insertCallBeforeInstruction(Fn, Fn.front().front(), InsertFnName);
}

static void insertCallAtAllFunctionExitPoints(Function &Fn,
                                              const char *InsertFnName) {
  for (auto &BB : Fn)
    for (auto &I : BB)
      if (isa<ReturnInst>(&I))
        insertCallBeforeInstruction(Fn, I, InsertFnName);
}

RealtimeSanitizerPass::RealtimeSanitizerPass(
    const RealtimeSanitizerOptions &Options) {}

PreservedAnalyses RealtimeSanitizerPass::run(Function &F,
                                             AnalysisManager<Function> &AM) {
  if (F.hasFnAttribute(Attribute::SanitizeRealtime)) {
    insertCallAtFunctionEntryPoint(F, "__rtsan_realtime_enter");
    insertCallAtAllFunctionExitPoints(F, "__rtsan_realtime_exit");

    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

  return PreservedAnalyses::all();
}
