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

#include "llvm/Demangle/Demangle.h"
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

static PreservedAnalyses rtsanPreservedCFGAnalyses() {
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

static void insertNotifyBlockingCallAtFunctionEntryPoint(Function &Fn) {
  IRBuilder<> Builder(&Fn.front().front());
  Value *NameArg = Builder.CreateGlobalString(demangle(Fn.getName()));

  FunctionType *FuncType =
      FunctionType::get(Type::getVoidTy(Fn.getContext()),
                        {PointerType::getUnqual(Fn.getContext())}, false);

  FunctionCallee Func = Fn.getParent()->getOrInsertFunction(
      "__rtsan_notify_blocking_call", FuncType);

  Builder.CreateCall(Func, {NameArg});
}

RealtimeSanitizerPass::RealtimeSanitizerPass(
    const RealtimeSanitizerOptions &Options) {}

PreservedAnalyses RealtimeSanitizerPass::run(Function &Fn,
                                             AnalysisManager<Function> &AM) {
  if (Fn.hasFnAttribute(Attribute::SanitizeRealtime)) {
    insertCallAtFunctionEntryPoint(Fn, "__rtsan_realtime_enter");
    insertCallAtAllFunctionExitPoints(Fn, "__rtsan_realtime_exit");
    return rtsanPreservedCFGAnalyses();
  }

  if (Fn.hasFnAttribute(Attribute::SanitizeRealtimeUnsafe)) {
    insertNotifyBlockingCallAtFunctionEntryPoint(Fn);
    return rtsanPreservedCFGAnalyses();
  }

  return PreservedAnalyses::all();
}
