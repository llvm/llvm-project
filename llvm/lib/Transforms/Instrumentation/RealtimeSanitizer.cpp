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
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/Transforms/Instrumentation/RealtimeSanitizer.h"

using namespace llvm;

const char kRtsanModuleCtorName[] = "rtsan.module_ctor";
const char kRtsanInitName[] = "__rtsan_ensure_initialized";

static SmallVector<Type *> getArgTypes(ArrayRef<Value *> FunctionArgs) {
  SmallVector<Type *> Types;
  for (Value *Arg : FunctionArgs)
    Types.push_back(Arg->getType());
  return Types;
}

static void insertCallBeforeInstruction(Function &Fn, Instruction &Instruction,
                                        const char *FunctionName,
                                        ArrayRef<Value *> FunctionArgs) {
  LLVMContext &Context = Fn.getContext();
  FunctionType *FuncType = FunctionType::get(Type::getVoidTy(Context),
                                             getArgTypes(FunctionArgs), false);
  FunctionCallee Func =
      Fn.getParent()->getOrInsertFunction(FunctionName, FuncType);
  IRBuilder<> Builder{&Instruction};
  Builder.CreateCall(Func, FunctionArgs);
}

static void insertCallAtFunctionEntryPoint(Function &Fn,
                                           const char *InsertFnName,
                                           ArrayRef<Value *> FunctionArgs) {
  insertCallBeforeInstruction(Fn, Fn.front().front(), InsertFnName,
                              FunctionArgs);
}

static void insertCallAtAllFunctionExitPoints(Function &Fn,
                                              const char *InsertFnName,
                                              ArrayRef<Value *> FunctionArgs) {
  for (auto &I : instructions(Fn))
    if (isa<ReturnInst>(&I))
      insertCallBeforeInstruction(Fn, I, InsertFnName, FunctionArgs);
}

static PreservedAnalyses rtsanPreservedCFGAnalyses() {
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

static PreservedAnalyses runSanitizeRealtime(Function &Fn) {
  insertCallAtFunctionEntryPoint(Fn, "__rtsan_realtime_enter", {});
  insertCallAtAllFunctionExitPoints(Fn, "__rtsan_realtime_exit", {});
  return rtsanPreservedCFGAnalyses();
}

static PreservedAnalyses runSanitizeRealtimeBlocking(Function &Fn) {
  IRBuilder<> Builder(&Fn.front().front());
  Value *Name = Builder.CreateGlobalString(demangle(Fn.getName()));
  insertCallAtFunctionEntryPoint(Fn, "__rtsan_notify_blocking_call", {Name});
  return rtsanPreservedCFGAnalyses();
}

PreservedAnalyses RealtimeSanitizerPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kRtsanModuleCtorName, kRtsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });

  for (Function &F : M) {
    if (F.hasFnAttribute(Attribute::SanitizeRealtime))
      runSanitizeRealtime(F);

    if (F.hasFnAttribute(Attribute::SanitizeRealtimeBlocking))
      runSanitizeRealtimeBlocking(F);
  }

  return PreservedAnalyses::none();
}
