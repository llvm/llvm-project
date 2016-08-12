//===- CoroInternal.h - Internal Coroutine interfaces ---------*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Common definitions/declarations used internally by coroutine lowering passes.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_COROUTINES_COROINTERNAL_H
#define LLVM_LIB_TRANSFORMS_COROUTINES_COROINTERNAL_H

#include "CoroInstr.h"
#include "llvm/Transforms/Coroutines.h"

namespace llvm {

class PassRegistry;

void initializeCoroEarlyPass(PassRegistry &);
void initializeCoroSplitPass(PassRegistry &);
void initializeCoroElidePass(PassRegistry &);
void initializeCoroCleanupPass(PassRegistry &);

// CoroEarly pass marks every function that has coro.begin with a string
// attribute "coroutine.presplit"="0". CoroSplit pass processes the coroutine
// twice. First, it lets it go through complete IPO optimization pipeline as a
// single function. It forces restart of the pipeline by inserting an indirect
// call to an empty function "coro.devirt.trigger" which is devirtualized by
// CoroElide pass that triggers a restart of the pipeline by CGPassManager.
// When CoroSplit pass sees the same coroutine the second time, it splits it up,
// adds coroutine subfunctions to the SCC to be processed by IPO pipeline.

#define CORO_PRESPLIT_ATTR "coroutine.presplit"
#define UNPREPARED_FOR_SPLIT "0"
#define PREPARED_FOR_SPLIT "1"

#define CORO_DEVIRT_TRIGGER_FN "coro.devirt.trigger"

namespace coro {

bool declaresIntrinsics(Module &M, std::initializer_list<StringRef>);
void replaceAllCoroAllocs(CoroBeginInst *CB, bool Replacement);
void replaceAllCoroFrees(CoroBeginInst *CB, Value *Replacement);

// Keeps data and helper functions for lowering coroutine intrinsics.
struct LowererBase {
  Module &TheModule;
  LLVMContext &Context;
  FunctionType *const ResumeFnType;

  LowererBase(Module &M);
  Value *makeSubFnCall(Value *Arg, int Index, Instruction *InsertPt);
};

} // End namespace coro.
} // End namespace llvm

#endif
