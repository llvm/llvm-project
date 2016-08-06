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

namespace coro {

bool declaresIntrinsics(Module &M, std::initializer_list<StringRef>);

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
