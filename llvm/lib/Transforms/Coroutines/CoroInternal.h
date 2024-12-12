//===- CoroInternal.h - Internal Coroutine interfaces ---------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Common definitions/declarations used internally by coroutine lowering passes.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_COROUTINES_COROINTERNAL_H
#define LLVM_LIB_TRANSFORMS_COROUTINES_COROINTERNAL_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"
#include "llvm/Transforms/Coroutines/CoroShape.h"

namespace llvm {

class CallGraph;

namespace coro {

bool isSuspendBlock(BasicBlock *BB);
bool declaresAnyIntrinsic(const Module &M);
bool declaresIntrinsics(const Module &M,
                        const std::initializer_list<StringRef>);
void replaceCoroFree(CoroIdInst *CoroId, bool Elide);

/// Replaces all @llvm.coro.alloc intrinsics calls associated with a given
/// call @llvm.coro.id instruction with boolean value false.
void suppressCoroAllocs(CoroIdInst *CoroId);
/// Replaces CoroAllocs with boolean value false.
void suppressCoroAllocs(LLVMContext &Context,
                        ArrayRef<CoroAllocInst *> CoroAllocs);

/// Attempts to rewrite the location operand of debug intrinsics in terms of
/// the coroutine frame pointer, folding pointer offsets into the DIExpression
/// of the intrinsic.
/// If the frame pointer is an Argument, store it into an alloca to enhance the
/// debugability.
void salvageDebugInfo(
    SmallDenseMap<Argument *, AllocaInst *, 4> &ArgToAllocaMap,
    DbgVariableIntrinsic &DVI, bool IsEntryPoint);
void salvageDebugInfo(
    SmallDenseMap<Argument *, AllocaInst *, 4> &ArgToAllocaMap,
    DbgVariableRecord &DVR, bool UseEntryValue);

// Keeps data and helper functions for lowering coroutine intrinsics.
struct LowererBase {
  Module &TheModule;
  LLVMContext &Context;
  PointerType *const Int8Ptr;
  FunctionType *const ResumeFnType;
  ConstantPointerNull *const NullPtr;

  LowererBase(Module &M);
  CallInst *makeSubFnCall(Value *Arg, int Index, Instruction *InsertPt);
};

bool defaultMaterializable(Instruction &V);
void normalizeCoroutine(Function &F, coro::Shape &Shape,
                        TargetTransformInfo &TTI);
CallInst *createMustTailCall(DebugLoc Loc, Function *MustTailCallFn,
                             TargetTransformInfo &TTI,
                             ArrayRef<Value *> Arguments, IRBuilder<> &);
} // End namespace coro.
} // End namespace llvm

#endif
