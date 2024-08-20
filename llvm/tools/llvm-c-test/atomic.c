/*===-- atomic.c - tool for testing libLLVM and llvm-c API ----------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file implements the --atomic-* commands in llvm-c-test.               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c-test.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int llvm_atomic_syncscope(void) {
  LLVMBuilderRef Builder = LLVMCreateBuilder();

  LLVMModuleRef M = LLVMModuleCreateWithName("Mod");
  LLVMTypeRef FT = LLVMFunctionType(LLVMVoidType(), NULL, 0, 0);
  LLVMValueRef F = LLVMAddFunction(M, "Fun", FT);
  LLVMBasicBlockRef BB = LLVMAppendBasicBlock(F, "Entry");
  LLVMPositionBuilderAtEnd(Builder, BB);

  // echo.cpp already tests the new SyncScope APIs, also test the old ones here

  // fence
  LLVMValueRef Fence =
      LLVMBuildFence(Builder, LLVMAtomicOrderingSequentiallyConsistent, 0, "");
  assert(!LLVMIsAtomicSingleThread(Fence));
  Fence =
      LLVMBuildFence(Builder, LLVMAtomicOrderingSequentiallyConsistent, 1, "");
  assert(LLVMIsAtomicSingleThread(Fence));

  // atomicrmw
  LLVMValueRef Ptr = LLVMConstPointerNull(LLVMPointerType(LLVMInt32Type(), 0));
  LLVMValueRef Val = LLVMConstInt(LLVMInt32Type(), 0, 0);
  LLVMValueRef AtomicRMW =
      LLVMBuildAtomicRMW(Builder, LLVMAtomicRMWBinOpXchg, Ptr, Val,
                         LLVMAtomicOrderingSequentiallyConsistent, 0);
  assert(!LLVMIsAtomicSingleThread(AtomicRMW));
  AtomicRMW = LLVMBuildAtomicRMW(Builder, LLVMAtomicRMWBinOpXchg, Ptr, Val,
                                 LLVMAtomicOrderingSequentiallyConsistent, 1);
  assert(LLVMIsAtomicSingleThread(AtomicRMW));

  // cmpxchg
  LLVMValueRef CmpXchg = LLVMBuildAtomicCmpXchg(
      Builder, Ptr, Val, Val, LLVMAtomicOrderingSequentiallyConsistent,
      LLVMAtomicOrderingSequentiallyConsistent, 0);
  assert(!LLVMIsAtomicSingleThread(CmpXchg));
  CmpXchg = LLVMBuildAtomicCmpXchg(Builder, Ptr, Val, Val,
                                   LLVMAtomicOrderingSequentiallyConsistent,
                                   LLVMAtomicOrderingSequentiallyConsistent, 1);
  assert(LLVMIsAtomicSingleThread(CmpXchg));

  LLVMDisposeBuilder(Builder);
  LLVMDisposeModule(M);

  return 0;
}
