//===-------- BasicOrcV2CBindings.c - Basic OrcV2 C Bindings Demo ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/Error.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/LLJIT.h"
#include "llvm-c/Support.h"
#include "llvm-c/Target.h"

#include <assert.h>
#include <stdio.h>

int handleError(LLVMErrorRef Err) {
  char *ErrMsg = LLVMGetErrorMessage(Err);
  fprintf(stderr, "Error: %s\n", ErrMsg);
  LLVMDisposeErrorMessage(ErrMsg);
  return 1;
}

int32_t add(int32_t X, int32_t Y) { return X + Y; }

int32_t mul(int32_t X, int32_t Y) { return X * Y; }

int allowedSymbols(void *Ctx, LLVMOrcSymbolStringPoolEntryRef Sym) {
  assert(Ctx && "Cannot call allowedSymbols with a null context");

  LLVMOrcSymbolStringPoolEntryRef *AllowList =
      (LLVMOrcSymbolStringPoolEntryRef *)Ctx;

  // If Sym appears in the allowed list then return true.
  LLVMOrcSymbolStringPoolEntryRef *P = AllowList;
  while (*P) {
    if (Sym == *P)
      return 1;
    ++P;
  }

  // otherwise return false.
  return 0;
}

LLVMOrcThreadSafeModuleRef createDemoModule(void) {
  // Create a new ThreadSafeContext and underlying LLVMContext.
  LLVMOrcThreadSafeContextRef TSCtx = LLVMOrcCreateNewThreadSafeContext();

  // Get a reference to the underlying LLVMContext.
  LLVMContextRef Ctx = LLVMOrcThreadSafeContextGetContext(TSCtx);

  // Create a new LLVM module.
  LLVMModuleRef M = LLVMModuleCreateWithNameInContext("demo", Ctx);

  // Add a "sum" function":
  //  - Create the function type and function instance.
  LLVMTypeRef I32BinOpParamTypes[] = {LLVMInt32Type(), LLVMInt32Type()};
  LLVMTypeRef I32BinOpFunctionType =
      LLVMFunctionType(LLVMInt32Type(), I32BinOpParamTypes, 2, 0);
  LLVMValueRef AddI32Function = LLVMAddFunction(M, "add", I32BinOpFunctionType);
  LLVMValueRef MulI32Function = LLVMAddFunction(M, "mul", I32BinOpFunctionType);

  LLVMTypeRef MulAddParamTypes[] = {LLVMInt32Type(), LLVMInt32Type(),
                                    LLVMInt32Type()};
  LLVMTypeRef MulAddFunctionType =
      LLVMFunctionType(LLVMInt32Type(), MulAddParamTypes, 3, 0);
  LLVMValueRef MulAddFunction =
      LLVMAddFunction(M, "mul_add", MulAddFunctionType);

  //  - Add a basic block to the function.
  LLVMBasicBlockRef EntryBB = LLVMAppendBasicBlock(MulAddFunction, "entry");

  //  - Add an IR builder and point it at the end of the basic block.
  LLVMBuilderRef Builder = LLVMCreateBuilder();
  LLVMPositionBuilderAtEnd(Builder, EntryBB);

  //  - Get the three function arguments and use them co construct calls to
  //    'mul' and 'add':
  //
  //    i32 mul_add(i32 %0, i32 %1, i32 %2) {
  //      %t = call i32 @mul(i32 %0, i32 %1)
  //      %r = call i32 @add(i32 %t, i32 %2)
  //      ret i32 %r
  //    }
  LLVMValueRef SumArg0 = LLVMGetParam(MulAddFunction, 0);
  LLVMValueRef SumArg1 = LLVMGetParam(MulAddFunction, 1);
  LLVMValueRef SumArg2 = LLVMGetParam(MulAddFunction, 2);

  LLVMValueRef MulArgs[] = {SumArg0, SumArg1};
  LLVMValueRef MulResult = LLVMBuildCall2(Builder, I32BinOpFunctionType,
                                          MulI32Function, MulArgs, 2, "t");

  LLVMValueRef AddArgs[] = {MulResult, SumArg2};
  LLVMValueRef AddResult = LLVMBuildCall2(Builder, I32BinOpFunctionType,
                                          AddI32Function, AddArgs, 2, "r");

  //  - Build the return instruction.
  LLVMBuildRet(Builder, AddResult);

  // Our demo module is now complete. Wrap it and our ThreadSafeContext in a
  // ThreadSafeModule.
  LLVMOrcThreadSafeModuleRef TSM = LLVMOrcCreateNewThreadSafeModule(M, TSCtx);

  // Dispose of our local ThreadSafeContext value. The underlying LLVMContext
  // will be kept alive by our ThreadSafeModule, TSM.
  LLVMOrcDisposeThreadSafeContext(TSCtx);

  // Return the result.
  return TSM;
}

int main(int argc, char *argv[]) {

  int MainResult = 0;

  // Parse command line arguments and initialize LLVM Core.
  LLVMParseCommandLineOptions(argc, (const char **)argv, "");
  LLVMInitializeCore(LLVMGetGlobalPassRegistry());

  // Initialize native target codegen and asm printer.
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();

  // Create the JIT instance.
  LLVMOrcLLJITRef J;
  {
    LLVMErrorRef Err;
    if ((Err = LLVMOrcCreateLLJIT(&J, 0))) {
      MainResult = handleError(Err);
      goto llvm_shutdown;
    }
  }

  // Build a filter to allow JIT'd code to only access allowed symbols.
  // This filter is optional: If a null value is suppled for the Filter
  // argument to LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess then
  // all process symbols will be reflected.
  LLVMOrcSymbolStringPoolEntryRef AllowList[] = {
      LLVMOrcLLJITMangleAndIntern(J, "mul"),
      LLVMOrcLLJITMangleAndIntern(J, "add"), 0};

  {
    LLVMOrcDefinitionGeneratorRef ProcessSymbolsGenerator = 0;
    LLVMErrorRef Err;
    if ((Err = LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
             &ProcessSymbolsGenerator, LLVMOrcLLJITGetGlobalPrefix(J),
             allowedSymbols, AllowList))) {
      MainResult = handleError(Err);
      goto jit_cleanup;
    }

    LLVMOrcJITDylibAddGenerator(LLVMOrcLLJITGetMainJITDylib(J),
                                ProcessSymbolsGenerator);
  }

  // Create our demo module.
  LLVMOrcThreadSafeModuleRef TSM = createDemoModule();

  // Add our demo module to the JIT.
  {
    LLVMOrcJITDylibRef MainJD = LLVMOrcLLJITGetMainJITDylib(J);
    LLVMErrorRef Err;
    if ((Err = LLVMOrcLLJITAddLLVMIRModule(J, MainJD, TSM))) {
      // If adding the ThreadSafeModule fails then we need to clean it up
      // ourselves. If adding it succeeds the JIT will manage the memory.
      LLVMOrcDisposeThreadSafeModule(TSM);
      MainResult = handleError(Err);
      goto jit_cleanup;
    }
  }

  // Look up the address of our demo entry point.
  LLVMOrcJITTargetAddress MulAddAddr;
  {
    LLVMErrorRef Err;
    if ((Err = LLVMOrcLLJITLookup(J, &MulAddAddr, "mul_add"))) {
      MainResult = handleError(Err);
      goto jit_cleanup;
    }
  }

  // If we made it here then everything succeeded. Execute our JIT'd code.
  int32_t (*MulAdd)(int32_t, int32_t, int32_t) =
      (int32_t(*)(int32_t, int32_t, int32_t))MulAddAddr;
  int32_t Result = MulAdd(3, 4, 5);

  // Print the result.
  printf("3 * 4 + 5 = %i\n", Result);

jit_cleanup:
  // Release all symbol string pool entries that we have allocated. In this
  // example that's just our allowed entries.
  {
    LLVMOrcSymbolStringPoolEntryRef *P = AllowList;
    while (*P)
      LLVMOrcReleaseSymbolStringPoolEntry(*P++);
  }

  // Destroy our JIT instance. This will clean up any memory that the JIT has
  // taken ownership of. This operation is non-trivial (e.g. it may need to
  // JIT static destructors) and may also fail. In that case we want to render
  // the error to stderr, but not overwrite any existing return value.
  {
    LLVMErrorRef Err;
    if ((Err = LLVMOrcDisposeLLJIT(J))) {
      int NewFailureResult = handleError(Err);
      if (MainResult == 0)
        MainResult = NewFailureResult;
    }
  }

llvm_shutdown:
  // Shut down LLVM.
  LLVMShutdown();

  return MainResult;
}
