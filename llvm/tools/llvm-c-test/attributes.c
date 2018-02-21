/*===-- attributes.c - tool for testing libLLVM and llvm-c API ------------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file implements the --test-attributes and --test-callsite-attributes  *|
|* commands in llvm-c-test.                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c-test.h"

#include <assert.h>
#include <stdlib.h>

int llvm_test_function_attributes(void) {
  LLVMEnablePrettyStackTrace();

  LLVMModuleRef M = llvm_load_module(false, true);

  LLVMValueRef F = LLVMGetFirstFunction(M);
  while (F) {
    // Read attributes
    int Idx, ParamCount;
    for (Idx = LLVMAttributeFunctionIndex, ParamCount = LLVMCountParams(F);
         Idx <= ParamCount; ++Idx) {
      int AttrCount = LLVMGetAttributeCountAtIndex(F, Idx);
      LLVMAttributeRef *Attrs =
          (LLVMAttributeRef *)malloc(AttrCount * sizeof(LLVMAttributeRef));
      assert(Attrs);
      LLVMGetAttributesAtIndex(F, Idx, Attrs);
      free(Attrs);
    }
    F = LLVMGetNextFunction(F);
  }

  LLVMDisposeModule(M);

  return 0;
}

int llvm_test_callsite_attributes(void) {
  LLVMEnablePrettyStackTrace();

  LLVMModuleRef M = llvm_load_module(false, true);

  LLVMValueRef F = LLVMGetFirstFunction(M);
  while (F) {
    LLVMBasicBlockRef BB;
    for (BB = LLVMGetFirstBasicBlock(F); BB; BB = LLVMGetNextBasicBlock(BB)) {
      LLVMValueRef I;
      for (I = LLVMGetFirstInstruction(BB); I; I = LLVMGetNextInstruction(I)) {
        if (LLVMIsACallInst(I)) {
          // Read attributes
          int Idx, ParamCount;
          for (Idx = LLVMAttributeFunctionIndex,
              ParamCount = LLVMCountParams(F);
               Idx <= ParamCount; ++Idx) {
            int AttrCount = LLVMGetCallSiteAttributeCount(I, Idx);
            LLVMAttributeRef *Attrs = (LLVMAttributeRef *)malloc(
                AttrCount * sizeof(LLVMAttributeRef));
            assert(Attrs);
            LLVMGetCallSiteAttributes(I, Idx, Attrs);
            free(Attrs);
          }
        }
      }
    }

    F = LLVMGetNextFunction(F);
  }

  LLVMDisposeModule(M);

  return 0;
}
