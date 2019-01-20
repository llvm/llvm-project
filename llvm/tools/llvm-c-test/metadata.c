/*===-- object.c - tool for testing libLLVM and llvm-c API ----------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file implements the --add-named-metadata-operand and --set-metadata   *|
|* commands in llvm-c-test.                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c-test.h"

int llvm_add_named_metadata_operand(void) {
  LLVMModuleRef m = LLVMModuleCreateWithName("Mod");
  LLVMValueRef values[] = { LLVMConstInt(LLVMInt32Type(), 0, 0) };

  // This used to trigger an assertion
  LLVMAddNamedMetadataOperand(m, "name", LLVMMDNode(values, 1));

  LLVMDisposeModule(m);

  return 0;
}

int llvm_set_metadata(void) {
  LLVMBuilderRef b = LLVMCreateBuilder();
  LLVMValueRef values[] = { LLVMConstInt(LLVMInt32Type(), 0, 0) };

  // This used to trigger an assertion
  LLVMSetMetadata(
      LLVMBuildRetVoid(b),
      LLVMGetMDKindID("kind", 4),
      LLVMMDNode(values, 1));

  LLVMDisposeBuilder(b);

  return 0;
}
