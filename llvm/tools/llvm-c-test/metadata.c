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

#include <assert.h>
#include <string.h>

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
  LLVMValueRef ret = LLVMBuildRetVoid(b);
  LLVMSetMetadata(ret, LLVMGetMDKindID("kind", 4), LLVMMDNode(values, 1));

  LLVMDisposeBuilder(b);
  LLVMDeleteInstruction(ret);

  return 0;
}

int llvm_replace_md_operand(void) {
  LLVMModuleRef m = LLVMModuleCreateWithName("Mod");
  LLVMContextRef context = LLVMGetModuleContext(m);
  unsigned int tmp = 0;

  LLVMMetadataRef metas[] = {LLVMMDStringInContext2(context, "foo", 3)};
  LLVMValueRef md =
      LLVMMetadataAsValue(context, LLVMMDNodeInContext2(context, metas, 1));

  LLVMReplaceMDNodeOperandWith(md, 0,
                               LLVMMDStringInContext2(context, "bar", 3));

  assert(!strncmp(LLVMGetMDString(LLVMGetOperand(md, 0), &tmp), "bar", 0));
  (void)tmp;

  LLVMDisposeModule(m);

  return 0;
}

int llvm_is_a_value_as_metadata(void) {
  LLVMModuleRef m = LLVMModuleCreateWithName("Mod");
  LLVMContextRef context = LLVMGetModuleContext(m);

  LLVMValueRef values[] = {LLVMConstInt(LLVMInt32Type(), 0, 0)};
  LLVMValueRef md = LLVMMDNode(values, 1);
  assert(LLVMIsAValueAsMetadata(md) == md);
  (void)md;

  LLVMMetadataRef metas[] = {LLVMMDStringInContext2(context, "foo", 3)};
  LLVMValueRef md2 =
      LLVMMetadataAsValue(context, LLVMMDNodeInContext2(context, metas, 1));
  assert(LLVMIsAValueAsMetadata(md2) == NULL);
  (void)md2;

  return 0;
}
