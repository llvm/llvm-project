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
#include "llvm-c/Types.h"

#include <assert.h>
#include <string.h>

int llvm_add_named_metadata_operand(void) {
  LLVMContextRef C = LLVMContextCreate();
  LLVMModuleRef M = LLVMModuleCreateWithNameInContext("Mod", C);
  LLVMValueRef Int = LLVMConstInt(LLVMInt32TypeInContext(C), 0, 0);

  // This used to trigger an assertion
  LLVMAddNamedMetadataOperand(M, "name", LLVMMDNodeInContext(C, &Int, 1));

  LLVMDisposeModule(M);
  LLVMContextDispose(C);

  return 0;
}

int llvm_set_metadata(void) {
  LLVMContextRef C = LLVMContextCreate();
  LLVMBuilderRef Builder = LLVMCreateBuilderInContext(C);

  // This used to trigger an assertion
  LLVMValueRef Return = LLVMBuildRetVoid(Builder);

  const char Name[] = "kind";
  LLVMValueRef Int = LLVMConstInt(LLVMInt32TypeInContext(C), 0, 0);
  LLVMSetMetadata(Return, LLVMGetMDKindIDInContext(C, Name, strlen(Name)),
                  LLVMMDNodeInContext(C, &Int, 1));

  LLVMDisposeBuilder(Builder);
  LLVMDeleteInstruction(Return);
  LLVMContextDispose(C);

  return 0;
}

int llvm_replace_md_operand(void) {
  LLVMContextRef Context = LLVMContextCreate();
  LLVMModuleRef M = LLVMModuleCreateWithNameInContext("Mod", Context);

  const char String1[] = "foo";
  LLVMMetadataRef String1MD =
      LLVMMDStringInContext2(Context, String1, strlen(String1));
  LLVMMetadataRef NodeMD = LLVMMDNodeInContext2(Context, &String1MD, 1);
  LLVMValueRef Value = LLVMMetadataAsValue(Context, NodeMD);

  const char String2[] = "bar";
  LLVMMetadataRef String2MD =
      LLVMMDStringInContext2(Context, String2, strlen(String2));
  LLVMReplaceMDNodeOperandWith(Value, 0, String2MD);

  LLVMValueRef Operand = LLVMGetOperand(Value, 0);

  unsigned int Len;
  const char *String = LLVMGetMDString(Operand, &Len);
  assert(Len == strlen(String2));
  assert(strncmp(String, String2, Len) == 0);
  (void)String;

  LLVMDisposeModule(M);
  LLVMContextDispose(Context);

  return 0;
}

int llvm_is_a_value_as_metadata(void) {
  LLVMContextRef Context = LLVMContextCreate();
  LLVMModuleRef M = LLVMModuleCreateWithNameInContext("Mod", Context);

  {
    LLVMValueRef Int = LLVMConstInt(LLVMInt32TypeInContext(Context), 0, 0);
    LLVMValueRef NodeMD = LLVMMDNodeInContext(Context, &Int, 1);
    assert(LLVMIsAValueAsMetadata(NodeMD) == NodeMD);
    (void)NodeMD;
  }

  {
    const char String[] = "foo";
    LLVMMetadataRef StringMD =
        LLVMMDStringInContext2(Context, String, strlen(String));
    LLVMMetadataRef NodeMD = LLVMMDNodeInContext2(Context, &StringMD, 1);
    LLVMValueRef Value = LLVMMetadataAsValue(Context, NodeMD);
    assert(LLVMIsAValueAsMetadata(Value) == NULL);
    (void)Value;
  }

  LLVMDisposeModule(M);
  LLVMContextDispose(Context);

  return 0;
}
