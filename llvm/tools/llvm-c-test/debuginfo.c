/*===-- debuginfo.c - tool for testing libLLVM and llvm-c API -------------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Tests for the LLVM C DebugInfo API                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c-test.h"
#include "llvm-c/Core.h"
#include "llvm-c/DebugInfo.h"
#include <stdio.h>
#include <string.h>

int llvm_test_dibuilder(void) {
  const char *Filename = "debuginfo.c";
  LLVMModuleRef M = LLVMModuleCreateWithName(Filename);
  LLVMDIBuilderRef DIB = LLVMCreateDIBuilder(M);

  LLVMMetadataRef File = LLVMDIBuilderCreateFile(DIB, Filename,
    strlen(Filename), ".", 1);

  LLVMMetadataRef CompileUnit = LLVMDIBuilderCreateCompileUnit(DIB,
    LLVMDWARFSourceLanguageC, File, "llvm-c-test", 11, 0, NULL, 0, 0,
    NULL, 0, LLVMDWARFEmissionFull, 0, 0, 0);

  LLVMMetadataRef Module =
    LLVMDIBuilderCreateModule(DIB, CompileUnit,
                              "llvm-c-test", 11,
                              "", 0,
                              "/test/include/llvm-c-test.h", 27,
                              "", 0);

  LLVMMetadataRef OtherModule =
    LLVMDIBuilderCreateModule(DIB, CompileUnit,
                              "llvm-c-test-import", 18,
                              "", 0,
                              "/test/include/llvm-c-test-import.h", 34,
                              "", 0);
  LLVMMetadataRef ImportedModule =
    LLVMDIBuilderCreateImportedModuleFromModule(DIB, Module, OtherModule,
                                                File, 42);
  LLVMDIBuilderCreateImportedModuleFromAlias(DIB, Module, ImportedModule,
                                             File, 42);

  LLVMMetadataRef Int64Ty =
    LLVMDIBuilderCreateBasicType(DIB, "Int64", 5, 64, 0);
  LLVMMetadataRef GlobalVarValueExpr =
    LLVMDIBuilderCreateConstantValueExpression(DIB, 0);
  LLVMDIBuilderCreateGlobalVariableExpression(DIB, Module, "global", 6,
                                              "", 0, File, 1, Int64Ty,
                                              true, GlobalVarValueExpr,
                                              NULL, 0);

  LLVMMetadataRef NameSpace =
    LLVMDIBuilderCreateNameSpace(DIB, Module, "NameSpace", 9, false);

  LLVMMetadataRef StructDbgElts[] = {Int64Ty, Int64Ty, Int64Ty};
  LLVMMetadataRef StructDbgTy =
    LLVMDIBuilderCreateStructType(DIB, NameSpace, "MyStruct",
    8, File, 0, 192, 0, 0, NULL, StructDbgElts, 3,
    LLVMDWARFSourceLanguageC, NULL, "MyStruct", 8);

  LLVMMetadataRef StructDbgPtrTy =
    LLVMDIBuilderCreatePointerType(DIB, StructDbgTy, 192, 0, 0, "", 0);

  LLVMAddNamedMetadataOperand(M, "FooType",
    LLVMMetadataAsValue(LLVMGetModuleContext(M), StructDbgPtrTy));


  LLVMTypeRef FooParamTys[] = {
    LLVMInt64Type(),
    LLVMInt64Type(),
    LLVMVectorType(LLVMInt64Type(), 10),
  };
  LLVMTypeRef FooFuncTy = LLVMFunctionType(LLVMInt64Type(), FooParamTys, 3, 0);
  LLVMValueRef FooFunction = LLVMAddFunction(M, "foo", FooFuncTy);
  LLVMBasicBlockRef FooEntryBlock = LLVMAppendBasicBlock(FooFunction, "entry");

  LLVMMetadataRef Subscripts[] = {
    LLVMDIBuilderGetOrCreateSubrange(DIB, 0, 10),
  };
  LLVMMetadataRef VectorTy =
    LLVMDIBuilderCreateVectorType(DIB, 64 * 10, 0,
                                  Int64Ty, Subscripts, 1);


  LLVMMetadataRef ParamTypes[] = {Int64Ty, Int64Ty, VectorTy};
  LLVMMetadataRef FunctionTy =
    LLVMDIBuilderCreateSubroutineType(DIB, File, ParamTypes, 3, 0);
  LLVMMetadataRef FunctionMetadata =
    LLVMDIBuilderCreateFunction(DIB, File, "foo", 3, "foo", 3,
                                File, 42, FunctionTy, true, true,
                                42, 0, false);

  LLVMMetadataRef FooParamLocation =
    LLVMDIBuilderCreateDebugLocation(LLVMGetGlobalContext(), 42, 0,
                                     FunctionMetadata, NULL);
  LLVMMetadataRef FooParamExpression =
    LLVMDIBuilderCreateExpression(DIB, NULL, 0);
  LLVMMetadataRef FooParamVar1 =
    LLVMDIBuilderCreateParameterVariable(DIB, FunctionMetadata, "a", 1, 1, File,
                                         42, Int64Ty, true, 0);
  LLVMDIBuilderInsertDeclareAtEnd(DIB, LLVMConstInt(LLVMInt64Type(), 0, false),
                                  FooParamVar1, FooParamExpression,
                                  FooParamLocation, FooEntryBlock);
  LLVMMetadataRef FooParamVar2 =
    LLVMDIBuilderCreateParameterVariable(DIB, FunctionMetadata, "b", 1, 2, File,
                                         42, Int64Ty, true, 0);
  LLVMDIBuilderInsertDeclareAtEnd(DIB, LLVMConstInt(LLVMInt64Type(), 0, false),
                                  FooParamVar2, FooParamExpression,
                                  FooParamLocation, FooEntryBlock);
  LLVMMetadataRef FooParamVar3 =
    LLVMDIBuilderCreateParameterVariable(DIB, FunctionMetadata, "c", 1, 3, File,
                                         42, VectorTy, true, 0);
  LLVMDIBuilderInsertDeclareAtEnd(DIB, LLVMConstInt(LLVMInt64Type(), 0, false),
                                  FooParamVar3, FooParamExpression,
                                  FooParamLocation, FooEntryBlock);

  LLVMSetSubprogram(FooFunction, FunctionMetadata);

  LLVMMetadataRef FooLexicalBlock =
    LLVMDIBuilderCreateLexicalBlock(DIB, FunctionMetadata, File, 42, 0);

  LLVMBasicBlockRef FooVarBlock = LLVMAppendBasicBlock(FooFunction, "vars");
  LLVMMetadataRef FooVarsLocation =
    LLVMDIBuilderCreateDebugLocation(LLVMGetGlobalContext(), 43, 0,
                                     FunctionMetadata, NULL);
  LLVMMetadataRef FooVar1 =
    LLVMDIBuilderCreateAutoVariable(DIB, FooLexicalBlock, "d", 1, File,
                                    43, Int64Ty, true, 0, 0);
  LLVMValueRef FooVal1 = LLVMConstInt(LLVMInt64Type(), 0, false);
  LLVMMetadataRef FooVarValueExpr =
    LLVMDIBuilderCreateConstantValueExpression(DIB, 0);

  LLVMDIBuilderInsertDbgValueAtEnd(DIB, FooVal1, FooVar1, FooVarValueExpr,
                                   FooVarsLocation, FooVarBlock);

  LLVMDIBuilderFinalize(DIB);

  char *MStr = LLVMPrintModuleToString(M);
  puts(MStr);
  LLVMDisposeMessage(MStr);

  LLVMDisposeDIBuilder(DIB);
  LLVMDisposeModule(M);

  return 0;
}
