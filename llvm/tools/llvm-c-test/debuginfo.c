/*===-- debuginfo.c - tool for testing libLLVM and llvm-c API -------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Tests for the LLVM C DebugInfo API                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/DebugInfo.h"
#include "llvm-c-test.h"
#include "llvm-c/Core.h"
#include "llvm-c/Types.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static LLVMMetadataRef
declare_objc_class(LLVMDIBuilderRef DIB, LLVMMetadataRef File) {
  LLVMMetadataRef Decl = LLVMDIBuilderCreateStructType(DIB, File, "TestClass", 9, File, 42, 64, 0, LLVMDIFlagObjcClassComplete, NULL, NULL, 0, 0, NULL, NULL, 0);
  LLVMMetadataRef SuperDecl = LLVMDIBuilderCreateStructType(DIB, File, "TestSuperClass", 14, File, 42, 64, 0, LLVMDIFlagObjcClassComplete, NULL, NULL, 0, 0, NULL, NULL, 0);
  LLVMDIBuilderCreateInheritance(DIB, Decl, SuperDecl, 0, 0, 0);
  LLVMMetadataRef TestProperty =
      LLVMDIBuilderCreateObjCProperty(DIB, "test", 4, File, 42, "getTest", 7, "setTest", 7, 0x20 /*copy*/ | 0x40 /*nonatomic*/, SuperDecl);
  LLVMDIBuilderCreateObjCIVar(DIB, "_test", 5, File, 42, 64, 0, 64, LLVMDIFlagPublic, SuperDecl, TestProperty);
  return Decl;
}

int llvm_test_dibuilder(void) {
  const char *Filename = "debuginfo.c";
  LLVMModuleRef M = LLVMModuleCreateWithName(Filename);

  LLVMSetIsNewDbgInfoFormat(M, true);
  assert(LLVMIsNewDbgInfoFormat(M));

  LLVMDIBuilderRef DIB = LLVMCreateDIBuilder(M);

  LLVMMetadataRef File = LLVMDIBuilderCreateFile(DIB, Filename,
    strlen(Filename), ".", 1);

  LLVMMetadataRef CompileUnit = LLVMDIBuilderCreateCompileUnit(
      DIB, LLVMDWARFSourceLanguageC, File, "llvm-c-test", 11, 0, NULL, 0, 0,
      NULL, 0, LLVMDWARFEmissionFull, 0, 0, 0, "/", 1, "", 0);

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
  LLVMMetadataRef ImportedModule = LLVMDIBuilderCreateImportedModuleFromModule(
      DIB, Module, OtherModule, File, 42, NULL, 0);
  LLVMDIBuilderCreateImportedModuleFromAlias(DIB, Module, ImportedModule, File,
                                             42, NULL, 0);

  LLVMMetadataRef ClassTy = declare_objc_class(DIB, File);
  LLVMMetadataRef GlobalClassValueExpr =
      LLVMDIBuilderCreateConstantValueExpression(DIB, 0);
  LLVMDIBuilderCreateGlobalVariableExpression(
      DIB, Module, "globalClass", 11, "", 0, File, 1, ClassTy, true,
      GlobalClassValueExpr, NULL, 0);

  LLVMMetadataRef Int64Ty =
      LLVMDIBuilderCreateBasicType(DIB, "Int64", 5, 64, 0, LLVMDIFlagZero);
  LLVMMetadataRef Int64TypeDef =
      LLVMDIBuilderCreateTypedef(DIB, Int64Ty, "int64_t", 7, File, 42, File, 0);

  LLVMMetadataRef GlobalVarValueExpr =
      LLVMDIBuilderCreateConstantValueExpression(DIB, 0);
  LLVMDIBuilderCreateGlobalVariableExpression(
      DIB, Module, "global", 6, "", 0, File, 1, Int64TypeDef, true,
      GlobalVarValueExpr, NULL, 0);

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

  LLVMMetadataRef ReplaceableFunctionMetadata =
    LLVMDIBuilderCreateReplaceableCompositeType(DIB, 0x15, "foo", 3,
                                                File, File, 42,
                                                0, 0, 0,
                                                LLVMDIFlagFwdDecl,
                                                "", 0);

  LLVMMetadataRef FooParamLocation =
    LLVMDIBuilderCreateDebugLocation(LLVMGetGlobalContext(), 42, 0,
                                     ReplaceableFunctionMetadata, NULL);
  LLVMMetadataRef FunctionMetadata = LLVMDIBuilderCreateFunction(
      DIB, File, "foo", 3, "foo", 3, File, 42, NULL, true, true, 42, 0, false);
  LLVMMetadataReplaceAllUsesWith(ReplaceableFunctionMetadata, FunctionMetadata);

  LLVMDISubprogramReplaceType(FunctionMetadata, FunctionTy);

  LLVMMetadataRef FooParamExpression =
    LLVMDIBuilderCreateExpression(DIB, NULL, 0);
  LLVMMetadataRef FooParamVar1 =
    LLVMDIBuilderCreateParameterVariable(DIB, FunctionMetadata, "a", 1, 1, File,
                                         42, Int64Ty, true, 0);

  LLVMDIBuilderInsertDeclareRecordAtEnd(
      DIB, LLVMConstInt(LLVMInt64Type(), 0, false), FooParamVar1,
      FooParamExpression, FooParamLocation, FooEntryBlock);

  LLVMMetadataRef FooParamVar2 =
    LLVMDIBuilderCreateParameterVariable(DIB, FunctionMetadata, "b", 1, 2, File,
                                         42, Int64Ty, true, 0);

  LLVMDIBuilderInsertDeclareRecordAtEnd(
      DIB, LLVMConstInt(LLVMInt64Type(), 0, false), FooParamVar2,
      FooParamExpression, FooParamLocation, FooEntryBlock);

  LLVMMetadataRef FooParamVar3 = LLVMDIBuilderCreateParameterVariable(
      DIB, FunctionMetadata, "c", 1, 3, File, 42, VectorTy, true, 0);

  LLVMDIBuilderInsertDeclareRecordAtEnd(
      DIB, LLVMConstInt(LLVMInt64Type(), 0, false), FooParamVar3,
      FooParamExpression, FooParamLocation, FooEntryBlock);

  LLVMSetSubprogram(FooFunction, FunctionMetadata);

  LLVMMetadataRef FooLabel1 = LLVMDIBuilderCreateLabel(DIB, FunctionMetadata,
    "label1", 6, File, 42, false);
  LLVMDIBuilderInsertLabelAtEnd(DIB, FooLabel1, FooParamLocation,
    FooEntryBlock);

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
  LLVMMetadataRef FooVarValueExpr1 =
      LLVMDIBuilderCreateConstantValueExpression(DIB, 0);

  LLVMDIBuilderInsertDbgValueRecordAtEnd(
      DIB, FooVal1, FooVar1, FooVarValueExpr1, FooVarsLocation, FooVarBlock);

  LLVMMetadataRef FooVar2 = LLVMDIBuilderCreateAutoVariable(
      DIB, FooLexicalBlock, "e", 1, File, 44, Int64Ty, true, 0, 0);
  LLVMValueRef FooVal2 = LLVMConstInt(LLVMInt64Type(), 1, false);
  LLVMMetadataRef FooVarValueExpr2 =
      LLVMDIBuilderCreateConstantValueExpression(DIB, 1);

  LLVMDIBuilderInsertDbgValueRecordAtEnd(
      DIB, FooVal2, FooVar2, FooVarValueExpr2, FooVarsLocation, FooVarBlock);

  LLVMMetadataRef MacroFile =
      LLVMDIBuilderCreateTempMacroFile(DIB, NULL, 0, File);
  LLVMDIBuilderCreateMacro(DIB, MacroFile, 0, LLVMDWARFMacinfoRecordTypeDefine,
                           "SIMPLE_DEFINE", 13, NULL, 0);
  LLVMDIBuilderCreateMacro(DIB, MacroFile, 0, LLVMDWARFMacinfoRecordTypeDefine,
                           "VALUE_DEFINE", 12, "1", 1);

  LLVMMetadataRef EnumeratorTestA =
      LLVMDIBuilderCreateEnumerator(DIB, "Test_A", strlen("Test_A"), 0, true);
  LLVMMetadataRef EnumeratorTestB =
      LLVMDIBuilderCreateEnumerator(DIB, "Test_B", strlen("Test_B"), 1, true);
  LLVMMetadataRef EnumeratorTestC =
      LLVMDIBuilderCreateEnumerator(DIB, "Test_B", strlen("Test_C"), 2, true);
  LLVMMetadataRef EnumeratorsTest[] = {EnumeratorTestA, EnumeratorTestB,
                                       EnumeratorTestC};
  LLVMMetadataRef EnumTest = LLVMDIBuilderCreateEnumerationType(
      DIB, NameSpace, "EnumTest", strlen("EnumTest"), File, 0, 64, 0,
      EnumeratorsTest, 3, Int64Ty);
  LLVMAddNamedMetadataOperand(
      M, "EnumTest", LLVMMetadataAsValue(LLVMGetModuleContext(M), EnumTest));

  LLVMMetadataRef UInt128Ty = LLVMDIBuilderCreateBasicType(
      DIB, "UInt128", strlen("UInt128"), 128, 0, LLVMDIFlagZero);
  const uint64_t WordsTestD[] = {0x098a224000000000ull, 0x4b3b4ca85a86c47aull};
  const uint64_t WordsTestE[] = {0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull};

  LLVMMetadataRef LargeEnumeratorTestD =
      LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision(
          DIB, "Test_D", strlen("Test_D"), 128, WordsTestD, false);
  LLVMMetadataRef LargeEnumeratorTestE =
      LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision(
          DIB, "Test_E", strlen("Test_E"), 128, WordsTestE, false);
  LLVMMetadataRef LargeEnumeratorsTest[] = {LargeEnumeratorTestD,
                                            LargeEnumeratorTestE};
  LLVMMetadataRef LargeEnumTest = LLVMDIBuilderCreateEnumerationType(
      DIB, NameSpace, "LargeEnumTest", strlen("LargeEnumTest"), File, 0, 128, 0,
      LargeEnumeratorsTest, 2, UInt128Ty);
  LLVMAddNamedMetadataOperand(
      M, "LargeEnumTest",
      LLVMMetadataAsValue(LLVMGetModuleContext(M), LargeEnumTest));

  LLVMValueRef FooVal3 = LLVMConstInt(LLVMInt64Type(), 8, false);
  LLVMValueRef FooVal4 = LLVMConstInt(LLVMInt64Type(), 4, false);
  LLVMMetadataRef lo = LLVMValueAsMetadata(FooVal1);
  LLVMMetadataRef hi = LLVMValueAsMetadata(FooVal2);
  LLVMMetadataRef strd = LLVMValueAsMetadata(FooVal3);
  LLVMMetadataRef bias = LLVMValueAsMetadata(FooVal4);
  LLVMMetadataRef SubrangeMetadataTy = LLVMDIBuilderCreateSubrangeType(
      DIB, File, "foo", 3, 42, File, 64, 0, 0, Int64Ty, lo, hi, strd, bias);
  LLVMAddNamedMetadataOperand(
      M, "SubrangeType",
      LLVMMetadataAsValue(LLVMGetModuleContext(M), SubrangeMetadataTy));

  LLVMMetadataRef SetMetadataTy1 = LLVMDIBuilderCreateSetType(
      DIB, File, "enumset", 7, File, 42, 64, 0, EnumTest);
  LLVMMetadataRef SetMetadataTy2 = LLVMDIBuilderCreateSetType(
      DIB, File, "subrangeset", 11, File, 42, 64, 0, SubrangeMetadataTy);
  LLVMAddNamedMetadataOperand(
      M, "SetType1",
      LLVMMetadataAsValue(LLVMGetModuleContext(M), SetMetadataTy1));
  LLVMAddNamedMetadataOperand(
      M, "SetType2",
      LLVMMetadataAsValue(LLVMGetModuleContext(M), SetMetadataTy2));

  LLVMMetadataRef DynSubscripts[] = {
      LLVMDIBuilderGetOrCreateSubrange(DIB, 0, 10),
  };
  LLVMMetadataRef Loc = LLVMDIBuilderCreateExpression(DIB, NULL, 0);
  LLVMMetadataRef Rank = LLVMDIBuilderCreateExpression(DIB, NULL, 0);
  LLVMMetadataRef DynamicArrayMetadataTy = LLVMDIBuilderCreateDynamicArrayType(
      DIB, File, "foo", 3, 42, File, 64 * 10, 0, Int64Ty, DynSubscripts, 1, Loc,
      FooVar1, NULL, Rank, NULL);
  LLVMAddNamedMetadataOperand(
      M, "DynType",
      LLVMMetadataAsValue(LLVMGetModuleContext(M), DynamicArrayMetadataTy));

  LLVMMetadataRef StructPTy = LLVMDIBuilderCreateForwardDecl(
      DIB, 2 /*DW_TAG_class_type*/, "Class1", 5, NameSpace, File, 0, 0, 192, 0,
      "FooClass", 8);

  LLVMMetadataRef Int32Ty =
      LLVMDIBuilderCreateBasicType(DIB, "Int32", 5, 32, 0, LLVMDIFlagZero);
  LLVMMetadataRef StructElts[] = {Int64Ty, Int64Ty, Int32Ty};
  LLVMMetadataRef ClassArr = LLVMDIBuilderGetOrCreateArray(DIB, StructElts, 3);
  LLVMReplaceArrays(DIB, &StructPTy, &ClassArr, 1);
  LLVMAddNamedMetadataOperand(
      M, "ClassType", LLVMMetadataAsValue(LLVMGetModuleContext(M), StructPTy));

  // Using the new debug format, debug records get attached to instructions.
  // Insert a `br` and `ret` now to absorb the debug records which are
  // currently "trailing", meaning that they're associated with a block
  // but no particular instruction, which is only valid as a transient state.
  LLVMContextRef Ctx = LLVMGetModuleContext(M);
  LLVMBuilderRef Builder = LLVMCreateBuilderInContext(Ctx);
  LLVMPositionBuilderAtEnd(Builder, FooEntryBlock);
  // Build `br label %vars` in entry.
  LLVMBuildBr(Builder, FooVarBlock);

  // Build another br for the sake of testing labels.
  LLVMMetadataRef FooLabel2 = LLVMDIBuilderCreateLabel(DIB, FunctionMetadata,
    "label2", 6, File, 42, false);
  LLVMDIBuilderInsertLabelBefore(DIB, FooLabel2, FooParamLocation,
    LLVMBuildBr(Builder, FooVarBlock));
  // label3 will be emitted, but label4 won't be emitted
  // because label3 is AlwaysPreserve and label4 is not.
  LLVMDIBuilderCreateLabel(DIB, FunctionMetadata,
    "label3", 6, File, 42, true);
  LLVMDIBuilderCreateLabel(DIB, FunctionMetadata,
    "label4", 6, File, 42, false);
  LLVMDIBuilderFinalize(DIB);

  // Build `ret i64 0` in vars.
  LLVMPositionBuilderAtEnd(Builder, FooVarBlock);
  LLVMTypeRef I64 = LLVMInt64TypeInContext(Ctx);
  LLVMValueRef Zero = LLVMConstInt(I64, 0, false);
  LLVMValueRef Ret = LLVMBuildRet(Builder, Zero);

  // Insert a `phi` before the `ret`. In the new debug info mode we need to
  // be careful to insert before debug records too, else the debug records
  // will come before the `phi` (and be absorbed onto it) which is an invalid
  // state.
  LLVMValueRef InsertPos = LLVMGetFirstInstruction(FooVarBlock);
  LLVMPositionBuilderBeforeInstrAndDbgRecords(Builder, InsertPos);
  LLVMValueRef Phi1 = LLVMBuildPhi(Builder, I64, "p1");
  LLVMAddIncoming(Phi1, &Zero, &FooEntryBlock, 1);

  // Do the same again using the other position-setting function.
  LLVMPositionBuilderBeforeDbgRecords(Builder, FooVarBlock, InsertPos);
  LLVMValueRef Phi2 = LLVMBuildPhi(Builder, I64, "p2");
  LLVMAddIncoming(Phi2, &Zero, &FooEntryBlock, 1);

  // Test that LLVMGetFirstDbgRecord and LLVMGetLastDbgRecord return NULL for
  // instructions without debug info.
  LLVMDbgRecordRef Phi1FirstDbgRecord = LLVMGetFirstDbgRecord(Phi1);
  (void)Phi1FirstDbgRecord;
  assert(Phi1FirstDbgRecord == NULL);
  LLVMDbgRecordRef Phi1LastDbgRecord = LLVMGetLastDbgRecord(Phi1);
  (void)Phi1LastDbgRecord;
  assert(Phi1LastDbgRecord == NULL);

  // Insert a non-phi before the `ret` but not before the debug records to
  // test that works as expected.
  LLVMPositionBuilder(Builder, FooVarBlock, Ret);
  LLVMValueRef Add = LLVMBuildAdd(Builder, Phi1, Phi2, "a");

  // Iterate over debug records in the add instruction. There should be two.
  LLVMDbgRecordRef AddDbgRecordFirst = LLVMGetFirstDbgRecord(Add);
  assert(AddDbgRecordFirst != NULL);
  LLVMDbgRecordRef AddDbgRecordSecond = LLVMGetNextDbgRecord(AddDbgRecordFirst);
  assert(AddDbgRecordSecond != NULL);
  LLVMDbgRecordRef AddDbgRecordLast = LLVMGetLastDbgRecord(Add);
  assert(AddDbgRecordLast != NULL);
  (void)AddDbgRecordLast;
  assert(AddDbgRecordSecond == AddDbgRecordLast);
  LLVMDbgRecordRef AddDbgRecordOverTheRange =
      LLVMGetNextDbgRecord(AddDbgRecordSecond);
  assert(AddDbgRecordOverTheRange == NULL);
  (void)AddDbgRecordOverTheRange;
  LLVMDbgRecordRef AddDbgRecordFirstPrev =
      LLVMGetPreviousDbgRecord(AddDbgRecordSecond);
  assert(AddDbgRecordFirstPrev != NULL);
  assert(AddDbgRecordFirst == AddDbgRecordFirstPrev);
  LLVMDbgRecordRef AddDbgRecordUnderTheRange =
      LLVMGetPreviousDbgRecord(AddDbgRecordFirstPrev);
  assert(AddDbgRecordUnderTheRange == NULL);
  (void)AddDbgRecordUnderTheRange;

  char *MStr = LLVMPrintModuleToString(M);
  puts(MStr);
  LLVMDisposeMessage(MStr);

  LLVMDisposeBuilder(Builder);
  LLVMDisposeDIBuilder(DIB);
  LLVMDisposeModule(M);

  return 0;
}

int llvm_get_di_tag(void) {
  LLVMModuleRef M = LLVMModuleCreateWithName("Mod");
  LLVMContextRef Context = LLVMGetModuleContext(M);

  const char String[] = "foo";
  LLVMMetadataRef StringMD =
      LLVMMDStringInContext2(Context, String, strlen(String));
  LLVMMetadataRef NodeMD = LLVMMDNodeInContext2(Context, &StringMD, 1);
  assert(LLVMGetDINodeTag(NodeMD) == 0);
  (void)NodeMD;

  LLVMDIBuilderRef Builder = LLVMCreateDIBuilder(M);
  const char Filename[] = "metadata.c";
  const char Directory[] = ".";
  LLVMMetadataRef File = LLVMDIBuilderCreateFile(
      Builder, Filename, strlen(Filename), Directory, strlen(Directory));
  const char Name[] = "TestClass";
  LLVMMetadataRef Struct = LLVMDIBuilderCreateStructType(
      Builder, File, Name, strlen(Name), File, 42, 64, 0,
      LLVMDIFlagObjcClassComplete, NULL, NULL, 0, 0, NULL, NULL, 0);
  assert(LLVMGetDINodeTag(Struct) == 0x13);
  (void)Struct;

  LLVMDisposeDIBuilder(Builder);
  LLVMDisposeModule(M);

  return 0;
}

int llvm_di_type_get_name(void) {
  LLVMModuleRef M = LLVMModuleCreateWithName("Mod");

  LLVMDIBuilderRef Builder = LLVMCreateDIBuilder(M);
  const char Filename[] = "metadata.c";
  const char Directory[] = ".";
  LLVMMetadataRef File = LLVMDIBuilderCreateFile(
      Builder, Filename, strlen(Filename), Directory, strlen(Directory));
  const char Name[] = "TestClass";
  LLVMMetadataRef Struct = LLVMDIBuilderCreateStructType(
      Builder, File, Name, strlen(Name), File, 42, 64, 0,
      LLVMDIFlagObjcClassComplete, NULL, NULL, 0, 0, NULL, NULL, 0);

  size_t Len;
  const char *TypeName = LLVMDITypeGetName(Struct, &Len);
  assert(Len == strlen(Name));
  assert(strncmp(TypeName, Name, Len) == 0);
  (void)TypeName;

  LLVMDisposeDIBuilder(Builder);
  LLVMDisposeModule(M);

  return 0;
}

int llvm_add_globaldebuginfo(void) {
  const char *Filename = "debuginfo.c";
  LLVMModuleRef M = LLVMModuleCreateWithName(Filename);
  LLVMDIBuilderRef Builder = LLVMCreateDIBuilder(M);
  LLVMMetadataRef File =
      LLVMDIBuilderCreateFile(Builder, Filename, strlen(Filename), ".", 1);

  LLVMMetadataRef GlobalVarValueExpr =
      LLVMDIBuilderCreateConstantValueExpression(Builder, 0);
  LLVMMetadataRef Int64Ty =
      LLVMDIBuilderCreateBasicType(Builder, "Int64", 5, 64, 0, LLVMDIFlagZero);
  LLVMMetadataRef Int64TypeDef = LLVMDIBuilderCreateTypedef(
      Builder, Int64Ty, "int64_t", 7, File, 42, File, 0);

  LLVMMetadataRef GVE = LLVMDIBuilderCreateGlobalVariableExpression(
      Builder, File, "global", 6, "", 0, File, 1, Int64TypeDef, true,
      GlobalVarValueExpr, NULL, 0);

  LLVMTypeRef RecType =
      LLVMStructCreateNamed(LLVMGetModuleContext(M), "struct");
  LLVMValueRef Global = LLVMAddGlobal(M, RecType, "global");

  LLVMGlobalAddDebugInfo(Global, GVE);
  // use AddMetadata to add twice
  int kindId = LLVMGetMDKindID("dbg", 3);
  LLVMGlobalAddMetadata(Global, kindId, GVE);
  size_t numEntries;
  LLVMValueMetadataEntry *ME = LLVMGlobalCopyAllMetadata(Global, &numEntries);
  assert(ME != NULL);
  assert(numEntries == 2);

  LLVMDisposeValueMetadataEntries(ME);
  LLVMDisposeDIBuilder(Builder);
  LLVMDisposeModule(M);

  return 0;
}
