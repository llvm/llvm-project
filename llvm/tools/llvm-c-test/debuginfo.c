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
      LLVMDWARFSourceLanguageC, File,"llvm-c-test", 11, 0, NULL, 0, 0,
      NULL, 0, LLVMDWARFEmissionFull, 0, 0, 0);

  LLVMMetadataRef Int64Ty =
    LLVMDIBuilderCreateBasicType(DIB, "Int64", 5, 64, 0);

  LLVMMetadataRef StructDbgElts[] = {Int64Ty, Int64Ty, Int64Ty};
  LLVMMetadataRef StructDbgTy =
    LLVMDIBuilderCreateStructType(DIB, CompileUnit, "MyStruct",
    8, File, 0, 192, 0, 0, NULL, StructDbgElts, 3,
    LLVMDWARFSourceLanguageC, NULL, "MyStruct", 8);

  LLVMMetadataRef StructDbgPtrTy =
    LLVMDIBuilderCreatePointerType(DIB, StructDbgTy, 192, 0, 0, "", 0);

  LLVMAddNamedMetadataOperand(M, "FooType",
    LLVMMetadataAsValue(LLVMGetModuleContext(M), StructDbgPtrTy));

  char *MStr = LLVMPrintModuleToString(M);
  puts(MStr);
  LLVMDisposeMessage(MStr);

  LLVMDisposeDIBuilder(DIB);
  LLVMDisposeModule(M);

  return 0;
}
