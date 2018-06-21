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

#include "llvm-c/DebugInfo.h"
#include <stdio.h>

int llvm_test_dibuilder(void) {
  LLVMModuleRef M = LLVMModuleCreateWithName("debuginfo.c");
  LLVMDIBuilderRef DIB = LLVMCreateDIBuilder(M);

  LLVMMetadataRef File = LLVMDIBuilderCreateFile(DIB, "debuginfo.c", 12,
                                                 ".", 1);

  LLVMDIBuilderCreateCompileUnit(DIB,
      LLVMDWARFSourceLanguageC, File,"llvm-c-test", 11, 0, NULL, 0, 0,
      NULL, 0, LLVMDWARFEmissionFull, 0, 0, 0);

  char *MStr = LLVMPrintModuleToString(M);
  puts(MStr);
  LLVMDisposeMessage(MStr);

  LLVMDisposeDIBuilder(DIB);
  LLVMDisposeModule(M);

  return 0;
}
