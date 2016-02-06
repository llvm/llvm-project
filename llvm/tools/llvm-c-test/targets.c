/*===-- targets.c - tool for testing libLLVM and llvm-c API ---------------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file implements the --targets command in llvm-c-test.                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/TargetMachine.h"
#include <stdio.h>

int llvm_targets_list(void) {
  LLVMTargetRef t;
  LLVMInitializeAllTargetInfos();
  LLVMInitializeAllTargets();

  for (t = LLVMGetFirstTarget(); t; t = LLVMGetNextTarget(t)) {
    printf("%s", LLVMGetTargetName(t));
    if (LLVMTargetHasJIT(t))
      printf(" (+jit)");
    printf("\n - %s\n", LLVMGetTargetDescription(t));
  }

  return 0;
}
