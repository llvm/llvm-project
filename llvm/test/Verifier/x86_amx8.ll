; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@GV = dso_local global x86_amx* null
; CHECK: pointer to this type is invalid
