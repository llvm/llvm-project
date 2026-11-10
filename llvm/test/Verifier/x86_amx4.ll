; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

@GV = external global x86_amx
; CHECK: invalid type for global variable
