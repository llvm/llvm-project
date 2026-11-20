; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

@GV = dso_local global [10 x x86_amx] zeroinitializer, align 16
; CHECK: invalid array element type
