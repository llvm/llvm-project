; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

@llvm.compiler.used = appending global [1 x i32] [i32 0], section "llvm.metadata"

; CHECK:       wrong type for intrinsic global variable
; CHECK-NEXT: ptr @llvm.compiler.used
