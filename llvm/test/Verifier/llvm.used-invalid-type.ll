; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

@llvm.used = appending global [1 x i32] [i32 0], section "llvm.metadata"

; CHECK:       wrong type for intrinsic global variable
; CHECK-NEXT: ptr @llvm.used
