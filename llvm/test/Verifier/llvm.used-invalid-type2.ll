; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
@llvm.used = appending global i32 0, section "llvm.metadata"

; CHECK: Only global arrays can have appending linkage!
; CHECK-NEXT: ptr @llvm.used
