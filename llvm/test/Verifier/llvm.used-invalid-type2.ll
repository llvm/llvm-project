; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
@llvm.used = appending global i32 0, section "llvm.metadata"

; CHECK: Only global arrays can have appending linkage!
; CHECK-NEXT: i32* @llvm.used
