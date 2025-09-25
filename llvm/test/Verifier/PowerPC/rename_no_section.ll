; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@a = global i32 1, !rename !0

!0 = !{}

; CHECK: global value with rename metadata must have section attribute
; CHECK: ptr @a
