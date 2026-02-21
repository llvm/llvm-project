; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@a = global i32 1, section "abc", !rename !0

!0 = !{!"Hello World!"}
; CHECK: rename metadata must have no operands
; CHECK: ptr @a
; CHECK: !0 = !{!"Hello World!"}
