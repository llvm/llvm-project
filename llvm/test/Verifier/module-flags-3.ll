; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!llvm.module.flags = !{!0}
!0 = !{i32 1, null, null}

; CHECK: invalid ID operand in module flag (expected metadata string)
