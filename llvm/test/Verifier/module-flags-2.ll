; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!llvm.module.flags = !{!0}
!0 = !{null, null, null}

; CHECK: invalid behavior operand in module flag (expected constant integer)
