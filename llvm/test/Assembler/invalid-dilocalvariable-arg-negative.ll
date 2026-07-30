; RUN: not llvm-as < %s 2>&1 | FileCheck %s

!0 = !DILocalVariable(scope: !{}, arg: 1)
!1 = !DILocalVariable(scope: !{})

; CHECK: <stdin>:[[@LINE+1]]:40: error: expected unsigned integer
!2 = !DILocalVariable(scope: !{}, arg: -1)
