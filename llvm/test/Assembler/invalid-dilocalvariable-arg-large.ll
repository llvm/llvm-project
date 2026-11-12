; RUN: not llvm-as < %s 2>&1 | FileCheck %s

!0 = !DILocalVariable(scope: !{}, arg: 65535)

; CHECK: <stdin>:[[@LINE+1]]:40: error: value for 'arg' too large, limit is 65535
!1 = !DILocalVariable(scope: !{}, arg: 65536)
