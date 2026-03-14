; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK-NOT: error:
!0 = !DIExpression(18446744073709551615)

; CHECK: <stdin>:[[@LINE+1]]:20: error: element too large, limit is 18446744073709551615
!1 = !DIExpression(18446744073709551616)
