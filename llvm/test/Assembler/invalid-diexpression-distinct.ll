; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:6: error: 'distinct' not allowed for !DIExpression
!0 = distinct !DIExpression()
