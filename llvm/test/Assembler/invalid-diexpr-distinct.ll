; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:15: error: 'distinct' not allowed for !DIExpr{{$}}
!0 = distinct !DIExpr()
