; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:18: error: invalid field 'bad'
!0 = !DILocation(bad: 0)
