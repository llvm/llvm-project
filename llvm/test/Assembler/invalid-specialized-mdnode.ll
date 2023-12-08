; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:6: error: expected metadata type
!0 = !Invalid(field: 0)
