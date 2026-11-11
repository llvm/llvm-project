; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:36: error: expected unsigned integer
!0 = !DIExpression(DW_OP_LLVM_arg, -1)
