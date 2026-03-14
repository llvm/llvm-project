; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: invalid declaration in uselistorder_bb
declare void @foo()
uselistorder_bb @foo, %bb, { 1, 0 }
