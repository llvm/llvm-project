; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: expected function name in uselistorder_bb
@global = global i1 0
uselistorder_bb @global, %bb, { 1, 0 }
