; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: invalid function forward reference in uselistorder_bb
uselistorder_bb @foo, %bb, { 1, 0 }
