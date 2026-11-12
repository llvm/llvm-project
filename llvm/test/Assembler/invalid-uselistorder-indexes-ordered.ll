; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: expected uselistorder indexes to change the order
@global = global i32 0
@alias1 = alias i32, ptr @global
@alias2 = alias i32, ptr @global
@alias3 = alias i32, ptr @global
uselistorder ptr @global, { 0, 1, 2 }
