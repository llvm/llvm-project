; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: wrong number of indexes, expected 2
@global = global i32 0
@alias1 = alias i32, ptr @global
@alias2 = alias i32, ptr @global
uselistorder ptr @global, { 1, 0, 2 }
