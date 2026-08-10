; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: value only has one use
@global = global i32 0
@alias = alias i32, ptr @global
uselistorder ptr @global, { 1, 0 }
