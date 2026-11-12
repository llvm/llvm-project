; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

@global = global i32 0

@alias = private hidden alias i32, ptr @global
; CHECK: symbol with local linkage must have default visibility
