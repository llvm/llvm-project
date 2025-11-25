; RUN: llvm-as  < %s | llvm-dis  | FileCheck %s

@0 = private constant i32 0
; CHECK: @0 = private constant i32 0
@1 = private constant i32 1
; CHECK: @1 = private constant i32 1

@2 = private alias i32, ptr @3
; CHECK: @2 = private alias i32, ptr @3
@3 = private alias i32, ptr @1
; CHECK: @3 = private alias i32, ptr @1
