; RUN: llvm-link %s %S/Inputs/alias.ll -S -o - | FileCheck %s
; RUN: llvm-link %S/Inputs/alias.ll %s -S -o - | FileCheck %s

@foo = weak global i32 0
; CHECK-DAG: @foo = alias i32, i32* @zed

@bar = alias i32, i32* @foo
; CHECK-DAG: @bar = alias i32, i32* @foo

@foo2 = weak global i32 0
; CHECK-DAG: @foo2 = alias i16, bitcast (i32* @zed to i16*)

@bar2 = alias i32, i32* @foo2
; CHECK-DAG: @bar2 = alias i32, bitcast (i16* @foo2 to i32*)

; CHECK-DAG: @zed = global i32 42
