; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: .text
; CHECK-LABEL: f:
define void @f(i32 %a, float %b, i128 %c, i1 %d) {
  tail call i32 @printi(i32 %a)
  tail call float @printf(float %b)
  tail call void @printv()
  tail call void @split_arg(i128 %c)
  tail call void @expanded_arg(i1 %d)
  tail call i1 @lowered_result()
  ret void
}

; CHECK-LABEL: .imports
; CHECK-NEXT:  .import printi "" printi (param i32) (result i32){{$}}
; CHECK-NEXT:  .import printf "" printf (param f32) (result f32){{$}}
; CHECK-NEXT:  .import printv "" printv{{$}}
; CHECK-NEXT:  .import add2 "" add2 (param i32 i32) (result i32){{$}}
; CHECK-NEXT:  .import split_arg "" split_arg (param i64 i64){{$}}
; CHECK-NEXT:  .import expanded_arg "" expanded_arg (param i32){{$}}
; CHECK-NEXT:  .import lowered_result "" lowered_result (result i32){{$}}
declare i32 @printi(i32)
declare float @printf(float)
declare void @printv()
declare i32 @add2(i32, i32)
declare void @split_arg(i128)
declare void @expanded_arg(i1)
declare i1 @lowered_result()
