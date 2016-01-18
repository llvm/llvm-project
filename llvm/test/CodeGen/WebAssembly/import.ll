; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: .text
; CHECK-LABEL: f:
define void @f(i32 %a, float %b) {
  tail call i32 @printi(i32 %a)
  tail call float @printf(float %b)
  tail call void @printv()
  ret void
}

; CHECK-LABEL: .imports
; CHECK-NEXT:  .import $printi "" "printi" (param i32) (result i32)
; CHECK-NEXT:  .import $printf "" "printf" (param f32) (result f32)
; CHECK-NEXT:  .import $printv "" "printv" (param)
; CHECK-NEXT:  .import $add2 "" "add2" (param i32 i32) (result i32)
declare i32 @printi(i32)
declare float @printf(float)
declare void @printv()
declare i32 @add2(i32, i32)
