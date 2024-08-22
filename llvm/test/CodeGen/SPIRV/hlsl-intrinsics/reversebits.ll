; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450

define noundef i32 @reversebits_i32(i32 noundef %a) {
entry:
; CHECK: %[[#]] = OpBitReverse %[[#]] %[[#]]
  %elt.bitreverse = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %elt.bitreverse
}

define noundef i16 @reversebits_i16(i16 noundef %a) {
entry:
; CHECK: %[[#]] = OpBitReverse %[[#]] %[[#]]
  %elt.bitreverse = call i16 @llvm.bitreverse.i16(i16 %a)
  ret i16 %elt.bitreverse
}

declare i16 @llvm.bitreverse.i16(i16)
declare i32 @llvm.bitreverse.i32(i32)
