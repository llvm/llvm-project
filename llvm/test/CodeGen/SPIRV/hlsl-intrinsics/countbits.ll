; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450

define noundef i32 @countbits_i32(i32 noundef %a) {
entry:
; CHECK: %[[#]] = OpBitCount %[[#]] %[[#]]
  %elt.bitreverse = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %elt.bitreverse
}

define noundef i16 @countbits_i16(i16 noundef %a) {
entry:
; CHECK: %[[#]] = OpBitCount %[[#]] %[[#]]
  %elt.ctpop = call i16 @llvm.ctpop.i16(i16 %a)
  ret i16 %elt.ctpop
}

declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
