; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450

define noundef i32 @firstbituhigh_i32(i32 noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindUMsb %[[#]]
  %elt.firstbituhigh = call i32 @llvm.spv.firstbituhigh.i32(i32 %a)
  ret i32 %elt.firstbituhigh
}

define noundef i16 @firstbituhigh_i16(i16 noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindUMsb %[[#]]
  %elt.firstbituhigh = call i16 @llvm.spv.firstbituhigh.i16(i16 %a)
  ret i16 %elt.firstbituhigh
}

define noundef i32 @firstbitshigh_i32(i32 noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindSMsb %[[#]]
  %elt.firstbitshigh = call i32 @llvm.spv.firstbitshigh.i32(i32 %a)
  ret i32 %elt.firstbitshigh
}

define noundef i16 @firstbitshigh_i16(i16 noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindSMsb %[[#]]
  %elt.firstbitshigh = call i16 @llvm.spv.firstbitshigh.i16(i16 %a)
  ret i16 %elt.firstbitshigh
}

declare i16 @llvm.spv.firstbituhigh.i16(i16)
declare i32 @llvm.spv.firstbituhigh.i32(i32)
declare i16 @llvm.spv.firstbitshigh.i16(i16)
declare i32 @llvm.spv.firstbitshigh.i32(i32)
