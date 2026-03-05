; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450

;CHECK-DAG: %[[#int_16:]] = OpTypeInt 16
;CHECK-DAG: %[[#int_32:]] = OpTypeInt 32

define noundef i32 @reversebits_i32(i32 noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#int_32]]
; CHECK-NOT: OpUConvert 
; CHECK: %[[#]] = OpBitReverse %[[#int_32]] %[[#param]]
  %elt.bitreverse = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %elt.bitreverse
}

define noundef i16 @reversebits_i16(i16 noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#int_16]]
; CHECK: %[[#conversion:]] = OpUConvert %[[#int_32]] %[[#param]]
; CHECK-NEXT: %[[#]] = OpBitReverse %[[#int_32]] %[[#conversion]]
  %elt.bitreverse = call i16 @llvm.bitreverse.i16(i16 %a)
  ret i16 %elt.bitreverse
}

declare i16 @llvm.bitreverse.i16(i16)
declare i32 @llvm.bitreverse.i32(i32)
