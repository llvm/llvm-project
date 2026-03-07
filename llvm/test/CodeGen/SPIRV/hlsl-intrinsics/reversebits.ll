; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450

;CHECK-DAG: %[[#int_16:]] = OpTypeInt 16
;CHECK-DAG: %[[#int_32:]] = OpTypeInt 32
;CHECK-DAG: %[[#vec_int_32:]] = OpTypeVector %[[#int_32]] 2
;CHECK-DAG: %[[#vec_int_16:]] = OpTypeVector %[[#int_16]] 2

;CHECK-DAG: %[[#const_16:]] = OpConstant %[[#int_32]] 16
;CHECK-DAG: %[[#composite:]] = OpConstantComposite %[[#vec_int_32]] %[[#const_16]] %[[#const_16]] 

define noundef i32 @reversebits_i32(i32 noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#int_32]]
; CHECK-NOT: OpUConvert 
; CHECK: %[[#]] = OpBitReverse %[[#int_32]] %[[#param]]
; CHECK-NOT: OpShiftRightLogical 
  %elt.bitreverse = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %elt.bitreverse
}

define noundef i16 @reversebits_i16(i16 noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#int_16]]
; CHECK: %[[#conversion:]] = OpUConvert %[[#int_32]] %[[#param]]
; CHECK-NEXT: %[[#bitrev:]] = OpBitReverse %[[#int_32]] %[[#conversion]]
; CHECK-NEXT: %[[#shift:]] = OpShiftRightLogical %[[#int_32]] %[[#bitrev]] %[[#const_16]]
; CHECK-NEXT: %[[#]] = OpUConvert %[[#int_16]] %[[#shift]]
  %elt.bitreverse = call i16 @llvm.bitreverse.i16(i16 %a)
  ret i16 %elt.bitreverse
}

define noundef <2 x i16> @reversebits_veci16(<2 x i16> noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#vec_int_16]]
; CHECK: %[[#conversion:]] = OpUConvert %[[#vec_int_32]] %[[#param]]
; CHECK-NEXT: %[[#bitrev:]] = OpBitReverse %[[#vec_int_32]] %[[#conversion]]
; CHECK-NEXT: %[[#shift:]] = OpShiftRightLogical %[[#vec_int_32]] %[[#bitrev]] %[[#composite]]
; CHECK-NEXT: %[[#]] = OpUConvert %[[#vec_int_16]] %[[#shift]]
  %elt.bitreverse = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  ret <2 x i16> %elt.bitreverse
}


declare i16 @llvm.bitreverse.i16(i16)
declare i32 @llvm.bitreverse.i32(i32)
declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>)
