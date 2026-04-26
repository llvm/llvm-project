; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450

;CHECK-DAG: %[[#int_16:]] = OpTypeInt 16
;CHECK-DAG: %[[#int_32:]] = OpTypeInt 32
;CHECK-DAG: %[[#int_64:]] = OpTypeInt 64
;CHECK-DAG: %[[#vec_int_32:]] = OpTypeVector %[[#int_32]] 2
;CHECK-DAG: %[[#vec_int_32x4:]] = OpTypeVector %[[#int_32]] 4
;CHECK-DAG: %[[#vec_int_64:]] = OpTypeVector %[[#int_64]] 2
;CHECK-DAG: %[[#vec_int_64x3:]] = OpTypeVector %[[#int_64]] 3
;CHECK-DAG: %[[#vec_int_64x4:]] = OpTypeVector %[[#int_64]] 4
;CHECK-DAG: %[[#vec_int_16:]] = OpTypeVector %[[#int_16]] 2

;CHECK-DAG: %[[#const_16:]] = OpConstant %[[#int_32]] 16
;CHECK-DAG: %[[#const_zero:]] = OpConstant %[[#int_32]] 0
;CHECK-DAG: %[[#const_one:]] = OpConstant %[[#int_32]] 1
;CHECK-DAG: %[[#const_two:]] = OpConstant %[[#int_64]] 2
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


define noundef i64 @reversebits_i64(i64 noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#int_64]]
; CHECK: %[[#vec_cast:]] = OpBitcast %[[#vec_int_32]] %[[#param]]
; CHECK: %[[#reverse:]] = OpBitReverse %[[#vec_int_32]] %[[#vec_cast]]
; CHECK: %[[#low:]] = OpVectorExtractDynamic %[[#int_32]] %[[#reverse]] %[[#const_one]]
; CHECK: %[[#high:]] = OpVectorExtractDynamic %[[#int_32]] %[[#reverse]] %[[#const_zero]]
; CHECK: %[[#vec_rebuild:]] = OpCompositeConstruct %[[#vec_int_32]] %[[#low]] %[[#high]]
; CHECK: %[[#]] = OpBitcast %[[#int_64]] %[[#vec_rebuild]]
  %elt.bitreverse = call i64 @llvm.bitreverse.i64(i64 %a)
  ret i64 %elt.bitreverse
}

define noundef <2 x i64> @reversebits_veci64x2(<2 x i64> noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#vec_int_64]]
; CHECK: %[[#vec_cast:]] = OpBitcast %[[#vec_int_32x4]] %[[#param]]
; CHECK: %[[#reverse:]] = OpBitReverse %[[#vec_int_32x4]] %[[#vec_cast]]
; CHECK: %[[#low:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 1 3
; CHECK: %[[#high:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 0 2
; CHECK: %[[#vec_rebuild:]] = OpCompositeConstruct %[[#vec_int_32x4]] %[[#low]] %[[#high]]
; CHECK: %[[#]] = OpBitcast %[[#vec_int_64]] %[[#vec_rebuild]]
  %elt.bitreverse = call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %a)
  ret <2 x i64> %elt.bitreverse
}

define noundef <3 x i64> @reversebits_veci64x3(<3 x i64> noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#vec_int_64x3]]
; CHECK: %[[#first_part:]] = OpVectorShuffle %[[#vec_int_64]] %[[#param]] %[[#param]] 0 1
; CHECK: %[[#vec_cast:]] = OpBitcast %[[#vec_int_32x4]] %[[#first_part]]
; CHECK: %[[#reverse:]] = OpBitReverse %[[#vec_int_32x4]] %[[#vec_cast]]
; CHECK: %[[#low:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 1 3
; CHECK: %[[#high:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 0 2
; CHECK: %[[#vec_rebuild:]] = OpCompositeConstruct %[[#vec_int_32x4]] %[[#low]] %[[#high]]
; CHECK: %[[#first_result:]] = OpBitcast %[[#vec_int_64]] %[[#vec_rebuild]]

; CHECK: %[[#third_element:]] = OpVectorExtractDynamic %[[#int_64]] %[[#param]] %[[#const_two]]
; CHECK: %[[#bitcast:]] = OpBitcast %[[#vec_int_32]] %[[#third_element]]
; CHECK: %[[#bitreverse:]] = OpBitReverse %[[#vec_int_32]] %[[#bitcast]]
; CHECK: %[[#extract_one:]] = OpVectorExtractDynamic %[[#int_32]] %[[#bitreverse]] %[[#const_one]]
; CHECK: %[[#extract_zero:]] = OpVectorExtractDynamic %[[#int_32]] %[[#bitreverse]] %[[#const_zero]]
; CHECK: %[[#composite:]] = OpCompositeConstruct %[[#vec_int_32]] %[[#extract_one]] %[[#extract_zero]]

; CHECK: %[[#second_result:]] = OpBitcast %[[#int_64]] %[[#composite]]
; CHECK: %[[#]] = OpCompositeConstruct %[[#vec_int_64x3]] %[[#first_result]] %[[#second_result]]
  %elt.bitreverse = call <3 x i64> @llvm.bitreverse.v3i64(<3 x i64> %a)
  ret <3 x i64> %elt.bitreverse
}

define noundef <4 x i64> @reversebits_veci64x4(<4 x i64> noundef %a) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#vec_int_64x4]]
; CHECK: %[[#first_part:]] = OpVectorShuffle %[[#vec_int_64]] %[[#param]] %[[#param]] 0 1
; CHECK: %[[#vec_cast:]] = OpBitcast %[[#vec_int_32x4]] %[[#first_part]]
; CHECK: %[[#reverse:]] = OpBitReverse %[[#vec_int_32x4]] %[[#vec_cast]]
; CHECK: %[[#low:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 1 3
; CHECK: %[[#high:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 0 2
; CHECK: %[[#vec_rebuild:]] = OpCompositeConstruct %[[#vec_int_32x4]] %[[#low]] %[[#high]]
; CHECK: %[[#first_result:]] = OpBitcast %[[#vec_int_64]] %[[#vec_rebuild]]

; CHECK: %[[#second_part:]] = OpVectorShuffle %[[#vec_int_64]] %[[#param]] %[[#param]] 2 3
; CHECK: %[[#vec_cast:]] = OpBitcast %[[#vec_int_32x4]] %[[#second_part]]
; CHECK: %[[#reverse:]] = OpBitReverse %[[#vec_int_32x4]] %[[#vec_cast]]
; CHECK: %[[#low:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 1 3
; CHECK: %[[#high:]] = OpVectorShuffle %[[#vec_int_32]] %[[#reverse]] %[[#reverse]] 0 2
; CHECK: %[[#vec_rebuild:]] = OpCompositeConstruct %[[#vec_int_32x4]] %[[#low]] %[[#high]]
; CHECK: %[[#second_result:]] = OpBitcast %[[#vec_int_64]] %[[#vec_rebuild]]

; CHECK: %[[#]] = OpCompositeConstruct %[[#vec_int_64x4]] %[[#first_result]] %[[#second_result]]
  %elt.bitreverse = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> %a)
  ret <4 x i64> %elt.bitreverse
}

declare i16 @llvm.bitreverse.i16(i16)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)
declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>)
declare <2 x i64> @llvm.bitreverse.v2i64(<2 x i64>)
declare <3 x i64> @llvm.bitreverse.v3i64(<3 x i64>)
declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>)
