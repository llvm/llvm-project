; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure lowering is correctly generating spirv code.

; CHECK-DAG: %[[#double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#vec_2_double:]] = OpTypeVector %[[#double]] 2
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#vec_2_int_32:]] = OpTypeVector %[[#int_32]] 2
; CHECK-DAG: %[[#vec_4_int_32:]] = OpTypeVector %[[#int_32]] 4


define spir_func noundef i32 @test_scalar(double noundef %D) local_unnamed_addr {
entry:
  ; CHECK-LABEL: ; -- Begin function test_scalar
  ; CHECK: %[[#param:]] = OpFunctionParameter %[[#double]]
  ; CHECK: %[[#bitcast:]] = OpBitcast %[[#vec_2_int_32]] %[[#param]]
  %0 = bitcast double %D to <2 x i32>
  ; CHECK: %[[#]] = OpCompositeExtract %[[#int_32]] %[[#bitcast]] 0
  %1 = extractelement <2 x i32> %0, i64 0
  ; CHECK: %[[#]] = OpCompositeExtract %[[#int_32]] %[[#bitcast]] 1
  %2 = extractelement <2 x i32> %0, i64 1
  %add = add i32 %1, %2
  ret i32 %add
}


define spir_func noundef <2 x i32> @test_vector(<2 x double> noundef %D) local_unnamed_addr {
entry:
  ; CHECK-LABEL: ; -- Begin function test_vector
  ; CHECK: %[[#param:]] = OpFunctionParameter %[[#vec_2_double]]
  ; CHECK: %[[#CAST1:]] = OpBitcast %[[#vec_4_int_32]] %[[#param]]
  ; CHECK: %[[#SHUFF2:]] = OpVectorShuffle %[[#vec_2_int_32]] %[[#CAST1]] %[[#]] 0 2
  ; CHECK: %[[#SHUFF3:]] = OpVectorShuffle %[[#vec_2_int_32]] %[[#CAST1]] %[[#]] 1 3
  %0 = bitcast <2 x double> %D to <4 x i32>
  %1 = shufflevector <4 x i32> %0, <4 x i32> poison, <2 x i32> <i32 0, i32 2>
  %2 = shufflevector <4 x i32> %0, <4 x i32> poison, <2 x i32> <i32 1, i32 3>
  %add = add <2 x i32> %1, %2
  ret <2 x i32> %add
}
