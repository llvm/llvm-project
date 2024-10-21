; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure lowering is correctly generating spirv code.

; CHECK-DAG: %[[#double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#scalar_function:]] = OpTypeFunction %[[#int_32]] %[[#double]]
; CHECK-DAG: %[[#vec_2_int_32:]] = OpTypeVector %[[#int_32]] 2
; CHECK-DAG: %[[#vec_4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec_3_int_32:]] = OpTypeVector %[[#int_32]] 3
; CHECK-DAG: %[[#vec_3_double:]] = OpTypeVector %[[#double]] 3
; CHECK-DAG: %[[#vector_function:]] = OpTypeFunction %[[#vec_3_int_32]] %[[#vec_3_double]]
; CHECK-DAG: %[[#vec_2_double:]] = OpTypeVector %[[#double]] 2


define spir_func noundef i32 @test_scalar(double noundef %D) local_unnamed_addr {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#int_32]] None %[[#scalar_function]]
  ; CHECK: %[[#param:]] = OpFunctionParameter %[[#double]]
  ; CHECK: %[[#bitcast:]] = OpBitcast %[[#vec_2_int_32]] %[[#param]]
  %0 = bitcast double %D to <2 x i32>
  ; CHECK: %[[#]] = OpCompositeExtract %[[#int_32:]] %[[#bitcast]] 0
  %1 = extractelement <2 x i32> %0, i64 0
  ; CHECK: %[[#]] = OpCompositeExtract %[[#int_32:]] %[[#bitcast]] 1
  %2 = extractelement <2 x i32> %0, i64 1
  %add = add i32 %1, %2
  ret i32 %add
}


define spir_func noundef <3 x i32> @test_vector(<3 x double> noundef %D) local_unnamed_addr {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec_3_int_32]] None %[[#vector_function]]
  ; CHECK: %[[#param:]] = OpFunctionParameter %[[#vec_3_double]]
  ; CHECK: %[[#shuf1:]] = OpVectorShuffle %[[#vec_2_double]] %[[#param]] %[[#]] 0 1
  %0 = shufflevector <3 x double> %D, <3 x double> poison, <2 x i32> <i32 0, i32 1>
  ; CHECK: %[[#shuf2:]] = OpVectorShuffle %[[#vec_2_double]] %[[#param]] %[[#]] 2 0  
  %1 = shufflevector <3 x double> %D, <3 x double> poison, <2 x i32> <i32 2, i32 0>
  ; CHECK: %[[#cast1:]] = OpBitcast %[[#vec_4_int_32]] %[[#shuf1]]  
  %2 = bitcast <2 x double> %0 to <4 x i32>
  ; CHECK: %[[#cast2:]] = OpBitcast %[[#vec_4_int_32]] %[[#shuf2]]  
  %3 = bitcast <2 x double> %1 to <4 x i32>
  ; CHECK: %[[#]] = OpVectorShuffle %[[#vec_3_int_32]] %[[#cast1]] %[[#cast2]] 0 2 4  
  %4 = shufflevector <4 x i32> %2, <4 x i32> %3, <3 x i32> <i32 0, i32 2, i32 4>
  ; CHECK: %[[#]] = OpVectorShuffle %[[#vec_3_int_32]] %[[#cast1]] %[[#cast2]] 1 3 5  
  %5 = shufflevector <4 x i32> %2, <4 x i32> %3, <3 x i32> <i32 1, i32 3, i32 5>
  %add = add <3 x i32> %4, %5
  ret <3 x i32> %add
}
