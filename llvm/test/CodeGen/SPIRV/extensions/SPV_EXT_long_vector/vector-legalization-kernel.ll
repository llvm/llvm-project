; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; spirv-val seems to have problems reading OpTypeVectorIdEXT correctly, enable once fixed
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#test_int32_double_conversion:]] "test_int32_double_conversion"
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#v18i32:]] = OpTypeVectorIdEXT %[[#int]] 18
; CHECK-DAG: %[[#v9i32:]] = OpTypeVector %[[#int]] 9
; CHECK-DAG: %[[#ptr_func_v18i32:]] = OpTypePointer Function %[[#v18i32]]

define spir_kernel void @test_int32_double_conversion(ptr %G_vec) {
; CHECK: %[[#test_int32_double_conversion]] = OpFunction
; CHECK: %[[#param:]] = OpFunctionParameter %[[#ptr_func_v18i32]]
entry:
  ; CHECK: %[[#LOAD:]] = OpLoad %[[#v18i32]] %[[#param]]
  ; CHECK: %[[#SHUF1:]] = OpVectorShuffle %[[#v9i32]] %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 0 2 4 6 8 10 12 14 16
  ; CHECK: %[[#SHUF2:]] = OpVectorShuffle %[[#v9i32]] %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 1 3 5 7 9 11 13 15 17
  ; CHECK: %[[#SHUF3:]] = OpVectorShuffle %[[#v18i32]] %[[#SHUF1]] %[[#SHUF2]] 0 9 1 10 2 11 3 12 4 13 5 14 6 15 7 16 8 17
  ; CHECK: OpStore %[[#param]] %[[#SHUF3]]

  %0 = load <18 x i32>, ptr %G_vec
  %1 = shufflevector <18 x i32> %0, <18 x i32> poison, <9 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16>
  %2 = shufflevector <18 x i32> %0, <18 x i32> poison, <9 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17>
  %3 = shufflevector <9 x i32> %1, <9 x i32> %2, <18 x i32> <i32 0, i32 9, i32 1, i32 10, i32 2, i32 11, i32 3, i32 12, i32 4, i32 13, i32 5, i32 14, i32 6, i32 15, i32 7, i32 16, i32 8, i32 17>
  store <18 x i32> %3, ptr %G_vec
  ret void
}
