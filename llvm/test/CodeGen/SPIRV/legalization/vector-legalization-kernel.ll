; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#test_int32_double_conversion:]] "test_int32_double_conversion"
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#v8i32:]] = OpTypeVector %[[#int]] 8
; CHECK-DAG: %[[#v4i32:]] = OpTypeVector %[[#int]] 4
; CHECK-DAG: %[[#ptr_func_v8i32:]] = OpTypePointer Function %[[#v8i32]]

; CHECK-DAG: OpName %[[#test_v3f64_conversion:]] "test_v3f64_conversion"
; CHECK-DAG: %[[#double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#v3f64:]] = OpTypeVector %[[#double]] 3
; CHECK-DAG: %[[#ptr_func_v3f64:]] = OpTypePointer Function %[[#v3f64]]
; CHECK-DAG: %[[#v4f64:]] = OpTypeVector %[[#double]] 4

; CHECK-DAG: OpName %[[#test_float_shuffle:]] "test_float_shuffle"
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v4f32:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#v8f32:]] = OpTypeVector %[[#float]] 8
; CHECK-DAG: %[[#undef_v4f32:]] = OpUndef %[[#v4f32]]

; CHECK-DAG: OpName %[[#test_int_shuffle:]] "test_int_shuffle"
; CHECK-DAG: %[[#undef_v4i32:]] = OpUndef %[[#v4i32]]

; CHECK-DAG: OpName %[[#test_bool_shuffle:]] "test_bool_shuffle"
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#v4bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG: %[[#v8bool:]] = OpTypeVector %[[#bool]] 8
; CHECK-DAG: %[[#undef_v4bool:]] = OpUndef %[[#v4bool]]

define spir_kernel void @test_int32_double_conversion(ptr %G_vec) {
; CHECK: %[[#test_int32_double_conversion]] = OpFunction
; CHECK: %[[#param:]] = OpFunctionParameter %[[#ptr_func_v8i32]]
entry:
  ; CHECK: %[[#LOAD:]] = OpLoad %[[#v8i32]] %[[#param]]
  ; CHECK: %[[#SHUF1:]] = OpVectorShuffle %[[#v4i32]] %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 0 2 4 6
  ; CHECK: %[[#SHUF2:]] = OpVectorShuffle %[[#v4i32]] %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 1 3 5 7
  ; CHECK: %[[#SHUF3:]] = OpVectorShuffle %[[#v8i32]] %[[#SHUF1]] %[[#SHUF2]] 0 4 1 5 2 6 3 7
  ; CHECK: OpStore %[[#param]] %[[#SHUF3]]

  %0 = load <8 x i32>, ptr %G_vec
  %1 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %2 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32> %3, ptr %G_vec
  ret void
}

define spir_kernel void @test_v3f64_conversion(ptr %G_vec) {
; CHECK: %[[#test_v3f64_conversion:]] = OpFunction
; CHECK: %[[#param_v3f64:]] = OpFunctionParameter %[[#ptr_func_v3f64]]
entry:
  ; CHECK: %[[#LOAD:]] = OpLoad %[[#v3f64]] %[[#param_v3f64]]
  %0 = load <3 x double>, ptr %G_vec

  ; The 6-element vector is not legal. It get expanded to 8.
  ; CHECK: %[[#EXTRACT1:]] = OpCompositeExtract %[[#double]] %[[#LOAD]] 0
  ; CHECK: %[[#EXTRACT2:]] = OpCompositeExtract %[[#double]] %[[#LOAD]] 1
  ; CHECK: %[[#EXTRACT3:]] = OpCompositeExtract %[[#double]] %[[#LOAD]] 2
  ; CHECK: %[[#CONSTRUCT1:]] = OpCompositeConstruct %[[#v4f64]] %[[#EXTRACT1]] %[[#EXTRACT2]] %[[#EXTRACT3]] %{{[a-zA-Z0-9_]+}}
  ; CHECK: %[[#BITCAST1:]] = OpBitcast %[[#v8i32]] %[[#CONSTRUCT1]]
  %1 = bitcast <3 x double> %0 to <6 x i32>

  ; CHECK: %[[#SHUFFLE1:]] = OpVectorShuffle %[[#v8i32]] %[[#BITCAST1]] %{{[a-zA-Z0-9_]+}} 0 2 4 0xFFFFFFFF 0xFFFFFFFF 0xFFFFFFFF 0xFFFFFFFF 0xFFFFFFFF
  %2 = shufflevector <6 x i32> %1, <6 x i32> poison, <3 x i32> <i32 0, i32 2, i32 4>

  ; CHECK: %[[#SHUFFLE2:]] = OpVectorShuffle %[[#v8i32]] %[[#BITCAST1]] %{{[a-zA-Z0-9_]+}} 1 3 5 0xFFFFFFFF 0xFFFFFFFF 0xFFFFFFFF 0xFFFFFFFF 0xFFFFFFFF
  %3 = shufflevector <6 x i32> %1, <6 x i32> poison, <3 x i32> <i32 1, i32 3, i32 5>

  ; CHECK: %[[#SHUFFLE3:]] = OpVectorShuffle %[[#v8i32]] %[[#SHUFFLE1]] %[[#SHUFFLE2]] 0 8 1 9 2 10 0xFFFFFFFF 0xFFFFFFFF
  %4 = shufflevector <3 x i32> %2, <3 x i32> %3, <6 x i32> <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>

  ; CHECK: %[[#BITCAST2:]] = OpBitcast %[[#v4f64]] %[[#SHUFFLE3]]
  ; CHECK: %[[#EXTRACT10:]] = OpCompositeExtract %[[#double]] %[[#BITCAST2]] 0
  ; CHECK: %[[#EXTRACT11:]] = OpCompositeExtract %[[#double]] %[[#BITCAST2]] 1
  ; CHECK: %[[#EXTRACT12:]] = OpCompositeExtract %[[#double]] %[[#BITCAST2]] 2
  ; CHECK: %[[#CONSTRUCT3:]] = OpCompositeConstruct %[[#v3f64]] %[[#EXTRACT10]] %[[#EXTRACT11]] %[[#EXTRACT12]]
  %5 = bitcast <6 x i32> %4 to <3 x double>
  
  ; CHECK: OpStore %[[#param_v3f64]] %[[#CONSTRUCT3]]
  store <3 x double> %5, ptr %G_vec
  ret void
}

define spir_kernel void @test_float_shuffle(ptr %G_vec) {
; CHECK: %[[#test_float_shuffle]] = OpFunction
entry:
  ; CHECK: %[[#LOAD:]] = OpLoad %[[#v4f32]]
  ; CHECK: OpCompositeConstruct %[[#v8f32]] %[[#LOAD]] %[[#undef_v4f32]]
  %0 = load <4 x float>, ptr %G_vec
  %1 = shufflevector <4 x float> %0, <4 x float> %0, <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %2 = shufflevector <6 x float> %1, <6 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 5>
  store <4 x float> %2, ptr %G_vec
  ret void
}

define spir_kernel void @test_int_shuffle(ptr %G_vec) {
; CHECK: %[[#test_int_shuffle]] = OpFunction
entry:
  ; CHECK: %[[#LOAD:]] = OpLoad %[[#v4i32]]
  ; CHECK: OpCompositeConstruct %[[#v8i32]] %[[#LOAD]] %[[#undef_v4i32]]
  %0 = load <4 x i32>, ptr %G_vec
  %1 = shufflevector <4 x i32> %0, <4 x i32> %0, <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %2 = shufflevector <6 x i32> %1, <6 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 5>
  store <4 x i32> %2, ptr %G_vec
  ret void
}

define spir_kernel void @test_bool_shuffle(ptr %G_vec) {
; CHECK: %[[#test_bool_shuffle]] = OpFunction
entry:
  ; CHECK: %[[#CMP:]] = OpIEqual %[[#v4bool]]
  ; CHECK: OpCompositeConstruct %[[#v8bool]] %[[#CMP]] %[[#undef_v4bool]]
  %0 = load <4 x i32>, ptr %G_vec
  %1 = icmp eq <4 x i32> %0, zeroinitializer
  %2 = shufflevector <4 x i1> %1, <4 x i1> %1, <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %3 = shufflevector <6 x i1> %2, <6 x i1> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 5>
  %4 = select <4 x i1> %3, <4 x i32> %0, <4 x i32> zeroinitializer
  store <4 x i32> %4, ptr %G_vec
  ret void
}
