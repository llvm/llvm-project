; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#test_int32_double_conversion:]] "test_int32_double_conversion"
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#v4i32:]] = OpTypeVector %[[#int]] 4
; CHECK-DAG: %[[#double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#v4f64:]] = OpTypeVector %[[#double]] 4
; CHECK-DAG: %[[#v2i32:]] = OpTypeVector %[[#int]] 2
; CHECK-DAG: %[[#ptr_private_v4i32:]] = OpTypePointer Private %[[#v4i32]]
; CHECK-DAG: %[[#ptr_private_v4f64:]] = OpTypePointer Private %[[#v4f64]]
; CHECK-DAG: %[[#global_double:]] = OpVariable %[[#ptr_private_v4f64]] Private

@G_16 = internal addrspace(10) global [16 x i32] zeroinitializer
@G_4_double = internal addrspace(10) global <4 x double> zeroinitializer
@G_4_int = internal addrspace(10) global <4 x i32> zeroinitializer


; This is the way matrices will be represented in HLSL. The memory type will be
; an array, but it will be loaded as a vector.
; TODO: Legalization for loads and stores of long vectors is not implemented yet.                                                                                                                                                                                    │
;define spir_func void @test_load_store_global() {                                                                                                                                                                                                                      │
;entry:                                                                                                                                                                                                                                                                 │
;  %0 = load <16 x i32>, ptr addrspace(10) @G_16, align 64                                                                                                                                                                                                               │
;  store <16 x i32> %0, ptr addrspace(10) @G_16, align 64                                                                                                                                                                                                                │
;  ret void                                                                                                                                                                                                                                                             │
;} 

; This is the code pattern that can be generated from the `asuint` and `asdouble`
; HLSL intrinsics.

; TODO: This cods not the best because instruction selection is not folding an
; extract from other intstruction. That needs to be handled.
define spir_func void @test_int32_double_conversion() {
; CHECK: %[[#test_int32_double_conversion]] = OpFunction
entry:
  ; CHECK: %[[#LOAD:]] = OpLoad %[[#v4f64]] %[[#global_double]]
  ; CHECK: %[[#VEC_SHUF1:]] = OpVectorShuffle %{{[a-zA-Z0-9_]+}} %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 0 1
  ; CHECK: %[[#VEC_SHUF2:]] = OpVectorShuffle %{{[a-zA-Z0-9_]+}} %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 2 3
  ; CHECK: %[[#BITCAST1:]] = OpBitcast %[[#v4i32]] %[[#VEC_SHUF1]]
  ; CHECK: %[[#BITCAST2:]] = OpBitcast %[[#v4i32]] %[[#VEC_SHUF2]]
  ; CHECK: %[[#EXTRACT1:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 0
  ; CHECK: %[[#EXTRACT2:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 2
  ; CHECK: %[[#EXTRACT3:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 0
  ; CHECK: %[[#EXTRACT4:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 2
  ; CHECK: %[[#CONSTRUCT1:]] = OpCompositeConstruct %[[#v4i32]] %[[#EXTRACT1]] %[[#EXTRACT2]] %[[#EXTRACT3]] %[[#EXTRACT4]]
  ; CHECK: %[[#EXTRACT5:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 1
  ; CHECK: %[[#EXTRACT6:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 3
  ; CHECK: %[[#EXTRACT7:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 1
  ; CHECK: %[[#EXTRACT8:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 3
  ; CHECK: %[[#CONSTRUCT2:]] = OpCompositeConstruct %[[#v4i32]] %[[#EXTRACT5]] %[[#EXTRACT6]] %[[#EXTRACT7]] %[[#EXTRACT8]]
  ; CHECK: %[[#EXTRACT9:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT1]] 0
  ; CHECK: %[[#EXTRACT10:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT2]] 0
  ; CHECK: %[[#EXTRACT11:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT1]] 1
  ; CHECK: %[[#EXTRACT12:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT2]] 1
  ; CHECK: %[[#EXTRACT13:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT1]] 2
  ; CHECK: %[[#EXTRACT14:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT2]] 2
  ; CHECK: %[[#EXTRACT15:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT1]] 3
  ; CHECK: %[[#EXTRACT16:]] = OpCompositeExtract %[[#int]] %[[#CONSTRUCT2]] 3
  ; CHECK: %[[#CONSTRUCT3:]] = OpCompositeConstruct %[[#v2i32]] %[[#EXTRACT9]] %[[#EXTRACT10]]
  ; CHECK: %[[#CONSTRUCT4:]] = OpCompositeConstruct %[[#v2i32]] %[[#EXTRACT11]] %[[#EXTRACT12]]
  ; CHECK: %[[#CONSTRUCT5:]] = OpCompositeConstruct %[[#v2i32]] %[[#EXTRACT13]] %[[#EXTRACT14]]
  ; CHECK: %[[#CONSTRUCT6:]] = OpCompositeConstruct %[[#v2i32]] %[[#EXTRACT15]] %[[#EXTRACT16]]
  ; CHECK: %[[#BITCAST3:]] = OpBitcast %[[#double]] %[[#CONSTRUCT3]]
  ; CHECK: %[[#BITCAST4:]] = OpBitcast %[[#double]] %[[#CONSTRUCT4]]
  ; CHECK: %[[#BITCAST5:]] = OpBitcast %[[#double]] %[[#CONSTRUCT5]]
  ; CHECK: %[[#BITCAST6:]] = OpBitcast %[[#double]] %[[#CONSTRUCT6]]
  ; CHECK: %[[#CONSTRUCT7:]] = OpCompositeConstruct %[[#v4f64]] %[[#BITCAST3]] %[[#BITCAST4]] %[[#BITCAST5]] %[[#BITCAST6]]
  ; CHECK: OpStore %[[#global_double]] %[[#CONSTRUCT7]] Aligned 32

  %0 = load <8 x i32>, ptr addrspace(10) @G_4_double
  %1 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %2 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32> %3, ptr addrspace(10) @G_4_double
  ret void
}

; Add a main function to make it a valid module for spirv-val
define void @main() #1 {
  ret void
}

attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
