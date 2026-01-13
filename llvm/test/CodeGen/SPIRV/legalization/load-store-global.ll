; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#v4i32:]] = OpTypeVector %[[#int]] 4
; CHECK-DAG: %[[#double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#v4f64:]] = OpTypeVector %[[#double]] 4
; CHECK-DAG: %[[#v2i32:]] = OpTypeVector %[[#int]] 2
; CHECK-DAG: %[[#ptr_private_v4i32:]] = OpTypePointer Private %[[#v4i32]]
; CHECK-DAG: %[[#ptr_private_v4f64:]] = OpTypePointer Private %[[#v4f64]]
; CHECK-DAG: %[[#global_double:]] = OpVariable %[[#ptr_private_v4f64]] Private
; CHECK-DAG: %[[#C15:]] = OpConstant %[[#int]] 15{{$}}
; CHECK-DAG: %[[#C14:]] = OpConstant %[[#int]] 14{{$}}
; CHECK-DAG: %[[#C13:]] = OpConstant %[[#int]] 13{{$}}
; CHECK-DAG: %[[#C12:]] = OpConstant %[[#int]] 12{{$}}
; CHECK-DAG: %[[#C11:]] = OpConstant %[[#int]] 11{{$}}
; CHECK-DAG: %[[#C10:]] = OpConstant %[[#int]] 10{{$}}
; CHECK-DAG: %[[#C9:]] = OpConstant %[[#int]] 9{{$}}
; CHECK-DAG: %[[#C8:]] = OpConstant %[[#int]] 8{{$}}
; CHECK-DAG: %[[#C7:]] = OpConstant %[[#int]] 7{{$}}
; CHECK-DAG: %[[#C6:]] = OpConstant %[[#int]] 6{{$}}
; CHECK-DAG: %[[#C5:]] = OpConstant %[[#int]] 5{{$}}
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#int]] 4{{$}}
; CHECK-DAG: %[[#C3:]] = OpConstant %[[#int]] 3{{$}}
; CHECK-DAG: %[[#C2:]] = OpConstant %[[#int]] 2{{$}}
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#int]] 1{{$}}
; CHECK-DAG: %[[#C0:]] = OpConstant %[[#int]] 0{{$}}

@G_16 = internal addrspace(10) global [16 x i32] zeroinitializer
@G_4_double = internal addrspace(10) global <4 x double> zeroinitializer
@G_4_int = internal addrspace(10) global <4 x i32> zeroinitializer


; This is the way matrices will be represented in HLSL. The memory type will be
; an array, but it will be loaded as a vector.
define spir_func void @test_load_store_global() {
entry:
; CHECK-DAG: %[[#PTR0:]] = OpAccessChain %[[#ptr_int:]] %[[#G16:]] %[[#C0]]
; CHECK-DAG: %[[#VAL0:]] = OpLoad %[[#int]] %[[#PTR0]] Aligned 4
; CHECK-DAG: %[[#PTR1:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C1]]
; CHECK-DAG: %[[#VAL1:]] = OpLoad %[[#int]] %[[#PTR1]] Aligned 4
; CHECK-DAG: %[[#PTR2:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C2]]
; CHECK-DAG: %[[#VAL2:]] = OpLoad %[[#int]] %[[#PTR2]] Aligned 4
; CHECK-DAG: %[[#PTR3:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C3]]
; CHECK-DAG: %[[#VAL3:]] = OpLoad %[[#int]] %[[#PTR3]] Aligned 4
; CHECK-DAG: %[[#PTR4:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C4]]
; CHECK-DAG: %[[#VAL4:]] = OpLoad %[[#int]] %[[#PTR4]] Aligned 4
; CHECK-DAG: %[[#PTR5:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C5]]
; CHECK-DAG: %[[#VAL5:]] = OpLoad %[[#int]] %[[#PTR5]] Aligned 4
; CHECK-DAG: %[[#PTR6:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C6]]
; CHECK-DAG: %[[#VAL6:]] = OpLoad %[[#int]] %[[#PTR6]] Aligned 4
; CHECK-DAG: %[[#PTR7:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C7]]
; CHECK-DAG: %[[#VAL7:]] = OpLoad %[[#int]] %[[#PTR7]] Aligned 4
; CHECK-DAG: %[[#PTR8:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C8]]
; CHECK-DAG: %[[#VAL8:]] = OpLoad %[[#int]] %[[#PTR8]] Aligned 4
; CHECK-DAG: %[[#PTR9:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C9]]
; CHECK-DAG: %[[#VAL9:]] = OpLoad %[[#int]] %[[#PTR9]] Aligned 4
; CHECK-DAG: %[[#PTR10:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C10]]
; CHECK-DAG: %[[#VAL10:]] = OpLoad %[[#int]] %[[#PTR10]] Aligned 4
; CHECK-DAG: %[[#PTR11:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C11]]
; CHECK-DAG: %[[#VAL11:]] = OpLoad %[[#int]] %[[#PTR11]] Aligned 4
; CHECK-DAG: %[[#PTR12:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C12]]
; CHECK-DAG: %[[#VAL12:]] = OpLoad %[[#int]] %[[#PTR12]] Aligned 4
; CHECK-DAG: %[[#PTR13:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C13]]
; CHECK-DAG: %[[#VAL13:]] = OpLoad %[[#int]] %[[#PTR13]] Aligned 4
; CHECK-DAG: %[[#PTR14:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C14]]
; CHECK-DAG: %[[#VAL14:]] = OpLoad %[[#int]] %[[#PTR14]] Aligned 4
; CHECK-DAG: %[[#PTR15:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C15]]
; CHECK-DAG: %[[#VAL15:]] = OpLoad %[[#int]] %[[#PTR15]] Aligned 4
  %0 = load <16 x i32>, ptr addrspace(10) @G_16, align 64
 
; CHECK: %[[#PTR0_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C0]]
; CHECK: OpStore %[[#PTR0_S]] %[[#VAL0]] Aligned 64
; CHECK: %[[#PTR1_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C1]]
; CHECK: OpStore %[[#PTR1_S]] %[[#VAL1]] Aligned 4
; CHECK: %[[#PTR2_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C2]]
; CHECK: OpStore %[[#PTR2_S]] %[[#VAL2]] Aligned 8
; CHECK: %[[#PTR3_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C3]]
; CHECK: OpStore %[[#PTR3_S]] %[[#VAL3]] Aligned 4
; CHECK: %[[#PTR4_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C4]]
; CHECK: OpStore %[[#PTR4_S]] %[[#VAL4]] Aligned 16
; CHECK: %[[#PTR5_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C5]]
; CHECK: OpStore %[[#PTR5_S]] %[[#VAL5]] Aligned 4
; CHECK: %[[#PTR6_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C6]]
; CHECK: OpStore %[[#PTR6_S]] %[[#VAL6]] Aligned 8
; CHECK: %[[#PTR7_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C7]]
; CHECK: OpStore %[[#PTR7_S]] %[[#VAL7]] Aligned 4
; CHECK: %[[#PTR8_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C8]]
; CHECK: OpStore %[[#PTR8_S]] %[[#VAL8]] Aligned 32
; CHECK: %[[#PTR9_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C9]]
; CHECK: OpStore %[[#PTR9_S]] %[[#VAL9]] Aligned 4
; CHECK: %[[#PTR10_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C10]]
; CHECK: OpStore %[[#PTR10_S]] %[[#VAL10]] Aligned 8
; CHECK: %[[#PTR11_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C11]]
; CHECK: OpStore %[[#PTR11_S]] %[[#VAL11]] Aligned 4
; CHECK: %[[#PTR12_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C12]]
; CHECK: OpStore %[[#PTR12_S]] %[[#VAL12]] Aligned 16
; CHECK: %[[#PTR13_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C13]]
; CHECK: OpStore %[[#PTR13_S]] %[[#VAL13]] Aligned 4
; CHECK: %[[#PTR14_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C14]]
; CHECK: OpStore %[[#PTR14_S]] %[[#VAL14]] Aligned 8
; CHECK: %[[#PTR15_S:]] = OpAccessChain %[[#ptr_int]] %[[#G16]] %[[#C15]]
; CHECK: OpStore %[[#PTR15_S]] %[[#VAL15]] Aligned 4
  store <16 x i32> %0, ptr addrspace(10) @G_16, align 64
  ret void
}

define spir_func void @test_int32_double_conversion() {
; CHECK: OpFunction
entry:
  ; CHECK: %[[#LOAD:]] = OpLoad %[[#v4f64]] %[[#global_double]]
  ; CHECK: %[[#VEC_SHUF1:]] = OpVectorShuffle %{{[a-zA-Z0-9_]+}} %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 0 1
  ; CHECK: %[[#VEC_SHUF2:]] = OpVectorShuffle %{{[a-zA-Z0-9_]+}} %[[#LOAD]] %{{[a-zA-Z0-9_]+}} 2 3
  ; CHECK: %[[#BITCAST1:]] = OpBitcast %[[#v4i32]] %[[#VEC_SHUF1]]
  ; CHECK: %[[#BITCAST2:]] = OpBitcast %[[#v4i32]] %[[#VEC_SHUF2]]
  %0 = load <8 x i32>, ptr addrspace(10) @G_4_double

  ; CHECK: %[[#EXTRACT1:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 0
  ; CHECK: %[[#EXTRACT2:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 2
  ; CHECK: %[[#EXTRACT3:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 0
  ; CHECK: %[[#EXTRACT4:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 2
  ; CHECK: %[[#CONSTRUCT1:]] = OpCompositeConstruct %[[#v4i32]] %[[#EXTRACT1]] %[[#EXTRACT2]] %[[#EXTRACT3]] %[[#EXTRACT4]]
  %1 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  
  ; CHECK: %[[#EXTRACT5:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 1
  ; CHECK: %[[#EXTRACT6:]] = OpCompositeExtract %[[#int]] %[[#BITCAST1]] 3
  ; CHECK: %[[#EXTRACT7:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 1
  ; CHECK: %[[#EXTRACT8:]] = OpCompositeExtract %[[#int]] %[[#BITCAST2]] 3
  ; CHECK: %[[#CONSTRUCT2:]] = OpCompositeConstruct %[[#v4i32]] %[[#EXTRACT5]] %[[#EXTRACT6]] %[[#EXTRACT7]] %[[#EXTRACT8]]
  %2 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

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
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>

  ; CHECK: %[[#BITCAST3:]] = OpBitcast %[[#double]] %[[#CONSTRUCT3]]
  ; CHECK: %[[#BITCAST4:]] = OpBitcast %[[#double]] %[[#CONSTRUCT4]]
  ; CHECK: %[[#BITCAST5:]] = OpBitcast %[[#double]] %[[#CONSTRUCT5]]
  ; CHECK: %[[#BITCAST6:]] = OpBitcast %[[#double]] %[[#CONSTRUCT6]]
  ; CHECK: %[[#CONSTRUCT7:]] = OpCompositeConstruct %[[#v4f64]] %[[#BITCAST3]] %[[#BITCAST4]] %[[#BITCAST5]] %[[#BITCAST6]]
  ; CHECK: OpStore %[[#global_double]] %[[#CONSTRUCT7]] Aligned 32
  store <8 x i32> %3, ptr addrspace(10) @G_4_double
  ret void
}

; CHECK: OpFunction
define spir_func void @test_double_to_int_implicit_conversion() {
entry:

; CHECK: %[[#LOAD_V4F64:]] = OpLoad %[[#V4F64_TYPE:]] %[[#GLOBAL_DOUBLE_VAR:]] Aligned 32
; CHECK: %[[#VEC_SHUF_01:]] = OpVectorShuffle %[[#V2F64_TYPE:]] %[[#LOAD_V4F64]] %[[#UNDEF_V2F64:]] 0 1
; CHECK: %[[#VEC_SHUF_23:]] = OpVectorShuffle %[[#V2F64_TYPE:]] %[[#LOAD_V4F64]] %[[#UNDEF_V2F64]] 2 3
; CHECK: %[[#BITCAST_V4I32_01:]] = OpBitcast %[[#V4I32_TYPE:]] %[[#VEC_SHUF_01]]
; CHECK: %[[#BITCAST_V4I32_23:]] = OpBitcast %[[#V4I32_TYPE]] %[[#VEC_SHUF_23]]
  %0 = load <8 x i32>, ptr addrspace(10) @G_4_double, align 64

; CHECK: %[[#VEC_SHUF_0_0:]] = OpVectorShuffle %[[#V2I32_TYPE:]] %[[#BITCAST_V4I32_01]] %[[#UNDEF_V2I32:]] 0 1
; CHECK: %[[#VEC_SHUF_0_1:]] = OpVectorShuffle %[[#V2I32_TYPE]] %[[#BITCAST_V4I32_01]] %[[#UNDEF_V2I32]] 2 3
; CHECK: %[[#VEC_SHUF_1_0:]] = OpVectorShuffle %[[#V2I32_TYPE]] %[[#BITCAST_V4I32_23]] %[[#UNDEF_V2I32]] 0 1
; CHECK: %[[#VEC_SHUF_1_1:]] = OpVectorShuffle %[[#V2I32_TYPE]] %[[#BITCAST_V4I32_23]] %[[#UNDEF_V2I32]] 2 3
; CHECK: %[[#BITCAST_DOUBLE_0_0:]] = OpBitcast %[[#DOUBLE_TYPE:]] %[[#VEC_SHUF_0_0]]
; CHECK: %[[#BITCAST_DOUBLE_0_1:]] = OpBitcast %[[#DOUBLE_TYPE]] %[[#VEC_SHUF_0_1]]
; CHECK: %[[#BITCAST_DOUBLE_1_0:]] = OpBitcast %[[#DOUBLE_TYPE]] %[[#VEC_SHUF_1_0]]
; CHECK: %[[#BITCAST_DOUBLE_1_1:]] = OpBitcast %[[#DOUBLE_TYPE]] %[[#VEC_SHUF_1_1]]
; CHECK: %[[#COMPOSITE_CONSTRUCT:]] = OpCompositeConstruct %[[#V4F64_TYPE]] %[[#BITCAST_DOUBLE_0_0]] %[[#BITCAST_DOUBLE_0_1]] %[[#BITCAST_DOUBLE_1_0]] %[[#BITCAST_DOUBLE_1_1]]
; CHECK: OpStore %[[#GLOBAL_DOUBLE_VAR]] %[[#COMPOSITE_CONSTRUCT]] Aligned 64
  store <8 x i32> %0, ptr addrspace(10) @G_4_double, align 64
  ret void
}

; Add a main function to make it a valid module for spirv-val
define void @main() #1 {
  ret void
}

attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
