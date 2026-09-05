; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Matrices with N=3 exercise the legalizer's handling of non-power-of-2 
; vector widths (legal for shader via allShaderFloatVectors and allShaderIntVectors).

; CHECK-NOT: OpCapability Vector16

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#vec4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec2_int_32:]] = OpTypeVector %[[#int_32]] 2
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec2_float_32:]] = OpTypeVector %[[#float_32]] 2

@wide_i32_6 = internal addrspace(10) global [6 x i32] zeroinitializer
@wide_i32_8 = internal addrspace(10) global [8 x i32] zeroinitializer
@wide_i32_9 = internal addrspace(10) global [9 x i32] zeroinitializer
@wide_i32_12 = internal addrspace(10) global [12 x i32] zeroinitializer
@wide_i32_16 = internal addrspace(10) global [16 x i32] zeroinitializer
@wide_u32_6 = internal addrspace(10) global [6 x i32] zeroinitializer
@wide_u32_8 = internal addrspace(10) global [8 x i32] zeroinitializer
@wide_u32_9 = internal addrspace(10) global [9 x i32] zeroinitializer
@wide_u32_12 = internal addrspace(10) global [12 x i32] zeroinitializer
@wide_u32_16 = internal addrspace(10) global [16 x i32] zeroinitializer
@wide_f32_6 = internal addrspace(10) global [6 x float] zeroinitializer
@wide_f32_8 = internal addrspace(10) global [8 x float] zeroinitializer
@wide_f32_9 = internal addrspace(10) global [9 x float] zeroinitializer
@wide_f32_12 = internal addrspace(10) global [12 x float] zeroinitializer
@wide_f32_16 = internal addrspace(10) global [16 x float] zeroinitializer

define internal void @min_int6() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] SMin
  ; CHECK: OpExtInst %[[#vec2_int_32]] %[[#op_ext_glsl]] SMin
  ; CHECK: OpFunctionEnd
  %va = load <6 x i32>, ptr addrspace(10) @wide_i32_6
  %vb = load <6 x i32>, ptr addrspace(10) @wide_i32_6
  %r = call <6 x i32> @llvm.smin.v6i32(<6 x i32> %va, <6 x i32> %vb)
  store <6 x i32> %r, ptr addrspace(10) @wide_i32_6
  ret void
}

define internal void @min_int8() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] SMin
  ; CHECK: OpFunctionEnd
  %va = load <8 x i32>, ptr addrspace(10) @wide_i32_8
  %vb = load <8 x i32>, ptr addrspace(10) @wide_i32_8
  %r = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %va, <8 x i32> %vb)
  store <8 x i32> %r, ptr addrspace(10) @wide_i32_8
  ret void
}

define internal void @min_int9() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] SMin
  ; CHECK: OpExtInst %[[#int_32]] %[[#op_ext_glsl]] SMin
  ; CHECK: OpFunctionEnd
  %va = load <9 x i32>, ptr addrspace(10) @wide_i32_9
  %vb = load <9 x i32>, ptr addrspace(10) @wide_i32_9
  %r = call <9 x i32> @llvm.smin.v9i32(<9 x i32> %va, <9 x i32> %vb)
  store <9 x i32> %r, ptr addrspace(10) @wide_i32_9
  ret void
}

define internal void @min_int12() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-3: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] SMin
  ; CHECK: OpFunctionEnd
  %va = load <12 x i32>, ptr addrspace(10) @wide_i32_12
  %vb = load <12 x i32>, ptr addrspace(10) @wide_i32_12
  %r = call <12 x i32> @llvm.smin.v12i32(<12 x i32> %va, <12 x i32> %vb)
  store <12 x i32> %r, ptr addrspace(10) @wide_i32_12
  ret void
}

define internal void @min_int16() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] SMin
  ; CHECK: OpFunctionEnd
  %va = load <16 x i32>, ptr addrspace(10) @wide_i32_16
  %vb = load <16 x i32>, ptr addrspace(10) @wide_i32_16
  %r = call <16 x i32> @llvm.smin.v16i32(<16 x i32> %va, <16 x i32> %vb)
  store <16 x i32> %r, ptr addrspace(10) @wide_i32_16
  ret void
}

define internal void @min_uint6() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] UMin
  ; CHECK: OpExtInst %[[#vec2_int_32]] %[[#op_ext_glsl]] UMin
  ; CHECK: OpFunctionEnd
  %va = load <6 x i32>, ptr addrspace(10) @wide_u32_6
  %vb = load <6 x i32>, ptr addrspace(10) @wide_u32_6
  %r = call <6 x i32> @llvm.umin.v6i32(<6 x i32> %va, <6 x i32> %vb)
  store <6 x i32> %r, ptr addrspace(10) @wide_u32_6
  ret void
}

define internal void @min_uint8() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] UMin
  ; CHECK: OpFunctionEnd
  %va = load <8 x i32>, ptr addrspace(10) @wide_u32_8
  %vb = load <8 x i32>, ptr addrspace(10) @wide_u32_8
  %r = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %va, <8 x i32> %vb)
  store <8 x i32> %r, ptr addrspace(10) @wide_u32_8
  ret void
}

define internal void @min_uint9() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] UMin
  ; CHECK: OpExtInst %[[#int_32]] %[[#op_ext_glsl]] UMin
  ; CHECK: OpFunctionEnd
  %va = load <9 x i32>, ptr addrspace(10) @wide_u32_9
  %vb = load <9 x i32>, ptr addrspace(10) @wide_u32_9
  %r = call <9 x i32> @llvm.umin.v9i32(<9 x i32> %va, <9 x i32> %vb)
  store <9 x i32> %r, ptr addrspace(10) @wide_u32_9
  ret void
}

define internal void @min_uint12() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-3: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] UMin
  ; CHECK: OpFunctionEnd
  %va = load <12 x i32>, ptr addrspace(10) @wide_u32_12
  %vb = load <12 x i32>, ptr addrspace(10) @wide_u32_12
  %r = call <12 x i32> @llvm.umin.v12i32(<12 x i32> %va, <12 x i32> %vb)
  store <12 x i32> %r, ptr addrspace(10) @wide_u32_12
  ret void
}

define internal void @min_uint16() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] UMin
  ; CHECK: OpFunctionEnd
  %va = load <16 x i32>, ptr addrspace(10) @wide_u32_16
  %vb = load <16 x i32>, ptr addrspace(10) @wide_u32_16
  %r = call <16 x i32> @llvm.umin.v16i32(<16 x i32> %va, <16 x i32> %vb)
  store <16 x i32> %r, ptr addrspace(10) @wide_u32_16
  ret void
}

define internal void @min_float6() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] NMin
  ; CHECK: OpExtInst %[[#vec2_float_32]] %[[#op_ext_glsl]] NMin
  ; CHECK: OpFunctionEnd
  %va = load <6 x float>, ptr addrspace(10) @wide_f32_6
  %vb = load <6 x float>, ptr addrspace(10) @wide_f32_6
  %r = call <6 x float> @llvm.minnum.v6f32(<6 x float> %va, <6 x float> %vb)
  store <6 x float> %r, ptr addrspace(10) @wide_f32_6
  ret void
}

define internal void @min_float8() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] NMin
  ; CHECK: OpFunctionEnd
  %va = load <8 x float>, ptr addrspace(10) @wide_f32_8
  %vb = load <8 x float>, ptr addrspace(10) @wide_f32_8
  %r = call <8 x float> @llvm.minnum.v8f32(<8 x float> %va, <8 x float> %vb)
  store <8 x float> %r, ptr addrspace(10) @wide_f32_8
  ret void
}

define internal void @min_float9() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] NMin
  ; CHECK: OpExtInst %[[#float_32]] %[[#op_ext_glsl]] NMin
  ; CHECK: OpFunctionEnd
  %va = load <9 x float>, ptr addrspace(10) @wide_f32_9
  %vb = load <9 x float>, ptr addrspace(10) @wide_f32_9
  %r = call <9 x float> @llvm.minnum.v9f32(<9 x float> %va, <9 x float> %vb)
  store <9 x float> %r, ptr addrspace(10) @wide_f32_9
  ret void
}

define internal void @min_float12() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-3: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] NMin
  ; CHECK: OpFunctionEnd
  %va = load <12 x float>, ptr addrspace(10) @wide_f32_12
  %vb = load <12 x float>, ptr addrspace(10) @wide_f32_12
  %r = call <12 x float> @llvm.minnum.v12f32(<12 x float> %va, <12 x float> %vb)
  store <12 x float> %r, ptr addrspace(10) @wide_f32_12
  ret void
}

define internal void @min_float16() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] NMin
  ; CHECK: OpFunctionEnd
  %va = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %vb = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %r = call <16 x float> @llvm.minnum.v16f32(<16 x float> %va, <16 x float> %vb)
  store <16 x float> %r, ptr addrspace(10) @wide_f32_16
  ret void
}

define void @main() #0 {
entry:
  call void @min_int6()
  call void @min_int8()
  call void @min_int9()
  call void @min_int12()
  call void @min_int16()
  call void @min_uint6()
  call void @min_uint8()
  call void @min_uint9()
  call void @min_uint12()
  call void @min_uint16()
  call void @min_float6()
  call void @min_float8()
  call void @min_float9()
  call void @min_float12()
  call void @min_float16()
  ret void
}

declare <6 x i32> @llvm.smin.v6i32(<6 x i32>, <6 x i32>)
declare <8 x i32> @llvm.smin.v8i32(<8 x i32>, <8 x i32>)
declare <9 x i32> @llvm.smin.v9i32(<9 x i32>, <9 x i32>)
declare <12 x i32> @llvm.smin.v12i32(<12 x i32>, <12 x i32>)
declare <16 x i32> @llvm.smin.v16i32(<16 x i32>, <16 x i32>)
declare <6 x i32> @llvm.umin.v6i32(<6 x i32>, <6 x i32>)
declare <8 x i32> @llvm.umin.v8i32(<8 x i32>, <8 x i32>)
declare <9 x i32> @llvm.umin.v9i32(<9 x i32>, <9 x i32>)
declare <12 x i32> @llvm.umin.v12i32(<12 x i32>, <12 x i32>)
declare <16 x i32> @llvm.umin.v16i32(<16 x i32>, <16 x i32>)
declare <6 x float> @llvm.minnum.v6f32(<6 x float>, <6 x float>)
declare <8 x float> @llvm.minnum.v8f32(<8 x float>, <8 x float>)
declare <9 x float> @llvm.minnum.v9f32(<9 x float>, <9 x float>)
declare <12 x float> @llvm.minnum.v12f32(<12 x float>, <12 x float>)
declare <16 x float> @llvm.minnum.v16f32(<16 x float>, <16 x float>)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
