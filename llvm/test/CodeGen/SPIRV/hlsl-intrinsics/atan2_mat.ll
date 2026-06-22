; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Vulkan/Shader does not allow the Vector16 capability, so an MxN HLSL matrix
; is represented as [M x <N x float>] in LLVM IR and elementwise atan2 is
; computed per-row as M OpExtInst Atan2 calls on <N x float> (and similarly
; for half). Matrices with N=3 exercise the legalizer's handling of
; non-power-of-2 vector widths (legal for shader via allShaderFloatVectors).

; CHECK-NOT: OpCapability Vector16

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec3_float_32:]] = OpTypeVector %[[#float_32]] 3
; CHECK-DAG: %[[#vec3_float_16:]] = OpTypeVector %[[#float_16]] 3
; CHECK-DAG: %[[#vec2_float_32:]] = OpTypeVector %[[#float_32]] 2
; CHECK-DAG: %[[#vec2_float_16:]] = OpTypeVector %[[#float_16]] 2
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#const_0:]] = OpConstant %[[#int_32]] 0
; CHECK-DAG: %[[#const_1:]] = OpConstant %[[#int_32]] 1
; CHECK-DAG: %[[#const_2:]] = OpConstant %[[#int_32]] 2
; CHECK-DAG: %[[#const_3:]] = OpConstant %[[#int_32]] 3
; CHECK-DAG: %[[#const_4:]] = OpConstant %[[#int_32]] 4
; CHECK-DAG: %[[#arr_f32:]] = OpTypeArray %[[#vec4_float_32]] %[[#const_4]]
; CHECK-DAG: %[[#arr_f16:]] = OpTypeArray %[[#vec4_float_16]] %[[#const_4]]
; CHECK-DAG: %[[#arr3_f32:]] = OpTypeArray %[[#vec3_float_32]] %[[#const_3]]
; CHECK-DAG: %[[#arr3_f16:]] = OpTypeArray %[[#vec3_float_16]] %[[#const_3]]
; CHECK-DAG: %[[#ptr_arr_f32:]] = OpTypePointer Private %[[#arr_f32]]
; CHECK-DAG: %[[#ptr_arr_f16:]] = OpTypePointer Private %[[#arr_f16]]
; CHECK-DAG: %[[#ptr_arr3_f32:]] = OpTypePointer Private %[[#arr3_f32]]
; CHECK-DAG: %[[#ptr_arr3_f16:]] = OpTypePointer Private %[[#arr3_f16]]
; CHECK-DAG: %[[#ptr_vec4_f32:]] = OpTypePointer Private %[[#vec4_float_32]]
; CHECK-DAG: %[[#ptr_vec4_f16:]] = OpTypePointer Private %[[#vec4_float_16]]
; CHECK-DAG: %[[#ptr_vec3_f32:]] = OpTypePointer Private %[[#vec3_float_32]]
; CHECK-DAG: %[[#ptr_vec3_f16:]] = OpTypePointer Private %[[#vec3_float_16]]
; CHECK-DAG: %[[#fn_f32:]] = OpTypeFunction %[[#void]] %[[#ptr_arr_f32]] %[[#ptr_arr_f32]] %[[#ptr_arr_f32]]
; CHECK-DAG: %[[#fn_f16:]] = OpTypeFunction %[[#void]] %[[#ptr_arr_f16]] %[[#ptr_arr_f16]] %[[#ptr_arr_f16]]
; CHECK-DAG: %[[#fn3_f32:]] = OpTypeFunction %[[#void]] %[[#ptr_arr3_f32]] %[[#ptr_arr3_f32]] %[[#ptr_arr3_f32]]
; CHECK-DAG: %[[#fn3_f16:]] = OpTypeFunction %[[#void]] %[[#ptr_arr3_f16]] %[[#ptr_arr3_f16]] %[[#ptr_arr3_f16]]

define internal void @atan2_float4x4(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None %[[#fn_f32]]
  ; CHECK: %[[#out_f32:]] = OpFunctionParameter %[[#ptr_arr_f32]]
  ; CHECK: %[[#a_f32:]] = OpFunctionParameter %[[#ptr_arr_f32]]
  ; CHECK: %[[#b_f32:]] = OpFunctionParameter %[[#ptr_arr_f32]]
  ; CHECK: %[[#a0_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#a_f32]] %[[#const_0]]
  ; CHECK: %[[#a1_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#a_f32]] %[[#const_1]]
  ; CHECK: %[[#a2_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#a_f32]] %[[#const_2]]
  ; CHECK: %[[#a3_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#a_f32]] %[[#const_3]]
  ; CHECK: %[[#b0_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#b_f32]] %[[#const_0]]
  ; CHECK: %[[#b1_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#b_f32]] %[[#const_1]]
  ; CHECK: %[[#b2_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#b_f32]] %[[#const_2]]
  ; CHECK: %[[#b3_ptr_f32:]] = OpAccessChain %[[#ptr_vec4_f32]] %[[#b_f32]] %[[#const_3]]
  ; CHECK: %[[#a0_f32:]] = OpLoad %[[#vec4_float_32]] %[[#a0_ptr_f32]]
  ; CHECK: %[[#a1_f32:]] = OpLoad %[[#vec4_float_32]] %[[#a1_ptr_f32]]
  ; CHECK: %[[#a2_f32:]] = OpLoad %[[#vec4_float_32]] %[[#a2_ptr_f32]]
  ; CHECK: %[[#a3_f32:]] = OpLoad %[[#vec4_float_32]] %[[#a3_ptr_f32]]
  ; CHECK: %[[#b0_f32:]] = OpLoad %[[#vec4_float_32]] %[[#b0_ptr_f32]]
  ; CHECK: %[[#b1_f32:]] = OpLoad %[[#vec4_float_32]] %[[#b1_ptr_f32]]
  ; CHECK: %[[#b2_f32:]] = OpLoad %[[#vec4_float_32]] %[[#b2_ptr_f32]]
  ; CHECK: %[[#b3_f32:]] = OpLoad %[[#vec4_float_32]] %[[#b3_ptr_f32]]
  ; CHECK: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan2 %[[#a0_f32]] %[[#b0_f32]]
  ; CHECK: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan2 %[[#a1_f32]] %[[#b1_f32]]
  ; CHECK: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan2 %[[#a2_f32]] %[[#b2_f32]]
  ; CHECK: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan2 %[[#a3_f32]] %[[#b3_f32]]
  %a0 = getelementptr [4 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [4 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [4 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 2
  %a3 = getelementptr [4 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 3
  %b0 = getelementptr [4 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [4 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [4 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 2
  %b3 = getelementptr [4 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 3
  %va0 = load <4 x float>, ptr addrspace(10) %a0
  %va1 = load <4 x float>, ptr addrspace(10) %a1
  %va2 = load <4 x float>, ptr addrspace(10) %a2
  %va3 = load <4 x float>, ptr addrspace(10) %a3
  %vb0 = load <4 x float>, ptr addrspace(10) %b0
  %vb1 = load <4 x float>, ptr addrspace(10) %b1
  %vb2 = load <4 x float>, ptr addrspace(10) %b2
  %vb3 = load <4 x float>, ptr addrspace(10) %b3
  %r0 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va0, <4 x float> %vb0)
  %r1 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va1, <4 x float> %vb1)
  %r2 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va2, <4 x float> %vb2)
  %r3 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va3, <4 x float> %vb3)
  %out0 = getelementptr [4 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [4 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [4 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 2
  %out3 = getelementptr [4 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 3
  store <4 x float> %r0, ptr addrspace(10) %out0
  store <4 x float> %r1, ptr addrspace(10) %out1
  store <4 x float> %r2, ptr addrspace(10) %out2
  store <4 x float> %r3, ptr addrspace(10) %out3
  ret void
}

define internal void @atan2_half4x4(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None %[[#fn_f16]]
  ; CHECK: %[[#out_f16:]] = OpFunctionParameter %[[#ptr_arr_f16]]
  ; CHECK: %[[#a_f16:]] = OpFunctionParameter %[[#ptr_arr_f16]]
  ; CHECK: %[[#b_f16:]] = OpFunctionParameter %[[#ptr_arr_f16]]
  ; CHECK: %[[#a0_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#a_f16]] %[[#const_0]]
  ; CHECK: %[[#a1_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#a_f16]] %[[#const_1]]
  ; CHECK: %[[#a2_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#a_f16]] %[[#const_2]]
  ; CHECK: %[[#a3_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#a_f16]] %[[#const_3]]
  ; CHECK: %[[#b0_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#b_f16]] %[[#const_0]]
  ; CHECK: %[[#b1_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#b_f16]] %[[#const_1]]
  ; CHECK: %[[#b2_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#b_f16]] %[[#const_2]]
  ; CHECK: %[[#b3_ptr_f16:]] = OpAccessChain %[[#ptr_vec4_f16]] %[[#b_f16]] %[[#const_3]]
  ; CHECK: %[[#a0_f16:]] = OpLoad %[[#vec4_float_16]] %[[#a0_ptr_f16]]
  ; CHECK: %[[#a1_f16:]] = OpLoad %[[#vec4_float_16]] %[[#a1_ptr_f16]]
  ; CHECK: %[[#a2_f16:]] = OpLoad %[[#vec4_float_16]] %[[#a2_ptr_f16]]
  ; CHECK: %[[#a3_f16:]] = OpLoad %[[#vec4_float_16]] %[[#a3_ptr_f16]]
  ; CHECK: %[[#b0_f16:]] = OpLoad %[[#vec4_float_16]] %[[#b0_ptr_f16]]
  ; CHECK: %[[#b1_f16:]] = OpLoad %[[#vec4_float_16]] %[[#b1_ptr_f16]]
  ; CHECK: %[[#b2_f16:]] = OpLoad %[[#vec4_float_16]] %[[#b2_ptr_f16]]
  ; CHECK: %[[#b3_f16:]] = OpLoad %[[#vec4_float_16]] %[[#b3_ptr_f16]]
  ; CHECK: OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Atan2 %[[#a0_f16]] %[[#b0_f16]]
  ; CHECK: OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Atan2 %[[#a1_f16]] %[[#b1_f16]]
  ; CHECK: OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Atan2 %[[#a2_f16]] %[[#b2_f16]]
  ; CHECK: OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Atan2 %[[#a3_f16]] %[[#b3_f16]]
  %a0 = getelementptr [4 x <4 x half>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [4 x <4 x half>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [4 x <4 x half>], ptr addrspace(10) %a, i32 0, i32 2
  %a3 = getelementptr [4 x <4 x half>], ptr addrspace(10) %a, i32 0, i32 3
  %b0 = getelementptr [4 x <4 x half>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [4 x <4 x half>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [4 x <4 x half>], ptr addrspace(10) %b, i32 0, i32 2
  %b3 = getelementptr [4 x <4 x half>], ptr addrspace(10) %b, i32 0, i32 3
  %va0 = load <4 x half>, ptr addrspace(10) %a0
  %va1 = load <4 x half>, ptr addrspace(10) %a1
  %va2 = load <4 x half>, ptr addrspace(10) %a2
  %va3 = load <4 x half>, ptr addrspace(10) %a3
  %vb0 = load <4 x half>, ptr addrspace(10) %b0
  %vb1 = load <4 x half>, ptr addrspace(10) %b1
  %vb2 = load <4 x half>, ptr addrspace(10) %b2
  %vb3 = load <4 x half>, ptr addrspace(10) %b3
  %r0 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %va0, <4 x half> %vb0)
  %r1 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %va1, <4 x half> %vb1)
  %r2 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %va2, <4 x half> %vb2)
  %r3 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %va3, <4 x half> %vb3)
  %out0 = getelementptr [4 x <4 x half>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [4 x <4 x half>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [4 x <4 x half>], ptr addrspace(10) %out, i32 0, i32 2
  %out3 = getelementptr [4 x <4 x half>], ptr addrspace(10) %out, i32 0, i32 3
  store <4 x half> %r0, ptr addrspace(10) %out0
  store <4 x half> %r1, ptr addrspace(10) %out1
  store <4 x half> %r2, ptr addrspace(10) %out2
  store <4 x half> %r3, ptr addrspace(10) %out3
  ret void
}

define internal void @atan2_float3x3(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None %[[#fn3_f32]]
  ; CHECK: %[[#out3_f32:]] = OpFunctionParameter %[[#ptr_arr3_f32]]
  ; CHECK: %[[#a3v_f32:]] = OpFunctionParameter %[[#ptr_arr3_f32]]
  ; CHECK: %[[#b3v_f32:]] = OpFunctionParameter %[[#ptr_arr3_f32]]
  ; CHECK: %[[#a3v0_ptr_f32:]] = OpAccessChain %[[#ptr_vec3_f32]] %[[#a3v_f32]] %[[#const_0]]
  ; CHECK: %[[#a3v1_ptr_f32:]] = OpAccessChain %[[#ptr_vec3_f32]] %[[#a3v_f32]] %[[#const_1]]
  ; CHECK: %[[#a3v2_ptr_f32:]] = OpAccessChain %[[#ptr_vec3_f32]] %[[#a3v_f32]] %[[#const_2]]
  ; CHECK: %[[#b3v0_ptr_f32:]] = OpAccessChain %[[#ptr_vec3_f32]] %[[#b3v_f32]] %[[#const_0]]
  ; CHECK: %[[#b3v1_ptr_f32:]] = OpAccessChain %[[#ptr_vec3_f32]] %[[#b3v_f32]] %[[#const_1]]
  ; CHECK: %[[#b3v2_ptr_f32:]] = OpAccessChain %[[#ptr_vec3_f32]] %[[#b3v_f32]] %[[#const_2]]
  ; CHECK: %[[#a3v0_f32:]] = OpLoad %[[#vec3_float_32]] %[[#a3v0_ptr_f32]]
  ; CHECK: %[[#a3v1_f32:]] = OpLoad %[[#vec3_float_32]] %[[#a3v1_ptr_f32]]
  ; CHECK: %[[#a3v2_f32:]] = OpLoad %[[#vec3_float_32]] %[[#a3v2_ptr_f32]]
  ; CHECK: %[[#b3v0_f32:]] = OpLoad %[[#vec3_float_32]] %[[#b3v0_ptr_f32]]
  ; CHECK: %[[#b3v1_f32:]] = OpLoad %[[#vec3_float_32]] %[[#b3v1_ptr_f32]]
  ; CHECK: %[[#b3v2_f32:]] = OpLoad %[[#vec3_float_32]] %[[#b3v2_ptr_f32]]
  ; CHECK: OpExtInst %[[#vec3_float_32]] %[[#op_ext_glsl]] Atan2 %[[#a3v0_f32]] %[[#b3v0_f32]]
  ; CHECK: OpExtInst %[[#vec3_float_32]] %[[#op_ext_glsl]] Atan2 %[[#a3v1_f32]] %[[#b3v1_f32]]
  ; CHECK: OpExtInst %[[#vec3_float_32]] %[[#op_ext_glsl]] Atan2 %[[#a3v2_f32]] %[[#b3v2_f32]]
  %a0 = getelementptr [3 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [3 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [3 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 2
  %b0 = getelementptr [3 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [3 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [3 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 2
  %va0 = load <3 x float>, ptr addrspace(10) %a0
  %va1 = load <3 x float>, ptr addrspace(10) %a1
  %va2 = load <3 x float>, ptr addrspace(10) %a2
  %vb0 = load <3 x float>, ptr addrspace(10) %b0
  %vb1 = load <3 x float>, ptr addrspace(10) %b1
  %vb2 = load <3 x float>, ptr addrspace(10) %b2
  %r0 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va0, <3 x float> %vb0)
  %r1 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va1, <3 x float> %vb1)
  %r2 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va2, <3 x float> %vb2)
  %out0 = getelementptr [3 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [3 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [3 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 2
  store <3 x float> %r0, ptr addrspace(10) %out0
  store <3 x float> %r1, ptr addrspace(10) %out1
  store <3 x float> %r2, ptr addrspace(10) %out2
  ret void
}

define internal void @atan2_half3x3(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None %[[#fn3_f16]]
  ; CHECK: %[[#out3_f16:]] = OpFunctionParameter %[[#ptr_arr3_f16]]
  ; CHECK: %[[#a3v_f16:]] = OpFunctionParameter %[[#ptr_arr3_f16]]
  ; CHECK: %[[#b3v_f16:]] = OpFunctionParameter %[[#ptr_arr3_f16]]
  ; CHECK: %[[#a3v0_ptr_f16:]] = OpAccessChain %[[#ptr_vec3_f16]] %[[#a3v_f16]] %[[#const_0]]
  ; CHECK: %[[#a3v1_ptr_f16:]] = OpAccessChain %[[#ptr_vec3_f16]] %[[#a3v_f16]] %[[#const_1]]
  ; CHECK: %[[#a3v2_ptr_f16:]] = OpAccessChain %[[#ptr_vec3_f16]] %[[#a3v_f16]] %[[#const_2]]
  ; CHECK: %[[#b3v0_ptr_f16:]] = OpAccessChain %[[#ptr_vec3_f16]] %[[#b3v_f16]] %[[#const_0]]
  ; CHECK: %[[#b3v1_ptr_f16:]] = OpAccessChain %[[#ptr_vec3_f16]] %[[#b3v_f16]] %[[#const_1]]
  ; CHECK: %[[#b3v2_ptr_f16:]] = OpAccessChain %[[#ptr_vec3_f16]] %[[#b3v_f16]] %[[#const_2]]
  ; CHECK: %[[#a3v0_f16:]] = OpLoad %[[#vec3_float_16]] %[[#a3v0_ptr_f16]]
  ; CHECK: %[[#a3v1_f16:]] = OpLoad %[[#vec3_float_16]] %[[#a3v1_ptr_f16]]
  ; CHECK: %[[#a3v2_f16:]] = OpLoad %[[#vec3_float_16]] %[[#a3v2_ptr_f16]]
  ; CHECK: %[[#b3v0_f16:]] = OpLoad %[[#vec3_float_16]] %[[#b3v0_ptr_f16]]
  ; CHECK: %[[#b3v1_f16:]] = OpLoad %[[#vec3_float_16]] %[[#b3v1_ptr_f16]]
  ; CHECK: %[[#b3v2_f16:]] = OpLoad %[[#vec3_float_16]] %[[#b3v2_ptr_f16]]
  ; CHECK: OpExtInst %[[#vec3_float_16]] %[[#op_ext_glsl]] Atan2 %[[#a3v0_f16]] %[[#b3v0_f16]]
  ; CHECK: OpExtInst %[[#vec3_float_16]] %[[#op_ext_glsl]] Atan2 %[[#a3v1_f16]] %[[#b3v1_f16]]
  ; CHECK: OpExtInst %[[#vec3_float_16]] %[[#op_ext_glsl]] Atan2 %[[#a3v2_f16]] %[[#b3v2_f16]]
  %a0 = getelementptr [3 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [3 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [3 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 2
  %b0 = getelementptr [3 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [3 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [3 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 2
  %va0 = load <3 x half>, ptr addrspace(10) %a0
  %va1 = load <3 x half>, ptr addrspace(10) %a1
  %va2 = load <3 x half>, ptr addrspace(10) %a2
  %vb0 = load <3 x half>, ptr addrspace(10) %b0
  %vb1 = load <3 x half>, ptr addrspace(10) %b1
  %vb2 = load <3 x half>, ptr addrspace(10) %b2
  %r0 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va0, <3 x half> %vb0)
  %r1 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va1, <3 x half> %vb1)
  %r2 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va2, <3 x half> %vb2)
  %out0 = getelementptr [3 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [3 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [3 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 2
  store <3 x half> %r0, ptr addrspace(10) %out0
  store <3 x half> %r1, ptr addrspace(10) %out1
  store <3 x half> %r2, ptr addrspace(10) %out2
  ret void
}

define internal void @atan2_float2x2(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec2_float_32]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [2 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [2 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %b0 = getelementptr [2 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [2 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %va0 = load <2 x float>, ptr addrspace(10) %a0
  %va1 = load <2 x float>, ptr addrspace(10) %a1
  %vb0 = load <2 x float>, ptr addrspace(10) %b0
  %vb1 = load <2 x float>, ptr addrspace(10) %b1
  %r0 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va0, <2 x float> %vb0)
  %r1 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va1, <2 x float> %vb1)
  %out0 = getelementptr [2 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [2 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 1
  store <2 x float> %r0, ptr addrspace(10) %out0
  store <2 x float> %r1, ptr addrspace(10) %out1
  ret void
}

define internal void @atan2_half2x2(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec2_float_16]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [2 x <2 x half>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [2 x <2 x half>], ptr addrspace(10) %a, i32 0, i32 1
  %b0 = getelementptr [2 x <2 x half>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [2 x <2 x half>], ptr addrspace(10) %b, i32 0, i32 1
  %va0 = load <2 x half>, ptr addrspace(10) %a0
  %va1 = load <2 x half>, ptr addrspace(10) %a1
  %vb0 = load <2 x half>, ptr addrspace(10) %b0
  %vb1 = load <2 x half>, ptr addrspace(10) %b1
  %r0 = call <2 x half> @llvm.atan2.v2f16(<2 x half> %va0, <2 x half> %vb0)
  %r1 = call <2 x half> @llvm.atan2.v2f16(<2 x half> %va1, <2 x half> %vb1)
  %out0 = getelementptr [2 x <2 x half>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [2 x <2 x half>], ptr addrspace(10) %out, i32 0, i32 1
  store <2 x half> %r0, ptr addrspace(10) %out0
  store <2 x half> %r1, ptr addrspace(10) %out1
  ret void
}

; 2x3: 2 rows of <3 x float> — non-power-of-2 vector width.
define internal void @atan2_float2x3(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec3_float_32]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [2 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [2 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %b0 = getelementptr [2 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [2 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %va0 = load <3 x float>, ptr addrspace(10) %a0
  %va1 = load <3 x float>, ptr addrspace(10) %a1
  %vb0 = load <3 x float>, ptr addrspace(10) %b0
  %vb1 = load <3 x float>, ptr addrspace(10) %b1
  %r0 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va0, <3 x float> %vb0)
  %r1 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va1, <3 x float> %vb1)
  %out0 = getelementptr [2 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [2 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 1
  store <3 x float> %r0, ptr addrspace(10) %out0
  store <3 x float> %r1, ptr addrspace(10) %out1
  ret void
}

define internal void @atan2_half2x3(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec3_float_16]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [2 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [2 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 1
  %b0 = getelementptr [2 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [2 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 1
  %va0 = load <3 x half>, ptr addrspace(10) %a0
  %va1 = load <3 x half>, ptr addrspace(10) %a1
  %vb0 = load <3 x half>, ptr addrspace(10) %b0
  %vb1 = load <3 x half>, ptr addrspace(10) %b1
  %r0 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va0, <3 x half> %vb0)
  %r1 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va1, <3 x half> %vb1)
  %out0 = getelementptr [2 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [2 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 1
  store <3 x half> %r0, ptr addrspace(10) %out0
  store <3 x half> %r1, ptr addrspace(10) %out1
  ret void
}

define internal void @atan2_float3x2(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-3: OpExtInst %[[#vec2_float_32]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [3 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [3 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [3 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 2
  %b0 = getelementptr [3 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [3 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [3 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 2
  %va0 = load <2 x float>, ptr addrspace(10) %a0
  %va1 = load <2 x float>, ptr addrspace(10) %a1
  %va2 = load <2 x float>, ptr addrspace(10) %a2
  %vb0 = load <2 x float>, ptr addrspace(10) %b0
  %vb1 = load <2 x float>, ptr addrspace(10) %b1
  %vb2 = load <2 x float>, ptr addrspace(10) %b2
  %r0 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va0, <2 x float> %vb0)
  %r1 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va1, <2 x float> %vb1)
  %r2 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va2, <2 x float> %vb2)
  %out0 = getelementptr [3 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [3 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [3 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 2
  store <2 x float> %r0, ptr addrspace(10) %out0
  store <2 x float> %r1, ptr addrspace(10) %out1
  store <2 x float> %r2, ptr addrspace(10) %out2
  ret void
}

define internal void @atan2_float2x4(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [2 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [2 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %b0 = getelementptr [2 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [2 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %va0 = load <4 x float>, ptr addrspace(10) %a0
  %va1 = load <4 x float>, ptr addrspace(10) %a1
  %vb0 = load <4 x float>, ptr addrspace(10) %b0
  %vb1 = load <4 x float>, ptr addrspace(10) %b1
  %r0 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va0, <4 x float> %vb0)
  %r1 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va1, <4 x float> %vb1)
  %out0 = getelementptr [2 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [2 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 1
  store <4 x float> %r0, ptr addrspace(10) %out0
  store <4 x float> %r1, ptr addrspace(10) %out1
  ret void
}

define internal void @atan2_float4x2(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec2_float_32]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [4 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [4 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [4 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 2
  %a3 = getelementptr [4 x <2 x float>], ptr addrspace(10) %a, i32 0, i32 3
  %b0 = getelementptr [4 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [4 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [4 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 2
  %b3 = getelementptr [4 x <2 x float>], ptr addrspace(10) %b, i32 0, i32 3
  %va0 = load <2 x float>, ptr addrspace(10) %a0
  %va1 = load <2 x float>, ptr addrspace(10) %a1
  %va2 = load <2 x float>, ptr addrspace(10) %a2
  %va3 = load <2 x float>, ptr addrspace(10) %a3
  %vb0 = load <2 x float>, ptr addrspace(10) %b0
  %vb1 = load <2 x float>, ptr addrspace(10) %b1
  %vb2 = load <2 x float>, ptr addrspace(10) %b2
  %vb3 = load <2 x float>, ptr addrspace(10) %b3
  %r0 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va0, <2 x float> %vb0)
  %r1 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va1, <2 x float> %vb1)
  %r2 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va2, <2 x float> %vb2)
  %r3 = call <2 x float> @llvm.atan2.v2f32(<2 x float> %va3, <2 x float> %vb3)
  %out0 = getelementptr [4 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [4 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [4 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 2
  %out3 = getelementptr [4 x <2 x float>], ptr addrspace(10) %out, i32 0, i32 3
  store <2 x float> %r0, ptr addrspace(10) %out0
  store <2 x float> %r1, ptr addrspace(10) %out1
  store <2 x float> %r2, ptr addrspace(10) %out2
  store <2 x float> %r3, ptr addrspace(10) %out3
  ret void
}

; 3x4: 3 rows of <4 x float>.
define internal void @atan2_float3x4(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-3: OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [3 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [3 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [3 x <4 x float>], ptr addrspace(10) %a, i32 0, i32 2
  %b0 = getelementptr [3 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [3 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [3 x <4 x float>], ptr addrspace(10) %b, i32 0, i32 2
  %va0 = load <4 x float>, ptr addrspace(10) %a0
  %va1 = load <4 x float>, ptr addrspace(10) %a1
  %va2 = load <4 x float>, ptr addrspace(10) %a2
  %vb0 = load <4 x float>, ptr addrspace(10) %b0
  %vb1 = load <4 x float>, ptr addrspace(10) %b1
  %vb2 = load <4 x float>, ptr addrspace(10) %b2
  %r0 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va0, <4 x float> %vb0)
  %r1 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va1, <4 x float> %vb1)
  %r2 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %va2, <4 x float> %vb2)
  %out0 = getelementptr [3 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [3 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [3 x <4 x float>], ptr addrspace(10) %out, i32 0, i32 2
  store <4 x float> %r0, ptr addrspace(10) %out0
  store <4 x float> %r1, ptr addrspace(10) %out1
  store <4 x float> %r2, ptr addrspace(10) %out2
  ret void
}

define internal void @atan2_half3x4(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-3: OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [3 x <4 x half>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [3 x <4 x half>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [3 x <4 x half>], ptr addrspace(10) %a, i32 0, i32 2
  %b0 = getelementptr [3 x <4 x half>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [3 x <4 x half>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [3 x <4 x half>], ptr addrspace(10) %b, i32 0, i32 2
  %va0 = load <4 x half>, ptr addrspace(10) %a0
  %va1 = load <4 x half>, ptr addrspace(10) %a1
  %va2 = load <4 x half>, ptr addrspace(10) %a2
  %vb0 = load <4 x half>, ptr addrspace(10) %b0
  %vb1 = load <4 x half>, ptr addrspace(10) %b1
  %vb2 = load <4 x half>, ptr addrspace(10) %b2
  %r0 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %va0, <4 x half> %vb0)
  %r1 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %va1, <4 x half> %vb1)
  %r2 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %va2, <4 x half> %vb2)
  %out0 = getelementptr [3 x <4 x half>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [3 x <4 x half>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [3 x <4 x half>], ptr addrspace(10) %out, i32 0, i32 2
  store <4 x half> %r0, ptr addrspace(10) %out0
  store <4 x half> %r1, ptr addrspace(10) %out1
  store <4 x half> %r2, ptr addrspace(10) %out2
  ret void
}

; 4x3: 4 rows of <3 x float> — non-power-of-2 vector width.
define internal void @atan2_float4x3(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec3_float_32]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [4 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [4 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [4 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 2
  %a3 = getelementptr [4 x <3 x float>], ptr addrspace(10) %a, i32 0, i32 3
  %b0 = getelementptr [4 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [4 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [4 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 2
  %b3 = getelementptr [4 x <3 x float>], ptr addrspace(10) %b, i32 0, i32 3
  %va0 = load <3 x float>, ptr addrspace(10) %a0
  %va1 = load <3 x float>, ptr addrspace(10) %a1
  %va2 = load <3 x float>, ptr addrspace(10) %a2
  %va3 = load <3 x float>, ptr addrspace(10) %a3
  %vb0 = load <3 x float>, ptr addrspace(10) %b0
  %vb1 = load <3 x float>, ptr addrspace(10) %b1
  %vb2 = load <3 x float>, ptr addrspace(10) %b2
  %vb3 = load <3 x float>, ptr addrspace(10) %b3
  %r0 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va0, <3 x float> %vb0)
  %r1 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va1, <3 x float> %vb1)
  %r2 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va2, <3 x float> %vb2)
  %r3 = call <3 x float> @llvm.atan2.v3f32(<3 x float> %va3, <3 x float> %vb3)
  %out0 = getelementptr [4 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [4 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [4 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 2
  %out3 = getelementptr [4 x <3 x float>], ptr addrspace(10) %out, i32 0, i32 3
  store <3 x float> %r0, ptr addrspace(10) %out0
  store <3 x float> %r1, ptr addrspace(10) %out1
  store <3 x float> %r2, ptr addrspace(10) %out2
  store <3 x float> %r3, ptr addrspace(10) %out3
  ret void
}

define internal void @atan2_half4x3(ptr addrspace(10) %out, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec3_float_16]] %[[#op_ext_glsl]] Atan2
  %a0 = getelementptr [4 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 0
  %a1 = getelementptr [4 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 1
  %a2 = getelementptr [4 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 2
  %a3 = getelementptr [4 x <3 x half>], ptr addrspace(10) %a, i32 0, i32 3
  %b0 = getelementptr [4 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 0
  %b1 = getelementptr [4 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 1
  %b2 = getelementptr [4 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 2
  %b3 = getelementptr [4 x <3 x half>], ptr addrspace(10) %b, i32 0, i32 3
  %va0 = load <3 x half>, ptr addrspace(10) %a0
  %va1 = load <3 x half>, ptr addrspace(10) %a1
  %va2 = load <3 x half>, ptr addrspace(10) %a2
  %va3 = load <3 x half>, ptr addrspace(10) %a3
  %vb0 = load <3 x half>, ptr addrspace(10) %b0
  %vb1 = load <3 x half>, ptr addrspace(10) %b1
  %vb2 = load <3 x half>, ptr addrspace(10) %b2
  %vb3 = load <3 x half>, ptr addrspace(10) %b3
  %r0 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va0, <3 x half> %vb0)
  %r1 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va1, <3 x half> %vb1)
  %r2 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va2, <3 x half> %vb2)
  %r3 = call <3 x half> @llvm.atan2.v3f16(<3 x half> %va3, <3 x half> %vb3)
  %out0 = getelementptr [4 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 0
  %out1 = getelementptr [4 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 1
  %out2 = getelementptr [4 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 2
  %out3 = getelementptr [4 x <3 x half>], ptr addrspace(10) %out, i32 0, i32 3
  store <3 x half> %r0, ptr addrspace(10) %out0
  store <3 x half> %r1, ptr addrspace(10) %out1
  store <3 x half> %r2, ptr addrspace(10) %out2
  store <3 x half> %r3, ptr addrspace(10) %out3
  ret void
}

declare <4 x float> @llvm.atan2.v4f32(<4 x float>, <4 x float>)
declare <4 x half> @llvm.atan2.v4f16(<4 x half>, <4 x half>)
declare <3 x float> @llvm.atan2.v3f32(<3 x float>, <3 x float>)
declare <3 x half> @llvm.atan2.v3f16(<3 x half>, <3 x half>)
declare <2 x float> @llvm.atan2.v2f32(<2 x float>, <2 x float>)
declare <2 x half> @llvm.atan2.v2f16(<2 x half>, <2 x half>)
