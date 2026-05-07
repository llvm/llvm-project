; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Vulkan/Shader does not allow the Vector16 capability, so a 4x4 matrix is
; represented as [4 x <4 x float>] in LLVM IR and the elementwise atan2 is
; computed per-row as 4 OpExtInst Atan2 calls on <4 x float> (and similarly
; for half).

; CHECK-NOT: OpCapability Vector16

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#const_0:]] = OpConstant %[[#int_32]] 0
; CHECK-DAG: %[[#const_1:]] = OpConstant %[[#int_32]] 1
; CHECK-DAG: %[[#const_2:]] = OpConstant %[[#int_32]] 2
; CHECK-DAG: %[[#const_3:]] = OpConstant %[[#int_32]] 3
; CHECK-DAG: %[[#const_4:]] = OpConstant %[[#int_32]] 4
; CHECK-DAG: %[[#arr_f32:]] = OpTypeArray %[[#vec4_float_32]] %[[#const_4]]
; CHECK-DAG: %[[#arr_f16:]] = OpTypeArray %[[#vec4_float_16]] %[[#const_4]]
; CHECK-DAG: %[[#ptr_arr_f32:]] = OpTypePointer Private %[[#arr_f32]]
; CHECK-DAG: %[[#ptr_arr_f16:]] = OpTypePointer Private %[[#arr_f16]]
; CHECK-DAG: %[[#ptr_vec4_f32:]] = OpTypePointer Private %[[#vec4_float_32]]
; CHECK-DAG: %[[#ptr_vec4_f16:]] = OpTypePointer Private %[[#vec4_float_16]]
; CHECK-DAG: %[[#fn_f32:]] = OpTypeFunction %[[#void]] %[[#ptr_arr_f32]] %[[#ptr_arr_f32]] %[[#ptr_arr_f32]]
; CHECK-DAG: %[[#fn_f16:]] = OpTypeFunction %[[#void]] %[[#ptr_arr_f16]] %[[#ptr_arr_f16]] %[[#ptr_arr_f16]]

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

declare <4 x float> @llvm.atan2.v4f32(<4 x float>, <4 x float>)
declare <4 x half> @llvm.atan2.v4f16(<4 x half>, <4 x half>)
