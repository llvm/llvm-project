; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Make sure llvm.frexp is lowered to GLSL.std.450 Frexp for Vulkan.

; CHECK-DAG: %[[#glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#vec4_float:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#vec4_int:]] = OpTypeVector %[[#int]] 4
; CHECK-DAG: %[[#ptr_int:]] = OpTypePointer Function %[[#int]]
; CHECK-DAG: %[[#ptr_vec4_int:]] = OpTypePointer Function %[[#vec4_int]]

define noundef float @frexp_float(float noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float]]
  ; CHECK: %[[#var:]] = OpVariable %[[#ptr_int]] Function
  ; CHECK: %[[#mant:]] = OpExtInst %[[#float]] %[[#glsl]] Frexp %[[#arg0]] %[[#var]]
  ; CHECK: %[[#exp:]] = OpLoad %[[#int]] %[[#var]]
  ; CHECK: %[[#expf:]] = OpConvertSToF %[[#float]] %[[#exp]]
  ; CHECK: %[[#]] = OpFAdd %[[#float]] %[[#mant]] %[[#expf]]
  %r = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  %m = extractvalue { float, i32 } %r, 0
  %x = extractvalue { float, i32 } %r, 1
  %xf = sitofp i32 %x to float
  %s = fadd float %m, %xf
  ret float %s
}

define noundef <4 x float> @frexp_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float]]
  ; CHECK: %[[#var:]] = OpVariable %[[#ptr_vec4_int]] Function
  ; CHECK: %[[#mant:]] = OpExtInst %[[#vec4_float]] %[[#glsl]] Frexp %[[#arg0]] %[[#var]]
  ; CHECK: %[[#exp:]] = OpLoad %[[#vec4_int]] %[[#var]]
  ; CHECK: %[[#expf:]] = OpConvertSToF %[[#vec4_float]] %[[#exp]]
  ; CHECK: %[[#]] = OpFAdd %[[#vec4_float]] %[[#mant]] %[[#expf]]
  %r = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %a)
  %m = extractvalue { <4 x float>, <4 x i32> } %r, 0
  %x = extractvalue { <4 x float>, <4 x i32> } %r, 1
  %xf = sitofp <4 x i32> %x to <4 x float>
  %s = fadd <4 x float> %m, %xf
  ret <4 x float> %s
}

; The exponent result is never extracted, e.g. when an HLSL out argument is
; dead, so its register has no SPIR-V type.
define noundef <4 x float> @frexp_exp_not_extracted(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float]]
  ; CHECK: %[[#var:]] = OpVariable %[[#ptr_vec4_int]] Function
  ; CHECK: %[[#mant:]] = OpExtInst %[[#vec4_float]] %[[#glsl]] Frexp %[[#arg0]] %[[#var]]
  ; CHECK-NOT: OpLoad %[[#vec4_int]] %[[#var]]
  ; CHECK: OpReturnValue %[[#mant]]
  %r = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %a)
  %m = extractvalue { <4 x float>, <4 x i32> } %r, 0
  ret <4 x float> %m
}
