; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#extinst_id:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32_type:]] = OpTypeFloat 32
; CHECK-DAG: %[[#int_32_type:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#fn_ptr_type_i32:]] = OpTypePointer Function %[[#int_32_type]]
; CHECK-DAG: %[[#vec2_float_type:]] = OpTypeVector %[[#float_32_type]] 2
; CHECK-DAG: %[[#vec2_int_type:]] = OpTypeVector %[[#int_32_type]] 2
; CHECK-DAG: %[[#fn_ptr_type_vec2_i32:]] = OpTypePointer Function %[[#vec2_int_type]]

; CHECK: %[[#x_var1:]] = OpFunctionParameter %[[#float_32_type]]
; CHECK: %[[#var1:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#]] = OpExtInst %[[#float_32_type]] %[[#extinst_id]] Frexp %[[#x_var1]] %[[#var1]]
; CHECK: %[[#exp_only:]] = OpLoad %[[#int_32_type]] %[[#var1]]
; CHECK: OpReturnValue %[[#exp_only]]
define i32 @frexp_exponent_only(float %x) {
  %ret = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %exp_part = extractvalue { float, i32 } %ret, 1
  ret i32 %exp_part
}

; CHECK: %[[#x_var2:]] = OpFunctionParameter %[[#vec2_float_type]]
; CHECK: %[[#var2:]] = OpVariable %[[#fn_ptr_type_vec2_i32]] Function
; CHECK: %[[#]] = OpExtInst %[[#vec2_float_type]] %[[#extinst_id]] Frexp %[[#x_var2]] %[[#var2]]
; CHECK: %[[#exp_only_vec:]] = OpLoad %[[#vec2_int_type]] %[[#var2]]
; CHECK: OpReturnValue %[[#exp_only_vec]]
define <2 x i32> @frexp_exponent_only_vector(<2 x float> %x) {
  %ret = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %x)
  %exp_part = extractvalue { <2 x float>, <2 x i32> } %ret, 1
  ret <2 x i32> %exp_part
}
