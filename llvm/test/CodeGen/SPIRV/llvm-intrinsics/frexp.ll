; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#float_32_type:]] = OpTypeFloat 32
; CHECK-DAG: %[[#int_32_type:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#fn_ptr_type_i32:]] = OpTypePointer Function %[[#int_32_type]]
; CHECK-DAG: %[[#const_negzero:]] = OpConstant %[[#float_32_type]] -0
; CHECK-DAG: %[[#vec2_float_type:]] = OpTypeVector %[[#float_32_type]] 2
; CHECK-DAG: %[[#vec2_int_type:]] = OpTypeVector %[[#int_32_type]] 2
; CHECK-DAG: %[[#fn_ptr_type_vec2_i32:]] = OpTypePointer Function %[[#vec2_int_type]]
; CHECK-DAG: %[[#vec2_null:]] = OpConstantNull %[[#vec2_float_type]]
; CHECK-DAG: %[[#scalar_null:]] = OpConstantNull %[[#float_32_type]]
; CHECK-DAG: %[[#const_composite1:]] = OpConstantComposite %[[#vec2_float_type]] %[[#scalar_null]] %[[#const_negzero]]
; CHECK-DAG: %[[#vec4_float_type:]] = OpTypeVector %[[#float_32_type]] 4
; CHECK-DAG: %[[#vec4_int_type:]] = OpTypeVector %[[#int_32_type]] 4
; CHECK-DAG: %[[#fn_ptr_type_vec4_i32:]] = OpTypePointer Function %[[#vec4_int_type]]
; CHECK-DAG: %[[#const_composite2:]] = OpConstantComposite %[[#vec4_float_type]] %[[#const_16:]] %[[#const_neg32:]] %[[#const_0:]] %[[#const_9999:]]
; CHECK-DAG: %[[#float_64_type:]] = OpTypeFloat 64
; CHECK-DAG: %[[#vec2_double_type:]] = OpTypeVector %[[#float_64_type]] 2

; CHECK: %[[#]] = OpFunctionParameter %[[#float_32_type]]
; CHECK: %[[#var1:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#extinst1:]] = OpExtInst %[[#float_32_type]] %[[#extinst_id]] frexp %[[#const_negzero]] %[[#var1]]
; CHECK: %[[#exp_part_var:]] = OpLoad %[[#int_32_type]] %[[#var1]]
; CHECK: OpReturnValue %[[#exp_part_var]]
define i32 @frexp_negzero(float %x) {
  %ret = call { float, i32 } @llvm.frexp.f32.i32(float -0.0)
  %f_part = extractvalue { float, i32 } %ret, 0
  %exp_part = extractvalue { float, i32 } %ret, 1
  ret i32 %exp_part
}

; CHECK: %[[#x_var4:]] = OpFunctionParameter %[[#float_32_type]]
; CHECK: %[[#var10:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#extinst10:]] = OpExtInst %[[#float_32_type]] %[[#extinst_id]] frexp %[[#x_var4]] %[[#var10]]
; CHECK: %[[#exp_part_var2:]] = OpLoad %[[#int_32_type]] %[[#var10]]
; CHECK: OpReturnValue %[[#exp_part_var2]]
define i32 @frexp_frexp_get_int(float %x) {
  %frexp0 = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %f_part = extractvalue { float, i32 } %frexp0, 0
  %exp_part = extractvalue { float, i32 } %frexp0, 1
  ret i32 %exp_part
}

; CHECK: %[[#var3:]] = OpVariable %[[#fn_ptr_type_vec2_i32]] Function
; CHECK: %[[#extinst3:]] = OpExtInst %[[#vec2_float_type]] %[[#extinst_id]] frexp %[[#vec2_null]] %[[#var3]]
; CHECK: %[[#f_part_var2:]] = OpLoad %[[#vec2_int_type]] %[[#var3]]
; CHECK: OpReturnValue %[[#extinst3]]
define <2 x float> @frexp_zero_vector() {
  %ret = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> zeroinitializer)
  %f_part = extractvalue { <2 x float>, <2 x i32> } %ret, 0
  %exp_part = extractvalue { <2 x float>, <2 x i32> } %ret, 1
  ret <2 x float> %f_part
}

; CHECK: %[[#var4:]] = OpVariable %[[#fn_ptr_type_vec2_i32]] Function
; CHECK: %[[#extinst4:]] = OpExtInst %[[#vec2_float_type]] %[[#extinst_id]] frexp %[[#const_composite1]] %[[#var4]]
; CHECK: %[[#f_part_var3:]] = OpLoad %[[#vec2_int_type]] %[[#var4]]
; CHECK: OpReturnValue %[[#extinst4]]
define <2 x float> @frexp_zero_negzero_vector() {
  %ret = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> <float 0.0, float -0.0>)
  %f_part = extractvalue { <2 x float>, <2 x i32> } %ret, 0
  %exp_part = extractvalue { <2 x float>, <2 x i32> } %ret, 1
  ret <2 x float> %f_part
}

; CHECK: %[[#var5:]] = OpVariable %[[#fn_ptr_type_vec4_i32]] Function
; CHECK: %[[#extinst5:]] = OpExtInst %[[#vec4_float_type]] %[[#extinst_id]] frexp %[[#const_composite2]] %[[#var5]]
; CHECK: %[[#f_part_var4:]] = OpLoad %[[#vec4_int_type]] %[[#var5]]
; CHECK: OpReturnValue %[[#extinst5]]
define <4 x float> @frexp_nonsplat_vector() {
    %ret = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> <float 16.0, float -32.0, float 0.0, float 9999.0>)
    %f_part = extractvalue { <4 x float>, <4 x i32> } %ret, 0
    %exp_part = extractvalue { <4 x float>, <4 x i32> } %ret, 1
  ret <4 x float> %f_part
}

; CHECK: %[[#x_var2:]] = OpFunctionParameter %[[#float_32_type]]
; CHECK: %[[#var6:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#var7:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#extinst6:]] = OpExtInst %[[#float_32_type]] %[[#extinst_id]] frexp %[[#x_var2]] %[[#var6]]
; CHECK: %[[#load1:]] = OpLoad %[[#int_32_type]] %[[#var6]]
; CHECK: %[[#extinst7:]] = OpExtInst %[[#float_32_type]] %[[#extinst_id]] frexp %[[#extinst6]] %[[#var7]]
; CHECK: %[[#f_part_var5:]] = OpLoad %[[#int_32_type]] %[[#var7]]
; CHECK: OpReturnValue %[[#extinst7]]
define float @frexp_frexp(float %x) {
  %frexp0 = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %frexp0_f_part = extractvalue { float, i32 } %frexp0, 0
  %frexp0_exp_part = extractvalue { float, i32 } %frexp0, 1
  %frexp1 = call { float, i32 } @llvm.frexp.f32.i32(float %frexp0_f_part)
  %frexp1_f_part = extractvalue { float, i32 } %frexp1, 0
  %frexp1_exp_part = extractvalue { float, i32 } %frexp1, 1
  ret float %frexp1_f_part
}

; CHECK: %[[#x_var3:]] = OpFunctionParameter %[[#vec2_double_type]]
; CHECK: %[[#var9:]] = OpVariable %[[#fn_ptr_type_vec2_i32]] Function
; CHECK: %[[#extinst9:]] = OpExtInst %[[#vec2_double_type]] %[[#extinst_id]] frexp %[[#x_var3]] %[[#var9]]
; CHECK: %[[#f_part_var6:]] = OpLoad %[[#vec2_int_type]] %[[#var9]]
; CHECK: OpReturnValue %[[#extinst9]]
define <2 x double> @frexp_frexp_vector(<2 x double> %x) {
  %frexp0 = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %x)
  %f_part = extractvalue { <2 x double>, <2 x i32> } %frexp0, 0
  %exp_part = extractvalue { <2 x double>, <2 x i32> } %frexp0, 1
  ret <2 x double> %f_part
}

declare { float, i32 } @llvm.frexp.f32.i32(float)
declare { double, i32 } @llvm.frexp.f64.i32(double)
declare { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float>)
declare { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float>)
declare { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double>)
declare  { float, i8 } @llvm.frexp.f32.i8(float)
