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
; CHECK-DAG: %[[#void_type:]] = OpTypeVoid
; CHECK-DAG: %[[#const_1:]] = OpConstant %[[#int_32_type]] 1
; CHECK-DAG: %[[#int_null:]] = OpConstantNull %[[#int_32_type]]
; CHECK-DAG: %[[#float_16_type:]] = OpTypeFloat 16
; CHECK-DAG: %[[#struct_f16_i32:]] = OpTypeStruct %[[#float_16_type]] %[[#int_32_type]]
; CHECK-DAG: %[[#fn_ptr_type_struct_f16:]] = OpTypePointer Function %[[#struct_f16_i32]]
; CHECK-DAG: %[[#fn_ptr_type_f16:]] = OpTypePointer Function %[[#float_16_type]]
; CHECK-DAG: %[[#struct_f32_i32:]] = OpTypeStruct %[[#float_32_type]] %[[#int_32_type]]
; CHECK-DAG: %[[#fn_ptr_type_struct_f32:]] = OpTypePointer Function %[[#struct_f32_i32]]
; CHECK-DAG: %[[#fn_ptr_type_f32:]] = OpTypePointer Function %[[#float_32_type]]
; CHECK-DAG: %[[#struct_f64_i32:]] = OpTypeStruct %[[#float_64_type]] %[[#int_32_type]]
; CHECK-DAG: %[[#fn_ptr_type_struct_f64:]] = OpTypePointer Function %[[#struct_f64_i32]]
; CHECK-DAG: %[[#fn_ptr_type_f64:]] = OpTypePointer Function %[[#float_64_type]]

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
; CHECK-NOT: OpLoad %[[#vec2_int_type]] %[[#var3]]
; CHECK: OpReturnValue %[[#extinst3]]
define <2 x float> @frexp_zero_vector() {
  %ret = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> zeroinitializer)
  %f_part = extractvalue { <2 x float>, <2 x i32> } %ret, 0
  %exp_part = extractvalue { <2 x float>, <2 x i32> } %ret, 1
  ret <2 x float> %f_part
}

; CHECK: %[[#var4:]] = OpVariable %[[#fn_ptr_type_vec2_i32]] Function
; CHECK: %[[#extinst4:]] = OpExtInst %[[#vec2_float_type]] %[[#extinst_id]] frexp %[[#const_composite1]] %[[#var4]]
; CHECK-NOT: OpLoad %[[#vec2_int_type]] %[[#var4]]
; CHECK: OpReturnValue %[[#extinst4]]
define <2 x float> @frexp_zero_negzero_vector() {
  %ret = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> <float 0.0, float -0.0>)
  %f_part = extractvalue { <2 x float>, <2 x i32> } %ret, 0
  %exp_part = extractvalue { <2 x float>, <2 x i32> } %ret, 1
  ret <2 x float> %f_part
}

; CHECK: %[[#var5:]] = OpVariable %[[#fn_ptr_type_vec4_i32]] Function
; CHECK: %[[#extinst5:]] = OpExtInst %[[#vec4_float_type]] %[[#extinst_id]] frexp %[[#const_composite2]] %[[#var5]]
; CHECK-NOT: OpLoad %[[#vec4_int_type]] %[[#var5]]
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
; CHECK-NOT: OpLoad %[[#int_32_type]] %[[#var6]]
; CHECK: %[[#extinst7:]] = OpExtInst %[[#float_32_type]] %[[#extinst_id]] frexp %[[#extinst6]] %[[#var7]]
; CHECK-NOT: OpLoad %[[#int_32_type]] %[[#var7]]
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
; CHECK-NOT: OpLoad %[[#vec2_int_type]] %[[#var9]]
; CHECK: OpReturnValue %[[#extinst9]]
define <2 x double> @frexp_frexp_vector(<2 x double> %x) {
  %frexp0 = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %x)
  %f_part = extractvalue { <2 x double>, <2 x i32> } %frexp0, 0
  %exp_part = extractvalue { <2 x double>, <2 x i32> } %frexp0, 1
  ret <2 x double> %f_part
}

; CHECK: %[[#out_param_f16:]] = OpFunctionParameter %[[#fn_ptr_type_struct_f16]]
; CHECK: %[[#x_var_f16:]] = OpFunctionParameter %[[#float_16_type]]
; CHECK: %[[#var_f16:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#extinst_f16:]] = OpExtInst %[[#float_16_type]] %[[#extinst_id]] frexp %[[#x_var_f16]] %[[#var_f16]]
; CHECK: %[[#exp_load_f16:]] = OpLoad %[[#int_32_type]] %[[#var_f16]]
; CHECK: %[[#mantissa_ptr_f16:]] = OpInBoundsPtrAccessChain %[[#fn_ptr_type_f16]] %[[#out_param_f16]] %[[#int_null]] %[[#int_null]]
; CHECK: %[[#exp_ptr_f16:]] = OpInBoundsPtrAccessChain %[[#fn_ptr_type_i32]] %[[#out_param_f16]] %[[#int_null]] %[[#const_1]]
; CHECK: OpStore %[[#mantissa_ptr_f16]] %[[#extinst_f16]] Aligned 2
; CHECK: OpStore %[[#exp_ptr_f16]] %[[#exp_load_f16]] Aligned 4
; CHECK: OpReturn
define void @frexp_f16(ptr %out, half %x) {
  %ret = call { half, i32 } @llvm.frexp.f16.i32(half %x)

  %mantissa.ptr = getelementptr inbounds { half, i32 }, ptr %out, i32 0, i32 0
  %exp.ptr      = getelementptr inbounds { half, i32 }, ptr %out, i32 0, i32 1
  %mantissa = extractvalue { half, i32 } %ret, 0
  %exp      = extractvalue { half, i32 } %ret, 1

  store half %mantissa, ptr %mantissa.ptr
  store i32 %exp, ptr %exp.ptr
  ret void
}

; CHECK: %[[#out_param_f32:]] = OpFunctionParameter %[[#fn_ptr_type_struct_f32]]
; CHECK: %[[#x_var_f32:]] = OpFunctionParameter %[[#float_32_type]]
; CHECK: %[[#var_f32:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#extinst_f32:]] = OpExtInst %[[#float_32_type]] %[[#extinst_id]] frexp %[[#x_var_f32]] %[[#var_f32]]
; CHECK: %[[#exp_load_f32:]] = OpLoad %[[#int_32_type]] %[[#var_f32]]
; CHECK: %[[#mantissa_ptr_f32:]] = OpInBoundsPtrAccessChain %[[#fn_ptr_type_f32]] %[[#out_param_f32]] %[[#int_null]] %[[#int_null]]
; CHECK: %[[#exp_ptr_f32:]] = OpInBoundsPtrAccessChain %[[#fn_ptr_type_i32]] %[[#out_param_f32]] %[[#int_null]] %[[#const_1]]
; CHECK: OpStore %[[#mantissa_ptr_f32]] %[[#extinst_f32]] Aligned 4
; CHECK: OpStore %[[#exp_ptr_f32]] %[[#exp_load_f32]] Aligned 4
; CHECK: OpReturn
define void @frexp_f32(ptr %out, float %x) {
  %ret = call { float, i32 } @llvm.frexp.f32.i32(float %x)

  %mantissa.ptr = getelementptr inbounds { float, i32 }, ptr %out, i32 0, i32 0
  %exp.ptr      = getelementptr inbounds { float, i32 }, ptr %out, i32 0, i32 1
  %mantissa = extractvalue { float, i32 } %ret, 0
  %exp      = extractvalue { float, i32 } %ret, 1

  store float %mantissa, ptr %mantissa.ptr
  store i32 %exp, ptr %exp.ptr
  ret void
}

; CHECK: %[[#out_param_f64:]] = OpFunctionParameter %[[#fn_ptr_type_struct_f64]]
; CHECK: %[[#x_var_f64:]] = OpFunctionParameter %[[#float_64_type]]
; CHECK: %[[#var_f64:]] = OpVariable %[[#fn_ptr_type_i32]] Function
; CHECK: %[[#extinst_f64:]] = OpExtInst %[[#float_64_type]] %[[#extinst_id]] frexp %[[#x_var_f64]] %[[#var_f64]]
; CHECK: %[[#exp_load_f64:]] = OpLoad %[[#int_32_type]] %[[#var_f64]]
; CHECK: %[[#mantissa_ptr_f64:]] = OpInBoundsPtrAccessChain %[[#fn_ptr_type_f64]] %[[#out_param_f64]] %[[#int_null]] %[[#int_null]]
; CHECK: %[[#exp_ptr_f64:]] = OpInBoundsPtrAccessChain %[[#fn_ptr_type_i32]] %[[#out_param_f64]] %[[#int_null]] %[[#const_1]]
; CHECK: OpStore %[[#mantissa_ptr_f64]] %[[#extinst_f64]] Aligned 8
; CHECK: OpStore %[[#exp_ptr_f64]] %[[#exp_load_f64]] Aligned 4
; CHECK: OpReturn
define void @frexp_f64(ptr %out, double %x) {
  %ret = call { double, i32 } @llvm.frexp.f64.i32(double %x)

  %mantissa.ptr = getelementptr inbounds { double, i32 }, ptr %out, i32 0, i32 0
  %exp.ptr      = getelementptr inbounds { double, i32 }, ptr %out, i32 0, i32 1
  %mantissa = extractvalue { double, i32 } %ret, 0
  %exp      = extractvalue { double, i32 } %ret, 1

  store double %mantissa, ptr %mantissa.ptr
  store i32 %exp, ptr %exp.ptr
  ret void
}
