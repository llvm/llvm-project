; This test checks that functions with `convert_` prefix are translated as
; OpenCL builtins only in case they match the specification. Otherwise, we
; expect such functions to be translated to SPIR-V FunctionCall.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpName %[[#Func:]] "_Z18convert_float_func"
; CHECK-SPIRV: OpName %[[#Func1:]] "_Z20convert_uint_satfunc"
; CHECK-SPIRV: OpName %[[#Func2:]] "_Z21convert_float_rtzfunc"
; CHECK-SPIRV-DAG: %[[#VoidTy:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#CharTy:]] = OpTypeInt 8
; CHECK-SPIRV-DAG: %[[#FloatTy:]] = OpTypeFloat 32

; CHECK-SPIRV: %[[#Func]] = OpFunction %[[#VoidTy]] None %[[#]]
; CHECK-SPIRV: %[[#ConvertId1:]] = OpUConvert %[[#CharTy]] %[[#]]
; CHECK-SPIRV: %[[#ConvertId2:]] = OpConvertSToF %[[#FloatTy]] %[[#]]
; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#VoidTy]] %[[#Func]] %[[#ConvertId2]]
; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#VoidTy]] %[[#Func1]] %[[#]]
; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#VoidTy]] %[[#Func2]] %[[#ConvertId2]]
; CHECK-SPIRV-NOT: OpFConvert
; CHECK-SPIRV-NOT: OpConvertUToF

define dso_local spir_func void @_Z18convert_float_func(float noundef %x) {
entry:
  %x.addr = alloca float, align 4
  store float %x, ptr %x.addr, align 4
  ret void
}

define dso_local spir_func void @_Z20convert_uint_satfunc(i32 noundef %x) {
entry:
  ret void
}

define dso_local spir_func void @_Z21convert_float_rtzfunc(float noundef %x) {
entry:
  ret void
}

define dso_local spir_func void @convert_int_bf16(i32 noundef %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  call spir_func signext i8 @_Z16convert_char_rtei(i32 noundef %0)
  %call = call spir_func float @_Z13convert_floati(i32 noundef %0)
  call spir_func void @_Z18convert_float_func(float noundef %call)
  call spir_func void @_Z20convert_uint_satfunc(i32 noundef %0)
  call spir_func void @_Z21convert_float_rtzfunc(float noundef %call)
  ret void
}

declare spir_func signext i8 @_Z16convert_char_rtei(i32 noundef)

declare spir_func float @_Z13convert_floati(i32 noundef)
