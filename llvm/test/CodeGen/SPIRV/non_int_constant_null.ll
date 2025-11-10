; RUN: llc -mtriple spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_float_controls2 -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_float_controls2 -o - -filetype=obj | spirv-val %}

@A = addrspace(1) constant [1 x i8] zeroinitializer

; CHECK: OpName %[[#FOO:]] "foo"
; CHECK: OpName %[[#A:]] "A"
; CHECK: OpDecorate %[[#A]] Constant
; CHECK: OpDecorate %[[#A]] LinkageAttributes "A" Export
; CHECK: %[[#INT8:]] = OpTypeInt 8 0
; CHECK: %[[#INT32:]] = OpTypeInt 32 0
; CHECK: %[[#ONE:]] = OpConstant %[[#INT32]] 1
; CHECK: %[[#ARR_INT8:]] = OpTypeArray %[[#INT8]] %7
; CHECK: %[[#ARR_INT8_PTR:]] = OpTypePointer CrossWorkgroup %[[#ARR_INT8]]
; CHECK: %[[#ARR_INT8_ZERO:]] = OpConstantNull %[[#ARR_INT8]]
; CHECK: %13 = OpVariable %[[#ARR_INT8_PTR]] CrossWorkgroup %[[#ARR_INT8_ZERO]]
; CHECK: %[[#FOO]] = OpFunction
; CHECK: = OpLabel
; CHECK: OpReturn
; CHECK: OpFunctionEnd

define spir_kernel void @foo() {
entry:
  ret void
}
