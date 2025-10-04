; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability VariableLengthArrayINTEL
; CHECK: OpExtension "SPV_INTEL_variable_length_array"

; CHECK-DAG: OpDecorate %[[#]] SpecId 0
; CHECK-DAG: OpDecorate %[[#]] SpecId 1
; CHECK-DAG: OpDecorate %[[#]] SpecId 2
; CHECK-DAG: OpDecorate %[[#A0:]] Alignment 4
; CHECK-DAG: OpDecorate %[[#A1:]] Alignment 2
; CHECK-DAG: OpDecorate %[[#A2:]] Alignment 16

; CHECK: %[[#VOID_TY:]] = OpTypeVoid
; CHECK: %[[#FUNC_TY:]] = OpTypeFunction %[[#VOID_TY]]
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#F64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#STRUCT_TY:]] = OpTypeStruct %[[#F64]] %[[#F64]]
; CHECK-DAG: %[[#PTR_STRUCT:]] = OpTypePointer Function %[[#STRUCT_TY]]
; CHECK-DAG: %[[#PTR_I8:]] = OpTypePointer Function %[[#I8]]
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#PTR_F32:]] = OpTypePointer Function %[[#F32]]

; CHECK-DAG: %[[#SC0:]] = OpSpecConstant %[[#I64]] 1
; CHECK-DAG: %[[#SC1:]] = OpSpecConstant %[[#I32]] 2
; CHECK-DAG: %[[#SC2:]] = OpSpecConstant %[[#I8]] 4

; CHECK: %[[#]] = OpFunction %[[#VOID_TY]] None %[[#FUNC_TY]]
; CHECK: %[[#LABEL:]] = OpLabel

; CHECK: %[[#A0]] = OpVariableLengthArrayINTEL %[[#PTR_F32]] %[[#SC0]]
; CHECK: %[[#A1]] = OpVariableLengthArrayINTEL %[[#PTR_I8]] %[[#SC1]]
; CHECK: %[[#A2]] = OpVariableLengthArrayINTEL %[[#PTR_STRUCT]] %[[#SC2]]

%struct_type = type { double, double }

define spir_kernel void @test() {
 entry:
  %length0 = call i64 @_Z20__spirv_SpecConstantix(i32 0, i64 1), !SYCL_SPEC_CONST_SYM_ID !0
  %length1 = call i32 @_Z20__spirv_SpecConstantii(i32 1, i32 2), !SYCL_SPEC_CONST_SYM_ID !1
  %length2 = call i8 @_Z20__spirv_SpecConstantic(i32 2, i8 4), !SYCL_SPEC_CONST_SYM_ID !2
  %scla0 = alloca float, i64 %length0, align 4
  %scla1 = alloca i8, i32 %length1, align 2
  %scla2 = alloca %struct_type, i8 %length2, align 16
  ret void
}

declare i8 @_Z20__spirv_SpecConstantic(i32, i8)
declare i32 @_Z20__spirv_SpecConstantii(i32, i32)
declare i64 @_Z20__spirv_SpecConstantix(i32, i64)

!0 = !{!"i64_spec_const", i32 0}
!1 = !{!"i32_spec_const", i32 1}
!2 = !{!"i8_spec_const", i32 2}
