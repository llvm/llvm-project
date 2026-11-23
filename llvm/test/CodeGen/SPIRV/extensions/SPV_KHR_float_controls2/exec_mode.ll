; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2,+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2,+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability FloatControls2
; CHECK: Extension "SPV_KHR_float_controls2"

define dso_local dllexport spir_kernel void @k_float_controls_half(half %h) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_bfloat(bfloat %b) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_float(float %f) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_double(double %d) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_all(half %h, bfloat %b, float %f, double %d) {
entry:
  ret void
}

!spirv.ExecutionMode = !{!17, !18, !19, !20, !22, !23, !24, !25}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_HALF:]] "k_float_controls_half"
!0 = !{ptr @k_float_controls_half, !"k_float_controls_half", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_BFLOAT:]] "k_float_controls_bfloat"
!1 = !{ptr @k_float_controls_bfloat, !"k_float_controls_bfloat", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_FLOAT:]] "k_float_controls_float"
!2 = !{ptr @k_float_controls_float, !"k_float_controls_float", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_DOUBLE:]] "k_float_controls_double"
!3 = !{ptr @k_float_controls_double, !"k_float_controls_double", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_ALL:]] "k_float_controls_all"
!5 = !{ptr @k_float_controls_all, !"k_float_controls_all", !6, i32 0, !6, !7, !8, i32 0, i32 0}
!6 = !{i32 2, i32 2}
!7 = !{i32 32, i32 36}
!8 = !{i32 0, i32 0}

; CHECK-DAG: OpExecutionModeId %[[#KERNEL_HALF]] FPFastMathDefault %[[#HALF_TYPE:]] %[[#CONST1:]]
!17 = !{ptr @k_float_controls_half, i32 6028, half poison, i32 1}

; CHECK-DAG: OpExecutionModeId %[[#KERNEL_BFLOAT]] FPFastMathDefault %[[#BFLOAT_TYPE:]] %[[#CONST2:]]
!18 = !{ptr @k_float_controls_bfloat, i32 6028, bfloat poison, i32 2}

; CHECK-DAG: OpExecutionModeId %[[#KERNEL_FLOAT]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST4:]]
!19 = !{ptr @k_float_controls_float, i32 6028, float poison, i32 4}

; CHECK-DAG: OpExecutionModeId %[[#KERNEL_DOUBLE]] FPFastMathDefault %[[#DOUBLE_TYPE:]] %[[#CONST7:]]
!20 = !{ptr @k_float_controls_double, i32 6028, double poison, i32 7}

; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#HALF_TYPE]] %[[#CONST131072:]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#FLOAT_TYPE]] %[[#CONST458752:]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#DOUBLE_TYPE]] %[[#CONST458752:]]
!22 = !{ptr @k_float_controls_all, i32 6028, half poison, i32 131072}
!23 = !{ptr @k_float_controls_all, i32 6028, bfloat poison, i32 131072}
!24 = !{ptr @k_float_controls_all, i32 6028, float poison, i32 458752}
!25 = !{ptr @k_float_controls_all, i32 6028, double poison, i32 458752}

; CHECK-DAG: %[[#INT32_TYPE:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#HALF_TYPE]] = OpTypeFloat 16
; CHECK-DAG: %[[#FLOAT_TYPE]] = OpTypeFloat 32
; CHECK-DAG: %[[#DOUBLE_TYPE]] = OpTypeFloat 64
; CHECK-DAG: %[[#CONST1]] = OpConstant %[[#INT32_TYPE]] 1
; CHECK-DAG: %[[#CONST2]] = OpConstant %[[#INT32_TYPE]] 2
; CHECK-DAG: %[[#CONST4]] = OpConstant %[[#INT32_TYPE]] 4
; CHECK-DAG: %[[#CONST7]] = OpConstant %[[#INT32_TYPE]] 7
; CHECK-DAG: %[[#CONST131072]] = OpConstant %[[#INT32_TYPE]] 131072
; CHECK-DAG: %[[#CONST458752]] = OpConstant %[[#INT32_TYPE]] 458752
