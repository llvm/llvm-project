; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability FloatControls2
; CHECK: Extension "SPV_KHR_float_controls2"

; CHECK: OpEntryPoint Kernel %[[#KERNEL_FLOAT:]] "k_float_controls_float"
; CHECK: OpEntryPoint Kernel %[[#KERNEL_ALL:]] "k_float_controls_all"
; CHECK: OpEntryPoint Kernel %[[#KERNEL_FLOAT_V:]] "k_float_controls_float_v"
; CHECK: OpEntryPoint Kernel %[[#KERNEL_ALL_V:]] "k_float_controls_all_v"

define dso_local dllexport spir_kernel void @k_float_controls_float(float %f) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_all(half %h, float %f, double %d) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_float_v(<2 x float> %f) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_all_v(<2 x half> %h, <2 x float> %f, <2 x double> %d) {
entry:
  ret void
}

!spirv.ExecutionMode = !{!19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34}

; CHECK-DAG: OpExecutionModeId %[[#KERNEL_FLOAT]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079:]]
!19 = !{ptr @k_float_controls_float, i32 6028, float poison, i32 131079}
; We expect 130179 for float type.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079]]
; We expect 0 for the rest of types because it's SignedZeroInfNanPreserve.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#HALF_TYPE:]] %[[#CONST0:]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#DOUBLE_TYPE:]] %[[#CONST0]]
!20 = !{ptr @k_float_controls_all, i32 6028, float poison, i32 131079}
; ContractionOff is now replaced with FPFastMathDefault with AllowContract bit set to false.
!21 = !{ptr @k_float_controls_float, i32 31}
!22 = !{ptr @k_float_controls_all, i32 31}
; SignedZeroInfNanPreserve is now replaced with FPFastMathDefault with flags 0.
!23 = !{ptr @k_float_controls_float, i32 4461, i32 32}
!24 = !{ptr @k_float_controls_all, i32 4461, i32 16}
!25 = !{ptr @k_float_controls_all, i32 4461, i32 32}
!26 = !{ptr @k_float_controls_all, i32 4461, i32 64}

; CHECK-DAG: OpExecutionModeId %[[#KERNEL_FLOAT_V]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079]]
!27 = !{ptr @k_float_controls_float_v, i32 6028, float poison, i32 131079}
; We expect 130179 for float type.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL_V]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079]]
; We expect 0 for the rest of types because it's SignedZeroInfNanPreserve.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL_V]] FPFastMathDefault %[[#HALF_TYPE:]] %[[#CONST0]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL_V]] FPFastMathDefault %[[#DOUBLE_TYPE:]] %[[#CONST0]]
!28 = !{ptr @k_float_controls_all_v, i32 6028, float poison, i32 131079}
; ContractionOff is now replaced with FPFastMathDefault with AllowContract bit set to false.
!29 = !{ptr @k_float_controls_float_v, i32 31}
!30 = !{ptr @k_float_controls_all_v, i32 31}
; SignedZeroInfNanPreserve is now replaced with FPFastMathDefault with flags 0.
!31 = !{ptr @k_float_controls_float_v, i32 4461, i32 32}
!32 = !{ptr @k_float_controls_all_v, i32 4461, i32 16}
!33 = !{ptr @k_float_controls_all_v, i32 4461, i32 32}
!34 = !{ptr @k_float_controls_all_v, i32 4461, i32 64}

; CHECK-DAG: %[[#INT32_TYPE:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#HALF_TYPE]] = OpTypeFloat 16
; CHECK-DAG: %[[#FLOAT_TYPE]] = OpTypeFloat 32
; CHECK-DAG: %[[#DOUBLE_TYPE]] = OpTypeFloat 64
; CHECK-DAG: %[[#CONST0]] = OpConstantNull %[[#INT32_TYPE]]
; CHECK-DAG: %[[#CONST131079]] = OpConstant %[[#INT32_TYPE]] 131079
