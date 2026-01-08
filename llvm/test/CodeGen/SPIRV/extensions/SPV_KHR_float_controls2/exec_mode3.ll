; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability FloatControls2
; CHECK: Extension "SPV_KHR_float_controls2"
; CHECK: OpEntryPoint Kernel %[[#KERNEL_FLOAT:]] "k_float_controls_float"
; CHECK: OpEntryPoint Kernel %[[#KERNEL_ALL:]] "k_float_controls_all"
; CHECK: OpEntryPoint Kernel %[[#KERNEL_FLOAT_V:]] "k_float_controls_float_v"
; CHECK: OpEntryPoint Kernel %[[#KERNEL_ALL_V:]] "k_float_controls_all_v"

; We expect 130179 for float type.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_FLOAT]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079:]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079]]
; We expect 0 for the rest of types because it's SignedZeroInfNanPreserve.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#HALF_TYPE:]] %[[#CONST0:]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL]] FPFastMathDefault %[[#DOUBLE_TYPE:]] %[[#CONST0]]

; We expect 130179 for float type.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_FLOAT_V]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL_V]] FPFastMathDefault %[[#FLOAT_TYPE:]] %[[#CONST131079]]
; We expect 0 for the rest of types because it's SignedZeroInfNanPreserve.
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL_V]] FPFastMathDefault %[[#HALF_TYPE:]] %[[#CONST0]]
; CHECK-DAG: OpExecutionModeId %[[#KERNEL_ALL_V]] FPFastMathDefault %[[#DOUBLE_TYPE:]] %[[#CONST0]]

; CHECK-DAG: OpDecorate %[[#addRes:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowReassoc
; CHECK-DAG: OpDecorate %[[#addResH:]] FPFastMathMode None
; CHECK-DAG: OpDecorate %[[#addResF:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowReassoc
; CHECK-DAG: OpDecorate %[[#addResD:]] FPFastMathMode None
; CHECK-DAG: OpDecorate %[[#addRes_V:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowReassoc
; CHECK-DAG: OpDecorate %[[#addResH_V:]] FPFastMathMode None
; CHECK-DAG: OpDecorate %[[#addResF_V:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowReassoc
; CHECK-DAG: OpDecorate %[[#addResD_V:]] FPFastMathMode None

; CHECK-DAG: %[[#INT32_TYPE:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#HALF_TYPE]] = OpTypeFloat 16
; CHECK-DAG: %[[#FLOAT_TYPE]] = OpTypeFloat 32
; CHECK-DAG: %[[#DOUBLE_TYPE]] = OpTypeFloat 64
; CHECK-DAG: %[[#CONST0]] = OpConstantNull %[[#INT32_TYPE]]
; CHECK-DAG: %[[#CONST131079]] = OpConstant %[[#INT32_TYPE]] 131079

; CHECK-DAG: %[[#HALF_V_TYPE:]] = OpTypeVector %[[#HALF_TYPE]]
; CHECK-DAG: %[[#FLOAT_V_TYPE:]] = OpTypeVector %[[#FLOAT_TYPE]]
; CHECK-DAG: %[[#DOUBLE_V_TYPE:]] = OpTypeVector %[[#DOUBLE_TYPE]]

define dso_local dllexport spir_kernel void @k_float_controls_float(float %f) {
entry:
; CHECK-DAG: %[[#addRes]] = OpFAdd %[[#FLOAT_TYPE]]
  %addRes = fadd float %f,  %f
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_all(half %h, float %f, double %d) {
entry:
; CHECK-DAG: %[[#addResH]] = OpFAdd %[[#HALF_TYPE]]
; CHECK-DAG: %[[#addResF]] = OpFAdd %[[#FLOAT_TYPE]]
; CHECK-DAG: %[[#addResD]] = OpFAdd %[[#DOUBLE_TYPE]]
  %addResH = fadd half %h,  %h
  %addResF = fadd float %f,  %f
  %addResD = fadd double %d,  %d
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_float_v(<2 x float> %f) {
entry:
; CHECK-DAG: %[[#addRes_V]] = OpFAdd %[[#FLOAT_V_TYPE]]
  %addRes = fadd <2 x float> %f,  %f
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_all_v(<2 x half> %h, <2 x float> %f, <2 x double> %d) {
entry:
; CHECK-DAG: %[[#addResH_V]] = OpFAdd %[[#HALF_V_TYPE]]
; CHECK-DAG: %[[#addResF_V]] = OpFAdd %[[#FLOAT_V_TYPE]]
; CHECK-DAG: %[[#addResD_V]] = OpFAdd %[[#DOUBLE_V_TYPE]]
  %addResH = fadd <2 x half> %h,  %h
  %addResF = fadd <2 x float> %f,  %f
  %addResD = fadd <2 x double> %d,  %d
  ret void
}

!spirv.ExecutionMode = !{!19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34}

!19 = !{ptr @k_float_controls_float, i32 6028, float poison, i32 131079}
!20 = !{ptr @k_float_controls_all, i32 6028, float poison, i32 131079}
; ContractionOff is now replaced with FPFastMathDefault with AllowContract bit set to false.
!21 = !{ptr @k_float_controls_float, i32 31}
!22 = !{ptr @k_float_controls_all, i32 31}
; SignedZeroInfNanPreserve is now replaced with FPFastMathDefault with flags 0.
!23 = !{ptr @k_float_controls_float, i32 4461, i32 32}
!24 = !{ptr @k_float_controls_all, i32 4461, i32 16}
!25 = !{ptr @k_float_controls_all, i32 4461, i32 32}
!26 = !{ptr @k_float_controls_all, i32 4461, i32 64}

!27 = !{ptr @k_float_controls_float_v, i32 6028, float poison, i32 131079}
!28 = !{ptr @k_float_controls_all_v, i32 6028, float poison, i32 131079}
; ContractionOff is now replaced with FPFastMathDefault with AllowContract bit set to false.
!29 = !{ptr @k_float_controls_float_v, i32 31}
!30 = !{ptr @k_float_controls_all_v, i32 31}
; SignedZeroInfNanPreserve is now replaced with FPFastMathDefault with flags 0.
!31 = !{ptr @k_float_controls_float_v, i32 4461, i32 32}
!32 = !{ptr @k_float_controls_all_v, i32 4461, i32 16}
!33 = !{ptr @k_float_controls_all_v, i32 4461, i32 32}
!34 = !{ptr @k_float_controls_all_v, i32 4461, i32 64}
