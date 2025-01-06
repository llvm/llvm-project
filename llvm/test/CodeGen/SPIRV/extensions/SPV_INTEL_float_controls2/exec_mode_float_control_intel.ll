; Adapted from https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/main/test/extensions/INTEL/SPV_INTEL_float_controls2

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_float_controls2 %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability RoundToInfinityINTEL
; CHECK-DAG: Capability FloatingPointModeINTEL
; CHECK: Extension "SPV_INTEL_float_controls2"

define dso_local dllexport spir_kernel void @k_float_controls_0(i32 %ibuf, i32 %obuf) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_1(i32 %ibuf, i32 %obuf) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_2(i32 %ibuf, i32 %obuf) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_3(i32 %ibuf, i32 %obuf) {
entry:
  ret void
}

!llvm.module.flags = !{!12}
!llvm.ident = !{!13}
!spirv.EntryPoint = !{}
!spirv.ExecutionMode = !{!15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}

; CHECK-DAG: OpEntryPoint Kernel %[[#KERNEL0:]] "k_float_controls_0"
; CHECK-DAG: OpEntryPoint Kernel %[[#KERNEL1:]] "k_float_controls_1"
; CHECK-DAG: OpEntryPoint Kernel %[[#KERNEL2:]] "k_float_controls_2"
; CHECK-DAG: OpEntryPoint Kernel %[[#KERNEL3:]] "k_float_controls_3"
!0 = !{ptr @k_float_controls_0, !"k_float_controls_0", !1, i32 0, !2, !3, !4, i32 0, i32 0}
!1 = !{i32 2, i32 2}
!2 = !{i32 32, i32 36}
!3 = !{i32 0, i32 0}
!4 = !{!"", !""}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 8.0.1"}
!14 = !{i32 1, i32 0}

; CHECK-DAG: OpExecutionMode %[[#KERNEL0]] RoundingModeRTPINTEL 64
!15 = !{ptr @k_float_controls_0, i32 5620, i32 64}
; CHECK-DAG: OpExecutionMode %[[#KERNEL0]] RoundingModeRTPINTEL 32
!16 = !{ptr @k_float_controls_0, i32 5620, i32 32}
; CHECK-DAG: OpExecutionMode %[[#KERNEL0]] RoundingModeRTPINTEL 16
!17 = !{ptr @k_float_controls_0, i32 5620, i32 16}

; CHECK-DAG: OpExecutionMode %[[#KERNEL1]] RoundingModeRTNINTEL 64
!18 = !{ptr @k_float_controls_1, i32 5621, i32 64}
; CHECK-DAG: OpExecutionMode %[[#KERNEL1]] RoundingModeRTNINTEL 32
!19 = !{ptr @k_float_controls_1, i32 5621, i32 32}
; CHECK-DAG: OpExecutionMode %[[#KERNEL1]] RoundingModeRTNINTEL 16
!20 = !{ptr @k_float_controls_1, i32 5621, i32 16}

; CHECK-DAG: OpExecutionMode %[[#KERNEL2]] FloatingPointModeALTINTEL 64
!21 = !{ptr @k_float_controls_2, i32 5622, i32 64}
; CHECK-DAG: OpExecutionMode %[[#KERNEL2]] FloatingPointModeALTINTEL 32
!22 = !{ptr @k_float_controls_2, i32 5622, i32 32}
; CHECK-DAG: OpExecutionMode %[[#KERNEL2]] FloatingPointModeALTINTEL 16
!23 = !{ptr @k_float_controls_2, i32 5622, i32 16}

; CHECK-DAG: OpExecutionMode %[[#KERNEL3]] FloatingPointModeIEEEINTEL 64
!24 = !{ptr @k_float_controls_3, i32 5623, i32 64}
; CHECK-DAG: OpExecutionMode %[[#KERNEL3]] FloatingPointModeIEEEINTEL 32
!25 = !{ptr @k_float_controls_3, i32 5623, i32 32}
; CHECK-DAG: OpExecutionMode %[[#KERNEL3]] FloatingPointModeIEEEINTEL 16
!26 = !{ptr @k_float_controls_3, i32 5623, i32 16}
