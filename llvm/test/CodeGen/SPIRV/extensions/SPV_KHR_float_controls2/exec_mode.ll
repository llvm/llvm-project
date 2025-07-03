; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

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

define dso_local dllexport spir_kernel void @k_float_controls_fp128(fp128 %fp) {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_all(half %h, bfloat %b, float %f, double %d, fp128 %fp) {
entry:
  ret void
}

!llvm.module.flags = !{!12}
!llvm.ident = !{!13}
!spirv.EntryPoint = !{}
!spirv.ExecutionMode = !{!17, !18, !19, !20, !21, !22, !23, !24, !25, !26}


; CHECK: OpEntryPoint Kernel %[[#KERNEL_HALF:]] "k_float_controls_half"
!0 = !{ptr @k_float_controls_half, !"k_float_controls_half", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_BFLOAT:]] "k_float_controls_bfloat"
!1 = !{ptr @k_float_controls_bfloat, !"k_float_controls_bfloat", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_FLOAT:]] "k_float_controls_float"
!2 = !{ptr @k_float_controls_float, !"k_float_controls_float", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_DOUBLE:]] "k_float_controls_double"
!3 = !{ptr @k_float_controls_double, !"k_float_controls_double", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_FP128:]] "k_float_controls_fp128"
!4 = !{ptr @k_float_controls_fp128, !"k_float_controls_fp128", !6, i32 0, !6, !7, !8, i32 0, i32 0}

; CHECK: OpEntryPoint Kernel %[[#KERNEL_ALL:]] "k_float_controls_all"
!5 = !{ptr @k_float_controls_all, !"k_float_controls_all", !6, i32 0, !6, !7, !8, i32 0, i32 0}
!6 = !{i32 2, i32 2}
!7 = !{i32 32, i32 36}
!8 = !{i32 0, i32 0}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 8.0.1"}
!14 = !{i32 1, i32 0}

; CHECK-DAG: OpExecutionMode %[[#KERNEL_HALF]] FPFastMathDefault %[[#HALF_TYPE:]] 1 
!17 = !{ptr @k_float_controls_half, i32 6028, half undef, i32 1}

; CHECK-DAG: OpExecutionMode %[[#KERNEL_BFLOAT]] FPFastMathDefault %[[#BFLOAT_TYPE:]] 2 
!18 = !{ptr @k_float_controls_bfloat, i32 6028, bfloat undef, i32 2}

; CHECK-DAG: OpExecutionMode %[[#KERNEL_FLOAT]] FPFastMathDefault %[[#FLOAT_TYPE:]] 4 
!19 = !{ptr @k_float_controls_float, i32 6028, float undef, i32 4}

; CHECK-DAG: OpExecutionMode %[[#KERNEL_DOUBLE]] FPFastMathDefault %[[#DOUBLE_TYPE:]] 7 
!20 = !{ptr @k_float_controls_double, i32 6028, double undef, i32 7}

; CHECK-DAG: OpExecutionMode %[[#KERNEL_FP128]] FPFastMathDefault %[[#FP128_TYPE:]] 65536
!21 = !{ptr @k_float_controls_fp128, i32 6028, fp128 undef, i32 65536}

; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#HALF_TYPE]] 131072 
; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#FLOAT_TYPE]] 262144 
; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#DOUBLE_TYPE]] 458752 
; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#FP128_TYPE]] 65543 
!22 = !{ptr @k_float_controls_all, i32 6028, half undef, i32 131072}
!23 = !{ptr @k_float_controls_all, i32 6028, bfloat undef, i32 131072}
!24 = !{ptr @k_float_controls_all, i32 6028, float undef, i32 262144}
!25 = !{ptr @k_float_controls_all, i32 6028, double undef, i32 458752}
!26 = !{ptr @k_float_controls_all, i32 6028, fp128 undef, i32 65543}

; CHECK: %[[#HALF_TYPE]] = OpTypeFloat 16
; CHECK: %[[#FLOAT_TYPE]] = OpTypeFloat 32
; CHECK: %[[#DOUBLE_TYPE]] = OpTypeFloat 64
; CHECK: %[[#FP128_TYPE]] = OpTypeFloat 128
