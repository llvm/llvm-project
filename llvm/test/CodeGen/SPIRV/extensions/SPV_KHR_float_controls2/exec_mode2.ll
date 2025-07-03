; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability FloatControls2 
; CHECK: Extension "SPV_KHR_float_controls2"

define dso_local dllexport spir_kernel void @k_float_controls_float(float %f) {
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
!spirv.ExecutionMode = !{!19, !20, !21, !22, !23, !24, !25, !26, !27}


; CHECK: OpEntryPoint Kernel %[[#KERNEL_FLOAT:]] "k_float_controls_float"
!2 = !{ptr @k_float_controls_float, !"k_float_controls_float", !6, i32 0, !6, !7, !8, i32 0, i32 0}
; CHECK: OpEntryPoint Kernel %[[#KERNEL_ALL:]] "k_float_controls_all"
!3 = !{ptr @k_float_controls_all, !"k_float_controls_all", !6, i32 0, !6, !7, !8, i32 0, i32 0}

!6 = !{i32 2, i32 2}
!7 = !{i32 32, i32 36}
!8 = !{i32 0, i32 0}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 8.0.1"}
!14 = !{i32 1, i32 0}

; CHECK-DAG: OpExecutionMode %[[#KERNEL_FLOAT]] FPFastMathDefault %[[#FLOAT_TYPE:]] 131079 
!19 = !{ptr @k_float_controls_float, i32 6028, float undef, i32 131079}
; We expect 130179 for float type.
; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#FLOAT_TYPE:]] 131079 
; We expect 7 for the rest of types because it's NotInf | NotNaN | NSZ set by SignedZeroInfNanPreserve.
; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#HALF_TYPE:]] 7 
; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#DOUBLE_TYPE:]] 7 
; CHECK-DAG: OpExecutionMode %[[#KERNEL_ALL]] FPFastMathDefault %[[#FP128_TYPE:]] 7 
!20 = !{ptr @k_float_controls_all, i32 6028, float undef, i32 131079}
; ContractionOff is now replaced with FPFastMathDefault with AllowContract bit set to false.
!21 = !{ptr @k_float_controls_float, i32 31}
!22 = !{ptr @k_float_controls_all, i32 31}
; SignedZeroInfNanPreserve is now replaced with FPFastMathDefault with flags NotInf, NotNaN and NSZ. 
!23 = !{ptr @k_float_controls_float, i32 4461, i32 32}
!24 = !{ptr @k_float_controls_all, i32 4461, i32 16}
!25 = !{ptr @k_float_controls_all, i32 4461, i32 32}
!26 = !{ptr @k_float_controls_all, i32 4461, i32 64}
!27 = !{ptr @k_float_controls_all, i32 4461, i32 128}

; CHECK-DAG: %[[#HALF_TYPE]] = OpTypeFloat 16
; CHECK-DAG: %[[#FLOAT_TYPE]] = OpTypeFloat 32
; CHECK-DAG: %[[#DOUBLE_TYPE]] = OpTypeFloat 64
; CHECK-DAG: %[[#FP128_TYPE]] = OpTypeFloat 128
