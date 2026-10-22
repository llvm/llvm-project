; Explicit modes yielding a zero mask still emit FPFastMathDefault.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability FloatControls2
; CHECK-DAG: OpExtension "SPV_KHR_float_controls2"
; CHECK-DAG: OpEntryPoint Kernel %[[#K1:]] "k_szinp"
; CHECK-DAG: OpEntryPoint Kernel %[[#K2:]] "k_fpfmd"
; CHECK-NOT: OpExecutionMode
; CHECK: OpExecutionModeId %[[#K1]] FPFastMathDefault %[[#FLOAT:]] %[[#ZERO:]]
; CHECK-NEXT: OpExecutionModeId %[[#K2]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; CHECK-NOT: OpExecutionMode
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FLOAT]] = OpTypeFloat 32
; CHECK-DAG: %[[#ZERO]] = OpConstantNull %[[#INT]]

define spir_kernel void @k_szinp(ptr addrspace(1) %p) {
  %v = load float, ptr addrspace(1) %p
  store float %v, ptr addrspace(1) %p
  ret void
}

define spir_kernel void @k_fpfmd(ptr addrspace(1) %p) {
  %v = load float, ptr addrspace(1) %p
  store float %v, ptr addrspace(1) %p
  ret void
}

!spirv.ExecutionMode = !{!0, !1}
!0 = !{ptr @k_szinp, i32 4461, i32 32}
!1 = !{ptr @k_fpfmd, i32 6028, float poison, i32 0}
