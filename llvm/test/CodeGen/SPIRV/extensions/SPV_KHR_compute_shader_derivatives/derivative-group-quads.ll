; RUN: llc -O0 -verify-machineinstrs \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-compute \
; RUN:   --spirv-ext=+SPV_KHR_compute_shader_derivatives \
; RUN:   %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv1.5-unknown-vulkan-compute \
; RUN:   --spirv-ext=-SPV_KHR_compute_shader_derivatives \
; RUN:   %s -o - 2>&1 | FileCheck %s --check-prefix=NO-EXT
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-compute \
; RUN:   --spirv-ext=+SPV_KHR_compute_shader_derivatives \
; RUN:   %s -o - -filetype=obj | spirv-val --target-env vulkan1.2 %}

; NO-EXT: DerivativeGroupQuadsKHR requires the following SPIR-V extension: SPV_KHR_compute_shader_derivatives

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability ComputeDerivativeGroupQuadsKHR
; CHECK: OpExtension "SPV_KHR_compute_shader_derivatives"
; CHECK-NOT: OpExtension
; CHECK: OpEntryPoint GLCompute %[[#MAIN:]] "main"
; CHECK: OpExecutionMode %[[#MAIN]] DerivativeGroupQuadsKHR
; CHECK: OpExecutionMode %[[#MAIN]] LocalSize 2 2 1

define void @main() #0 {
  ret void
}

attributes #0 = { "hlsl.numthreads"="2,2,1" "hlsl.shader"="compute" }

!spirv.ExecutionMode = !{!0}
!0 = !{ptr @main, i32 5289}
