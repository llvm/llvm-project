; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.5-unknown-vulkan-pixel \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv1.5-unknown-vulkan-pixel \
; RUN:   --spirv-ext=-SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - 2>&1 | FileCheck %s --check-prefix=NO-EXT
; RUN: not llc -O0 -mtriple=spirv1.5-unknown-vulkan-pixel \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,-SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - 2>&1 | FileCheck %s --check-prefix=NO-RECONVERGENCE
; RUN: not llc -O0 -mtriple=spirv32v1.2-unknown-vulkan-pixel \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - 2>&1 | FileCheck %s --check-prefix=VERSION
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-pixel \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - -filetype=obj | spirv-val %}

; NO-EXT: QuadDerivativesKHR requires the following SPIR-V extension: SPV_KHR_quad_control
; NO-RECONVERGENCE: SPV_KHR_quad_control requires the following SPIR-V extension: SPV_KHR_maximal_reconvergence
; VERSION: QuadDerivativesKHR requires SPIR-V 1.3

; CHECK-DAG: OpCapability QuadControlKHR
; CHECK-DAG: OpExtension "SPV_KHR_quad_control"
; CHECK-DAG: OpExtension "SPV_KHR_maximal_reconvergence"
; CHECK: OpEntryPoint Fragment %[[#MAIN:]] "main"
; CHECK: OpExecutionMode %[[#MAIN]] QuadDerivativesKHR

define void @main() #0 {
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!spirv.ExecutionMode = !{!0}
!0 = !{ptr @main, i32 5088}
