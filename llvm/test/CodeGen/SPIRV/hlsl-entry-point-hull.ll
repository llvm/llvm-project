; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3-hull %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-hull %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability Tessellation
; CHECK-DAG: OpEntryPoint TessellationControl %[[#HS:]] "hs_main"
; CHECK-DAG: OpExecutionMode %[[#HS]] OutputVertices 3

define void @hs_main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.shader"="hull" }

!spirv.ExecutionMode = !{!0}
!0 = !{ptr @hs_main, i32 26, i32 3}
