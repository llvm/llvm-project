; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3-domain %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-domain %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability Tessellation
; CHECK-DAG: OpEntryPoint TessellationEvaluation %[[#DS:]] "ds_main"
; CHECK-DAG: OpExecutionMode %[[#DS]] Triangles

define void @ds_main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.shader"="domain" }

!spirv.ExecutionMode = !{!0}
!0 = !{ptr @ds_main, i32 22}
