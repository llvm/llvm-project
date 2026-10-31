; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; A Sample-decorated fragment input requires SampleRateShading even without
; sample-related instructions or built-ins.
; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability SampleRateShading
; CHECK-DAG: OpName %[[#INPUT:]] "input"
; CHECK-DAG: OpDecorate %[[#INPUT]] Location 0
; CHECK-DAG: OpDecorate %[[#INPUT]] Sample

@input = external addrspace(7) global float, !spirv.Decorations !0
@output = external addrspace(8) global float, !spirv.Decorations !1

define void @main() #0 {
  %value = load float, ptr addrspace(7) @input
  store float %value, ptr addrspace(8) @output
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!2, !3}
!1 = !{!2}
!2 = !{i32 30, i32 0} ; Location 0
!3 = !{i32 17} ; Sample
