; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@sv_position = external hidden thread_local local_unnamed_addr addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations !0

; CHECK-NOT: OpDecorate %[[#var]] LinkageAttributes "sv_position" Import

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float4:]] = OpTypeVector %[[#float]]
; CHECK-DAG: %[[#type:]] = OpTypePointer Input %[[#float4]]
; CHECK-DAG: %[[#var:]] = OpVariable %[[#type]] Input

; CHECK-NOT: OpDecorate %[[#var]] LinkageAttributes "sv_position" Import

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { "hlsl.shader"="pixel" }

!0 = !{!1}
!1 = !{i32 11, i32 0}
