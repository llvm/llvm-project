; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3-geometry %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-geometry %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability Geometry
; CHECK-DAG: OpEntryPoint Geometry %[[#GS:]] "gs_main"
; CHECK-DAG: OpExecutionMode %[[#GS]] Triangles
; CHECK-DAG: OpExecutionMode %[[#GS]] OutputTriangleStrip
; CHECK-DAG: OpExecutionMode %[[#GS]] OutputVertices 3
; CHECK-DAG: OpExecutionMode %[[#GS]] Invocations 1

define void @gs_main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.shader"="geometry" }

!spirv.ExecutionMode = !{!0, !1, !2, !3}
!0 = !{ptr @gs_main, i32 22}
!1 = !{ptr @gs_main, i32 29}
!2 = !{ptr @gs_main, i32 26, i32 3}
!3 = !{ptr @gs_main, i32 0, i32 1}
