; RUN: split-file %s %t
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %t/pixel.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %t/helper.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-vertex %t/vertex.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-vertex %t/vertex.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3 %t/mixed.ll -o - | FileCheck %s --check-prefix=MIXED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3 %t/mixed.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; BuiltIn PointSize variables cannot be used in Fragment shaders, directly or
; through a called function. A Vertex entry point in the same module can still
; write one.

; ERROR: error: {{.*}}BuiltIn PointSize cannot be used in a Fragment entry point (VUID-PointSize-PointSize-04314)

; CHECK-DAG: OpEntryPoint Vertex
; CHECK-DAG: OpDecorate %[[#PSIZE:]] BuiltIn PointSize
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Output %[[#FLOAT]]
; CHECK-DAG: %[[#PSIZE]] = OpVariable %[[#PTR]] Output
; CHECK: OpStore %[[#PSIZE]]

; MIXED-DAG: OpEntryPoint Vertex %[[#]] "vs"
; MIXED-DAG: OpEntryPoint Fragment %[[#]] "ps"
; MIXED-DAG: OpDecorate %[[#]] BuiltIn PointSize

;--- pixel.ll
@PSIZE = external hidden thread_local addrspace(7) externally_initialized constant float, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !2

define void @main() #0 {
entry:
  %0 = load float, ptr addrspace(7) @PSIZE, align 4
  store float %0, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1}
!1 = !{i32 11, i32 1}
!2 = !{!3}
!3 = !{i32 30, i32 0}

;--- helper.ll
@PSIZE = external hidden thread_local addrspace(7) externally_initialized constant float, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !2

define internal spir_func float @read_size() #1 {
entry:
  %0 = load float, ptr addrspace(7) @PSIZE, align 4
  ret float %0
}

define void @main() #0 {
entry:
  %0 = call spir_func float @read_size()
  store float %0, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }
attributes #1 = { noinline }

!0 = !{!1}
!1 = !{i32 11, i32 1}
!2 = !{!3}
!3 = !{i32 30, i32 0}

;--- vertex.ll
@SV_Position = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations !0
@PSIZE = external hidden thread_local addrspace(8) global float, !spirv.Decorations !2

define void @main() #0 {
entry:
  store <4 x float> zeroinitializer, ptr addrspace(8) @SV_Position, align 16
  store float 1.000000e+00, ptr addrspace(8) @PSIZE, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!0 = !{!1}
!1 = !{i32 11, i32 0}
!2 = !{!3}
!3 = !{i32 11, i32 1}

;--- mixed.ll
@SV_Position = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations !0
@PSIZE = external hidden thread_local addrspace(8) global float, !spirv.Decorations !2
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !4

define void @vs() #0 {
entry:
  store <4 x float> zeroinitializer, ptr addrspace(8) @SV_Position, align 16
  store float 1.000000e+00, ptr addrspace(8) @PSIZE, align 4
  ret void
}

define void @ps() #1 {
entry:
  store float 1.000000e+00, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }
attributes #1 = { "hlsl.shader"="pixel" }

!0 = !{!1}
!1 = !{i32 11, i32 0}
!2 = !{!3}
!3 = !{i32 11, i32 1}
!4 = !{!5}
!5 = !{i32 30, i32 0}
