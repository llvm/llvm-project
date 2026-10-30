; RUN: split-file %s %t
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-vertex -spirv-ext=+SPV_EXT_shader_viewport_index_layer %t/vertex.ll -o - | FileCheck %s --check-prefix=VS
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-vertex %t/vertex.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %t/pixel.ll -o - | FileCheck %s --check-prefix=PS --implicit-check-not=OpExtension
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-vertex -spirv-ext=+SPV_EXT_shader_viewport_index_layer %t/vertex.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %t/pixel.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; SV_RenderTargetArrayIndex is lowered to the Layer built-in. Writing it
; from a vertex shader requires ShaderViewportIndexLayerEXT, while reading it
; in a fragment shader requires Geometry.

; VS-DAG: OpCapability ShaderViewportIndexLayerEXT
; VS-DAG: OpExtension "SPV_EXT_shader_viewport_index_layer"
; VS-DAG: OpDecorate %[[#LAYER:]] BuiltIn Layer
; VS-DAG: %[[#LAYER]] = OpVariable %[[#]] Output
; ERROR: LLVM ERROR: Writing ViewportIndex or Layer requires the following SPIR-V extension: SPV_EXT_shader_viewport_index_layer

; PS-DAG: OpCapability Geometry
; PS-DAG: OpDecorate %[[#LAYER:]] BuiltIn Layer
; PS-DAG: OpDecorate %[[#LAYER]] Flat
; PS-DAG: %[[#LAYER]] = OpVariable %[[#]] Input

;--- vertex.ll
@SV_Position = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations !0
@SV_RenderTargetArrayIndex = external hidden thread_local addrspace(8) global i32, !spirv.Decorations !2

define void @main() #0 {
entry:
  store <4 x float> zeroinitializer, ptr addrspace(8) @SV_Position, align 16
  store i32 1, ptr addrspace(8) @SV_RenderTargetArrayIndex, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!0 = !{!1}
!1 = !{i32 11, i32 0}
!2 = !{!3, !4}
!3 = !{i32 11, i32 9}
!4 = !{i32 14}

;--- pixel.ll
@SV_RenderTargetArrayIndex = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !3

define void @main() #0 {
entry:
  %0 = load i32, ptr addrspace(7) @SV_RenderTargetArrayIndex, align 4
  %1 = uitofp i32 %0 to float
  store float %1, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1, !2}
!1 = !{i32 11, i32 9}
!2 = !{i32 14}
!3 = !{!4}
!4 = !{i32 30, i32 0}
