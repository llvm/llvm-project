; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-vertex %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-vertex %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpCapability Shader
; CHECK: OpEntryPoint Vertex %[[#entry:]] "main"

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { "hlsl.shader"="vertex" }
