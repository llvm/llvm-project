; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpEntryPoint Fragment %[[#entry:]] "main"
; CHECK-DAG: OpExecutionMode %[[#entry]] OriginUpperLeft

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { "hlsl.shader"="pixel" }
