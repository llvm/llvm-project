; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpEntryPoint GLCompute %[[#entry:]] "main"
; CHECK-DAG: OpExecutionMode %[[#entry]] LocalSize 4 8 16

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
