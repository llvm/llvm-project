; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpEntryPoint GLCompute %[[#entry:]] "main"
; CHECK-DAG: OpExecutionMode %[[#entry]] LocalSize 4 8 16

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
