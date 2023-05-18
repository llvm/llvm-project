; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpCapability Shader
;; Ensure no other capability is listed.
; CHECK-NOT: OpCapability

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
