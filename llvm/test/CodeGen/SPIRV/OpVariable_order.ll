; REQUIRES: spirv-tools
; RUN: llc -O0 -mtriple=spirv-unknown-linux %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck %s

; TODO(#66261): The SPIR-V backend should reorder OpVariable instructions so this doesn't fail,
;     but in the meantime it's a good example of the spirv-val tool working as intended.

; CHECK: All OpVariable instructions in a function must be the first instructions in the first block.

define void @main() #1 {
entry:
  %0 = alloca <2 x i32>, align 4
  %1 = getelementptr <2 x i32>, ptr %0, i32 0, i32 0
  %2 = alloca float, align 4
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
