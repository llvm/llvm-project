; All OpVariable instructions in a function must be the first instructions in the first block

; RUN: llc -O0 -mtriple=spirv32-unknown-linux %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-linux %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: OpVariable
; CHECK-SPIRV-NEXT: OpVariable
; CHECK-SPIRV: OpReturn
; CHECK-SPIRV: OpFunctionEnd

define void @main() #1 {
entry:
  %0 = alloca <2 x i32>, align 4
  %1 = getelementptr <2 x i32>, ptr %0, i32 0, i32 0
  %2 = alloca float, align 4
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
