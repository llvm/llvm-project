; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

%A = type {
  i32,
  i32
}

%B = type {
  %A,
  i32,
  %A
}

; Make sure all struct types are removed.
; CHECK-NOT: OpTypeStruct

; Make sure the GEPs and the function scope variable are removed.
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define void @main() #1 {
entry:
  %0 = alloca %B, align 4
  %1 = getelementptr %B, ptr %0, i32 0, i32 2
  %2 = getelementptr %A, ptr %1, i32 0, i32 1
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
