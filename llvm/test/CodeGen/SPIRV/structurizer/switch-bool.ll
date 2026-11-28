; Check that a switch on an i1 is lowered to a conditional branch before
; structurization, so that OpSelectionMerge directly precedes it.

; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --implicit-check-not=OpSwitch
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O2 %s -o - | FileCheck %s --implicit-check-not=OpSwitch
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#COND:]] = OpINotEqual
; CHECK-NEXT: OpSelectionMerge %[[#MERGE:]] None
; CHECK-NEXT: OpBranchConditional %[[#COND]] %[[#]] %[[#]]
; CHECK: %[[#MERGE]] = OpLabel

@in = internal addrspace(10) global i32 0
@out = internal addrspace(10) global i32 0

define void @main() #0 {
entry:
  %t = call token @llvm.experimental.convergence.entry()
  %v = load i32, ptr addrspace(10) @in
  %c = trunc i32 %v to i1
  switch i1 %c, label %default [
    i1 false, label %false
    i1 true, label %true
  ]
false:
  store i32 1, ptr addrspace(10) @out
  br label %exit
true:
  store i32 2, ptr addrspace(10) @out
  br label %exit
default:
  store i32 3, ptr addrspace(10) @out
  br label %exit
exit:
  ret void
}

declare token @llvm.experimental.convergence.entry()

attributes #0 = { convergent "hlsl.shader"="compute" "hlsl.numthreads"="1,1,1" }
