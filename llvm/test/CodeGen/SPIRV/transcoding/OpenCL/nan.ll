; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check OpenCL built-in nan translation.

; CHECK-SPIRV: %[[#]] = OpExtInst %[[#]] %[[#]] nan %[[#]]

define dso_local spir_kernel void @test(ptr addrspace(1) align 4 %a, i32 %b) {
entry:
  %call = tail call spir_func float @_Z3nanj(i32 %b)
  store float %call, ptr addrspace(1) %a, align 4
  ret void
}

declare spir_func float @_Z3nanj(i32)
