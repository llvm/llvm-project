; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.5-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.5-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that nofpclass attributes on non-OpExtInst instructions (regular
; function calls) do NOT produce FPFastMathMode decorations in Kernel
; environments before SPIR-V 1.6 without SPV_KHR_float_controls2.

; CHECK-NOT: FPFastMathMode
; CHECK:     OpFunctionCall

declare spir_func noundef nofpclass(nan inf) float @regular_func(float noundef nofpclass(nan inf))

define spir_kernel void @test(ptr addrspace(1) %data, ptr addrspace(1) %a) {
entry:
  %0 = load float, ptr addrspace(1) %a, align 4
  %reg = call spir_func noundef nofpclass(nan inf) float @regular_func(float noundef nofpclass(nan inf) %0)
  store float %reg, ptr addrspace(1) %data, align 4
  ret void
}
