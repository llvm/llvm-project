; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: OpCapability FPFastMathModeINTEL
; CHECK-SPIRV-NOT: OpDecorate %[[#]] FPFastMathMode AllowContractFastINTEL
; CHECK-SPIRV-NOT: OpDecorate %[[#]] FPFastMathMode AllowReassocINTEL

define spir_kernel void @test(float %a, float %b) {
entry:
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = load float, ptr %a.addr, align 4
  %mul = fmul contract float %0, %1
  store float %mul, ptr %b.addr, align 4
  %2 = load float, ptr %b.addr, align 4
  %3 = load float, ptr %b.addr, align 4
  %sub = fsub reassoc float %2, %3
  store float %sub, ptr %b.addr, align 4
  ret void
}
