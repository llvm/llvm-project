; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: OpCapability FPFastMathModeINTEL
; CHECK-SPIRV:     OpName %[[#mu:]] "mul"
; CHECK-SPIRV:     OpName %[[#su:]] "sub"
; CHECK-SPIRV-NOT: OpDecorate %[[#mu]] FPFastMathMode AllowContractFastINTEL
; CHECK-SPIRV-NOT: OpDecorate %[[#su]] FPFastMathMode AllowReassocINTEL

define spir_kernel void @test(float %a, float %b) {
entry:
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, float* %a.addr, align 4
  store float %b, float* %b.addr, align 4
  %0 = load float, float* %a.addr, align 4
  %1 = load float, float* %a.addr, align 4
  %mul = fmul contract float %0, %1
  store float %mul, float* %b.addr, align 4
  %2 = load float, float* %b.addr, align 4
  %3 = load float, float* %b.addr, align 4
  %sub = fsub reassoc float %2, %3
  store float %sub, float* %b.addr, align 4
  ret void
}
