; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-EXT-OFF
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_fp_fast_math_mode %s -o - | FileCheck %s --check-prefix=CHECK-EXT-ON
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_fp_fast_math_mode %s -o - -filetype=obj | spirv-val %}
; XFAIL: *

; CHECK-EXT-ON: OpCapability FPFastMathModeINTEL
; CHECK-EXT-ON: SPV_INTEL_fp_fast_math_mode
; CHECK-EXT-ON: OpName %[[#mul:]] "mul"
; CHECK-EXT-ON: OpName %[[#sub:]] "sub"
; CHECK-EXT-ON: OpDecorate %[[#mu:]] FPFastMathMode AllowContract
; CHECK-EXT-ON: OpDecorate %[[#su:]] FPFastMathMode AllowReassoc

; CHECK-EXT-OFF-NOT: OpCapability FPFastMathModeINTEL
; CHECK-EXT-OFF-NOT: SPV_INTEL_fp_fast_math_mode
; CHECK-EXT-OFF: OpName %[[#mul:]] "mul"
; CHECK-EXT-OFF: OpName %[[#sub:]] "sub"
; CHECK-EXT-OFF-NOT: 4 Decorate %[[#mul]] FPFastMathMode AllowContract
; CHECK-EXT-OFF-NOT: 4 Decorate %[[#sub]] FPFastMathMode AllowReassoc

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
