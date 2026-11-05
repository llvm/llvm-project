; bfloat is not an IEEE 754 type, so constrained fdiv/sqrt on it get no
; FPRoundingMode even with SPV_INTEL_rounded_divide_sqrt enabled.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt,+SPV_KHR_bfloat16,+SPV_INTEL_bfloat16_arithmetic %s -o - | FileCheck %s --implicit-check-not=FPRoundingMode --implicit-check-not=RoundedDivideSqrtINTEL --implicit-check-not=SPV_INTEL_rounded_divide_sqrt
; spirv-val does not know BFloat16ArithmeticINTEL yet.
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt,+SPV_KHR_bfloat16,+SPV_INTEL_bfloat16_arithmetic %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFDiv
; CHECK: OpExtInst {{.*}} sqrt

define spir_kernel void @test(bfloat %a, bfloat %b, ptr addrspace(1) %p0, ptr addrspace(1) %p1) strictfp {
entry:
  %div = call bfloat @llvm.experimental.constrained.fdiv.bf16(bfloat %a, bfloat %b, metadata !"round.towardzero", metadata !"fpexcept.strict")
  %sqrt = call bfloat @llvm.experimental.constrained.sqrt.bf16(bfloat %a, metadata !"round.towardzero", metadata !"fpexcept.strict")
  store bfloat %div, ptr addrspace(1) %p0
  store bfloat %sqrt, ptr addrspace(1) %p1
  ret void
}

declare bfloat @llvm.experimental.constrained.fdiv.bf16(bfloat, bfloat, metadata, metadata)
declare bfloat @llvm.experimental.constrained.sqrt.bf16(bfloat, metadata, metadata)
