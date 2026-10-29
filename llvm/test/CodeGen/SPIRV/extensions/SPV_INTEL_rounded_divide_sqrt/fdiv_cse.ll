; This test catches a bug as a result of MachineCSE: with the default pipeline (-O2),
; MachineCSE dropped the 2nd fdiv that differ only in rounding mode as redundant computation
; (most likely because the rounding mode becomes a separate OpDecorate instruction instead
; of an operand).
; The RTN divide below is dropped and only the RTP one survives. It passes with -O0 or
; -disable-machine-cse.
; Remove XFAIL once fixed.
; XFAIL: *

; RUN: llc -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - | FileCheck %s

; CHECK-DAG: OpDecorate %[[#RTP:]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#RTN:]] FPRoundingMode RTN
; CHECK-DAG: %[[#RTP]] = OpFDiv %[[#]] %[[#A:]] %[[#B:]]
; CHECK-DAG: %[[#RTN]] = OpFDiv %[[#]] %[[#A]] %[[#B]]

define spir_kernel void @test(double %a, double %b, ptr addrspace(1) %p0, ptr addrspace(1) %p1) strictfp {
entry:
  %rtp = call double @llvm.experimental.constrained.fdiv.f64(double %a, double %b, metadata !"round.upward", metadata !"fpexcept.strict")
  %rtn = call double @llvm.experimental.constrained.fdiv.f64(double %a, double %b, metadata !"round.downward", metadata !"fpexcept.strict")
  store double %rtp, ptr addrspace(1) %p0
  store double %rtn, ptr addrspace(1) %p1
  ret void
}

declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
