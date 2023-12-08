; RUN: llc < %s -march=nvptx64 -mcpu=sm_75 -mattr=+ptx70 | FileCheck %s
; RUN: %if ptxas-11.0 %{ llc < %s -march=nvptx64 -mcpu=sm_75 -mattr=+ptx70 | %ptxas-verify -arch=sm_75 %}

declare half @llvm.nvvm.ex2.approx.f16(half)
declare <2 x half> @llvm.nvvm.ex2.approx.f16x2(<2 x half>)

; CHECK-LABEL: exp2_half
define half @exp2_half(half %0) {
  ; CHECK-NOT: call
  ; CHECK: ex2.approx.f16
  %res = call half @llvm.nvvm.ex2.approx.f16(half %0);
  ret half %res
}

; CHECK-LABEL: exp2_2xhalf
define <2 x half> @exp2_2xhalf(<2 x half> %0) {
  ; CHECK-NOT: call
  ; CHECK: ex2.approx.f16x2
  %res = call <2 x half> @llvm.nvvm.ex2.approx.f16x2(<2 x half> %0);
  ret <2 x half> %res
}
