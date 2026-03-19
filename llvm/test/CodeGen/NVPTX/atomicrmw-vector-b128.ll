; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx82 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx84 | FileCheck %s --check-prefix=CHECK
; RUN: %if ptxas-sm_90 && ptxas-isa-8.4 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx84 | %ptxas-verify -arch=sm_90 %}

target triple = "nvptx64-nvidia-cuda"

; ERROR: error: unsupported cmpxchg

define <4 x float> @fadd_v4f32_monotonic(ptr %addr, <4 x float> %val) {
; CHECK-LABEL: fadd_v4f32_monotonic(
; CHECK: atom.relaxed.sys.cas.b128
; CHECK-NOT: __atomic_compare_exchange_16
entry:
  %old = atomicrmw fadd ptr %addr, <4 x float> %val monotonic, align 16
  ret <4 x float> %old
}

define <4 x float> @fadd_v4f32_seq_cst(ptr %addr, <4 x float> %val) {
; CHECK-LABEL: fadd_v4f32_seq_cst(
; CHECK: fence.sc.sys
; CHECK: atom.acquire.sys.cas.b128
; CHECK-NOT: __atomic_compare_exchange_16
entry:
  %old = atomicrmw fadd ptr %addr, <4 x float> %val seq_cst, align 16
  ret <4 x float> %old
}
