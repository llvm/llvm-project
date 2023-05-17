; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs -o - %s | FileCheck %s
; LiveRangeEdit::eliminateDeadDef did not update LiveInterval sub ranges
; properly.

; Just make sure this test doesn't crash.
; CHECK-LABEL: foobar:
; CHECK: s_endpgm
define amdgpu_kernel void @foobar() {
  %v0 = icmp eq <4 x i32> undef, <i32 0, i32 1, i32 2, i32 3>
  %v3 = sext <4 x i1> %v0 to <4 x i32>
  %v4 = extractelement <4 x i32> %v3, i32 1
  %v5 = icmp ne i32 %v4, 0
  %v6 = select i1 %v5, i32 undef, i32 0
  %v15 = insertelement <2 x i32> undef, i32 %v6, i32 1
  store <2 x i32> %v15, ptr addrspace(1) undef, align 8
  ret void
}

declare double @llvm.fma.f64(double, double, double)
