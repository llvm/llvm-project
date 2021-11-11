; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s

target triple = "nvptx-unknown-nvcl"

declare i32 @llvm.nvvm.suld.1d.i32.trap(i64, i32)

; CHECK: .entry foo
define void @foo(i64 %img, float* %red, i32 %idx) {
; CHECK: suld.b.1d.b32.trap {%r[[RED:[0-9]+]]}, [foo_param_0, {%r{{[0-9]+}}}]
  %val = tail call i32 @llvm.nvvm.suld.1d.i32.trap(i64 %img, i32 %idx)
; CHECK: cvt.rn.f32.s32 %f[[REDF:[0-9]+]], %r[[RED]]
  %ret = sitofp i32 %val to float
; CHECK: st.f32 [%r{{[0-9]+}}], %f[[REDF]]
  store float %ret, float* %red
  ret void
}

!nvvm.annotations = !{!1, !2}
!1 = !{void (i64, float*, i32)* @foo, !"kernel", i32 1}
!2 = !{void (i64, float*, i32)* @foo, !"rdwrimage", i32 0}
