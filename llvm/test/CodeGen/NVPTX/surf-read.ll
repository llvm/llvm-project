; RUN: llc < %s -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target triple = "nvptx64-unknown-nvcl"

declare i32 @llvm.nvvm.suld.1d.i32.trap(i64, i32)

; CHECK: .entry foo
define ptx_kernel void @foo(i64 %img, ptr %red, i32 %idx) {
; CHECK: .param .surfref foo_param_0
; CHECK: suld.b.1d.b32.trap {%r[[RED:[0-9]+]]}, [foo_param_0, {%r{{[0-9]+}}}]
  %val = tail call i32 @llvm.nvvm.suld.1d.i32.trap(i64 %img, i32 %idx)
; CHECK: cvt.rn.f32.s32 %r[[REDF:[0-9]+]], %r[[RED]]
  %ret = sitofp i32 %val to float
; CHECK: st.b32 [%rd{{[0-9]+}}], %r[[REDF]]
  store float %ret, ptr %red
  ret void
}

!nvvm.annotations = !{!1}
!1 = !{ptr @foo, !"rdwrimage", i32 0}
