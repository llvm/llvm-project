; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s --check-prefix=SM20
; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | FileCheck %s --check-prefix=SM30
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | %ptxas-verify %}

target triple = "nvptx-unknown-cuda"

declare i32 @llvm.nvvm.suld.1d.i32.trap(i64, i32)
declare i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1))


; SM20-LABEL: .entry foo
; SM30-LABEL: .entry foo
define void @foo(i64 %img, ptr %red, i32 %idx) {
; SM20: ld.param.u64    %rd[[SURFREG:[0-9]+]], [foo_param_0];
; SM20: suld.b.1d.b32.trap {%r[[RED:[0-9]+]]}, [%rd[[SURFREG]], {%r{{[0-9]+}}}]
; SM30: ld.param.u64    %rd[[SURFREG:[0-9]+]], [foo_param_0];
; SM30: suld.b.1d.b32.trap {%r[[RED:[0-9]+]]}, [%rd[[SURFREG]], {%r{{[0-9]+}}}]
  %val = tail call i32 @llvm.nvvm.suld.1d.i32.trap(i64 %img, i32 %idx)
; SM20: cvt.rn.f32.s32 %f[[REDF:[0-9]+]], %r[[RED]]
; SM30: cvt.rn.f32.s32 %f[[REDF:[0-9]+]], %r[[RED]]
  %ret = sitofp i32 %val to float
; SM20: st.global.f32 [%rd{{[0-9]+}}], %f[[REDF]]
; SM30: st.global.f32 [%rd{{[0-9]+}}], %f[[REDF]]
  store float %ret, ptr %red
  ret void
}

@surf0 = internal addrspace(1) global i64 0, align 8

; SM20-LABEL: .entry bar
; SM30-LABEL: .entry bar
define void @bar(ptr %red, i32 %idx) {
; SM30: mov.u64 %rd[[SURFHANDLE:[0-9]+]], surf0
  %surfHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @surf0)
; SM20: suld.b.1d.b32.trap {%r[[RED:[0-9]+]]}, [surf0, {%r{{[0-9]+}}}]
; SM30: suld.b.1d.b32.trap {%r[[RED:[0-9]+]]}, [%rd[[SURFHANDLE]], {%r{{[0-9]+}}}]
  %val = tail call i32 @llvm.nvvm.suld.1d.i32.trap(i64 %surfHandle, i32 %idx)
; SM20: cvt.rn.f32.s32 %f[[REDF:[0-9]+]], %r[[RED]]
; SM30: cvt.rn.f32.s32 %f[[REDF:[0-9]+]], %r[[RED]]
  %ret = sitofp i32 %val to float
; SM20: st.global.f32 [%rd{{[0-9]+}}], %f[[REDF]]
; SM30: st.global.f32 [%rd{{[0-9]+}}], %f[[REDF]]
  store float %ret, ptr %red
  ret void
}




!nvvm.annotations = !{!1, !2, !3}
!1 = !{ptr @foo, !"kernel", i32 1}
!2 = !{ptr @bar, !"kernel", i32 1}
!3 = !{ptr addrspace(1) @surf0, !"surface", i32 1}

