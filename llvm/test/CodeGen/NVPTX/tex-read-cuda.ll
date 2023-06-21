; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s --check-prefix=SM20
; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | FileCheck %s --check-prefix=SM30
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | %ptxas-verify %}


target triple = "nvptx-unknown-cuda"

declare { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64, i32)
declare i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1))

; SM20-LABEL: .entry foo
; SM30-LABEL: .entry foo
define void @foo(i64 %img, ptr %red, i32 %idx) {
; SM20: ld.param.u64    %rd[[TEXREG:[0-9]+]], [foo_param_0];
; SM20: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [%rd[[TEXREG]], {%r{{[0-9]+}}}]
; SM30: ld.param.u64    %rd[[TEXREG:[0-9]+]], [foo_param_0];
; SM30: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [%rd[[TEXREG]], {%r{{[0-9]+}}}]
  %val = tail call { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64 %img, i32 %idx)
  %ret = extractvalue { float, float, float, float } %val, 0
; SM20: st.global.f32 [%rd{{[0-9]+}}], %f[[RED]]
; SM30: st.global.f32 [%rd{{[0-9]+}}], %f[[RED]]
  store float %ret, ptr %red
  ret void
}


@tex0 = internal addrspace(1) global i64 0, align 8

; SM20-LABEL: .entry bar
; SM30-LABEL: .entry bar
define void @bar(ptr %red, i32 %idx) {
; SM30: mov.u64 %rd[[TEXHANDLE:[0-9]+]], tex0 
  %texHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @tex0)
; SM20: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [tex0, {%r{{[0-9]+}}}]
; SM30: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [%rd[[TEXHANDLE]], {%r{{[0-9]+}}}]
  %val = tail call { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64 %texHandle, i32 %idx)
  %ret = extractvalue { float, float, float, float } %val, 0
; SM20: st.global.f32 [%rd{{[0-9]+}}], %f[[RED]]
; SM30: st.global.f32 [%rd{{[0-9]+}}], %f[[RED]]
  store float %ret, ptr %red
  ret void
}

declare float @texfunc(i64)

; SM20-LABEL: .entry baz
; SM30-LABEL: .entry baz
define void @baz(ptr %red, i32 %idx) {
; SM30: mov.u64 %rd[[TEXHANDLE:[0-9]+]], tex0
  %texHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @tex0)
; SM20: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [tex0, {%r{{[0-9]+}}}]
; SM30: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [%rd[[TEXHANDLE]], {%r{{[0-9]+}}}]
  %val = tail call { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64 %texHandle, i32 %idx)
  %ret = extractvalue { float, float, float, float } %val, 0
; SM20: call.uni ([[RETVAL:.*]]),
; SM30: call.uni ([[RETVAL:.*]]),
; SM20: texfunc,
; SM30: texfunc,
  %texcall = tail call float @texfunc(i64 %texHandle)
; SM20: ld.param.f32 %f[[TEXCALL:[0-9]+]], [[[RETVAL]]+0]
; SM30: ld.param.f32 %f[[TEXCALL:[0-9]+]], [[[RETVAL]]+0]
; SM20: add.rn.f32 %f[[RET2:[0-9]+]], %f[[RED]], %f[[TEXCALL]]
; SM30: add.rn.f32 %f[[RET2:[0-9]+]], %f[[RED]], %f[[TEXCALL]]
  %ret2 = fadd float %ret, %texcall
; SM20: st.global.f32 [%rd{{[0-9]+}}], %f[[RET2]]
; SM30: st.global.f32 [%rd{{[0-9]+}}], %f[[RET2]]
  store float %ret2, ptr %red
  ret void
}

!nvvm.annotations = !{!1, !2, !3, !4}
!1 = !{ptr @foo, !"kernel", i32 1}
!2 = !{ptr @bar, !"kernel", i32 1}
!3 = !{ptr addrspace(1) @tex0, !"texture", i32 1}
!4 = !{ptr @baz, !"kernel", i32 1}
