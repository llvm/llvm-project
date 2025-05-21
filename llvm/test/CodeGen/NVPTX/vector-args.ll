; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

define float @foo(<2 x float> %a) {
; CHECK: .func (.param .b32 func_retval0) foo
; CHECK: .param .align 8 .b8 foo_param_0[8]
; CHECK: ld.param.v2.b32 {%r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = fmul <2 x float> %a, %a
  %t2 = extractelement <2 x float> %t1, i32 0
  %t3 = extractelement <2 x float> %t1, i32 1
  %t4 = fadd float %t2, %t3
  ret float %t4
}


define float @bar(<4 x float> %a) {
; CHECK: .func (.param .b32 func_retval0) bar
; CHECK: .param .align 16 .b8 bar_param_0[16]
; CHECK: ld.param.v4.b32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = fmul <4 x float> %a, %a
  %t2 = extractelement <4 x float> %t1, i32 0
  %t3 = extractelement <4 x float> %t1, i32 1
  %t4 = fadd float %t2, %t3
  ret float %t4
}


define <4 x float> @baz(<4 x float> %a) {
; CHECK: .func  (.param .align 16 .b8 func_retval0[16]) baz
; CHECK: .param .align 16 .b8 baz_param_0[16]
; CHECK: ld.param.v4.b32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
; CHECK: st.param.v4.b32 [func_retval0], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = fmul <4 x float> %a, %a
  ret <4 x float> %t1
}
