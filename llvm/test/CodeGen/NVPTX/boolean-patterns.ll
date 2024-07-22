; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target triple = "nvptx64-nvidia-cuda"

define i1 @m2and_rr(i1 %a, i1 %b) {
; CHECK: and.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK-NOT: mul
  %r = mul i1 %a, %b
  ret i1 %r
}

define i1 @m2and_ri(i1 %a) {
; CHECK-LABEL: m2and_ri(
; CHECK:       {
; CHECK-NEXT:    .reg .b32 %r<3>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.u8 %r1, [m2and_ri_param_0];
; CHECK-NEXT:    and.b32 %r2, %r1, 1;
; CHECK-NEXT:    st.param.b32 [func_retval0+0], %r2;
; CHECK-NEXT:    ret;
  %r = mul i1 %a, 1
  ret i1 %r
}

define i1 @select2or(i1 %a, i1 %b) {
; CHECK-LABEL: select2or(
; CHECK:       {
; CHECK-NEXT:    .reg .pred %p<4>;
; CHECK-NEXT:    .reg .b16 %rs<5>;
; CHECK-NEXT:    .reg .b32 %r<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.u8 %rs1, [select2or_param_1];
; CHECK-NEXT:    and.b16 %rs2, %rs1, 1;
; CHECK-NEXT:    setp.eq.b16 %p1, %rs2, 1;
; CHECK-NEXT:    ld.param.u8 %rs3, [select2or_param_0];
; CHECK-NEXT:    and.b16 %rs4, %rs3, 1;
; CHECK-NEXT:    setp.eq.b16 %p2, %rs4, 1;
; CHECK-NEXT:    or.pred %p3, %p2, %p1;
; CHECK-NEXT:    selp.u32 %r1, 1, 0, %p3;
; CHECK-NEXT:    st.param.b32 [func_retval0+0], %r1;
; CHECK-NEXT:    ret;
  %r = select i1 %a, i1 1, i1 %b
  ret i1 %r
}

define i1 @select2and(i1 %a, i1 %b) {
; CHECK-LABEL: select2and(
; CHECK:       {
; CHECK-NEXT:    .reg .pred %p<4>;
; CHECK-NEXT:    .reg .b16 %rs<5>;
; CHECK-NEXT:    .reg .b32 %r<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.u8 %rs1, [select2and_param_1];
; CHECK-NEXT:    and.b16 %rs2, %rs1, 1;
; CHECK-NEXT:    setp.eq.b16 %p1, %rs2, 1;
; CHECK-NEXT:    ld.param.u8 %rs3, [select2and_param_0];
; CHECK-NEXT:    and.b16 %rs4, %rs3, 1;
; CHECK-NEXT:    setp.eq.b16 %p2, %rs4, 1;
; CHECK-NEXT:    and.pred %p3, %p2, %p1;
; CHECK-NEXT:    selp.u32 %r1, 1, 0, %p3;
; CHECK-NEXT:    st.param.b32 [func_retval0+0], %r1;
; CHECK-NEXT:    ret;
  %r = select i1 %a, i1 %b, i1 0
  ret i1 %r
}
