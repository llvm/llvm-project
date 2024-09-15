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
; CHECK-NEXT:    .reg .b16 %rs<5>;
; CHECK-NEXT:    .reg .b32 %r<3>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.u8 %rs1, [select2or_param_0];
; CHECK-NEXT:    ld.param.u8 %rs2, [select2or_param_1];
; CHECK-NEXT:    or.b16 %rs4, %rs1, %rs2;
; CHECK-NEXT:    cvt.u32.u16 %r1, %rs4;
; CHECK-NEXT:    and.b32 %r2, %r1, 1;
; CHECK-NEXT:    st.param.b32 [func_retval0+0], %r2;
; CHECK-NEXT:    ret;
  %r = select i1 %a, i1 1, i1 %b
  ret i1 %r
}

define i1 @select2and(i1 %a, i1 %b) {
; CHECK-LABEL: select2and(
; CHECK:       {
; CHECK-NEXT:    .reg .b16 %rs<5>;
; CHECK-NEXT:    .reg .b32 %r<3>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.u8 %rs1, [select2and_param_0];
; CHECK-NEXT:    ld.param.u8 %rs2, [select2and_param_1];
; CHECK-NEXT:    and.b16 %rs4, %rs1, %rs2;
; CHECK-NEXT:    cvt.u32.u16 %r1, %rs4;
; CHECK-NEXT:    and.b32 %r2, %r1, 1;
; CHECK-NEXT:    st.param.b32 [func_retval0+0], %r2;
; CHECK-NEXT:    ret;
  %r = select i1 %a, i1 %b, i1 0
  ret i1 %r
}
