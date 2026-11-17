; RUN: llc -march=hexagon -mattr=+hvxv81,+hvx-length128b %s -o - | FileCheck %s
;
; Verify that ordered-equal (setoeq) on HVX float vectors uses the float
; comparison instruction V6_veqsf/V6_veqhf rather than the integer word/half
; comparison V6_veqw/V6_veqh.  The integer variants treat NaN as equal to
; itself (same bit pattern), which breaks NaN detection in vector reductions.

define <32 x float> @setoeq_v32f32(<32 x float> %a, <32 x float> %b,
                                    <32 x float> %c, <32 x float> %d) {
  %cmp = fcmp oeq <32 x float> %a, %b
  %r   = select <32 x i1> %cmp, <32 x float> %c, <32 x float> %d
  ret <32 x float> %r
}
; CHECK-LABEL: setoeq_v32f32
; CHECK: vcmp.eq({{.*}}.sf,{{.*}}.sf)
; CHECK-NOT: vcmp.eq({{.*}}.w,{{.*}}.w)

define <64 x float> @setoeq_v64f32(<64 x float> %a, <64 x float> %b,
                                    <64 x float> %c, <64 x float> %d) {
  %cmp = fcmp oeq <64 x float> %a, %b
  %r   = select <64 x i1> %cmp, <64 x float> %c, <64 x float> %d
  ret <64 x float> %r
}
; CHECK-LABEL: setoeq_v64f32
; CHECK-COUNT-2: vcmp.eq({{.*}}.sf,{{.*}}.sf)
; CHECK-NOT: vcmp.eq({{.*}}.w,{{.*}}.w)

define <64 x half> @setoeq_v64f16(<64 x half> %a, <64 x half> %b,
                                   <64 x half> %c, <64 x half> %d) {
  %cmp = fcmp oeq <64 x half> %a, %b
  %r   = select <64 x i1> %cmp, <64 x half> %c, <64 x half> %d
  ret <64 x half> %r
}
; CHECK-LABEL: setoeq_v64f16
; CHECK: vcmp.eq({{.*}}.hf,{{.*}}.hf)
; CHECK-NOT: vcmp.eq({{.*}}.h,{{.*}}.h)

define <128 x half> @setoeq_v128f16(<128 x half> %a, <128 x half> %b,
                                     <128 x half> %c, <128 x half> %d) {
  %cmp = fcmp oeq <128 x half> %a, %b
  %r   = select <128 x i1> %cmp, <128 x half> %c, <128 x half> %d
  ret <128 x half> %r
}
; CHECK-LABEL: setoeq_v128f16
; CHECK-COUNT-2: vcmp.eq({{.*}}.hf,{{.*}}.hf)
; CHECK-NOT: vcmp.eq({{.*}}.h,{{.*}}.h)
