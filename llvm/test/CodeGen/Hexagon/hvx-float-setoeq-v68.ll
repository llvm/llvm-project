; RUN: llc -march=hexagon -mattr=+hvxv68,+hvx-length128b,+hvx-ieee-fp %s -o - | FileCheck %s
;
; Verify that ordered-equal (setoeq) on HVX float vectors uses IEEE-754 float
; greater-than (V6_vgtsf/V6_vgthf) combined with integer NaN detection rather
; than the integer word/half comparison V6_veqw/V6_veqh.
;
; The integer variants treat NaN as equal to itself (same bit pattern), which
; breaks NaN detection.  The correct expansion is:
;   oeq(a, b) = NOT(ogt(a,b) OR ogt(b,a) OR isNaN(a) OR isNaN(b))
; where isNaN uses integer bit manipulation.
;

; Single-register f32: result predicate is v32i1
define <32 x float> @setoeq_v32f32(<32 x float> %a, <32 x float> %b,
                                    <32 x float> %c, <32 x float> %d) {
  %cmp = fcmp oeq <32 x float> %a, %b
  %r   = select <32 x i1> %cmp, <32 x float> %c, <32 x float> %d
  ret <32 x float> %r
}
; CHECK-LABEL: setoeq_v32f32
; Float GT comparisons (a > b) and (b > a)
; CHECK-DAG: vcmp.gt({{.*}}.sf,{{.*}}.sf)
; Integer AND to mask sign bit for NaN detection
; CHECK-DAG: vand(
; Integer GT against NaN threshold
; CHECK-DAG: vcmp.gt({{.*}}.w,{{.*}}.w)
; Must NOT use integer equality for the comparison
; CHECK-NOT: vcmp.eq({{.*}}.w,{{.*}}.w)

; Double-register f32: split into two single-register operations
define <64 x float> @setoeq_v64f32(<64 x float> %a, <64 x float> %b,
                                    <64 x float> %c, <64 x float> %d) {
  %cmp = fcmp oeq <64 x float> %a, %b
  %r   = select <64 x i1> %cmp, <64 x float> %c, <64 x float> %d
  ret <64 x float> %r
}
; CHECK-LABEL: setoeq_v64f32
; CHECK-COUNT-2: vcmp.gt({{.*}}.sf,{{.*}}.sf)
; CHECK-NOT: vcmp.eq({{.*}}.w,{{.*}}.w)

; Single-register f16
define <64 x half> @setoeq_v64f16(<64 x half> %a, <64 x half> %b,
                                   <64 x half> %c, <64 x half> %d) {
  %cmp = fcmp oeq <64 x half> %a, %b
  %r   = select <64 x i1> %cmp, <64 x half> %c, <64 x half> %d
  ret <64 x half> %r
}
; CHECK-LABEL: setoeq_v64f16
; CHECK-DAG: vcmp.gt({{.*}}.hf,{{.*}}.hf)
; CHECK-DAG: vand(
; CHECK-DAG: vcmp.gt({{.*}}.h,{{.*}}.h)
; CHECK-NOT: vcmp.eq({{.*}}.h,{{.*}}.h)

; Double-register f16
define <128 x half> @setoeq_v128f16(<128 x half> %a, <128 x half> %b,
                                     <128 x half> %c, <128 x half> %d) {
  %cmp = fcmp oeq <128 x half> %a, %b
  %r   = select <128 x i1> %cmp, <128 x half> %c, <128 x half> %d
  ret <128 x half> %r
}
; CHECK-LABEL: setoeq_v128f16
; CHECK-COUNT-2: vcmp.gt({{.*}}.hf,{{.*}}.hf)
; CHECK-NOT: vcmp.eq({{.*}}.h,{{.*}}.h)

