; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; cacheKey bit-packing coverage. The wrapper computes
;   cacheKey = funcIdx(32b) | dim[0](8b) | dim[1](8b) | dim[2](8b) | dim[3](8b)
; Each dimension is truncated to i8 (so an out-of-range index cannot bleed into
; an adjacent dimension's bit-field), zero-extended, then shifted by
; (list-position * 8) — note the shift uses the dimension's ORDER in the
; metadata list, NOT the function-argument index. These cases pin down the
; trunc/zext/shl/or sequence for the common and corner argument widths.

; --- i8 dim (the canonical uint8_t cellIdx): trunc to i8 is a no-op and must
;     be elided, leaving a bare zext. dim[1] must be shifted left by 8. ---
; CHECK-LABEL: define void @two_dim_i8(i8 %cell, i8 %trp)
; CHECK: jit_entry:
; CHECK-NOT: trunc i8 %cell
; CHECK: %[[Z0:.*]] = zext i8 %cell to i64
; CHECK: %[[O0:.*]] = or i64 {{-?[0-9]+}}, %[[Z0]]
; CHECK: %[[Z1:.*]] = zext i8 %trp to i64
; CHECK: %[[S1:.*]] = shl i64 %[[Z1]], 8
; CHECK: %[[O1:.*]] = or i64 %[[O0]], %[[S1]]
; CHECK: call ptr @ejit_compile_or_get(i64 %[[O1]], ptr null)
define void @two_dim_i8(i8 %cell, i8 %trp) !ejit.metadata !0 {
entry:
  ret void
}

; --- single dim attached to argument index 1 (not 0): it is still the first
;     (and only) dimension in the list, so it must occupy bits 0-7 with NO
;     shift, and must read %cell (arg 1), not the ignored arg 0. ---
; CHECK-LABEL: define void @dim_on_arg1(i32 %ignored, i8 %cell)
; CHECK: jit_entry:
; CHECK: %[[Z:.*]] = zext i8 %cell to i64
; CHECK-NOT: shl
; CHECK: %[[O:.*]] = or i64 {{-?[0-9]+}}, %[[Z]]
; CHECK: call ptr @ejit_compile_or_get(i64 %[[O]], ptr null)
define void @dim_on_arg1(i32 %ignored, i8 %cell) !ejit.metadata !1 {
entry:
  ret void
}

; --- i64 dim: a wider-than-i8 index must be truncated to i8 before zext, so a
;     value > 255 cannot pollute neighbouring dimension bits. ---
; CHECK-LABEL: define void @dim_i64(i64 %cell)
; CHECK: jit_entry:
; CHECK: %[[T:.*]] = trunc i64 %cell to i8
; CHECK: %[[Z:.*]] = zext i8 %[[T]] to i64
; CHECK: %[[O:.*]] = or i64 {{-?[0-9]+}}, %[[Z]]
; CHECK: call ptr @ejit_compile_or_get(i64 %[[O]], ptr null)
define void @dim_i64(i64 %cell) !ejit.metadata !2 {
entry:
  ret void
}

!0 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}, !{!"ejit_period_arr_ind", !"trp", i32 1}}
!1 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 1}}
!2 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
