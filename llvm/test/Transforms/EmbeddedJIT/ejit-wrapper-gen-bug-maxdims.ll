; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s
;
; XFAIL: *
;
; KNOWN BUG (recorded now, fix tracked separately).
;
; The cacheKey loop shifts dimension i by (i * 8) and never clamps the number of
; dimensions to the documented maximum of 4. The cacheKey layout is
;     funcIdx(32b) | dim0(8b) | dim1(8b) | dim2(8b) | dim3(8b)
; so only 4 dims fit in the low 32 bits. With more than 4 period_arr_ind entries
; (corrupt / Sema-bypassed metadata) the pass emits:
;   - dim4..dim7: shl by 32/40/48/56 -> overwrites the funcIdx bits 32..63
;   - dim8:       shl i64 %x, 64      -> shift-by-bitwidth is POISON (UB)
;
; The fix should clamp the dimension count to 4 (matching SPEC's limit and the
; key layout). The CHECK-NOT below encodes the critical invariant (no poison
; shift); today the pass emits `shl i64 %x, 64`, so it fails -> XFAIL.

define void @too_many_dims(i8 %a, i8 %b, i8 %c, i8 %d,
                           i8 %e, i8 %f, i8 %g, i8 %h, i8 %i)
                           !ejit.metadata !0 {
entry:
  ret void
}

!0 = distinct !{!{!"ejit_entry"},
  !{!"ejit_period_arr_ind", !"d0", i32 0}, !{!"ejit_period_arr_ind", !"d1", i32 1},
  !{!"ejit_period_arr_ind", !"d2", i32 2}, !{!"ejit_period_arr_ind", !"d3", i32 3},
  !{!"ejit_period_arr_ind", !"d4", i32 4}, !{!"ejit_period_arr_ind", !"d5", i32 5},
  !{!"ejit_period_arr_ind", !"d6", i32 6}, !{!"ejit_period_arr_ind", !"d7", i32 7},
  !{!"ejit_period_arr_ind", !"d8", i32 8}}

; No dimension may be shifted to or past the i64 width, and dims beyond the 4th
; must not be encoded at all (they would clobber the funcIdx bits 32..63).
; CHECK-LABEL: define void @too_many_dims
; CHECK-NOT: shl i64 {{.*}}, 64
; CHECK-NOT: shl i64 {{.*}}, 56
; CHECK-NOT: shl i64 {{.*}}, 48
; CHECK-NOT: shl i64 {{.*}}, 40
; CHECK-NOT: shl i64 {{.*}}, 32
