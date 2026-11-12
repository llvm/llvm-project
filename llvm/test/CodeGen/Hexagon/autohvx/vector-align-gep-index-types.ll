; RUN: opt -mtriple=hexagon -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b \
; RUN:   -hexagon-vc -hvc-ld-min-group-size-for-alignment=2 -S < %s \
; RUN:   | FileCheck %s

; HexagonVectorCombine::calculatePointerDifference falls back to subtracting
; the indices of two single-index GEPs with the same base. Only indices already
; in the pointer's index type (i32 here) are subtracted. Other index types used
; to assert ("Cannot create binary operator with two operands of differing
; type!") when they differed, and to compute a wrong difference when they were
; narrower than the index type and the subtraction wrapped. Such GEPs are now
; left alone.

; Different index types should not crash.

; CHECK-LABEL: @mixed_i64_i32(
; CHECK: store i8 0, ptr %a, align 1
; CHECK: store i8 1, ptr %b, align 1
define void @mixed_i64_i32(ptr %p, i64 %i, i32 %j) {
  %a = getelementptr i8, ptr %p, i64 %i
  %b = getelementptr i8, ptr %p, i32 %j
  store i8 0, ptr %a, align 1
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK-LABEL: @mixed_i32_i16(
; CHECK: store i8 0, ptr %a, align 1
; CHECK: store i8 1, ptr %b, align 1
define void @mixed_i32_i16(ptr %p, i32 %i, i16 %j) {
  %a = getelementptr i8, ptr %p, i32 %i
  %b = getelementptr i8, ptr %p, i16 %j
  store i8 0, ptr %a, align 1
  store i8 1, ptr %b, align 1
  ret void
}

; Index wider than 64 bits.
; CHECK-LABEL: @mixed_i128_i32(
; CHECK: store i8 0, ptr %a, align 1
; CHECK: store i8 1, ptr %b, align 1
define void @mixed_i128_i32(ptr %p, i128 %i, i32 %j) {
  %a = getelementptr i8, ptr %p, i128 %i
  %b = getelementptr i8, ptr %p, i32 %j
  store i8 0, ptr %a, align 1
  store i8 1, ptr %b, align 1
  ret void
}

; Same (narrow) index type, but %x - %y wraps in i8: the real difference
; between %b and %a is -128 for %x >= 0 and +128 for %x < 0. The loads must
; not be combined as if they were at a constant offset.

; CHECK-LABEL: @narrow_wrap(
; CHECK: %va = load <128 x i8>, ptr %a, align 1
; CHECK: %vb = load <128 x i8>, ptr %b, align 1
; CHECK-NOT: valignb
; CHECK: ret <128 x i8>
define <128 x i8> @narrow_wrap(ptr %p, i8 %x) {
  %y = add i8 %x, -128
  %a = getelementptr i8, ptr %p, i8 %x
  %b = getelementptr i8, ptr %p, i8 %y
  %va = load <128 x i8>, ptr %a, align 1
  %vb = load <128 x i8>, ptr %b, align 1
  %s = add <128 x i8> %va, %vb
  ret <128 x i8> %s
}

; Same as above, but with nsw the difference is a constant -128 that SCEV can
; prove, so the loads are still combined (the GEP fallback is not reached).

; CHECK-LABEL: @narrow_nsw(
; CHECK: load <128 x i8>, ptr %{{.*}}, align 128
; CHECK: valignb
; CHECK: ret <128 x i8>
define <128 x i8> @narrow_nsw(ptr %p, i8 %x) {
  %y = add nsw i8 %x, -128
  %a = getelementptr i8, ptr %p, i8 %x
  %b = getelementptr i8, ptr %p, i8 %y
  %va = load <128 x i8>, ptr %a, align 1
  %vb = load <128 x i8>, ptr %b, align 1
  %s = add <128 x i8> %va, %vb
  ret <128 x i8> %s
}
