; RUN: llc -march=hexagon < %s | FileCheck %s

; Widening of ISD::CTTZ_ELTS / CTTZ_ELTS_ZERO_POISON when the operand
; vector type is illegal (<3 x i32> widens to <4 x i32>). Padded lanes
; must be all-ones so the count never falls into the synthetic lane.

; All-zero input must return OrigElts (3), never WideElts (4).
; CHECK-LABEL: cttz_elts_zero_v3i32:
; CHECK:        r0 = #3
; CHECK-NEXT:   jumpr r31
; CHECK-NOT:    r0 = #4
define i32 @cttz_elts_zero_v3i32() {
  %res = call i32 @llvm.experimental.cttz.elts.i32.v3i32(<3 x i32> zeroinitializer, i1 false)
  ret i32 %res
}

; First non-zero element at the highest *original* lane (lane 2 of
; <3 x i32>). The padded lane 3 must not steal the result: answer is 2.
; CHECK-LABEL: cttz_elts_high_lane_v3i32:
; CHECK:        r0 = #2
; CHECK-NEXT:   jumpr r31
; CHECK-NOT:    r0 = #3
define i32 @cttz_elts_high_lane_v3i32() {
  %res = call i32 @llvm.experimental.cttz.elts.i32.v3i32(<3 x i32> <i32 0, i32 0, i32 9>, i1 false)
  ret i32 %res
}

; Symbolic input: confirm the operand is actually widened to <4 x i32>.
; Two vcmpw.eq compares against the zero pair cover both halves of the
; widened vector, and the final sub(#4, ...) reflects the widened lane
; count. A regression that fails to widen, or widens without all-ones
; padding, would change these.
; CHECK-LABEL: cttz_elts_v3i32:
; CHECK:        r{{[0-9]+}}:{{[0-9]+}} = combine(#0,#0)
; CHECK:        vcmpw.eq(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
; CHECK:        vcmpw.eq(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
; CHECK:        r0 = sub(#4,r{{[0-9]+}})
; CHECK-NEXT:   jumpr r31
define i32 @cttz_elts_v3i32(<3 x i32> %v) {
  %res = call i32 @llvm.experimental.cttz.elts.i32.v3i32(<3 x i32> %v, i1 false)
  ret i32 %res
}

; Same shape for the zero-poison variant; padding must still be emitted.
; CHECK-LABEL: cttz_elts_zero_poison_v3i32:
; CHECK:        r{{[0-9]+}}:{{[0-9]+}} = combine(#0,#0)
; CHECK:        vcmpw.eq(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
; CHECK:        vcmpw.eq(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
; CHECK:        r0 = sub(#4,r{{[0-9]+}})
; CHECK-NEXT:   jumpr r31
define i32 @cttz_elts_zero_poison_v3i32(<3 x i32> %v) {
  %res = call i32 @llvm.experimental.cttz.elts.i32.v3i32(<3 x i32> %v, i1 true)
  ret i32 %res
}

declare i32 @llvm.experimental.cttz.elts.i32.v3i32(<3 x i32>, i1)
