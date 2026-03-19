; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i32 @trunc_i65_to_i32(i65 %0) {
; CHECK-LABEL: trunc_i65_to_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i65 %0 to i32
  ret i32 %2
}

define i65 @sext_i32_to_i65(i32 signext %0) {
; CHECK-LABEL: sext_i32_to_i65:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i32 %0 to i65
  ret i65 %2
}

define i65 @zext_i32_to_i65(i32 zeroext %0) {
; CHECK-LABEL: zext_i32_to_i65:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i32 %0 to i65
  ret i65 %2
}

define signext i17 @trunc_i64_to_i17(i64 %0) {
; CHECK-LABEL: trunc_i64_to_i17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 47
; CHECK-NEXT:    sra.l %s0, %s0, 47
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i17
  ret i17 %2
}

define i64 @sext_i17_to_i64(i17 signext %0) {
; CHECK-LABEL: sext_i17_to_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i17 %0 to i64
  ret i64 %2
}

define i64 @zext_i17_to_i64(i17 zeroext %0) {
; CHECK-LABEL: zext_i17_to_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i17 %0 to i64
  ret i64 %2
}

define i128 @sext_i77_to_i128(i77 %0) {
; CHECK-LABEL: sext_i77_to_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s1, %s1, 51
; CHECK-NEXT:    sra.l %s1, %s1, 51
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i77 %0 to i128
  ret i128 %2
}

define i128 @zext_i77_to_i128(i77 %0) {
; CHECK-LABEL: zext_i77_to_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s1, %s1, (51)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i77 %0 to i128
  ret i128 %2
}
