; Test that the case of (64 - shift) used by a shift/rotate instruction is
; implemented with an lcr. This should also work for any multiple of 64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define i64 @f1(i64 %in, i64 %sh) {
; CHECK-LABEL: f1:
; CHECK: lcr %r1, %r3
; CHECK: sllg %r2, %r2, 0(%r1)
  %sub = sub i64 64, %sh
  %shl = shl i64 %in, %sub
  ret i64 %shl
}

define i64 @f2(i64 %in, i64 %sh) {
; CHECK-LABEL: f2:
; CHECK: lcr %r1, %r3
; CHECK: srag %r2, %r2, 0(%r1)
  %sub = sub i64 64, %sh
  %shl = ashr i64 %in, %sub
  ret i64 %shl
}

define i64 @f3(i64 %in, i64 %sh) {
; CHECK-LABEL: f3:
; CHECK: lcr %r1, %r3
; CHECK: srlg %r2, %r2, 0(%r1)
  %sub = sub i64 64, %sh
  %shl = lshr i64 %in, %sub
  ret i64 %shl
}

define i64 @f4(i64 %in, i64 %sh) {
; CHECK-LABEL: f4:
; CHECK: lcr %r1, %r3
; CHECK: rllg %r2, %r2, 0(%r1)
  %shr = lshr i64 %in, %sh
  %sub = sub i64 64, %sh
  %shl = shl i64 %in, %sub
  %or = or i64 %shl, %shr
  ret i64 %or
}

define i64 @f5(i64 %in, i64 %sh) {
; CHECK-LABEL: f5:
; CHECK: lcr %r1, %r3
; CHECK: sllg %r2, %r2, 0(%r1)
  %sub = sub i64 128, %sh
  %shl = shl i64 %in, %sub
  ret i64 %shl
}

define i64 @f6(i64 %in, i64 %sh) {
; CHECK-LABEL: f6:
; CHECK: lcr %r1, %r3
; CHECK: srag %r2, %r2, 0(%r1)
  %sub = sub i64 256, %sh
  %shl = ashr i64 %in, %sub
  ret i64 %shl
}

define i64 @f7(i64 %in, i64 %sh) {
; CHECK-LABEL: f7:
; CHECK: lcr %r1, %r3
; CHECK: srlg %r2, %r2, 0(%r1)
  %sub = sub i64 512, %sh
  %shl = lshr i64 %in, %sub
  ret i64 %shl
}

define i64 @f8(i64 %in, i64 %sh) {
; CHECK-LABEL: f8:
; CHECK: lcr %r1, %r3
; CHECK: srlg %r0, %r2, 0(%r3)
; CHECK: sllg %r2, %r2, 0(%r1)
; CHECK: ogr %r2, %r0
  %shr = lshr i64 %in, %sh
  %sub = sub i64 1024, %sh
  %shl = shl i64 %in, %sub
  %or = or i64 %shl, %shr
  ret i64 %or
}
