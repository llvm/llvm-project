; RUN: llc < %s -march=arm64 -mcpu=cyclone | FileCheck %s

define i128 @shl(i128 %r, i128 %s) nounwind readnone {
; CHECK-LABEL: shl:
; CHECK: lsl  [[XREG_0:x[0-9]+]], x1, x2
; CHECK-NEXT: orr w[[XREG_1:[0-9]+]], wzr, #0x40
; CHECK-NEXT: sub [[XREG_2:x[0-9]+]], x[[XREG_1]], x2
; CHECK-NEXT: lsr  [[XREG_3:x[0-9]+]], x0, [[XREG_2]]
; CHECK-NEXT: orr [[XREG_6:x[0-9]+]], [[XREG_3]], [[XREG_0]]
; CHECK-NEXT: sub [[XREG_4:x[0-9]+]], x2, #64
; CHECK-NEXT: lsl  [[XREG_5:x[0-9]+]], x0, [[XREG_4]]
; CHECK-NEXT: cmp   [[XREG_4]], #0
; CHECK-NEXT: csel  x1, [[XREG_5]], [[XREG_6]], ge
; CHECK-NEXT: lsl  [[SMALLSHIFT_LO:x[0-9]+]], x0, x2
; CHECK-NEXT: csel  x0, xzr, [[SMALLSHIFT_LO]], ge
; CHECK-NEXT: ret

  %shl = shl i128 %r, %s
  ret i128 %shl
}

define i128 @ashr(i128 %r, i128 %s) nounwind readnone {
; CHECK-LABEL: ashr:
; CHECK: lsr  [[XREG_0:x[0-9]+]], x0, x2
; CHECK-NEXT: orr w[[XREG_1:[0-9]+]], wzr, #0x40
; CHECK-NEXT: sub [[XREG_2:x[0-9]+]], x[[XREG_1]], x2
; CHECK-NEXT: lsl  [[XREG_3:x[0-9]+]], x1, [[XREG_2]]
; CHECK-NEXT: orr [[XREG_4:x[0-9]+]], [[XREG_0]], [[XREG_3]]
; CHECK-NEXT: sub [[XREG_5:x[0-9]+]], x2, #64
; CHECK-NEXT: asr  [[XREG_6:x[0-9]+]], x1, [[XREG_5]]
; CHECK-NEXT: cmp   [[XREG_5]], #0
; CHECK-NEXT: csel  x0, [[XREG_6]], [[XREG_4]], ge
; CHECK-NEXT: asr  [[SMALLSHIFT_HI:x[0-9]+]], x1, x2
; CHECK-NEXT: asr [[BIGSHIFT_HI:x[0-9]+]], x1, #63
; CHECK-NEXT: csel x1, [[BIGSHIFT_HI]], [[SMALLSHIFT_HI]], ge
; CHECK-NEXT: ret

  %shr = ashr i128 %r, %s
  ret i128 %shr
}

define i128 @lshr(i128 %r, i128 %s) nounwind readnone {
; CHECK-LABEL: lshr:
; CHECK: lsr  [[XREG_0:x[0-9]+]], x0, x2
; CHECK-NEXT: orr w[[XREG_1:[0-9]+]], wzr, #0x40
; CHECK-NEXT: sub [[XREG_2:x[0-9]+]], x[[XREG_1]], x2
; CHECK-NEXT: lsl  [[XREG_3:x[0-9]+]], x1, [[XREG_2]]
; CHECK-NEXT: orr [[XREG_4:x[0-9]+]], [[XREG_0]], [[XREG_3]]
; CHECK-NEXT: sub [[XREG_5:x[0-9]+]], x2, #64
; CHECK-NEXT: lsr  [[XREG_6:x[0-9]+]], x1, [[XREG_5]]
; CHECK-NEXT: cmp   [[XREG_5]], #0
; CHECK-NEXT: csel  x0, [[XREG_6]], [[XREG_4]], ge
; CHECK-NEXT: lsr  [[SMALLSHIFT_HI:x[0-9]+]], x1, x2
; CHECK-NEXT: csel x1, xzr, [[SMALLSHIFT_HI]], ge
; CHECK-NEXT: ret

  %shr = lshr i128 %r, %s
  ret i128 %shr
}
