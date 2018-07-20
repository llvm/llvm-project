; RUN: llc < %s -mtriple=arm64-eabi -mcpu=cyclone | FileCheck %s

define i128 @shl(i128 %r, i128 %s) nounwind readnone {
; CHECK-LABEL: shl:
; CHECK: neg [[REV_SHIFT:x[0-9]+]], x2
; CHECK: lsr  [[LO_FOR_HI_NORMAL:x[0-9]+]], x0, [[REV_SHIFT]]
; CHECK: cmp x2, #0
; CHECK: csel [[LO_FOR_HI:x[0-9]+]], xzr, [[LO_FOR_HI_NORMAL]], eq
; CHECK: lsl  [[HI_FOR_HI:x[0-9]+]], x1, x2
; CHECK: orr [[HI_NORMAL:x[0-9]+]], [[LO_FOR_HI]], [[HI_FOR_HI]]
; CHECK: lsl  [[HI_BIG_SHIFT:x[0-9]+]], x0, x2
; CHECK: sub [[EXTRA_SHIFT:x[0-9]+]], x2, #64
; CHECK: cmp   [[EXTRA_SHIFT]], #0
; CHECK: csel  x1, [[HI_BIG_SHIFT]], [[HI_NORMAL]], ge
; CHECK: csel  x0, xzr, [[HI_BIG_SHIFT]], ge
; CHECK: ret

  %shl = shl i128 %r, %s
  ret i128 %shl
}

define i128 @ashr(i128 %r, i128 %s) nounwind readnone {
; CHECK-LABEL: ashr:
; CHECK: neg [[REV_SHIFT:x[0-9]+]], x2
; CHECK: lsl  [[HI_FOR_LO_NORMAL:x[0-9]+]], x1, [[REV_SHIFT]]
; CHECK: cmp x2, #0
; CHECK: csel [[HI_FOR_LO:x[0-9]+]], xzr, [[HI_FOR_LO_NORMAL]], eq
; CHECK: lsr  [[LO_FOR_LO:x[0-9]+]], x0, x2
; CHECK: orr [[LO_NORMAL:x[0-9]+]], [[LO_FOR_LO]], [[HI_FOR_LO]]
; CHECK: asr  [[LO_BIG_SHIFT:x[0-9]+]], x1, x2
; CHECK: sub [[EXTRA_SHIFT:x[0-9]+]], x2, #64
; CHECK: cmp   [[EXTRA_SHIFT]], #0
; CHECK: csel  x0, [[LO_BIG_SHIFT]], [[LO_NORMAL]], ge
; CHECK: asr [[BIGSHIFT_HI:x[0-9]+]], x1, #63
; CHECK: csel x1, [[BIGSHIFT_HI]], [[LO_BIG_SHIFT]], ge
; CHECK: ret

  %shr = ashr i128 %r, %s
  ret i128 %shr
}

define i128 @lshr(i128 %r, i128 %s) nounwind readnone {
; CHECK-LABEL: lshr:
; CHECK: neg [[REV_SHIFT:x[0-9]+]], x2
; CHECK: lsl  [[HI_FOR_LO_NORMAL:x[0-9]+]], x1, [[REV_SHIFT]]
; CHECK: cmp x2, #0
; CHECK: csel [[HI_FOR_LO:x[0-9]+]], xzr, [[HI_FOR_LO_NORMAL]], eq
; CHECK: lsr  [[LO_FOR_LO:x[0-9]+]], x0, x2
; CHECK: orr [[LO_NORMAL:x[0-9]+]], [[LO_FOR_LO]], [[HI_FOR_LO]]
; CHECK: lsr  [[LO_BIG_SHIFT:x[0-9]+]], x1, x2
; CHECK: cmp   [[EXTRA_SHIFT]], #0
; CHECK: csel  x0, [[LO_BIG_SHIFT]], [[LO_NORMAL]], ge
; CHECK: csel x1, xzr, [[LO_BIG_SHIFT]], ge
; CHECK: ret

  %shr = lshr i128 %r, %s
  ret i128 %shr
}
