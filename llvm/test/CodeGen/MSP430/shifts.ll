; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-n8:16"
target triple = "msp430-elf"

define zeroext i8 @lshr8(i8 zeroext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK: lshr8:
; CHECK: rrc.b
  %shr = lshr i8 %a, %cnt
  ret i8 %shr
}

define signext i8 @ashr8(i8 signext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK: ashr8:
; CHECK: rra.b
  %shr = ashr i8 %a, %cnt
  ret i8 %shr
}

define zeroext i8 @shl8(i8 zeroext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK: shl8
; CHECK: rla.b
  %shl = shl i8 %a, %cnt
  ret i8 %shl
}

define zeroext i16 @lshr16(i16 zeroext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK: lshr16:
; CHECK: rrc.w
  %shr = lshr i16 %a, %cnt
  ret i16 %shr
}

define signext i16 @ashr16(i16 signext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK: ashr16:
; CHECK: rra.w
  %shr = ashr i16 %a, %cnt
  ret i16 %shr
}

define zeroext i16 @shl16(i16 zeroext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK: shl16:
; CHECK: rla.w
  %shl = shl i16 %a, %cnt
  ret i16 %shl
}
