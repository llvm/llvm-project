; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-n8:16"
target triple = "msp430-elf"

define zeroext i8 @lshr8(i8 zeroext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: lshr8:
; CHECK: clrc
; CHECK: rrc.b
  %shr = lshr i8 %a, %cnt
  ret i8 %shr
}

define signext i8 @ashr8(i8 signext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: ashr8:
; CHECK: rra.b
  %shr = ashr i8 %a, %cnt
  ret i8 %shr
}

define zeroext i8 @shl8(i8 zeroext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK: shl8
; CHECK: add.b
  %shl = shl i8 %a, %cnt
  ret i8 %shl
}

define zeroext i16 @lshr16(i16 zeroext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: lshr16:
; CHECK: clrc
; CHECK: rrc
  %shr = lshr i16 %a, %cnt
  ret i16 %shr
}

define signext i16 @ashr16(i16 signext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: ashr16:
; CHECK: rra
  %shr = ashr i16 %a, %cnt
  ret i16 %shr
}

define zeroext i16 @shl16(i16 zeroext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: shl16:
; CHECK: add
  %shl = shl i16 %a, %cnt
  ret i16 %shl
}

define i16 @ashr10_i16(i16 %a) #0 {
entry:
; CHECK-LABEL: ashr10_i16:
; CHECK:      swpb	r12
; CHECK-NEXT: sxt	r12
; CHECK-NEXT: rra	r12
; CHECK-NEXT: rra	r12
  %shr = ashr i16 %a, 10
  ret i16 %shr
}

define i16 @lshr10_i16(i16 %a) #0 {
entry:
; CHECK-LABEL: lshr10_i16:
; CHECK:      swpb	r12
; CHECK-NEXT: mov.b	r12, r12
; CHECK-NEXT: clrc
; CHECK-NEXT: rrc	r12
; CHECK-NEXT: rra	r12
  %shr = lshr i16 %a, 10
  ret i16 %shr
}
