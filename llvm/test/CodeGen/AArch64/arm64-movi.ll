; RUN: llc < %s -march=arm64 | FileCheck %s

;==--------------------------------------------------------------------------==
; Tests for MOV-immediate implemented with ORR-immediate.
;==--------------------------------------------------------------------------==

; 64-bit immed with 32-bit pattern size, rotated by 0.
define i64 @test64_32_rot0() nounwind {
; CHECK-LABEL: test64_32_rot0:
; CHECK: mov x0, #30064771079
  ret i64 30064771079
}

; 64-bit immed with 32-bit pattern size, rotated by 2.
define i64 @test64_32_rot2() nounwind {
; CHECK-LABEL: test64_32_rot2:
; CHECK: mov x0, #-4611686002321260541
  ret i64 13835058071388291075
}

; 64-bit immed with 4-bit pattern size, rotated by 3.
define i64 @test64_4_rot3() nounwind {
; CHECK-LABEL: test64_4_rot3:
; CHECK: mov  x0, #-1229782938247303442
  ret i64 17216961135462248174
}

; 32-bit immed with 32-bit pattern size, rotated by 16.
define i32 @test32_32_rot16() nounwind {
; CHECK-LABEL: test32_32_rot16:
; CHECK: orr w0, wzr, #0xff0000
  ret i32 16711680
}

; 32-bit immed with 2-bit pattern size, rotated by 1.
define i32 @test32_2_rot1() nounwind {
; CHECK-LABEL: test32_2_rot1:
; CHECK: mov w0, #-1431655766
  ret i32 2863311530
}

;==--------------------------------------------------------------------------==
; Tests for MOVZ with MOVK.
;==--------------------------------------------------------------------------==

define i32 @movz() nounwind {
; CHECK-LABEL: movz:
; CHECK: mov w0, #5
  ret i32 5
}

define i64 @movz_3movk() nounwind {
; CHECK-LABEL: movz_3movk:
; CHECK:      mov x0, #1407374883553280
; CHECK-NEXT: movk x0, #4660, lsl #32
; CHECK-NEXT: movk x0, #43981, lsl #16
; CHECK-NEXT: movk x0, #22136
  ret i64 1427392313513592
}

define i64 @movz_movk_skip1() nounwind {
; CHECK-LABEL: movz_movk_skip1:
; CHECK:      mov x0, #21474836480
; CHECK-NEXT: movk x0, #17185, lsl #16
  ret i64 22601072640
}

define i64 @movz_skip1_movk() nounwind {
; CHECK-LABEL: movz_skip1_movk:
; CHECK:      mov x0, #147695335374848
; CHECK-NEXT: movk x0, #4660
  ret i64 147695335379508
}

;==--------------------------------------------------------------------------==
; Tests for MOVN with MOVK.
;==--------------------------------------------------------------------------==

define i64 @movn() nounwind {
; CHECK-LABEL: movn:
; CHECK: mov x0, #-42
  ret i64 -42
}

define i64 @movn_skip1_movk() nounwind {
; CHECK-LABEL: movn_skip1_movk:
; CHECK:      mov x0, #-176093659137
; CHECK-NEXT: movk x0, #4660
  ret i64 -176093720012
}

;==--------------------------------------------------------------------------==
; Tests for ORR with MOVK.
;==--------------------------------------------------------------------------==
; rdar://14987673

define i64 @orr_movk1() nounwind {
; CHECK-LABEL: orr_movk1:
; CHECK: mov x0, #72056494543077120
; CHECK: movk x0, #57005, lsl #16
  ret i64 72056498262245120
}

define i64 @orr_movk2() nounwind {
; CHECK-LABEL: orr_movk2:
; CHECK: mov x0, #72056494543077120
; CHECK: movk x0, #57005, lsl #48
  ret i64 -2400982650836746496
}

define i64 @orr_movk3() nounwind {
; CHECK-LABEL: orr_movk3:
; CHECK: mov x0, #72056494543077120
; CHECK: movk x0, #57005, lsl #32
  ret i64 72020953688702720
}

define i64 @orr_movk4() nounwind {
; CHECK-LABEL: orr_movk4:
; CHECK: mov x0, #72056494543077120
; CHECK: movk x0, #57005
  ret i64 72056494543068845
}

; rdar://14987618
define i64 @orr_movk5() nounwind {
; CHECK-LABEL: orr_movk5:
; CHECK: mov x0, #-71777214294589696
; CHECK: movk x0, #57005, lsl #16
  ret i64 -71777214836900096
}

define i64 @orr_movk6() nounwind {
; CHECK-LABEL: orr_movk6:
; CHECK: mov x0, #-71777214294589696
; CHECK: movk x0, #57005, lsl #16
; CHECK: movk x0, #57005, lsl #48
  ret i64 -2400982647117578496
}

define i64 @orr_movk7() nounwind {
; CHECK-LABEL: orr_movk7:
; CHECK: mov x0, #-71777214294589696
; CHECK: movk x0, #57005, lsl #48
  ret i64 -2400982646575268096
}

define i64 @orr_movk8() nounwind {
; CHECK-LABEL: orr_movk8:
; CHECK: mov x0, #-71777214294589696
; CHECK: movk x0, #57005
; CHECK: movk x0, #57005, lsl #48
  ret i64 -2400982646575276371
}

; rdar://14987715
define i64 @orr_movk9() nounwind {
; CHECK-LABEL: orr_movk9:
; CHECK: mov x0, #1152921435887370240
; CHECK: movk x0, #65280
; CHECK: movk x0, #57005, lsl #16
  ret i64 1152921439623315200
}

define i64 @orr_movk10() nounwind {
; CHECK-LABEL: orr_movk10:
; CHECK: mov x0, #1152921504606846720
; CHECK: movk x0, #57005, lsl #16
  ret i64 1152921504047824640
}

define i64 @orr_movk11() nounwind {
; CHECK-LABEL: orr_movk11:
; CHECK: mov x0, #-4503599627370241
; CHECK: movk x0, #57005, lsl #16
; CHECK: movk x0, #65535, lsl #32
  ret i64 -4222125209747201
}

define i64 @orr_movk12() nounwind {
; CHECK-LABEL: orr_movk12:
; CHECK: mov x0, #-4503599627370241
; CHECK: movk x0, #57005, lsl #32
  ret i64 -4258765016661761
}

define i64 @orr_movk13() nounwind {
; CHECK-LABEL: orr_movk13:
; CHECK: mov x0, #17592169267200
; CHECK: movk x0, #57005
; CHECK: movk x0, #57005, lsl #48
  ret i64 -2401245434149282131
}

; rdar://13944082
define i64 @g() nounwind {
; CHECK-LABEL: g:
; CHECK: mov x0, #-281474976710656
; CHECK: movk x0, #2
entry:
  ret i64 -281474976710654
}
