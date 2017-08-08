; RUN: llc < %s -march=avr | FileCheck %s

; Bit rotation tests.

; CHECK-LABEL: rol8:
define i8 @rol8(i8 %val, i8 %amt) {
  ; CHECK:      andi r22, 7

  ; CHECK-NEXT: cpi r22, 0
  ; CHECK-NEXT: breq LBB0_2

; CHECK-NEXT: LBB0_1:
  ; CHECK-NEXT: rol r24
  ; CHECK-NEXT: subi r22, 1
  ; CHECK-NEXT: brne LBB0_1

; CHECK-NEXT:LBB0_2:
  ; CHECK-NEXT: ret
  %mod = urem i8 %amt, 8

  %inv = sub i8 8, %mod
  %parta = shl i8 %val, %mod
  %partb = lshr i8 %val, %inv

  %rotl = or i8 %parta, %partb

  ret i8 %rotl
}


; CHECK-LABEL: ror8:
define i8 @ror8(i8 %val, i8 %amt) {
  ; CHECK:      andi r22, 7

  ; CHECK-NEXT: cpi r22, 0
  ; CHECK-NEXT: breq LBB1_2

; CHECK-NEXT: LBB1_1:
  ; CHECK-NEXT: ror r24
  ; CHECK-NEXT: subi r22, 1
  ; CHECK-NEXT: brne LBB1_1

; CHECK-NEXT:LBB1_2:
  ; CHECK-NEXT: ret
  %mod = urem i8 %amt, 8

  %inv = sub i8 8, %mod
  %parta = lshr i8 %val, %mod
  %partb = shl i8 %val, %inv

  %rotr = or i8 %parta, %partb

  ret i8 %rotr
}

