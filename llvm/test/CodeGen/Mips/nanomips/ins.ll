; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @ins0(i32 %a) {
; CHECK-NOT: li $a1, 4294902015
; CHECK-NOT: and $a0, $a0, $a1
; CHECK: ins $a0, $zero, 8, 8
  %and = and i32 %a, 4294902015 ; 0xffff00ff
  ret i32 %and
}

define i32 @ins1(i32 %a, i32 %b) {
; CHECK: ins $a0, $a1, 8, 8
  %and1 = and i32 %a, 4294902015 ; 0xffff00ff
  %lshift = shl i32 %b, 8
  %and2 = and i32 %lshift, 65280 ; 0x0000ff00
  %or = or i32 %and1, %and2
  ret i32 %or
}

define i32 @ins2(i32 %a, i32 %b) {
; CHECK: ins $a0, $a1, 8, 8
  %and1 = and i32 %a, 4294902015 ; 0xffff00ff
  %lshift = shl i32 %b, 8
  %and2 = and i32 %lshift, 65280 ; 0x0000ff00
  %or = or i32 %and2, %and1
  ret i32 %or
}

define i32 @ins3(i32 %a, i32 %b) {
; CHECK: ins $a0, $a1, 8, 8
  %and1 = and i32 %a, 4294902015 ; 0xffff00ff
  %and2 = and i32 %b, 255        ; 0x000000ff
  %lshift = shl i32 %and2, 8
  %or = or i32 %and1, %lshift
  ret i32 %or
}

define i32 @ins4(i32 %a, i32 %b) {
; CHECK: ins $a0, $a1, 0, 16
  %and1 = and i32 %a, 4294901760 ; 0xffff0000
  %and2 = and i32 %b, 65535      ; 0x0000ffff
  %or = or i32 %and1, %and2
  ret i32 %or
}

define i32 @ins5(i32 %a, i32 %b) {
; CHECK: ins $a1, $a0, 0, 16
; CHECK: move $a0, $a1
; CHECK-NOT: ins $a0, $a1, 16, 16
  %and1 = and i32 %a, 65535      ; 0x0000ffff
  %and2 = and i32 %b, 4294901760 ; 0xffff0000
  %or = or i32 %and1, %and2
  ret i32 %or
}

define i32 @ins6(i32 %a, i32 %b) {
; CHECK: ins $a0, $a1, 20, 12
  %and1 = and i32 %a, 1048575    ; 0x000fffff
  %shift = shl i32 %b, 20
  %or = or i32 %and1, %shift
  ret i32 %or
}

define i32 @ins7(i32 %a, i8 %b) {
  ; CHECK: ins $a0, $a1, 16, 8
  %and1 = and i32 %a, 4278255615 ; 0xff00ffff
  %conv = zext i8 %b to i32
  %shift = shl i32 %conv, 16
  %or = or i32 %and1, %shift
  ret i32 %or
}

define i32 @ins8(i32 %a, i16 %b) {
  ; CHECK: ins $a0, $a1, 8, 16
  %and1 = and i32 %a, 4278190335 ; 0xff0000ff
  %conv = zext i16 %b to i32
  %shift = shl i32 %conv, 8
  %or = or i32 %and1, %shift
  ret i32 %or
}

define i32 @ins9(i32 %a, i8* %b) {
  ; CHECK: ins $a0, $a1, 16, 8
  %val = load i8, i8* %b
  %and1 = and i32 %a, 4278255615 ; 0xff00ffff
  %conv = zext i8 %val to i32
  %shift = shl i32 %conv, 16
  %or = or i32 %and1, %shift
  ret i32 %or
}

define i32 @ins10(i32 %a, i16* %b) {
  ; CHECK: ins $a0, $a1, 8, 16
  %val = load i16, i16* %b
  %and1 = and i32 %a, 4278190335 ; 0xff0000ff
  %conv = zext i16 %val to i32
  %shift = shl i32 %conv, 8
  %or = or i32 %and1, %shift
  ret i32 %or
}


