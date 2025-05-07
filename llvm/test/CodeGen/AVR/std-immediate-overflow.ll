; RUN: llc -mtriple=avr -filetype=asm -O1 < %s | FileCheck %s

define void @check60(ptr %1) {
; CHECK-LABEL: check60:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: ldi r18, 0
; CHECK-NEXT: ldi r19, 0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+63, r19
; CHECK-NEXT: std Z+62, r18
; CHECK-NEXT: ldi r24, 210
; CHECK-NEXT: ldi r25, 4
; CHECK-NEXT: std Z+61, r25
; CHECK-NEXT: std Z+60, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i8 60
  store i32 1234, ptr %2
  ret void
}

define void @check61(ptr %1) {
; CHECK-LABEL: check61:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: ldi r18, 210
; CHECK-NEXT: ldi r19, 4
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+62, r19
; CHECK-NEXT: std Z+61, r18
; CHECK-NEXT: adiw r24, 63
; CHECK-NEXT: ldi r18, 0
; CHECK-NEXT: ldi r19, 0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+1, r19
; CHECK-NEXT: st Z, r18

bb0:
  %2 = getelementptr i8, ptr %1, i8 61
  store i32 1234, ptr %2
  ret void
}

define void @check62(ptr %1) {
; CHECK-LABEL: check62:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: ldi r18, 210
; CHECK-NEXT: ldi r19, 4
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+63, r19
; CHECK-NEXT: std Z+62, r18
; CHECK-NEXT: adiw r24, 62
; CHECK-NEXT: ldi r18, 0
; CHECK-NEXT: ldi r19, 0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+3, r19
; CHECK-NEXT: std Z+2, r18
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i8 62
  store i32 1234, ptr %2
  ret void
}

define void @check63(ptr %1) {
; CHECK-LABEL: check63:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: adiw r24, 63
; CHECK-NEXT: ldi r18, 0
; CHECK-NEXT: ldi r19, 0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+3, r19
; CHECK-NEXT: std Z+2, r18
; CHECK-NEXT: ldi r24, 210
; CHECK-NEXT: ldi r25, 4
; CHECK-NEXT: std Z+1, r25
; CHECK-NEXT: st Z, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i8 63
  store i32 1234, ptr %2
  ret void
}

define void @check64(ptr %1) {
; CHECK-LABEL: check64:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: subi r24, 192
; CHECK-NEXT: sbci r25, 255
; CHECK-NEXT: ldi r18, 0
; CHECK-NEXT: ldi r19, 0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+3, r19
; CHECK-NEXT: std Z+2, r18
; CHECK-NEXT: ldi r24, 210
; CHECK-NEXT: ldi r25, 4
; CHECK-NEXT: std Z+1, r25
; CHECK-NEXT: st Z, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i8 64
  store i32 1234, ptr %2
  ret void
}

define void @check65(ptr %1) {
; CHECK-LABEL: check65:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: subi r24, 191
; CHECK-NEXT: sbci r25, 255
; CHECK-NEXT: ldi r18, 0
; CHECK-NEXT: ldi r19, 0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: std Z+3, r19
; CHECK-NEXT: std Z+2, r18
; CHECK-NEXT: ldi r24, 210
; CHECK-NEXT: ldi r25, 4
; CHECK-NEXT: std Z+1, r25
; CHECK-NEXT: st Z, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i8 65
  store i32 1234, ptr %2
  ret void
}
