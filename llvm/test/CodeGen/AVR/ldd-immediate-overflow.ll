; RUN: llc -march=avr -filetype=asm -O1 < %s | FileCheck %s

define void @check60(ptr %1) {
; CHECK-LABEL: check60:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ldd r24, Z+60
; CHECK-NEXT: ldd r25, Z+61
; CHECK-NEXT: ldd r18, Z+62
; CHECK-NEXT: ldd r19, Z+63
; CHECK-NEXT: sts 3, r19
; CHECK-NEXT: sts 2, r18
; CHECK-NEXT: sts 1, r25
; CHECK-NEXT: sts 0, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i16 60
  %3 = load i32, ptr %2, align 1
  store i32 %3, ptr null, align 1
  ret void
}

define void @check61(ptr %1) {
; CHECK-LABEL: check61:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ldd r18, Z+61
; CHECK-NEXT: ldd r19, Z+62
; CHECK-NEXT: adiw r24, 63
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ld r24, Z
; CHECK-NEXT: ldd r25, Z+1
; CHECK-NEXT: sts 3, r25
; CHECK-NEXT: sts 2, r24
; CHECK-NEXT: sts 1, r19
; CHECK-NEXT: sts 0, r18
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i16 61
  %3 = load i32, ptr %2, align 1
  store i32 %3, ptr null, align 1
  ret void
}

define void @check62(ptr %1) {
; CHECK-LABEL: check62:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ldd r18, Z+62
; CHECK-NEXT: ldd r19, Z+63
; CHECK-NEXT: adiw r24, 62
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ldd r24, Z+2
; CHECK-NEXT: ldd r25, Z+3
; CHECK-NEXT: sts 3, r25
; CHECK-NEXT: sts 2, r24
; CHECK-NEXT: sts 1, r19
; CHECK-NEXT: sts 0, r18
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i16 62
  %3 = load i32, ptr %2, align 1
  store i32 %3, ptr null, align 1
  ret void
}

define void @check63(ptr %1) {
; CHECK-LABEL: check63:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: adiw r24, 63
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ld r24, Z
; CHECK-NEXT: ldd r25, Z+1
; CHECK-NEXT: ldd r18, Z+2
; CHECK-NEXT: ldd r19, Z+3
; CHECK-NEXT: sts 3, r19
; CHECK-NEXT: sts 2, r18
; CHECK-NEXT: sts 1, r25
; CHECK-NEXT: sts 0, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i16 63
  %3 = load i32, ptr %2, align 1
  store i32 %3, ptr null, align 1
  ret void
}

define void @check64(ptr %1) {
; CHECK-LABEL: check64:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: subi r24, 192
; CHECK-NEXT: sbci r25, 255
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ld r24, Z
; CHECK-NEXT: ldd r25, Z+1
; CHECK-NEXT: ldd r18, Z+2
; CHECK-NEXT: ldd r19, Z+3
; CHECK-NEXT: sts 3, r19
; CHECK-NEXT: sts 2, r18
; CHECK-NEXT: sts 1, r25
; CHECK-NEXT: sts 0, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i16 64
  %3 = load i32, ptr %2, align 1
  store i32 %3, ptr null, align 1
  ret void
}

define void @check65(ptr %1) {
; CHECK-LABEL: check65:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: subi r24, 191
; CHECK-NEXT: sbci r25, 255
; CHECK-NEXT: mov r30, r24
; CHECK-NEXT: mov r31, r25
; CHECK-NEXT: ld r24, Z
; CHECK-NEXT: ldd r25, Z+1
; CHECK-NEXT: ldd r18, Z+2
; CHECK-NEXT: ldd r19, Z+3
; CHECK-NEXT: sts 3, r19
; CHECK-NEXT: sts 2, r18
; CHECK-NEXT: sts 1, r25
; CHECK-NEXT: sts 0, r24
; CHECK-NEXT: ret

bb0:
  %2 = getelementptr i8, ptr %1, i16 65
  %3 = load i32, ptr %2, align 1
  store i32 %3, ptr null, align 1
  ret void
}
