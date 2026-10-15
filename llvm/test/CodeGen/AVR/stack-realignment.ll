; RUN: llc -mtriple=avr -mcpu=atmega328 -O1 -verify-machineinstrs < %s | FileCheck %s

declare void @use(ptr %x);

; Case: One 2-byte variable with alignment of 1.
;
; This shouldn't activate stack realignment - this function exists so that it's
; easy to see the difference when stack realignment gets actually activated (see
; the next test case).
define i16 @no_alignment() {
; CHECK-LABEL: no_alignment:
; CHECK-NEXT: ; %bb.0:
;
;; prologue
; CHECK-NEXT: push r28
; CHECK-NEXT: push r29
; CHECK-NEXT: in r28, 61
; CHECK-NEXT: in r29, 62
; CHECK-NEXT: sbiw r28, 2
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
;
;; call void @use(ptr %1)
; CHECK-NEXT: movw r24, r28
; CHECK-NEXT: adiw r24, 1
; CHECK-NEXT: call use
;
;; %2 = load i16, ptr %1, align 1
; CHECK-NEXT: ldd r24, Y+1
; CHECK-NEXT: ldd r25, Y+2
;
;; epilogue
; CHECK-NEXT: adiw r28, 2
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; CHECK-NEXT: pop r29
; CHECK-NEXT: pop r28
; CHECK-NEXT: ret

  %1 = alloca i16, align 1
  call void @use(ptr %1)
  %2 = load i16, ptr %1, align 1

  ret i16 %2
}

; Case: One 2-byte variable with alignment of 16.
;
; This reserves 2 + 16 - 1 = 17 bytes of stack space and allocates an extra
; aligned stack pointer (in this case into r24:r25).
define i16 @some_alignment() {
; CHECK-LABEL: some_alignment:
; CHECK-NEXT: %bb.0:
;
;; prologue
; CHECK-NEXT: push r16
; CHECK-NEXT: push r17
; CHECK-NEXT: push r28
; CHECK-NEXT: push r29
; CHECK-NEXT: in r28, 61
; CHECK-NEXT: in r29, 62
; CHECK-NEXT: sbiw r28, 17
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
;
;; SP allocation
; CHECK-NEXT: movw r24, r28
; CHECK-NEXT: adiw r24, 16
; CHECK-NEXT: andi r24, 240
;
;; call void @use (ptr %1)
; CHECK-NEXT: movw r16, r24
; CHECK-NEXT: movw r24, r16
; CHECK-NEXT: call use
;
;; %2 = load i16, ptr %1, align 16
; CHECK-NEXT: movw r30, r16
; CHECK-NEXT: ld r24, Z
; CHECK-NEXT: ldd r25, Z+1
;
;; epilogue
; CHECK-NEXT: adiw r28, 17
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; CHECK-NEXT: pop r29
; CHECK-NEXT: pop r28
; CHECK-NEXT: pop r17
; CHECK-NEXT: pop r16
; CHECK-NEXT: ret

  %1 = alloca i16, align 16
  call void @use(ptr %1)
  %2 = load i16, ptr %1, align 16

  ret i16 %2
}

; Case: One variable with no alignment, another variable with alignment of 16.
;
; This creates two separate stack spaces - an unaligned one (r28:r29) and an
; aligned one (r16:r17).
define i16 @mixed_alignment() {
; CHECK-LABEL: mixed_alignment:
; CHECK-NEXT: ; %bb.0:
;
;; prologue
; CHECK-NEXT: push r16
; CHECK-NEXT: push r17
; CHECK-NEXT: push r28
; CHECK-NEXT: push r29
; CHECK-NEXT: in r28, 61
; CHECK-NEXT: in r29, 62
; CHECK-NEXT: sbiw r28, 19
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
;
;; SP allocation
; CHECK-NEXT: movw r16, r28
; CHECK-NEXT: subi r16, 238
; CHECK-NEXT: sbci r17, 255
; CHECK-NEXT: andi r16, 240
;
;; call void @use(ptr %1)
; CHECK-NEXT: movw r24, r28
; CHECK-NEXT: adiw r24, 1
; CHECK-NEXT: call use
;
;; call void @use(ptr %2)
; CHECK-NEXT: movw r24, r16
; CHECK-NEXT: call use
;
;; %4 = load i16, ptr %2, align 16
; CHECK-NEXT: movw r30, r16
; CHECK-NEXT: ld r18, Z
; CHECK-NEXT: ldd r19, Z+1
;
;; %3 = load i16, ptr %1, align 1
; CHECK-NEXT: ldd r24, Y+1
; CHECK-NEXT: ldd r25, Y+2
;
;; %5 = or i16 %3, %4
; CHECK-NEXT: or r24, r18
; CHECK-NEXT: or r25, r19
;
;; epilogue
; CHECK-NEXT: adiw r28, 19
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; CHECK-NEXT: pop r29
; CHECK-NEXT: pop r28
; CHECK-NEXT: pop r17
; CHECK-NEXT: pop r16
; CHECK-NEXT: ret

  %1 = alloca i16, align 1
  %2 = alloca i16, align 16

  call void @use(ptr %1)
  call void @use(ptr %2)

  %3 = load i16, ptr %1, align 1
  %4 = load i16, ptr %2, align 16
  %5 = or i16 %3, %4

  ret i16 %5
}

; Case: getelementptr referring to an aligned variable.
define i16 @gep() {
; CHECK-LABEL: gep:
; CHECK-NEXT: ; %bb.0:
;
;; prologue
; CHECK-NEXT: push r16
; CHECK-NEXT: push r17
; CHECK-NEXT: push r28
; CHECK-NEXT: push r29
; CHECK-NEXT: in r28, 61
; CHECK-NEXT: in r29, 62
; CHECK-NEXT: sbiw r28, 23
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
;
;; SP allocation
; CHECK-NEXT: movw r16, r28
; CHECK-NEXT: subi r16, 248
; CHECK-NEXT: sbci r17, 255
; CHECK-NEXT: andi r16, 248
;
;; %2 = getelementptr inbounds nuw i8, ptr %1, i16 8
; CHECK-NEXT: movw r24, r16
; CHECK-NEXT: adiw r24, 8
;
;; call void @use(ptr %2)
; CHECK-NEXT: call use
;
;; %3 = getelementptr inbounds nuw i8, ptr %1, i16 8
; CHECK-NEXT: movw r24, r16
; CHECK-NEXT: adiw r24, 8
;
;; %4 = load i16, ptr %3, align 4
;; ret i16 %4
; CHECK-NEXT: movw r30, r24
; CHECK-NEXT: ld r24, Z
; CHECK-NEXT: ldd r25, Z+1
; CHECK-NEXT: adiw r28, 23
;
;; epilogue
; CHECK-NEXT: in  r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; CHECK-NEXT: pop r29
; CHECK-NEXT: pop r28
; CHECK-NEXT: pop r17
; CHECK-NEXT: pop r16
; CHECK-NEXT: ret

  %1 = alloca [8 x i16], align 8
  %2 = getelementptr inbounds nuw i8, ptr %1, i16 8

  call void @use(ptr %2)

  %3 = getelementptr inbounds nuw i8, ptr %1, i16 8
  %4 = load i16, ptr %3, align 4

  ret i16 %4
}
