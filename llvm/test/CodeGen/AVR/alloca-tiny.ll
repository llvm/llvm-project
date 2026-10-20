; Test that stores to stack slots work on reduced tiny cores, which don't
; support the `std` instruction. The store pseudo must materialize the frame
; index as SP and be expanded using pointer adjustments instead of `std`.

; RUN: llc -mtriple=avr -mcpu=attiny10 -o - %s | FileCheck %s --check-prefix=TINY
; RUN: llc -mtriple=avr -mcpu=attiny2313 -o - %s | FileCheck %s --check-prefix=FULL

define void @bytealloca() {
entry:
  %a = alloca i8, align 1
  store i8 0, ptr %a, align 1
  ret void
}

; TINY: bytealloca:
; TINY-NEXT: %bb.0:
; TINY-NEXT: push r28
; TINY-NEXT: push r29
; TINY-NEXT: in r28, 61
; TINY-NEXT: clr r29
; TINY-NEXT: subi r28, 1
; TINY-NEXT: sbci r29, 0
; TINY-NEXT: out 61, r28
; TINY-NEXT: in r16, 63
; TINY-NEXT: subi r28, 255
; TINY-NEXT: sbci r29, 255
; TINY-NEXT: st Y, r17
; TINY-NEXT: subi r28, 1
; TINY-NEXT: sbci r29, 0
; TINY-NEXT: out 63, r16
; TINY-NEXT: subi r28, 255
; TINY-NEXT: sbci r29, 255
; TINY-NEXT: out 61, r28
; TINY-NEXT: pop r29
; TINY-NEXT: pop r28
; TINY-NEXT: ret

; FULL: bytealloca:
; FULL-NEXT: %bb.0:
; FULL-NEXT: push r28
; FULL-NEXT: push r29
; FULL-NEXT: in r28, 61
; FULL-NEXT: clr r29
; FULL-NEXT: sbiw r28, 1
; FULL-NEXT: out 61, r28
; FULL-NEXT: std Y+1, r1
; FULL-NEXT: adiw r28, 1
; FULL-NEXT: out 61, r28
; FULL-NEXT: pop r29
; FULL-NEXT: pop r28
; FULL-NEXT: ret

define i16 @wordalloca() {
entry:
  %a = alloca i16, align 1
  store i16 7, ptr %a, align 1
  ret i16 7
}

; TINY: wordalloca:
; TINY-NEXT: %bb.0:
; TINY-NEXT: push r28
; TINY-NEXT: push r29
; TINY-NEXT: in r28, 61
; TINY-NEXT: clr r29
; TINY-NEXT: subi r28, 2
; TINY-NEXT: sbci r29, 0
; TINY-NEXT: out 61, r28
; TINY-NEXT: ldi r24, 7
; TINY-NEXT: ldi r25, 0
; TINY-NEXT: in r16, 63
; TINY-NEXT: subi r28, 255
; TINY-NEXT: sbci r29, 255
; TINY-NEXT: st Y+, r24
; TINY-NEXT: st Y+, r25
; TINY-NEXT: subi r28, 2
; TINY-NEXT: sbci r29, 0
; TINY-NEXT: subi r28, 1
; TINY-NEXT: sbci r29, 0
; TINY-NEXT: out 63, r16
; TINY-NEXT: subi r28, 254
; TINY-NEXT: sbci r29, 255
; TINY-NEXT: out 61, r28
; TINY-NEXT: pop r29
; TINY-NEXT: pop r28
; TINY-NEXT: ret

; FULL: wordalloca:
; FULL-NEXT: %bb.0:
; FULL-NEXT: push r28
; FULL-NEXT: push r29
; FULL-NEXT: in r28, 61
; FULL-NEXT: clr r29
; FULL-NEXT: sbiw r28, 2
; FULL-NEXT: out 61, r28
; FULL-NEXT: ldi r24, 7
; FULL-NEXT: ldi r25, 0
; FULL-NEXT: std Y+2, r25
; FULL-NEXT: std Y+1, r24
; FULL-NEXT: adiw r28, 2
; FULL-NEXT: out 61, r28
; FULL-NEXT: pop r29
; FULL-NEXT: pop r28
; FULL-NEXT: ret
