; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; CHECK-LABEL: atomic_store16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: std [[RD:(X|Y|Z)]]+1, [[RR:r[0-9]+]]
; CHECK-NEXT: st [[RD:(X|Y|Z)]], [[RR:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define void @atomic_store16(ptr %foo) {
  store atomic i16 1, ptr %foo unordered, align 2
  ret void
}

; CHECK-LABEL: monotonic
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: std Z+1, r25
; CHECK-NEXT: st Z, r24
; CHECK-NEXT: out 63, r0
define void @monotonic(i16) {
entry-block:
  store atomic i16 %0, ptr undef monotonic, align 2
  ret void
}

