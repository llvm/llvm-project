; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; Checks that atomic fences are simply removed from IR.
; AVR is always singlethreaded so fences do nothing.

; CHECK-LABEL: atomic_fence8
; CHECK:      ; %bb.0:
; CHECK-NEXT:   ret
define void @atomic_fence8() {
  fence acquire
  ret void
}

