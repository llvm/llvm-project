; RUN: llc -mattr=avr6 < %s -mtriple=avr | FileCheck %s

; CHECK-LABEL: atomic_store8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: st [[RD:(X|Y|Z)]], [[RR:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define void @atomic_store8(ptr %foo) {
  store atomic i8 1, ptr %foo unordered, align 1
  ret void
}

; CHECK-LABEL: atomic_store16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: std [[RD:(X|Y|Z)]]+1, [[RR:r[0-9]+]]
; CHECK-NEXT: st [[RD]], [[RR:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define void @atomic_store16(ptr %foo) {
  store atomic i16 1, ptr %foo unordered, align 2
  ret void
}

; CHECK-LABEL: atomic_store32
; CHECK: call __sync_lock_test_and_set_4
define void @atomic_store32(ptr %foo) {
  store atomic i32 1, ptr %foo unordered, align 4
  ret void
}

; CHECK-LABEL: atomic_store64
; CHECK: call __sync_lock_test_and_set_8
define void @atomic_store64(ptr %foo) {
  store atomic i64 1, ptr %foo unordered, align 8
  ret void
}

