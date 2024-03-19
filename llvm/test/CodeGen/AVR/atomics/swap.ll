; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; CHECK-LABEL: atomic_swap8
; CHECK: call __sync_lock_test_and_set_1
define i8 @atomic_swap8(ptr %foo) {
  %val = atomicrmw xchg ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_swap16
; CHECK: call __sync_lock_test_and_set_2
define i16 @atomic_swap16(ptr %foo) {
  %val = atomicrmw xchg ptr %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_swap32
; CHECK: call __sync_lock_test_and_set_4
define i32 @atomic_swap32(ptr %foo) {
  %val = atomicrmw xchg ptr %foo, i32 13 seq_cst
  ret i32 %val
}

; CHECK-LABEL: atomic_swap64
; CHECK: call __sync_lock_test_and_set_8
define i64 @atomic_swap64(ptr %foo) {
  %val = atomicrmw xchg ptr %foo, i64 13 seq_cst
  ret i64 %val
}

