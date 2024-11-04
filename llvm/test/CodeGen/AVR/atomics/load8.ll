; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; Tests atomic operations on AVR

; CHECK-LABEL: atomic_load8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR:r[0-9]+]], [[RD:(X|Y|Z)]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load8(ptr %foo) {
  %val = load atomic i8, ptr %foo unordered, align 1
  ret i8 %val
}

; CHECK-LABEL: atomic_load_swap8
; CHECK: call __sync_lock_test_and_set_1
define i8 @atomic_load_swap8(ptr %foo) {
  %val = atomicrmw xchg ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_cmp_swap8
; CHECK: call __sync_val_compare_and_swap_1
define i8 @atomic_load_cmp_swap8(ptr %foo) {
  %val = cmpxchg ptr %foo, i8 5, i8 10 acq_rel monotonic
  %value_loaded = extractvalue { i8, i1 } %val, 0
  ret i8 %value_loaded
}

; CHECK-LABEL: atomic_load_add8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: add [[RR1:r[0-9]+]], [[RD]]
; CHECK-NEXT: st [[RR]], [[RR1]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_add8(ptr %foo) {
  %val = atomicrmw add ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_sub8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: mov [[TMP:r[0-9]+]], [[RD]]
; CHECK-NEXT: sub [[TMP]], [[RR1:r[0-9]+]]
; CHECK-NEXT: st [[RR]], [[TMP]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_sub8(ptr %foo) {
  %val = atomicrmw sub ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_and8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: and [[RR1:r[0-9]+]], [[RD]]
; CHECK-NEXT: st [[RR]], [[RR1]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_and8(ptr %foo) {
  %val = atomicrmw and ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_or8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: or [[RR1:r[0-9]+]], [[RD]]
; CHECK-NEXT: st [[RR]], [[RR1]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_or8(ptr %foo) {
  %val = atomicrmw or ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_xor8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: eor [[RR1:r[0-9]+]], [[RD]]
; CHECK-NEXT: st [[RR]], [[RR1]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_xor8(ptr %foo) {
  %val = atomicrmw xor ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_nand8
; CHECK: call __sync_fetch_and_nand_1
define i8 @atomic_load_nand8(ptr %foo) {
  %val = atomicrmw nand ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_max8
; CHECK: call __sync_fetch_and_max_1
define i8 @atomic_load_max8(ptr %foo) {
  %val = atomicrmw max ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_min8
; CHECK: call __sync_fetch_and_min_1
define i8 @atomic_load_min8(ptr %foo) {
  %val = atomicrmw min ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_umax8
; CHECK: call __sync_fetch_and_umax_1
define i8 @atomic_load_umax8(ptr %foo) {
  %val = atomicrmw umax ptr %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_umin8
; CHECK: call __sync_fetch_and_umin_1
define i8 @atomic_load_umin8(ptr %foo) {
  %val = atomicrmw umin ptr %foo, i8 13 seq_cst
  ret i8 %val
}

