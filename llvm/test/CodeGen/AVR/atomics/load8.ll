; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; Tests atomic operations on AVR

; CHECK-LABEL: atomic_load8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR:r[0-9]+]], [[RD:(X|Y|Z)]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load8(i8* %foo) {
  %val = load atomic i8, i8* %foo unordered, align 1
  ret i8 %val
}

; CHECK-LABEL: atomic_load_swap8
; CHECK: call __sync_lock_test_and_set_1
define i8 @atomic_load_swap8(i8* %foo) {
  %val = atomicrmw xchg i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_cmp_swap8
; CHECK: call __sync_val_compare_and_swap_1
define i8 @atomic_load_cmp_swap8(i8* %foo) {
  %val = cmpxchg i8* %foo, i8 5, i8 10 acq_rel monotonic
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
define i8 @atomic_load_add8(i8* %foo) {
  %val = atomicrmw add i8* %foo, i8 13 seq_cst
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
define i8 @atomic_load_sub8(i8* %foo) {
  %val = atomicrmw sub i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_and8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: and [[RR1:r[0-9]+]], [[RD]]
; CHECK-NEXT: st [[RR]], [[RR1]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_and8(i8* %foo) {
  %val = atomicrmw and i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_or8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: or [[RR1:r[0-9]+]], [[RD]]
; CHECK-NEXT: st [[RR]], [[RR1]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_or8(i8* %foo) {
  %val = atomicrmw or i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_xor8
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RD:r[0-9]+]], [[RR:(X|Y|Z)]]
; CHECK-NEXT: eor [[RR1:r[0-9]+]], [[RD]]
; CHECK-NEXT: st [[RR]], [[RR1]]
; CHECK-NEXT: out 63, r0
define i8 @atomic_load_xor8(i8* %foo) {
  %val = atomicrmw xor i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_nand8
; CHECK: call __sync_fetch_and_nand_1
define i8 @atomic_load_nand8(i8* %foo) {
  %val = atomicrmw nand i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_max8
; CHECK: call __sync_fetch_and_max_1
define i8 @atomic_load_max8(i8* %foo) {
  %val = atomicrmw max i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_min8
; CHECK: call __sync_fetch_and_min_1
define i8 @atomic_load_min8(i8* %foo) {
  %val = atomicrmw min i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_umax8
; CHECK: call __sync_fetch_and_umax_1
define i8 @atomic_load_umax8(i8* %foo) {
  %val = atomicrmw umax i8* %foo, i8 13 seq_cst
  ret i8 %val
}

; CHECK-LABEL: atomic_load_umin8
; CHECK: call __sync_fetch_and_umin_1
define i8 @atomic_load_umin8(i8* %foo) {
  %val = atomicrmw umin i8* %foo, i8 13 seq_cst
  ret i8 %val
}

