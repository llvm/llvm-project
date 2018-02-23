; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; CHECK-LABEL: atomic_load16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR:r[0-9]+]], [[RD:(X|Y|Z)]]
; CHECK-NEXT: ldd [[RR:r[0-9]+]], [[RD:(X|Y|Z)]]+
; CHECK-NEXT: out 63, r0
define i16 @atomic_load16(i16* %foo) {
  %val = load atomic i16, i16* %foo unordered, align 2
  ret i16 %val
}

; CHECK-LABEL: atomic_load_swap16
; CHECK: call __sync_lock_test_and_set_2
define i16 @atomic_load_swap16(i16* %foo) {
  %val = atomicrmw xchg i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_cmp_swap16
; CHECK: call __sync_val_compare_and_swap_2
define i16 @atomic_load_cmp_swap16(i16* %foo) {
  %val = cmpxchg i16* %foo, i16 5, i16 10 acq_rel monotonic
  %value_loaded = extractvalue { i16, i1 } %val, 0
  ret i16 %value_loaded
}

; CHECK-LABEL: atomic_load_add16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR1:r[0-9]+]], [[RD1:(X|Y|Z)]]
; CHECK-NEXT: ldd [[RR2:r[0-9]+]], [[RD2:(X|Y|Z)]]+
; CHECK-NEXT: add [[RR1]], [[TMP:r[0-9]+]]
; CHECK-NEXT: adc [[RR2]], [[TMP:r[0-9]+]]
; CHECK-NEXT: st [[RD1]], [[RR1]]
; CHECK-NEXT: std [[RD1]]+1, [[A:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define i16 @atomic_load_add16(i16* %foo) {
  %val = atomicrmw add i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_sub16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR1:r[0-9]+]], [[RD1:(X|Y|Z)]]
; CHECK-NEXT: ldd [[RR2:r[0-9]+]], [[RD2:(X|Y|Z)]]+
; CHECK-NEXT: sub [[RR1]], [[TMP:r[0-9]+]]
; CHECK-NEXT: sbc [[RR2]], [[TMP:r[0-9]+]]
; CHECK-NEXT: st [[RD1]], [[RR1]]
; CHECK-NEXT: std [[RD1]]+1, [[A:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define i16 @atomic_load_sub16(i16* %foo) {
  %val = atomicrmw sub i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_and16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR1:r[0-9]+]], [[RD1:(X|Y|Z)]]
; CHECK-NEXT: ldd [[RR2:r[0-9]+]], [[RD2:(X|Y|Z)]]+
; CHECK-NEXT: and [[RR1]], [[TMP:r[0-9]+]]
; CHECK-NEXT: and [[RR2]], [[TMP:r[0-9]+]]
; CHECK-NEXT: st [[RD1]], [[RR1]]
; CHECK-NEXT: std [[RD1]]+1, [[A:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define i16 @atomic_load_and16(i16* %foo) {
  %val = atomicrmw and i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_or16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR1:r[0-9]+]], [[RD1:(X|Y|Z)]]
; CHECK-NEXT: ldd [[RR2:r[0-9]+]], [[RD2:(X|Y|Z)]]+
; CHECK-NEXT: or [[RR1]], [[TMP:r[0-9]+]]
; CHECK-NEXT: or [[RR2]], [[TMP:r[0-9]+]]
; CHECK-NEXT: st [[RD1]], [[RR1]]
; CHECK-NEXT: std [[RD1]]+1, [[A:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define i16 @atomic_load_or16(i16* %foo) {
  %val = atomicrmw or i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_xor16
; CHECK:      in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: ld [[RR1:r[0-9]+]], [[RD1:(X|Y|Z)]]
; CHECK-NEXT: ldd [[RR2:r[0-9]+]], [[RD2:(X|Y|Z)]]+
; CHECK-NEXT: eor [[RR1]], [[TMP:r[0-9]+]]
; CHECK-NEXT: eor [[RR2]], [[TMP:r[0-9]+]]
; CHECK-NEXT: st [[RD1]], [[RR1]]
; CHECK-NEXT: std [[RD1]]+1, [[A:r[0-9]+]]
; CHECK-NEXT: out 63, r0
define i16 @atomic_load_xor16(i16* %foo) {
  %val = atomicrmw xor i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_nand16
; CHECK: call __sync_fetch_and_nand_2
define i16 @atomic_load_nand16(i16* %foo) {
  %val = atomicrmw nand i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_max16
; CHECK: call __sync_fetch_and_max_2
define i16 @atomic_load_max16(i16* %foo) {
  %val = atomicrmw max i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_min16
; CHECK: call __sync_fetch_and_min_2
define i16 @atomic_load_min16(i16* %foo) {
  %val = atomicrmw min i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_umax16
; CHECK: call __sync_fetch_and_umax_2
define i16 @atomic_load_umax16(i16* %foo) {
  %val = atomicrmw umax i16* %foo, i16 13 seq_cst
  ret i16 %val
}

; CHECK-LABEL: atomic_load_umin16
; CHECK: call __sync_fetch_and_umin_2
define i16 @atomic_load_umin16(i16* %foo) {
  %val = atomicrmw umin i16* %foo, i16 13 seq_cst
  ret i16 %val
}
