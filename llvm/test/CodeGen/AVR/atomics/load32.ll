; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; CHECK-LABEL: atomic_load32
; CHECK: call __sync_val_compare_and_swap_4
define i32 @atomic_load32(i32* %foo) {
  %val = load atomic i32, i32* %foo unordered, align 4
  ret i32 %val
}

; CHECK-LABEL: atomic_load_sub32
; CHECK: call __sync_fetch_and_sub_4
define i32 @atomic_load_sub32(i32* %foo) {
  %val = atomicrmw sub i32* %foo, i32 13 seq_cst
  ret i32 %val
}

