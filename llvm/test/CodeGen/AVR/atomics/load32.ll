; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

; CHECK-LABEL: atomic_load32
; CHECK: call __sync_val_compare_and_swap_4
define i32 @atomic_load32(ptr %foo) {
  %val = load atomic i32, ptr %foo unordered, align 4
  ret i32 %val
}

; CHECK-LABEL: atomic_load_sub32
; CHECK: call __sync_fetch_and_sub_4
define i32 @atomic_load_sub32(ptr %foo) {
  %val = atomicrmw sub ptr %foo, i32 13 seq_cst
  ret i32 %val
}

