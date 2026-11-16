; RUN: llc -mattr=avr6 < %s -mtriple=avr | FileCheck %s

; CHECK-LABEL: atomic_load64
; CHECK: call __atomic_load_8
define i64 @atomic_load64(ptr %foo) {
  %val = load atomic i64, ptr %foo unordered, align 8
  ret i64 %val
}

; CHECK-LABEL: atomic_load_sub64
; CHECK: call __atomic_fetch_sub_8
define i64 @atomic_load_sub64(ptr %foo) {
  %val = atomicrmw sub ptr %foo, i64 13 seq_cst
  ret i64 %val
}

