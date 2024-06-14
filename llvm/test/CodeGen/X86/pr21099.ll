; RUN: llc < %s -O2 -mtriple=x86_64-- -verify-machineinstrs | FileCheck %s

define void @pr21099(ptr %p) {
; CHECK-LABEL: pr21099
; CHECK: lock addq $-2147483648
; This number is INT32_MIN: 0x80000000UL
  %1 = atomicrmw add ptr %p, i64 -2147483648 seq_cst
  ret void
}
