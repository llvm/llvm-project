; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7 -mattr=-cx16 | FileCheck %s
define void @test(ptr %a) nounwind {
entry:
; CHECK: __sync_val_compare_and_swap_16
  %0 = cmpxchg ptr %a, i128 1, i128 1 seq_cst seq_cst
; CHECK: __sync_lock_test_and_set_16
  %1 = atomicrmw xchg ptr %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_add_16
  %2 = atomicrmw add ptr %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_sub_16
  %3 = atomicrmw sub ptr %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_and_16
  %4 = atomicrmw and ptr %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_nand_16
  %5 = atomicrmw nand ptr %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_or_16
  %6 = atomicrmw or ptr %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_xor_16
  %7 = atomicrmw xor ptr %a, i128 1 seq_cst
  ret void
}
