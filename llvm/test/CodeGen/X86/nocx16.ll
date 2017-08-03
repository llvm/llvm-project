; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mattr=-cx16 | FileCheck %s
define void @test(i128* %a) nounwind {
entry:
; CHECK: __sync_val_compare_and_swap_16
  %0 = cmpxchg i128* %a, i128 1, i128 1 seq_cst seq_cst
; CHECK: __sync_lock_test_and_set_16
  %1 = atomicrmw xchg i128* %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_add_16
  %2 = atomicrmw add i128* %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_sub_16
  %3 = atomicrmw sub i128* %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_and_16
  %4 = atomicrmw and i128* %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_nand_16
  %5 = atomicrmw nand i128* %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_or_16
  %6 = atomicrmw or i128* %a, i128 1 seq_cst
; CHECK: __sync_fetch_and_xor_16
  %7 = atomicrmw xor i128* %a, i128 1 seq_cst
  ret void
}
