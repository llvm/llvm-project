; RUN: llc < %s -mtriple=x86_64-- -verify-machineinstrs -mcpu=corei7 -mattr=-cx16 | FileCheck %s
; RUN: llc < %s -mtriple=i386-linux-gnu -verify-machineinstrs -mattr=cx16 | FileCheck -check-prefix=CHECK32 %s

;; Verify that 128-bit atomics emit a libcall without cx16
;; available.
;;
;; We test 32-bit mode with -mattr=cx16, because it should have no
;; effect for 32-bit mode.

; CHECK-LABEL: test:
define void @test(ptr %a) nounwind {
entry:
; CHECK: __atomic_compare_exchange_16
; CHECK32: __atomic_compare_exchange
  %0 = cmpxchg ptr %a, i128 1, i128 1 seq_cst seq_cst
; CHECK: __atomic_exchange_16
; CHECK32: __atomic_exchange
  %1 = atomicrmw xchg ptr %a, i128 1 seq_cst
; CHECK: __atomic_fetch_add_16
; CHECK32: __atomic_compare_exchange
  %2 = atomicrmw add ptr %a, i128 1 seq_cst
; CHECK: __atomic_fetch_sub_16
; CHECK32: __atomic_compare_exchange
  %3 = atomicrmw sub ptr %a, i128 1 seq_cst
; CHECK: __atomic_fetch_and_16
; CHECK32: __atomic_compare_exchange
  %4 = atomicrmw and ptr %a, i128 1 seq_cst
; CHECK: __atomic_fetch_nand_16
; CHECK32: __atomic_compare_exchange
  %5 = atomicrmw nand ptr %a, i128 1 seq_cst
; CHECK: __atomic_fetch_or_16
; CHECK32: __atomic_compare_exchange
  %6 = atomicrmw or ptr %a, i128 1 seq_cst
; CHECK: __atomic_fetch_xor_16
; CHECK32: __atomic_compare_exchange
  %7 = atomicrmw xor ptr %a, i128 1 seq_cst
; CHECK: __atomic_load_16
; CHECK32: __atomic_load
  %8 = load atomic i128, ptr %a seq_cst, align 16
; CHECK: __atomic_store_16
; CHECK32: __atomic_store
  store atomic i128 %8, ptr %a seq_cst, align 16
  ret void
}

; CHECK-LABEL: test_fp:
define void @test_fp(fp128* %a) nounwind {
entry:
; CHECK: __atomic_exchange_16
; CHECK32: __atomic_exchange
  %0 = atomicrmw xchg fp128* %a, fp128 0xL00000000000000004000900000000000 seq_cst
; CHECK: __atomic_compare_exchange_16
; CHECK32: __atomic_compare_exchange
  %1 = atomicrmw fadd fp128* %a, fp128 0xL00000000000000004000900000000000 seq_cst
; CHECK: __atomic_compare_exchange_16
; CHECK32: __atomic_compare_exchange
  %2 = atomicrmw fsub fp128* %a, fp128 0xL00000000000000004000900000000000 seq_cst
; CHECK: __atomic_load_16
; CHECK32: __atomic_load
  %3 = load atomic fp128, fp128* %a seq_cst, align 16
; CHECK: __atomic_store_16
; CHECK32: __atomic_store
  store atomic fp128 %3, fp128* %a seq_cst, align 16
  ret void
}
