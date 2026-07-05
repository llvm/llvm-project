; RUN: llc -mtriple=x86_64 < %s | FileCheck %s

; Quick test to ensure that atomics which are not naturally-aligned
; emit unsized libcalls, and aren't emitted as native instructions or
; sized libcalls.
define void @test_i32(ptr %a) nounwind {
; CHECK-LABEL: test_i32:
; CHECK: callq __atomic_load
; CHECK: callq __atomic_store
; CHECK: callq __atomic_exchange
; CHECK: callq __atomic_compare_exchange
; CHECK: callq __atomic_compare_exchange
  %t0 = load atomic i32, ptr %a seq_cst, align 2
  store atomic i32 1, ptr %a seq_cst, align 2
  %t1 = atomicrmw xchg ptr %a, i32 1 seq_cst, align 2
  %t3 = atomicrmw add ptr %a, i32 2 seq_cst, align 2
  %t2 = cmpxchg ptr %a, i32 0, i32 1 seq_cst seq_cst, align 2
  ret void
}

define void @test_i128(ptr %a) nounwind {
; CHECK-LABEL: test_i128:
; CHECK: callq __atomic_load
; CHECK: callq __atomic_store
; CHECK: callq __atomic_exchange
; CHECK: callq __atomic_compare_exchange
  %t0 = load atomic i128, ptr %a seq_cst, align 8
  store atomic i128 1, ptr %a seq_cst, align 8
  %t1 = atomicrmw xchg ptr %a, i128 1 seq_cst, align 8
  %t2 = atomicrmw add ptr %a, i128 2 seq_cst, align 8
  %t3 = cmpxchg ptr %a, i128 0, i128 1 seq_cst seq_cst, align 8
  ret void
}
