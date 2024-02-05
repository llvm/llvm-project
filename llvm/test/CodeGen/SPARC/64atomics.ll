; RUN: llc < %s -march=sparcv9 -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: test_atomic_i64
; CHECK:       ldx [%o0]
; CHECK:       membar
; CHECK:       ldx [%o1]
; CHECK:       membar
; CHECK:       membar
; CHECK:       stx {{.+}}, [%o2]
define i64 @test_atomic_i64(ptr %ptr1, ptr %ptr2, ptr %ptr3) {
entry:
  %0 = load atomic i64, ptr %ptr1 acquire, align 8
  %1 = load atomic i64, ptr %ptr2 acquire, align 8
  %2 = add i64 %0, %1
  store atomic i64 %2, ptr %ptr3 release, align 8
  ret i64 %2
}

; CHECK-LABEL: test_cmpxchg_i64
; CHECK:       mov 123, [[R:%[gilo][0-7]]]
; CHECK:       casx [%o1], %o0, [[R]]

define i64 @test_cmpxchg_i64(i64 %a, ptr %ptr) {
entry:
  %pair = cmpxchg ptr %ptr, i64 %a, i64 123 monotonic monotonic
  %b = extractvalue { i64, i1 } %pair, 0
  ret i64 %b
}

; CHECK-LABEL: test_swap_i64
; CHECK:       casx [%o1],

define i64 @test_swap_i64(i64 %a, ptr %ptr) {
entry:
  %b = atomicrmw xchg ptr %ptr, i64 42 monotonic
  ret i64 %b
}

; CHECK-LABEL: test_load_sub_64
; CHECK: membar
; CHECK: sub
; CHECK: casx [%o0]
; CHECK: membar
define zeroext i64 @test_load_sub_64(ptr %p, i64 zeroext %v) {
entry:
  %0 = atomicrmw sub ptr %p, i64 %v seq_cst
  ret i64 %0
}

; CHECK-LABEL: test_load_max_64
; CHECK: membar
; CHECK: cmp
; CHECK: movg %xcc
; CHECK: casx [%o0]
; CHECK: membar
define zeroext i64 @test_load_max_64(ptr %p, i64 zeroext %v) {
entry:
  %0 = atomicrmw max ptr %p, i64 %v seq_cst
  ret i64 %0
}
