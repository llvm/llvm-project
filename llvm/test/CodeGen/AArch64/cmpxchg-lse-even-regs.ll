; RUN: llc -mtriple arm64-apple-ios -mattr=+lse %s -o - | FileCheck %s
; RUN: llc -mtriple arm64-apple-ios -mattr=+lse -mattr=+outline-atomics %s -o - | FileCheck %s

; Only "even,even+1" pairs are valid for CASP instructions. Make sure LLVM
; doesn't allocate odd ones and that it can copy them around properly. N.b. we
; don't actually check that they're sequential because FileCheck can't; odd/even
; will have to be good enough.
define void @test_atomic_cmpxchg_i128_register_shuffling(ptr %addr, i128 %desired, i128 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i128_register_shuffling:
; CHECK-DAG: mov [[DESIRED_LO:x[0-9]*[02468]]], x1
; CHECK-DAG: mov [[DESIRED_HI:x[0-9]*[13579]]], x2
; CHECK-DAG: mov [[NEW_LO:x[0-9]*[02468]]], x3
; CHECK-DAG: mov [[NEW_HI:x[0-9]*[13579]]], x4
; CHECK: caspal [[DESIRED_LO]], [[DESIRED_HI]], [[NEW_LO]], [[NEW_HI]], [x0]

  %res = cmpxchg ptr %addr, i128 %desired, i128 %new seq_cst seq_cst
  ret void
}
