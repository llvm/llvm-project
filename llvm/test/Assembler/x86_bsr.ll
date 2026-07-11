; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Test that x86_bsr type round-trips through bitcode correctly.
; x86_bsr can only appear in intrinsic signatures, so we test it via
; intrinsic calls inside regular functions that use normal types in
; their own signatures.

; CHECK-LABEL: define void @test_bsr_round_trip(<16 x i32> %lo, <16 x i32> %hi)
define void @test_bsr_round_trip(<16 x i32> %lo, <16 x i32> %hi) {
; CHECK: %bsr = call x86_bsr @llvm.x86.bsr.create(<16 x i32> %lo, <16 x i32> %hi)
  %bsr = call x86_bsr @llvm.x86.bsr.create(<16 x i32> %lo, <16 x i32> %hi)
; CHECK: %lo.out = call <16 x i32> @llvm.x86.bsr.get.lo(x86_bsr %bsr)
  %lo.out = call <16 x i32> @llvm.x86.bsr.get.lo(x86_bsr %bsr)
; CHECK: %hi.out = call <16 x i32> @llvm.x86.bsr.get.hi(x86_bsr %bsr)
  %hi.out = call <16 x i32> @llvm.x86.bsr.get.hi(x86_bsr %bsr)
  ret void
}

; CHECK-LABEL: define void @test_bsr_set(<16 x i32> %new_lo, <16 x i32> %new_hi)
define void @test_bsr_set(<16 x i32> %new_lo, <16 x i32> %new_hi) {
; CHECK: %bsr = call x86_bsr @llvm.x86.bsr.create(<16 x i32> %new_lo, <16 x i32> %new_hi)
  %bsr = call x86_bsr @llvm.x86.bsr.create(<16 x i32> %new_lo, <16 x i32> %new_hi)
; CHECK: %bsr2 = call x86_bsr @llvm.x86.bsr.set.lo(x86_bsr %bsr, <16 x i32> %new_lo)
  %bsr2 = call x86_bsr @llvm.x86.bsr.set.lo(x86_bsr %bsr, <16 x i32> %new_lo)
; CHECK: %bsr3 = call x86_bsr @llvm.x86.bsr.set.hi(x86_bsr %bsr2, <16 x i32> %new_hi)
  %bsr3 = call x86_bsr @llvm.x86.bsr.set.hi(x86_bsr %bsr2, <16 x i32> %new_hi)
  ret void
}

; CHECK: declare x86_bsr @llvm.x86.bsr.create(<16 x i32>, <16 x i32>)
; CHECK: declare <16 x i32> @llvm.x86.bsr.get.lo(x86_bsr)
; CHECK: declare <16 x i32> @llvm.x86.bsr.get.hi(x86_bsr)
; CHECK: declare x86_bsr @llvm.x86.bsr.set.lo(x86_bsr, <16 x i32>)
; CHECK: declare x86_bsr @llvm.x86.bsr.set.hi(x86_bsr, <16 x i32>)

declare x86_bsr @llvm.x86.bsr.create(<16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.x86.bsr.get.lo(x86_bsr)
declare <16 x i32> @llvm.x86.bsr.get.hi(x86_bsr)
declare x86_bsr @llvm.x86.bsr.set.lo(x86_bsr, <16 x i32>)
declare x86_bsr @llvm.x86.bsr.set.hi(x86_bsr, <16 x i32>)
