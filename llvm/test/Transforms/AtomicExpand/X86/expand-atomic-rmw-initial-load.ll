; RUN: opt -S %s -atomic-expand -mtriple=i686-linux-gnu | FileCheck %s

; This file tests the function `llvm::expandAtomicRMWToCmpXchg`.
; It isn't technically target specific, but is exposed through a pass that is.

define i8 @test_initial_load(ptr %ptr, i8 %value) {
  %res = atomicrmw nand ptr %ptr, i8 %value seq_cst
  ret i8 %res
}
; CHECK-LABEL: @test_initial_load
; CHECK-NEXT:    %1 = load i8, ptr %ptr, align 1
