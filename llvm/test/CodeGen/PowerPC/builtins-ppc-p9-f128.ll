; RUN: llc -verify-machineinstrs -mcpu=pwr9 -enable-ppc-quad-precision \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

@A = common global fp128 0xL00000000000000000000000000000000, align 16
@B = common global fp128 0xL00000000000000000000000000000000, align 16
@C = common global fp128 0xL00000000000000000000000000000000, align 16

define fp128 @testSqrtOdd() {
entry:
  %0 = load fp128, fp128* @A, align 16
  %1 = call fp128 @llvm.ppc.sqrtf128.round.to.odd(fp128 %0)
  ret fp128 %1
; CHECK-LABEL: testSqrtOdd
; CHECK: xssqrtqpo
}

declare fp128 @llvm.ppc.sqrtf128.round.to.odd(fp128)

define fp128 @testFMAOdd() {
entry:
  %0 = load fp128, fp128* @A, align 16
  %1 = load fp128, fp128* @B, align 16
  %2 = load fp128, fp128* @C, align 16
  %3 = call fp128 @llvm.ppc.fmaf128.round.to.odd(fp128 %0, fp128 %1, fp128 %2)
  ret fp128 %3
; CHECK-LABEL: testFMAOdd
; CHECK: xsmaddqpo
}

declare fp128 @llvm.ppc.fmaf128.round.to.odd(fp128, fp128, fp128)

define fp128 @testAddOdd() {
entry:
  %0 = load fp128, fp128* @A, align 16
  %1 = load fp128, fp128* @B, align 16
  %2 = call fp128 @llvm.ppc.addf128.round.to.odd(fp128 %0, fp128 %1)
  ret fp128 %2
; CHECK-LABEL: testAddOdd
; CHECK: xsaddqpo
}

declare fp128 @llvm.ppc.addf128.round.to.odd(fp128, fp128)

define fp128 @testSubOdd() {
entry:
  %0 = load fp128, fp128* @A, align 16
  %1 = load fp128, fp128* @B, align 16
  %2 = call fp128 @llvm.ppc.subf128.round.to.odd(fp128 %0, fp128 %1)
  ret fp128 %2
; CHECK-LABEL: testSubOdd
; CHECK: xssubqpo
}

; Function Attrs: nounwind readnone
declare fp128 @llvm.ppc.subf128.round.to.odd(fp128, fp128)

; Function Attrs: noinline nounwind optnone
define fp128 @testMulOdd() {
entry:
  %0 = load fp128, fp128* @A, align 16
  %1 = load fp128, fp128* @B, align 16
  %2 = call fp128 @llvm.ppc.mulf128.round.to.odd(fp128 %0, fp128 %1)
  ret fp128 %2
; CHECK-LABEL: testMulOdd
; CHECK: xsmulqpo
}

; Function Attrs: nounwind readnone
declare fp128 @llvm.ppc.mulf128.round.to.odd(fp128, fp128)

define fp128 @testDivOdd() {
entry:
  %0 = load fp128, fp128* @A, align 16
  %1 = load fp128, fp128* @B, align 16
  %2 = call fp128 @llvm.ppc.divf128.round.to.odd(fp128 %0, fp128 %1)
  ret fp128 %2
; CHECK-LABEL: testDivOdd
; CHECK: xsdivqpo
}

declare fp128 @llvm.ppc.divf128.round.to.odd(fp128, fp128)

define double @testTruncOdd() {
entry:
  %0 = load fp128, fp128* @A, align 16
  %1 = call double @llvm.ppc.truncf128.round.to.odd(fp128 %0)
  ret double %1
  ; CHECK-LABEL: testTruncOdd
  ; CHECK: xscvqpdpo
}

declare double @llvm.ppc.truncf128.round.to.odd(fp128)
