; RUN: llc -verify-machineinstrs -mcpu=pwr9 -enable-ppc-quad-precision \
; RUN:   -mtriple=powerpc64le-unknown-unknown -ppc-vsr-nums-as-vr \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s

@A = common global fp128 0xL00000000000000000000000000000000, align 16
@B = common global fp128 0xL00000000000000000000000000000000, align 16
@C = common global fp128 0xL00000000000000000000000000000000, align 16
@D = common global fp128 0xL00000000000000000000000000000000, align 16

define fp128 @testSqrtOdd(fp128 %a) {
entry:
  %0 = call fp128 @llvm.ppc.sqrtf128.round.to.odd(fp128 %a)
  ret fp128 %0
; CHECK-LABEL: testSqrtOdd
; CHECK: xssqrtqpo v2, v2
; CHECK: blr
}

declare fp128 @llvm.ppc.sqrtf128.round.to.odd(fp128)

define void @testFMAOdd(fp128 %a, fp128 %b, fp128 %c) {
entry:
  %0 = call fp128 @llvm.ppc.fmaf128.round.to.odd(fp128 %a, fp128 %b, fp128 %c)
  store fp128 %0, fp128* @A, align 16
  %sub = fsub fp128 0xL00000000000000008000000000000000, %c
  %1 = call fp128 @llvm.ppc.fmaf128.round.to.odd(fp128 %a, fp128 %b, fp128 %sub)
  store fp128 %1, fp128* @B, align 16
  %2 = call fp128 @llvm.ppc.fmaf128.round.to.odd(fp128 %a, fp128 %b, fp128 %c)
  %sub1 = fsub fp128 0xL00000000000000008000000000000000, %2
  store fp128 %sub1, fp128* @C, align 16
  %sub2 = fsub fp128 0xL00000000000000008000000000000000, %c
  %3 = call fp128 @llvm.ppc.fmaf128.round.to.odd(fp128 %a, fp128 %b, fp128 %sub2)
  %sub3 = fsub fp128 0xL00000000000000008000000000000000, %3
  store fp128 %sub3, fp128* @D, align 16
  ret void
; CHECK-LABEL: testFMAOdd
; CHECK-DAG: xsmaddqpo v{{[0-9]+}}, v2, v3
; CHECK-DAG: xsmsubqpo v{{[0-9]+}}, v2, v3
; CHECK-DAG: xsnmaddqpo v{{[0-9]+}}, v2, v3
; CHECK-DAG: xsnmsubqpo v{{[0-9]+}}, v2, v3
; CHECK: blr
}

declare fp128 @llvm.ppc.fmaf128.round.to.odd(fp128, fp128, fp128)

define fp128 @testAddOdd(fp128 %a, fp128 %b) {
entry:
  %0 = call fp128 @llvm.ppc.addf128.round.to.odd(fp128 %a, fp128 %b)
  ret fp128 %0
; CHECK-LABEL: testAddOdd
; CHECK: xsaddqpo v2, v2, v3
; CHECK: blr
}

declare fp128 @llvm.ppc.addf128.round.to.odd(fp128, fp128)

define fp128 @testSubOdd(fp128 %a, fp128 %b) {
entry:
  %0 = call fp128 @llvm.ppc.subf128.round.to.odd(fp128 %a, fp128 %b)
  ret fp128 %0
; CHECK-LABEL: testSubOdd
; CHECK: xssubqpo v2, v2, v3
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare fp128 @llvm.ppc.subf128.round.to.odd(fp128, fp128)

; Function Attrs: noinline nounwind optnone
define fp128 @testMulOdd(fp128 %a, fp128 %b) {
entry:
  %0 = call fp128 @llvm.ppc.mulf128.round.to.odd(fp128 %a, fp128 %b)
  ret fp128 %0
; CHECK-LABEL: testMulOdd
; CHECK: xsmulqpo v2, v2, v3
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare fp128 @llvm.ppc.mulf128.round.to.odd(fp128, fp128)

define fp128 @testDivOdd(fp128 %a, fp128 %b) {
entry:
  %0 = call fp128 @llvm.ppc.divf128.round.to.odd(fp128 %a, fp128 %b)
  ret fp128 %0
; CHECK-LABEL: testDivOdd
; CHECK: xsdivqpo v2, v2, v3
; CHECK: blr
}

declare fp128 @llvm.ppc.divf128.round.to.odd(fp128, fp128)

define double @testTruncOdd(fp128 %a) {
entry:
  %0 = call double @llvm.ppc.truncf128.round.to.odd(fp128 %a)
  ret double %0
; CHECK-LABEL: testTruncOdd
; CHECK: xscvqpdpo v2, v2
; CHECK: xxlor f1, v2, v2
; CHECK: blr
}

declare double @llvm.ppc.truncf128.round.to.odd(fp128)
