; Test 128-bit floating-point signaling comparison.  The tests assume a z10
; implementation of select, using conditional branches rather than LOCGR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; There is no memory form of 128-bit comparison.
define i64 @f1(i64 %a, i64 %b, ptr %ptr, float %f2) #0 {
; CHECK-LABEL: f1:
; CHECK-DAG: lxebr %f0, %f0
; CHECK-DAG: ld %f1, 0(%r4)
; CHECK-DAG: ld %f3, 8(%r4)
; CHECK: kxbr %f1, %f0
; CHECK-NEXT: ber %r14
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f2x = fpext float %f2 to fp128
  %f1 = load fp128, ptr %ptr
  %cond = call i1 @llvm.experimental.constrained.fcmps.f128(
                                               fp128 %f1, fp128 %f2x,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check comparison with zero - cannot use LOAD AND TEST.
define i64 @f2(i64 %a, i64 %b, ptr %ptr) #0 {
; CHECK-LABEL: f2:
; CHECK-DAG: ld %f0, 0(%r4)
; CHECK-DAG: ld %f2, 8(%r4)
; CHECK-DAG: lzxr [[REG:%f[0-9]+]]
; CHECK-NEXT: kxbr %f0, [[REG]]
; CHECK-NEXT: ber %r14
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f = load fp128, ptr %ptr
  %cond = call i1 @llvm.experimental.constrained.fcmps.f128(
                                               fp128 %f, fp128 0xL00000000000000000000000000000000,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

attributes #0 = { strictfp }

declare i1 @llvm.experimental.constrained.fcmps.f128(fp128, fp128, metadata, metadata)

