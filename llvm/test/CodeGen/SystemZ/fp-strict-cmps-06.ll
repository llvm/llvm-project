; Test f128 signaling comparisons on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; There is no memory form of 128-bit comparison.
define i64 @f1(i64 %a, i64 %b, fp128 *%ptr1, fp128 *%ptr2) #0 {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r4)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r5)
; CHECK: wfkxb [[REG1]], [[REG2]]
; CHECK-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f1 = load fp128, fp128 *%ptr1
  %f2 = load fp128, fp128 *%ptr2
  %cond = call i1 @llvm.experimental.constrained.fcmps.f128(
                                               fp128 %f1, fp128 %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check comparison with zero -- it is not worthwhile to copy to
; FP pairs just so we can use LTXBR, so simply load up a zero.
define i64 @f2(i64 %a, i64 %b, fp128 *%ptr) #0 {
; CHECK-LABEL: f2:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r4)
; CHECK-DAG: vzero [[REG2:%v[0-9]+]]
; CHECK: wfkxb [[REG1]], [[REG2]]
; CHECK-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f = load fp128, fp128 *%ptr
  %cond = call i1 @llvm.experimental.constrained.fcmps.f128(
                                               fp128 %f, fp128 0xL00000000000000000000000000000000,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

attributes #0 = { strictfp }

declare i1 @llvm.experimental.constrained.fcmps.f128(fp128, fp128, metadata, metadata)

