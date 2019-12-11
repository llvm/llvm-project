; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.fma.f128(fp128 %f1, fp128 %f2, fp128 %f3, metadata, metadata)

define void @f1(fp128 *%ptr1, fp128 *%ptr2, fp128 *%ptr3, fp128 *%dst) #0 {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, fmal
; CHECK: br %r14
  %f1 = load fp128, fp128 *%ptr1
  %f2 = load fp128, fp128 *%ptr2
  %f3 = load fp128, fp128 *%ptr3
  %res = call fp128 @llvm.experimental.constrained.fma.f128 (
                        fp128 %f1, fp128 %f2, fp128 %f3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

attributes #0 = { strictfp }

