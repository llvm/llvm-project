; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.fma.f128(fp128 %f1, fp128 %f2, fp128 %f3)

define void @f1(fp128 *%ptr1, fp128 *%ptr2, fp128 *%ptr3, fp128 *%dst) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, fmal
; CHECK: br %r14
  %f1 = load fp128, fp128 *%ptr1
  %f2 = load fp128, fp128 *%ptr2
  %f3 = load fp128, fp128 *%ptr3
  %res = call fp128 @llvm.fma.f128 (fp128 %f1, fp128 %f2, fp128 %f3)
  store fp128 %res, fp128 *%dst
  ret void
}

