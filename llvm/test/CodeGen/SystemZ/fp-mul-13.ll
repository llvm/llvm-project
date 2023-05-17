; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.fma.f128(fp128 %f1, fp128 %f2, fp128 %f3)

define void @f1(ptr %ptr1, ptr %ptr2, ptr %ptr3, ptr %dst) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, fmal
; CHECK: br %r14
  %f1 = load fp128, ptr %ptr1
  %f2 = load fp128, ptr %ptr2
  %f3 = load fp128, ptr %ptr3
  %res = call fp128 @llvm.fma.f128 (fp128 %f1, fp128 %f2, fp128 %f3)
  store fp128 %res, ptr %dst
  ret void
}

