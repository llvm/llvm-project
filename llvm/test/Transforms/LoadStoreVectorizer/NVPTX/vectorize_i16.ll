; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; CHECK-LABEL: @int16x2
; CHECK: load <2 x i16>
; CHECK: store <2 x i16>
define void @int16x2(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i16, ptr %ptr, i64 0
  %ptr1 = getelementptr i16, ptr %ptr, i64 1

  %l0 = load i16, ptr %ptr0, align 4
  %l1 = load i16, ptr %ptr1, align 2

  store i16 %l1, ptr %ptr0, align 4
  store i16 %l0, ptr %ptr1, align 2

  ret void
}
