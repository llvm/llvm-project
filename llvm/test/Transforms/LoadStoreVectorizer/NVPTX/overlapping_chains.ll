; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; CHECK-LABEL: @overlapping_stores
; CHECK: store i16
; CHECK: store i16
; CHECK: store i16
define void @overlapping_stores(ptr nocapture align 2 %ptr) {
  %ptr0 = getelementptr i16, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i16, ptr %ptr, i64 1

  store i16 0, ptr %ptr0, align 2
  store i16 0, ptr %ptr1, align 1
  store i16 0, ptr %ptr2, align 2

  ret void
}
