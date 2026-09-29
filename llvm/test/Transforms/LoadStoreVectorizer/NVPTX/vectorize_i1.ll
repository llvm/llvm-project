; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

define void @i1x8(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2
  %ptr3 = getelementptr i8, ptr %ptr, i64 3

  %l0 = load <8 x i1>, ptr %ptr0, align 4
  %l1 = load <8 x i1>, ptr %ptr1, align 1
  %l2 = load <8 x i1>, ptr %ptr2, align 2
  %l3 = load <8 x i1>, ptr %ptr3, align 1

  ret void

; CHECK-LABEL: @i1x8
; CHECK-DAG: load <32 x i1>
}

define void @i1x8x16x8(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 3

  %l0 = load <8 x i1>,  ptr %ptr0, align 4
  %l2 = load <16 x i1>, ptr %ptr1, align 1
  %l3 = load <8 x i1>,  ptr %ptr2, align 1

  ret void

; CHECK-LABEL: @i1x8x16x8
; CHECK-DAG: load <32 x i1>
}
