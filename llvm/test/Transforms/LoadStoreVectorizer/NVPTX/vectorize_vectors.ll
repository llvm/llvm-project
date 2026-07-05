; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; CHECK-LABEL: @int8x3Plus1
; CHECK: load <4 x i8>
; CHECK: store <4 x i8>
define void @int8x3Plus1(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr3 = getelementptr i8, ptr %ptr, i64 3

  %l0 = load <3 x i8>, ptr %ptr0, align 4
  %l1 = load i8, ptr %ptr3, align 1

  store <3 x i8> <i8 0, i8 0, i8 0>, ptr %ptr0, align 4
  store i8 0, ptr %ptr3, align 1

  ret void
}
