; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; We don't need to vectorize this.  Just make sure it doesn't crash.

; CHECK-LABEL: @int24x2
; CHECK: load i24
; CHECK: load i24
; CHECK: store i24
; CHECK: store i24
define void @int24x2(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i24, ptr %ptr, i64 0
  %ptr1 = getelementptr i24, ptr %ptr, i64 1

  %l0 = load i24, ptr %ptr0, align 4
  %l1 = load i24, ptr %ptr1, align 1

  store i24 %l1, ptr %ptr0, align 4
  store i24 %l0, ptr %ptr1, align 1

  ret void
}
