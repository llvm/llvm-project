; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; Vectorize and emit valid code (Issue #54896).

define void @int8x3a2(ptr nocapture align 2 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2

  %l0 = load i8, ptr %ptr0, align 2
  %l1 = load i8, ptr %ptr1, align 1
  %l2 = load i8, ptr %ptr2, align 2

  store i8 %l2, ptr %ptr0, align 2
  store i8 %l1, ptr %ptr1, align 1
  store i8 %l0, ptr %ptr2, align 2

  ret void

; CHECK-LABEL: @int8x3a2
; CHECK-DAG: load <2 x i8>
; CHECK-DAG: load i8
; CHECK-DAG: store <2 x i8>
; CHECK-DAG: store i8
}

define void @int8x3a4(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2

  %l0 = load i8, ptr %ptr0, align 4
  %l1 = load i8, ptr %ptr1, align 1
  %l2 = load i8, ptr %ptr2, align 2

  store i8 %l2, ptr %ptr0, align 2
  store i8 %l1, ptr %ptr1, align 1
  store i8 %l0, ptr %ptr2, align 4

  ret void

; CHECK-LABEL: @int8x3a4
; CHECK: load <2 x i8>
; CHECK: load i8
; CHECK: store <2 x i8>
; CHECK: store i8
}

define void @int8x12a4(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2
  %ptr3 = getelementptr i8, ptr %ptr, i64 3
  %ptr4 = getelementptr i8, ptr %ptr, i64 4
  %ptr5 = getelementptr i8, ptr %ptr, i64 5
  %ptr6 = getelementptr i8, ptr %ptr, i64 6
  %ptr7 = getelementptr i8, ptr %ptr, i64 7
  %ptr8 = getelementptr i8, ptr %ptr, i64 8
  %ptr9 = getelementptr i8, ptr %ptr, i64 9
  %ptra = getelementptr i8, ptr %ptr, i64 10
  %ptrb = getelementptr i8, ptr %ptr, i64 11

  %l0 = load i8, ptr %ptr0, align 4
  %l1 = load i8, ptr %ptr1, align 1
  %l2 = load i8, ptr %ptr2, align 2
  %l3 = load i8, ptr %ptr3, align 1
  %l4 = load i8, ptr %ptr4, align 4
  %l5 = load i8, ptr %ptr5, align 1
  %l6 = load i8, ptr %ptr6, align 2
  %l7 = load i8, ptr %ptr7, align 1
  %l8 = load i8, ptr %ptr8, align 4
  %l9 = load i8, ptr %ptr9, align 1
  %la = load i8, ptr %ptra, align 2
  %lb = load i8, ptr %ptrb, align 1

  store i8 %lb, ptr %ptr0, align 4
  store i8 %la, ptr %ptr1, align 1
  store i8 %l9, ptr %ptr2, align 2
  store i8 %l8, ptr %ptr3, align 1
  store i8 %l7, ptr %ptr4, align 4
  store i8 %l6, ptr %ptr5, align 1
  store i8 %l5, ptr %ptr6, align 2
  store i8 %l4, ptr %ptr7, align 1
  store i8 %l3, ptr %ptr8, align 4
  store i8 %l2, ptr %ptr9, align 1
  store i8 %l1, ptr %ptra, align 2
  store i8 %l0, ptr %ptrb, align 1

  ret void

; CHECK-LABEL: @int8x12a4
; CHECK: load <4 x i8>
; CHECK: load <4 x i8>
; CHECK: load <4 x i8>
; CHECK: store <4 x i8>
; CHECK: store <4 x i8>
; CHECK: store <4 x i8>
}


define void @int8x16a4(ptr nocapture align 4 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2
  %ptr3 = getelementptr i8, ptr %ptr, i64 3
  %ptr4 = getelementptr i8, ptr %ptr, i64 4
  %ptr5 = getelementptr i8, ptr %ptr, i64 5
  %ptr6 = getelementptr i8, ptr %ptr, i64 6
  %ptr7 = getelementptr i8, ptr %ptr, i64 7
  %ptr8 = getelementptr i8, ptr %ptr, i64 8
  %ptr9 = getelementptr i8, ptr %ptr, i64 9
  %ptra = getelementptr i8, ptr %ptr, i64 10
  %ptrb = getelementptr i8, ptr %ptr, i64 11
  %ptrc = getelementptr i8, ptr %ptr, i64 12
  %ptrd = getelementptr i8, ptr %ptr, i64 13
  %ptre = getelementptr i8, ptr %ptr, i64 14
  %ptrf = getelementptr i8, ptr %ptr, i64 15

  %l0 = load i8, ptr %ptr0, align 4
  %l1 = load i8, ptr %ptr1, align 1
  %l2 = load i8, ptr %ptr2, align 2
  %l3 = load i8, ptr %ptr3, align 1
  %l4 = load i8, ptr %ptr4, align 4
  %l5 = load i8, ptr %ptr5, align 1
  %l6 = load i8, ptr %ptr6, align 2
  %l7 = load i8, ptr %ptr7, align 1
  %l8 = load i8, ptr %ptr8, align 4
  %l9 = load i8, ptr %ptr9, align 1
  %la = load i8, ptr %ptra, align 2
  %lb = load i8, ptr %ptrb, align 1
  %lc = load i8, ptr %ptrc, align 4
  %ld = load i8, ptr %ptrd, align 1
  %le = load i8, ptr %ptre, align 2
  %lf = load i8, ptr %ptrf, align 1

  store i8 %lf, ptr %ptrc, align 4
  store i8 %le, ptr %ptrd, align 1
  store i8 %ld, ptr %ptre, align 2
  store i8 %lc, ptr %ptrf, align 1
  store i8 %lb, ptr %ptr0, align 4
  store i8 %la, ptr %ptr1, align 1
  store i8 %l9, ptr %ptr2, align 2
  store i8 %l8, ptr %ptr3, align 1
  store i8 %l7, ptr %ptr4, align 4
  store i8 %l6, ptr %ptr5, align 1
  store i8 %l5, ptr %ptr6, align 2
  store i8 %l4, ptr %ptr7, align 1
  store i8 %l3, ptr %ptr8, align 4
  store i8 %l2, ptr %ptr9, align 1
  store i8 %l1, ptr %ptra, align 2
  store i8 %l0, ptr %ptrb, align 1

  ret void

; CHECK-LABEL: @int8x16a4
; CHECK: load <4 x i8>
; CHECK: load <4 x i8>
; CHECK: load <4 x i8>
; CHECK: load <4 x i8>
; CHECK: store <4 x i8>
; CHECK: store <4 x i8>
; CHECK: store <4 x i8>
; CHECK: store <4 x i8>
}

define void @int8x8a8(ptr nocapture align 8 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2
  %ptr3 = getelementptr i8, ptr %ptr, i64 3
  %ptr4 = getelementptr i8, ptr %ptr, i64 4
  %ptr5 = getelementptr i8, ptr %ptr, i64 5
  %ptr6 = getelementptr i8, ptr %ptr, i64 6
  %ptr7 = getelementptr i8, ptr %ptr, i64 7

  %l0 = load i8, ptr %ptr0, align 8
  %l1 = load i8, ptr %ptr1, align 1
  %l2 = load i8, ptr %ptr2, align 2
  %l3 = load i8, ptr %ptr3, align 1
  %l4 = load i8, ptr %ptr4, align 4
  %l5 = load i8, ptr %ptr5, align 1
  %l6 = load i8, ptr %ptr6, align 2
  %l7 = load i8, ptr %ptr7, align 1

  store i8 %l7, ptr %ptr0, align 8
  store i8 %l6, ptr %ptr1, align 1
  store i8 %l5, ptr %ptr2, align 2
  store i8 %l4, ptr %ptr3, align 1
  store i8 %l3, ptr %ptr4, align 4
  store i8 %l2, ptr %ptr5, align 1
  store i8 %l1, ptr %ptr6, align 2
  store i8 %l0, ptr %ptr7, align 1

  ret void

; CHECK-LABEL: @int8x8a8
; CHECK: load <8 x i8>
; CHECK: store <8 x i8>
}

define void @int8x12a8(ptr nocapture align 8 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2
  %ptr3 = getelementptr i8, ptr %ptr, i64 3
  %ptr4 = getelementptr i8, ptr %ptr, i64 4
  %ptr5 = getelementptr i8, ptr %ptr, i64 5
  %ptr6 = getelementptr i8, ptr %ptr, i64 6
  %ptr7 = getelementptr i8, ptr %ptr, i64 7
  %ptr8 = getelementptr i8, ptr %ptr, i64 8
  %ptr9 = getelementptr i8, ptr %ptr, i64 9
  %ptra = getelementptr i8, ptr %ptr, i64 10
  %ptrb = getelementptr i8, ptr %ptr, i64 11

  %l0 = load i8, ptr %ptr0, align 8
  %l1 = load i8, ptr %ptr1, align 1
  %l2 = load i8, ptr %ptr2, align 2
  %l3 = load i8, ptr %ptr3, align 1
  %l4 = load i8, ptr %ptr4, align 4
  %l5 = load i8, ptr %ptr5, align 1
  %l6 = load i8, ptr %ptr6, align 2
  %l7 = load i8, ptr %ptr7, align 1
  %l8 = load i8, ptr %ptr8, align 8
  %l9 = load i8, ptr %ptr9, align 1
  %la = load i8, ptr %ptra, align 2
  %lb = load i8, ptr %ptrb, align 1

  store i8 %lb, ptr %ptr0, align 8
  store i8 %la, ptr %ptr1, align 1
  store i8 %l9, ptr %ptr2, align 2
  store i8 %l8, ptr %ptr3, align 1
  store i8 %l7, ptr %ptr4, align 4
  store i8 %l6, ptr %ptr5, align 1
  store i8 %l5, ptr %ptr6, align 2
  store i8 %l4, ptr %ptr7, align 1
  store i8 %l3, ptr %ptr8, align 8
  store i8 %l2, ptr %ptr9, align 1
  store i8 %l1, ptr %ptra, align 2
  store i8 %l0, ptr %ptrb, align 1

  ret void

; CHECK-LABEL: @int8x12a8
; CHECK-DAG: load <8 x i8>
; CHECK-DAG: load <4 x i8>
; CHECK-DAG: store <8 x i8>
; CHECK-DAG: store <4 x i8>
}


define void @int8x16a8(ptr nocapture align 8 %ptr) {
  %ptr0 = getelementptr i8, ptr %ptr, i64 0
  %ptr1 = getelementptr i8, ptr %ptr, i64 1
  %ptr2 = getelementptr i8, ptr %ptr, i64 2
  %ptr3 = getelementptr i8, ptr %ptr, i64 3
  %ptr4 = getelementptr i8, ptr %ptr, i64 4
  %ptr5 = getelementptr i8, ptr %ptr, i64 5
  %ptr6 = getelementptr i8, ptr %ptr, i64 6
  %ptr7 = getelementptr i8, ptr %ptr, i64 7
  %ptr8 = getelementptr i8, ptr %ptr, i64 8
  %ptr9 = getelementptr i8, ptr %ptr, i64 9
  %ptra = getelementptr i8, ptr %ptr, i64 10
  %ptrb = getelementptr i8, ptr %ptr, i64 11
  %ptrc = getelementptr i8, ptr %ptr, i64 12
  %ptrd = getelementptr i8, ptr %ptr, i64 13
  %ptre = getelementptr i8, ptr %ptr, i64 14
  %ptrf = getelementptr i8, ptr %ptr, i64 15

  %l0 = load i8, ptr %ptr0, align 8
  %l1 = load i8, ptr %ptr1, align 1
  %l2 = load i8, ptr %ptr2, align 2
  %l3 = load i8, ptr %ptr3, align 1
  %l4 = load i8, ptr %ptr4, align 4
  %l5 = load i8, ptr %ptr5, align 1
  %l6 = load i8, ptr %ptr6, align 2
  %l7 = load i8, ptr %ptr7, align 1
  %l8 = load i8, ptr %ptr8, align 8
  %l9 = load i8, ptr %ptr9, align 1
  %la = load i8, ptr %ptra, align 2
  %lb = load i8, ptr %ptrb, align 1
  %lc = load i8, ptr %ptrc, align 4
  %ld = load i8, ptr %ptrd, align 1
  %le = load i8, ptr %ptre, align 2
  %lf = load i8, ptr %ptrf, align 1

  store i8 %lf, ptr %ptr0, align 8
  store i8 %le, ptr %ptr1, align 1
  store i8 %ld, ptr %ptr2, align 2
  store i8 %lc, ptr %ptr3, align 1
  store i8 %lb, ptr %ptr4, align 4
  store i8 %la, ptr %ptr5, align 1
  store i8 %l9, ptr %ptr6, align 2
  store i8 %l8, ptr %ptr7, align 1
  store i8 %l7, ptr %ptr8, align 8
  store i8 %l6, ptr %ptr9, align 1
  store i8 %l5, ptr %ptra, align 2
  store i8 %l4, ptr %ptrb, align 1
  store i8 %l3, ptr %ptrc, align 4
  store i8 %l2, ptr %ptrd, align 1
  store i8 %l1, ptr %ptre, align 2
  store i8 %l0, ptr %ptrf, align 1

  ret void

; CHECK-LABEL: @int8x16a8
; CHECK: load <8 x i8>
; CHECK: load <8 x i8>
; CHECK: store <8 x i8>
; CHECK: store <8 x i8>
}
