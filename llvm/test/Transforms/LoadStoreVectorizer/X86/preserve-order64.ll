; RUN: opt -mtriple=x86_64-unknown-linux-gnu -load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

%struct.buffer_t = type { i64, i8* }
%struct.nested.buffer = type { %struct.buffer_t, %struct.buffer_t }

; Check an i64 and i8* get vectorized, and that the two accesses
; (load into buff.val and store to buff.p) preserve their order.
; Vectorized loads should be inserted at the position of the first load,
; and instructions which were between the first and last load should be
; reordered preserving their relative order inasmuch as possible.

; CHECK-LABEL: @preserve_order_64(
; CHECK: load <2 x i64>
; CHECK: %buff.val = load i8
; CHECK: store i8 0
define void @preserve_order_64(%struct.buffer_t* noalias %buff) #0 {
entry:
  %tmp1 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i64 0, i32 1
  %buff.p = load i8*, i8** %tmp1
  %buff.val = load i8, i8* %buff.p
  store i8 0, i8* %buff.p, align 8
  %tmp0 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i64 0, i32 0
  %buff.int = load i64, i64* %tmp0, align 16
  ret void
}

; Check reordering recurses correctly.

; CHECK-LABEL: @transitive_reorder(
; CHECK: load <2 x i64>
; CHECK: %buff.val = load i8
; CHECK: store i8 0
define void @transitive_reorder(%struct.buffer_t* noalias %buff, %struct.nested.buffer* noalias %nest) #0 {
entry:
  %nest0_0 = getelementptr inbounds %struct.nested.buffer, %struct.nested.buffer* %nest, i64 0, i32 0
  %tmp1 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %nest0_0, i64 0, i32 1
  %buff.p = load i8*, i8** %tmp1
  %buff.val = load i8, i8* %buff.p
  store i8 0, i8* %buff.p, align 8
  %nest1_0 = getelementptr inbounds %struct.nested.buffer, %struct.nested.buffer* %nest, i64 0, i32 0
  %tmp0 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %nest1_0, i64 0, i32 0
  %buff.int = load i64, i64* %tmp0, align 16
  ret void
}

; Check for no vectorization over phi node

; CHECK-LABEL: @no_vect_phi(
; CHECK: load i8*
; CHECK: load i8
; CHECK: store i8 0
; CHECK: load i64
define void @no_vect_phi(i32* noalias %ptr, %struct.buffer_t* noalias %buff) {
entry:
  %tmp1 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i64 0, i32 1
  %buff.p = load i8*, i8** %tmp1
  %buff.val = load i8, i8* %buff.p
  store i8 0, i8* %buff.p, align 8
  br label %"for something"

"for something":
  %index = phi i64 [ 0, %entry ], [ %index.next, %"for something" ]

  %tmp0 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i64 0, i32 0
  %buff.int = load i64, i64* %tmp0, align 16

  %index.next = add i64 %index, 8
  %cmp_res = icmp eq i64 %index.next, 8
  br i1 %cmp_res, label %ending, label %"for something"

ending:
  ret void
}

attributes #0 = { nounwind }
