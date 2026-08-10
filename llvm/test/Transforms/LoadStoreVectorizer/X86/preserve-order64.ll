; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=load-store-vectorizer -S -o - %s | FileCheck %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

%struct.buffer_t = type { i64, ptr }
%struct.nested.buffer = type { %struct.buffer_t, %struct.buffer_t }

; Check an i64 and ptr get vectorized, and that the two accesses
; (load into buff.val and store to buff.p) preserve their order.
; Vectorized loads should be inserted at the position of the first load,
; and instructions which were between the first and last load should be
; reordered preserving their relative order inasmuch as possible.

; CHECK-LABEL: @preserve_order_64(
; CHECK: load <2 x i64>
; CHECK: %buff.val = load i8
; CHECK: store i8 0
define void @preserve_order_64(ptr noalias %buff) #0 {
entry:
  %tmp1 = getelementptr inbounds %struct.buffer_t, ptr %buff, i64 0, i32 1
  %buff.p = load ptr, ptr %tmp1
  %buff.val = load i8, ptr %buff.p
  store i8 0, ptr %buff.p, align 8
  %buff.int = load i64, ptr %buff, align 16
  ret void
}

; Check reordering recurses correctly.

; CHECK-LABEL: @transitive_reorder(
; CHECK: load <2 x i64>
; CHECK: %buff.val = load i8
; CHECK: store i8 0
define void @transitive_reorder(ptr noalias %buff, ptr noalias %nest) #0 {
entry:
  %tmp1 = getelementptr inbounds %struct.buffer_t, ptr %nest, i64 0, i32 1
  %buff.p = load ptr, ptr %tmp1
  %buff.val = load i8, ptr %buff.p
  store i8 0, ptr %buff.p, align 8
  %buff.int = load i64, ptr %nest, align 16
  ret void
}

; Check for no vectorization over phi node

; CHECK-LABEL: @no_vect_phi(
; CHECK: load ptr
; CHECK: load i8
; CHECK: store i8 0
; CHECK: load i64
define void @no_vect_phi(ptr noalias %ptr, ptr noalias %buff) {
entry:
  %tmp1 = getelementptr inbounds %struct.buffer_t, ptr %buff, i64 0, i32 1
  %buff.p = load ptr, ptr %tmp1
  %buff.val = load i8, ptr %buff.p
  store i8 0, ptr %buff.p, align 8
  br label %"for something"

"for something":
  %index = phi i64 [ 0, %entry ], [ %index.next, %"for something" ]

  %buff.int = load i64, ptr %buff, align 16

  %index.next = add i64 %index, 8
  %cmp_res = icmp eq i64 %index.next, 8
  br i1 %cmp_res, label %ending, label %"for something"

ending:
  ret void
}

attributes #0 = { nounwind }
