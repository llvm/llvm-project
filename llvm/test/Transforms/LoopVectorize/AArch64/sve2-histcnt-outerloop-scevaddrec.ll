; RUN: opt < %s -mattr=+sve2 -passes=loop-vectorize,instcombine -enable-histogram-loop-vectorization -sve-gather-overhead=2 -sve-scatter-overhead=2 -debug-only=loop-vectorize --disable-output -S 2>&1 | FileCheck %s
; REQUIRES: asserts

target triple = "aarch64-unknown-linux-gnu"

;; Make sure we don't detect a histogram operation if the index address is
;; loop invariant.
; CHECK: LV: Checking for a histogram on:   store i32 %inc, ptr %gep.bucket, align 4
; CHECK-NEXT: LV: Can't vectorize due to memory conflicts
; CHECK-NEXT: LV: Not vectorizing: Cannot prove legality.

define void @outer_loop_scevaddrec(ptr noalias %buckets, ptr readonly %indices, i64 %N, i64 %M) {
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  %gep.indices = getelementptr inbounds i32, ptr %indices, i64 %outer.iv
  br label %inner.body

inner.body:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.body ]
  %l.idx = load i32, ptr %gep.indices, align 4
  %idxprom1 = zext i32 %l.idx to i64
  %gep.bucket = getelementptr inbounds i32, ptr %buckets, i64 %idxprom1
  %l.bucket = load i32, ptr %gep.bucket, align 4
  %inc = add nsw i32 %l.bucket, 1
  store i32 %inc, ptr %gep.bucket, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %outer.latch, label %inner.body

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.exitcond = icmp eq i64 %outer.iv.next, %M
  br i1 %outer.exitcond, label %outer.exit, label %outer.header

outer.exit:
  ret void
}
