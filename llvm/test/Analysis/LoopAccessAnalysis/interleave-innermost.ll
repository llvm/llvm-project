; RUN: opt -loop-vectorize -force-vector-interleave=1 -S < %s | FileCheck %s
; CHECK-LABEL: TestFoo
; CHECK-NOT: %wide.vec

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define void @TestFoo(i1 %X, i1 %Y) {
bb:
  br label %.loopexit5.outer

.loopexit5.outer:
  br label %.lr.ph12

.loopexit:
  br i1 %X, label %.loopexit5.outer, label %.lr.ph12

.lr.ph12:
  %f.110 = phi i32* [ %tmp1, %.loopexit ], [ null, %.loopexit5.outer ]
  %tmp1 = getelementptr inbounds i32, i32* %f.110, i64 -2
  br i1 %Y, label %bb4, label %.loopexit

bb4:
  %j.27 = phi i32 [ 0, %.lr.ph12 ], [ %tmp7, %bb4 ]
  %tmp5 = load i32, i32* %f.110, align 4
  %tmp7 = add nsw i32 %j.27, 1
  %exitcond = icmp eq i32 %tmp7, 0
  br i1 %exitcond, label %.loopexit, label %bb4
}
