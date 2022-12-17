; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=2 -S -mtriple=xcore | FileCheck %s

target datalayout = "e-p:32:32:32-a0:0:32-n32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f16:16:32-f32:32:32-f64:32:32"
target triple = "xcore"
; The xcore target has no vector registers, so loop should not be vectorized.
;CHECK-LABEL: @f(
;CHECK: entry:
;CHECK-NOT: vector.body
;CHECK-NEXT: br label %do.body
define void @f(i8* nocapture %ptr, i32 %len) {
entry:
  br label %do.body
do.body:
  %ptr.addr.0 = phi i8* [ %ptr, %entry ], [ %incdec.ptr, %do.body ]
  %len.addr.0 = phi i32 [ %len, %entry ], [ %dec, %do.body ]
  %incdec.ptr = getelementptr inbounds i8, i8* %ptr.addr.0, i32 1
  store i8 0, i8* %ptr.addr.0, align 1
  %dec = add nsw i32 %len.addr.0, -1
  %tobool = icmp eq i32 %len.addr.0, 0
  br i1 %tobool, label %do.end, label %do.body
do.end:
  ret void
}
