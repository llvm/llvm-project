; RUN: opt -S -passes=indvars < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@a = common global double* null, align 8
@b = common global double 0.000000e+00, align 8

define void @fn1(i32 %p1) {
; CHECK-LABEL: @fn1(
entry:
  %ld = load double*, double** @a, align 8
  br label %outer.loop

outer.loop:
  %iv.outer = phi i32 [ %p1, %entry ], [ %iv.outer.dec, %outer.be ]
  %idxprom = sext i32 %iv.outer to i64
  %arrayidx = getelementptr inbounds double, double* %ld, i64 %idxprom
  %arrayidx.bc = bitcast double* %arrayidx to i64*
  br label %inner.loop

inner.loop:
  %iv.inner = phi i32 [ %iv.outer, %outer.loop ], [ %iv.inner.dec, %inner.loop ]
  %ld.arr = load i64, i64* %arrayidx.bc, align 8
  store i64 %ld.arr, i64* bitcast (double* @b to i64*), align 8
  %iv.inner.dec = add nsw i32 %iv.inner, -1
  %cmp = icmp slt i32 %iv.outer, %iv.inner.dec
  br i1 %cmp, label %outer.be, label %inner.loop

outer.be:
  %iv.outer.dec = add nsw i32 %iv.outer, -1
  br label %outer.loop
}
