; RUN: opt < %s -S -passes='loop(indvars),loop-unroll' -verify-dom-info | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; Make sure that this test doesn't crash because of dangling pointer in SCEV.
declare void @llvm.experimental.guard(i1, ...)

define void @test(i32* %p, i8** %p2, i64* %dest) {

; CHECK-LABEL: @test(

entry:
  br label %outer.loop

outer.loop:                                           ; preds = %outer.latch, %entry
  %local_2_ = phi i32 [ 10, %entry ], [ %tmp2, %outer.latch ]
  %tmp1 = icmp eq i32 %local_2_, 0
  br label %inner.loop

outer.latch:                                          ; preds = %inner.latch
  %tmp2 = add i32 %local_2_, 1
  br label %outer.loop

inner.loop:                                           ; preds = %inner.latch, %outer.loop
  %local_4_20 = phi i32 [ 7, %outer.loop ], [ %tmp15, %inner.latch ]
  %tmp6 = icmp eq i32 %local_4_20, 0
  call void (i1, ...) @llvm.experimental.guard(i1 %tmp6) [ "deopt"() ]
  br label %innermost.loop

store.block:                                          ; preds = %innermost.loop
  store i64 %tmp20, i64* %dest, align 8
  br i1 %tmp1, label %exit, label %inner.latch

inner.latch:                                   ; preds = %store.block
  %tmp15 = add i32 %local_4_20, 4
  %tmp16 = icmp sgt i32 %tmp15, 263
  br i1 %tmp16, label %outer.latch, label %inner.loop

innermost.loop:                                          ; preds = %innermost.loop, %inner.loop
  %tmp17 = phi i64 [ 0, %inner.loop ], [ %tmp20, %innermost.loop ]
  %local_6_51 = phi i32 [ 1, %inner.loop ], [ %tmp21, %innermost.loop ]
  %ze = zext i32 %local_6_51 to i64
  %tmp20 = add i64 %tmp17, %ze
  %tmp21 = add nuw nsw i32 %local_6_51, 1
  %tmp22 = icmp ugt i32 %local_6_51, 5
  br i1 %tmp22, label %store.block, label %innermost.loop

exit:                                           ; preds = %store.block
  ret void
}
