; RUN: opt -S -irce -irce-print-changed-loops=true < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; REQUIRES: asserts
; XFAIL: *

define void @test() {
bb:
  br label %outer_header

outer_latch:                                      ; preds = %inner_exit
  %tmp = or i32 %tmp5, 1
  %tmp2 = add nuw nsw i32 %tmp5, 1
  %tmp3 = icmp eq i32 %tmp8, 0
  br i1 %tmp3, label %ret2, label %outer_header

outer_header:                                     ; preds = %outer_latch, %bb
  %tmp5 = phi i32 [ 0, %bb ], [ %tmp2, %outer_latch ]
  br label %inner_header

inner_exit:                                       ; preds = %inner_header
  %tmp12.lcssa = phi i32 [ %tmp12, %inner_header ]
  %tmp7 = or i32 %tmp12.lcssa, %tmp5
  %tmp8 = add nuw i32 %tmp12.lcssa, %tmp5
  %tmp9 = icmp ult i32 %tmp5, 0
  br i1 %tmp9, label %outer_latch, label %ret1

ret1:                                             ; preds = %inner_exit
  ret void

inner_header:                                     ; preds = %inner_header, %outer_header
  %tmp12 = phi i32 [ %tmp14, %inner_header ], [ 0, %outer_header ]
  %tmp13 = or i32 %tmp12, 1
  %tmp14 = add nuw nsw i32 %tmp12, 1
  br i1 true, label %inner_exit, label %inner_header

ret2:                                             ; preds = %outer_latch
  ret void
}
