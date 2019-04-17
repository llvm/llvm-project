; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -dce -instcombine < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; PR16073

; Because we were caching value pointers across a function call that could RAUW
; we would generate an undefined value store below:
; SCEVExpander::expandCodeFor would change a value (the start value of an
; induction) that we cached in the induction variable list.

; CHECK-LABEL: @test_vh(
; CHECK-NOT: store <4 x i8> undef

define void @test_vh(i32* %ptr265, i32* %ptr266, i32 %sub267) {
entry:
  br label %loop

loop:
  %inc = phi i32 [ %sub267, %entry ], [ %add, %loop]
  %ext.inc = sext i32 %inc to i64
  %add.ptr265 = getelementptr inbounds i32, i32* %ptr265, i64 %ext.inc
  %add.ptr266 = getelementptr inbounds i32, i32* %ptr266, i64 %ext.inc
  %add = add i32 %inc, 9
  %cmp = icmp slt i32 %add, 140
  br i1 %cmp, label %block1, label %loop

block1:
  %sub267.lcssa = phi i32 [ %add, %loop ]
  %add.ptr266.lcssa = phi i32* [ %add.ptr266, %loop ]
  %add.ptr265.lcssa = phi i32* [ %add.ptr265, %loop ]
  %tmp29 = bitcast i32* %add.ptr265.lcssa to i8*
  %tmp30 = bitcast i32* %add.ptr266.lcssa to i8*
  br label %do.body272

do.body272:
  %row_width.5 = phi i32 [ %sub267.lcssa, %block1 ], [ %dec, %do.body272 ]
  %sp.4 = phi i8* [ %tmp30, %block1 ], [ %incdec.ptr273, %do.body272 ]
  %dp.addr.4 = phi i8* [ %tmp29, %block1 ], [ %incdec.ptr274, %do.body272 ]
  %incdec.ptr273 = getelementptr inbounds i8, i8* %sp.4, i64 1
  %tmp31 = load i8, i8* %sp.4, align 1
  %incdec.ptr274 = getelementptr inbounds i8, i8* %dp.addr.4, i64 1
  store i8 %tmp31, i8* %dp.addr.4, align 1
  %dec = add i32 %row_width.5, -1
  %cmp276 = icmp eq i32 %dec, 0
  br i1 %cmp276, label %loop.exit, label %do.body272

loop.exit:
  ret void
}
