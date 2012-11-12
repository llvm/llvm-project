; RUN: opt < %s -simplifycfg -phi-node-folding-threshold=2 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test1(i32 %a, i32 %b, i32 %c) nounwind  {
; CHECK: @test1
entry:
        %tmp1 = icmp eq i32 %b, 0
        br i1 %tmp1, label %bb1, label %bb3

bb1:            ; preds = %entry
	%tmp2 = icmp sgt i32 %c, 1
	br i1 %tmp2, label %bb2, label %bb3
; CHECK: bb1:
; CHECK-NEXT: icmp sgt i32 %c, 1
; CHECK-NEXT: add i32 %a, 1
; CHECK-NEXT: select i1 %tmp2, i32 %tmp3, i32 %a
; CHECK-NEXT: br label %bb3

bb2:		; preds = bb1
	%tmp3 = add i32 %a, 1
	br label %bb3

bb3:		; preds = %bb2, %entry
	%tmp4 = phi i32 [ %b, %entry ], [ %a, %bb1 ], [ %tmp3, %bb2 ]
        %tmp5 = sub i32 %tmp4, 1
	ret i32 %tmp5
}

declare i8 @llvm.cttz.i8(i8, i1)

define i8 @test2(i8 %a) {
; CHECK: @test2
  br i1 undef, label %bb_true, label %bb_false
bb_true:
  %b = tail call i8 @llvm.cttz.i8(i8 %a, i1 false)
  br label %join
bb_false:
  br label %join
join:
  %c = phi i8 [%b, %bb_true], [%a, %bb_false]
; CHECK: select
  ret i8 %c
}

