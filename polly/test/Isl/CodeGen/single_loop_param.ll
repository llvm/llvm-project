; RUN: opt %loadPolly %defaultOpts -polly-ast -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

@A = common global [1024 x i32] zeroinitializer, align 16 ; <[1024 x i32]*> [#uses=3]

define void @bar(i64 %n) nounwind {
bb:
  fence seq_cst
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp, %bb3 ]       ; <i64> [#uses=3]
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %i.0 ; <i32*> [#uses=1]
  %exitcond = icmp ne i64 %i.0, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb4

bb2:                                              ; preds = %bb1
  store i32 1, i32* %scevgep
  br label %bb3

bb3:                                              ; preds = %bb2
  %tmp = add nsw i64 %i.0, 1                      ; <i64> [#uses=1]
  br label %bb1

bb4:                                              ; preds = %bb1
  fence seq_cst
  ret void
}

define i32 @main() nounwind {
bb:
  call void @llvm.memset.p0i8.i64(i8* bitcast ([1024 x i32]* @A to i8*), i8 0, i64 4096, i32 1, i1 false)
  call void @bar(i64 1024)
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %indvar = phi i64 [ %indvar.next, %bb7 ], [ 0, %bb ] ; <i64> [#uses=3]
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar ; <i32*> [#uses=1]
  %i.0 = trunc i64 %indvar to i32                 ; <i32> [#uses=1]
  %tmp = icmp slt i32 %i.0, 1024                  ; <i1> [#uses=1]
  br i1 %tmp, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp3 = load i32* %scevgep                      ; <i32> [#uses=1]
  %tmp4 = icmp ne i32 %tmp3, 1                    ; <i1> [#uses=1]
  br i1 %tmp4, label %bb5, label %bb6

bb5:                                              ; preds = %bb2
  br label %bb9

bb6:                                              ; preds = %bb2
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %bb1

bb8:                                              ; preds = %bb1
  br label %bb9

bb9:                                              ; preds = %bb8, %bb5
  %.0 = phi i32 [ 1, %bb5 ], [ 0, %bb8 ]          ; <i32> [#uses=1]
  ret i32 %.0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

; CHECK: for (int c1 = 0; c1 < n; c1 += 1)
; CHECK:   Stmt_bb2(c1);

