; RUN: opt < %s -passes=gvn -enable-load-pre -S | FileCheck %s
; CHECK-NOT: pre1
; GVN load pre was hoisting the loads at %13 and %16 up to bb4.outer.
; This is invalid as it bypasses the check for %m.0.ph==null in bb4.
; ModuleID = 'mbuf.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
  %struct.mbuf = type { ptr, ptr, i32, ptr, i16, i16, i32 }

define void @m_adj(ptr %mp, i32 %req_len) nounwind optsize {
entry:
  %0 = icmp eq ptr %mp, null    ; <i1> [#uses=1]
  %1 = icmp slt i32 %req_len, 0   ; <i1> [#uses=1]
  %or.cond = or i1 %1, %0   ; <i1> [#uses=1]
  br i1 %or.cond, label %return, label %bb4.preheader

bb4.preheader:    ; preds = %entry
  br label %bb4.outer

bb2:    ; preds = %bb1
  %2 = sub i32 %len.0, %13   ; <i32> [#uses=1]
  %3 = getelementptr %struct.mbuf, ptr %m.0.ph, i32 0, i32 2    ; <ptr> [#uses=1]
  store i32 0, ptr %3, align 4
  %4 = getelementptr %struct.mbuf, ptr %m.0.ph, i32 0, i32 0    ; <ptr> [#uses=1]
  %5 = load ptr, ptr %4, align 4    ; <ptr> [#uses=1]
  br label %bb4.outer

bb4.outer:    ; preds = %bb4.preheader, %bb2
  %m.0.ph = phi ptr [ %5, %bb2 ], [ %mp, %bb4.preheader ]   ; <ptr> [#uses=7]
  %len.0.ph = phi i32 [ %2, %bb2 ], [ %req_len, %bb4.preheader ]    ; <i32> [#uses=1]
  %6 = icmp ne ptr %m.0.ph, null    ; <i1> [#uses=1]
  %7 = getelementptr %struct.mbuf, ptr %m.0.ph, i32 0, i32 2    ; <ptr> [#uses=1]
  %8 = getelementptr %struct.mbuf, ptr %m.0.ph, i32 0, i32 2   ; <ptr> [#uses=1]
  %9 = getelementptr %struct.mbuf, ptr %m.0.ph, i32 0, i32 3   ; <ptr> [#uses=1]
  %10 = getelementptr %struct.mbuf, ptr %m.0.ph, i32 0, i32 3   ; <ptr> [#uses=1]
  br label %bb4

bb4:    ; preds = %bb4.outer, %bb3
  %len.0 = phi i32 [ 0, %bb3 ], [ %len.0.ph, %bb4.outer ]   ; <i32> [#uses=6]
  %11 = icmp sgt i32 %len.0, 0    ; <i1> [#uses=1]
  %12 = and i1 %11, %6    ; <i1> [#uses=1]
  br i1 %12, label %bb1, label %bb7

bb1:    ; preds = %bb4
  %13 = load i32, ptr %7, align 4    ; <i32> [#uses=3]
  %14 = icmp sgt i32 %13, %len.0    ; <i1> [#uses=1]
  br i1 %14, label %bb3, label %bb2

bb3:    ; preds = %bb1
  %15 = sub i32 %13, %len.0    ; <i32> [#uses=1]
  store i32 %15, ptr %8, align 4
  %16 = load ptr, ptr %9, align 4    ; <ptr> [#uses=1]
  %17 = getelementptr i8, ptr %16, i32 %len.0   ; <ptr> [#uses=1]
  store ptr %17, ptr %10, align 4
  br label %bb4

bb7:    ; preds = %bb4
  %18 = getelementptr %struct.mbuf, ptr %mp, i32 0, i32 5   ; <ptr> [#uses=1]
  %19 = load i16, ptr %18, align 2    ; <i16> [#uses=1]
  %20 = zext i16 %19 to i32   ; <i32> [#uses=1]
  %21 = and i32 %20, 2    ; <i32> [#uses=1]
  %22 = icmp eq i32 %21, 0    ; <i1> [#uses=1]
  br i1 %22, label %return, label %bb8

bb8:    ; preds = %bb7
  %23 = sub i32 %req_len, %len.0    ; <i32> [#uses=1]
  %24 = getelementptr %struct.mbuf, ptr %mp, i32 0, i32 6   ; <ptr> [#uses=1]
  store i32 %23, ptr %24, align 4
  ret void

return:   ; preds = %bb7, %entry
  ret void
}
