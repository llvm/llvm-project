; RUN: llc -march=x86 < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"
; PR7651

; CHECK: align
; CHECK: align
; CHECK: align
; CHECK: movl  $0, (%e
; CHECK-NEXT: addl  $4, %e
; CHECK-NEXT: decl  %e
; CHECK-NEXT: jne

%struct.anon = type { [72 x i32], i32 }

@mp2grad_ = external global %struct.anon

define void @chomp2g_setup_(i32 %n, i32 %m) nounwind {
entry:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %indvar11 = phi i32 [ %indvar.next12, %bb6 ], [ 0, %entry ] ; <i32> [#uses=2]
  %tmp21 = add i32 %indvar11, 1                   ; <i32> [#uses=1]
  %t = load i32* getelementptr inbounds (%struct.anon* @mp2grad_, i32 0, i32 1)
  %tmp15 = mul i32 %n, %t                      ; <i32> [#uses=1]
  %tmp16 = add i32 %tmp21, %tmp15                 ; <i32> [#uses=1]
  %tmp17 = shl i32 %tmp16, 3                      ; <i32> [#uses=1]
  %tmp18 = add i32 %tmp17, -8                     ; <i32> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %bb2, %bb2.preheader
  %indvar = phi i32 [ 0, %bb1 ], [ %indvar.next, %bb2 ] ; <i32> [#uses=2]
  %tmp19 = add i32 %tmp18, %indvar                ; <i32> [#uses=1]
  %scevgep = getelementptr %struct.anon* @mp2grad_, i32 0, i32 0, i32 %tmp19 ; <i32*> [#uses=1]
  store i32 0, i32* %scevgep
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=1]
  %c = icmp ne i32 %indvar.next, %m
  br i1 %c, label %bb2, label %bb6

bb6:                                              ; preds = %bb2, %bb1
  %indvar.next12 = add i32 %indvar11, 1           ; <i32> [#uses=1]
  br label %bb1
}
