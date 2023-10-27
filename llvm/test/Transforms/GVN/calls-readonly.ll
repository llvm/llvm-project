; RUN: opt < %s -passes=gvn -S | FileCheck %s
; Should delete the second call to strlen even though the intervening strchr call exists.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define ptr @test(ptr %P, ptr %Q, i32 %x, i32 %y) nounwind readonly {
entry:
  %0 = tail call i32 @strlen(ptr %P), !prof !0    ; <i32> [#uses=2]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %bb, label %bb1

bb:                                               ; preds = %entry
  %2 = sdiv i32 %x, %y                            ; <i32> [#uses=1]
  br label %bb1

bb1:                                              ; preds = %bb, %entry
  %x_addr.0 = phi i32 [ %2, %bb ], [ %x, %entry ] ; <i32> [#uses=1]
  %3 = tail call ptr @strchr(ptr %Q, i32 97)      ; <ptr> [#uses=1]
  %4 = tail call i32 @strlen(ptr %P) , !prof !1   ; <i32> [#uses=1]
  %5 = add i32 %x_addr.0, %0                      ; <i32> [#uses=1]
  %.sum = sub i32 %5, %4                          ; <i32> [#uses=1]
  %6 = getelementptr i8, ptr %3, i32 %.sum            ; <ptr> [#uses=1]
  ret ptr %6
}

; CHECK: define ptr @test(ptr %P, ptr %Q, i32 %x, i32 %y) #0 {
; CHECK: entry:
; CHECK-NEXT:   %0 = tail call i32 @strlen(ptr %P), !prof !0
; CHECK-NEXT:   %1 = icmp eq i32 %0, 0
; CHECK-NEXT:   br i1 %1, label %bb, label %bb1
; CHECK: bb:
; CHECK-NEXT:   %2 = sdiv i32 %x, %y
; CHECK-NEXT:   br label %bb1
; CHECK: bb1:
; CHECK-NEXT:   %x_addr.0 = phi i32 [ %2, %bb ], [ %x, %entry ]
; CHECK-NEXT:   %3 = tail call ptr @strchr(ptr %Q, i32 97)
; CHECK-NEXT:   %4 = add i32 %x_addr.0, %0
; CHECK-NEXT:   %5 = getelementptr i8, ptr %3, i32 %x_addr.0
; CHECK-NEXT:   ret ptr %5
; CHECK: }

declare i32 @strlen(ptr) nounwind readonly

declare ptr @strchr(ptr, i32) nounwind readonly

!0 = !{!"branch_weights", i32 95}
!1 = !{!"branch_weights", i32 95}
