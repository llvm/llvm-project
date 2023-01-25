; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -O3 | FileCheck %s

@.str = private constant [4 x i8] c"%d\0A\00", align 4 ; <ptr> [#uses=1]

define internal fastcc i32 @Callee(i32 %i) nounwind {
entry:
; CHECK-LABEL: Callee:
; CHECK: push
; CHECK: mov r4, sp
; CHECK: sub.w [[R12:r[0-9]+]], r4, #1000
; CHECK: mov sp, [[R12]]
  %0 = icmp eq i32 %i, 0                          ; <i1> [#uses=1]
  br i1 %0, label %bb2, label %bb

bb:                                               ; preds = %entry
  %1 = alloca [1000 x i8], align 4                ; <ptr> [#uses=1]
  %.sub = getelementptr inbounds [1000 x i8], ptr %1, i32 0, i32 0 ; <ptr> [#uses=2]
  %2 = call i32 (ptr, i32, i32, ptr, ...) @__sprintf_chk(ptr %.sub, i32 0, i32 1000, ptr @.str, i32 %i) nounwind ; <i32> [#uses=0]
  %3 = load i8, ptr %.sub, align 4                    ; <i8> [#uses=1]
  %4 = sext i8 %3 to i32                          ; <i32> [#uses=1]
  ret i32 %4

bb2:                                              ; preds = %entry
; Must restore sp from fp here. Make sure not to leave sp in a temporarily invalid
; state though. rdar://8465407
; CHECK-NOT: mov sp, r7
; CHECK: sub.w r4, r7, #8
; CHECK: mov sp, r4
; CHECK: pop
  ret i32 0
}

declare i32 @__sprintf_chk(ptr, i32, i32, ptr, ...) nounwind

define i32 @main() nounwind {
; CHECK-LABEL: main:
bb.nph:
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %0 = phi i32 [ 0, %bb.nph ], [ %3, %bb ]        ; <i32> [#uses=2]
  %j.01 = phi i32 [ 0, %bb.nph ], [ %2, %bb ]     ; <i32> [#uses=1]
  %1 = tail call fastcc i32 @Callee(i32 %0) nounwind ; <i32> [#uses=1]
  %2 = add nsw i32 %1, %j.01                      ; <i32> [#uses=2]
  %3 = add nsw i32 %0, 1                          ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %3, 10000               ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb
; No need to restore sp from fp here.
; CHECK: printf
; CHECK-NOT: mov sp, r7
; CHECK-NOT: sub sp, #12
; CHECK: pop
  %4 = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %2) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(ptr nocapture, ...) nounwind
