; RUN: llc < %s -mtriple=thumbv7-apple-darwin -O3 -relocation-model=pic -frame-pointer=all -mcpu=cortex-a8
; rdar://8110842

declare arm_apcscc i32 @__maskrune(i32, i32)

define arm_apcscc i32 @strncmpic(i8* nocapture %s1, i8* nocapture %s2, i32 %n) nounwind {
entry:
  br i1 undef, label %bb11, label %bb19

bb11:                                             ; preds = %entry
  %0 = sext i8 0 to i32                           ; <i32> [#uses=1]
  br i1 undef, label %bb.i.i10, label %bb1.i.i11

bb.i.i10:                                         ; preds = %bb11
  br label %isupper144.exit12

bb1.i.i11:                                        ; preds = %bb11
  %1 = tail call arm_apcscc  i32 @__maskrune(i32 %0, i32 32768) nounwind ; <i32> [#uses=1]
  %2 = icmp ne i32 %1, 0                          ; <i1> [#uses=1]
  %3 = zext i1 %2 to i32                          ; <i32> [#uses=1]
  %.pre = load i8, i8* undef, align 1                 ; <i8> [#uses=1]
  br label %isupper144.exit12

isupper144.exit12:                                ; preds = %bb1.i.i11, %bb.i.i10
  %4 = phi i8 [ %.pre, %bb1.i.i11 ], [ 0, %bb.i.i10 ] ; <i8> [#uses=1]
  %5 = phi i32 [ %3, %bb1.i.i11 ], [ undef, %bb.i.i10 ] ; <i32> [#uses=1]
  %6 = icmp eq i32 %5, 0                          ; <i1> [#uses=1]
  %7 = sext i8 %4 to i32                          ; <i32> [#uses=1]
  %storemerge1 = select i1 %6, i32 %7, i32 undef  ; <i32> [#uses=1]
  %8 = sub nsw i32 %storemerge1, 0                ; <i32> [#uses=1]
  ret i32 %8

bb19:                                             ; preds = %entry
  ret i32 0
}
