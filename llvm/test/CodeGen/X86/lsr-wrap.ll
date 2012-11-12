; RUN: llc -march=x86-64 < %s | FileCheck %s

; LSR would like to use a single IV for both of these, however it's
; not safe due to wraparound.

; CHECK: addb  $-4, %
; CHECK: decw  %

@g_19 = common global i32 0                       ; <i32*> [#uses=2]

declare i32 @func_8(i8 zeroext) nounwind

declare i32 @func_3(i8 signext) nounwind

define void @func_1() nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %indvar = phi i16 [ 0, %entry ], [ %indvar.next, %bb ] ; <i16> [#uses=2]
  %tmp = sub i16 0, %indvar                       ; <i16> [#uses=1]
  %tmp27 = trunc i16 %tmp to i8                   ; <i8> [#uses=1]
  %tmp1 = load i32* @g_19, align 4                ; <i32> [#uses=2]
  %tmp2 = add i32 %tmp1, 1                        ; <i32> [#uses=1]
  store i32 %tmp2, i32* @g_19, align 4
  %tmp3 = trunc i32 %tmp1 to i8                   ; <i8> [#uses=1]
  %tmp4 = tail call i32 @func_8(i8 zeroext %tmp3) nounwind ; <i32> [#uses=0]
  %tmp5 = shl i8 %tmp27, 2                        ; <i8> [#uses=1]
  %tmp6 = add i8 %tmp5, -112                      ; <i8> [#uses=1]
  %tmp7 = tail call i32 @func_3(i8 signext %tmp6) nounwind ; <i32> [#uses=0]
  %indvar.next = add i16 %indvar, 1               ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %indvar.next, -28       ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb
  ret void
}
