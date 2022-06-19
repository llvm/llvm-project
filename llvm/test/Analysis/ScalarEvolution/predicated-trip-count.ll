; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = weak global [1000 x i32] zeroinitializer, align 32

; The resulting predicate is i16 {0,+,1} <nssw>, meanining
; that the resulting backedge expression will be valid for:
;   (1 + (-1 smax %M)) <= MAX_INT16
;
; At the limit condition for M (MAX_INT16 - 1) we have in the
; last iteration:
;    i0 <- MAX_INT16
;    i0.ext <- MAX_INT16
;
; and therefore no wrapping happend for i0 or i0.ext
; throughout the execution of the loop. The resulting predicated
; backedge taken count is correct.

; CHECK: Classifying expressions for: @test1
; CHECK: %i.0.ext = sext i16 %i.0 to i32
; CHECK-NEXT:  -->  (sext i16 {0,+,1}<%bb3> to i32)
; CHECK:      Loop %bb3: Unpredictable backedge-taken count.
; CHECK-NEXT: Loop %bb3: Unpredictable max backedge-taken count.
; CHECK-NEXT: Loop %bb3: Predicated backedge-taken count is (1 + (-1 smax %M))
; CHECK-NEXT: Predicates:
; CHECK-NEXT:    {0,+,1}<%bb3> Added Flags: <nssw>
define void @test1(i32 %N, i32 %M) {
entry:
        br label %bb3

bb:             ; preds = %bb3
        %tmp = getelementptr [1000 x i32], [1000 x i32]* @A, i32 0, i16 %i.0          ; <i32*> [#uses=1]
        store i32 123, i32* %tmp
        %tmp2 = add i16 %i.0, 1         ; <i32> [#uses=1]
        br label %bb3

bb3:            ; preds = %bb, %entry
        %i.0 = phi i16 [ 0, %entry ], [ %tmp2, %bb ]            ; <i32> [#uses=3]
        %i.0.ext = sext i16 %i.0 to i32
        %tmp3 = icmp sle i32 %i.0.ext, %M          ; <i1> [#uses=1]
        br i1 %tmp3, label %bb, label %bb5

bb5:            ; preds = %bb3
        br label %return

return:         ; preds = %bb5
        ret void
}

; The predicated backedge taken count is:
;    (2 + (zext i16 %Start to i32) + ((-2 + (-1 * (sext i16 %Start to i32)))
;                                     smax (-1 + (-1 * %M)))
;    )

; -1 + (-1 * %M) <= (-2 + (-1 * (sext i16 %Start to i32))
; The predicated backedge taken count is 0.
; From the IR, this is correct since we will bail out at the
; first iteration.


; * -1 + (-1 * %M) > (-2 + (-1 * (sext i16 %Start to i32))
; or: %M < 1 + (sext i16 %Start to i32)
;
; The predicated backedge taken count is 1 + (zext i16 %Start to i32) - %M
;
; If %M >= MIN_INT + 1, this predicated backedge taken count would be correct (even
; without predicates). However, for %M < MIN_INT this would be an infinite loop.
; In these cases, the {%Start,+,-1} <nusw> predicate would be false, as the
; final value of the expression {%Start,+,-1} expression (%M - 1) would not be
; representable as an i16.

; There is also a limit case here where the value of %M is MIN_INT. In this case
; we still have an infinite loop, since icmp sge %x, MIN_INT will always return
; true.

; CHECK: Classifying expressions for: @test2

; CHECK:      %i.0.ext = sext i16 %i.0 to i32
; CHECK-NEXT:    -->  (sext i16 {%Start,+,-1}<%bb3> to i32)
; CHECK:       Loop %bb3: Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %bb3: Unpredictable max backedge-taken count.
; CHECK-NEXT:  Loop %bb3: Predicated backedge-taken count is (1 + (sext i16 %Start to i32) + (-1 * ((1 + (sext i16 %Start to i32))<nsw> smin %M)))
; CHECK-NEXT:  Predicates:
; CHECK-NEXT:    {%Start,+,-1}<%bb3> Added Flags: <nssw>

define void @test2(i32 %N, i32 %M, i16 %Start) {
entry:
        br label %bb3

bb:             ; preds = %bb3
        %tmp = getelementptr [1000 x i32], [1000 x i32]* @A, i32 0, i16 %i.0          ; <i32*> [#uses=1]
        store i32 123, i32* %tmp
        %tmp2 = sub i16 %i.0, 1         ; <i32> [#uses=1]
        br label %bb3

bb3:            ; preds = %bb, %entry
        %i.0 = phi i16 [ %Start, %entry ], [ %tmp2, %bb ]            ; <i32> [#uses=3]
        %i.0.ext = sext i16 %i.0 to i32
        %tmp3 = icmp sge i32 %i.0.ext, %M          ; <i1> [#uses=1]
        br i1 %tmp3, label %bb, label %bb5

bb5:            ; preds = %bb3
        br label %return

return:         ; preds = %bb5
        ret void
}

