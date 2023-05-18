; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s
; PR1597

; CHECK: Loop %bb: backedge-taken count is (-1 + (-1 * %x) + %y)

define i32 @f(i32 %x, i32 %y) {
entry:
        %tmp63 = icmp ult i32 %x, %y            ; <i1> [#uses=1]
        br i1 %tmp63, label %bb.preheader, label %bb8

bb.preheader:           ; preds = %entry
        br label %bb

bb:             ; preds = %bb3, %bb.preheader
        %x_addr.0 = phi i32 [ %tmp2, %bb3 ], [ %x, %bb.preheader ]              ; <i32> [#uses=1]
        %tmp2 = add i32 %x_addr.0, 1            ; <i32> [#uses=3]
        br label %bb3

bb3:            ; preds = %bb
        %tmp6 = icmp ult i32 %tmp2, %y          ; <i1> [#uses=1]
        br i1 %tmp6, label %bb, label %bb8.loopexit

bb8.loopexit:           ; preds = %bb3
        br label %bb8

bb8:            ; preds = %bb8.loopexit, %entry
        %x_addr.1 = phi i32 [ %x, %entry ], [ %tmp2, %bb8.loopexit ]            ; <i32> [#uses=1]
        br label %return

return:         ; preds = %bb8
        ret i32 %x_addr.1
}
