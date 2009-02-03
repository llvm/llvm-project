; This test case was reduced from MultiSource/Applications/hbd. It makes sure
; that folding doesn't happen in case a zext is applied where a sext should have
; been when a setcc is used with two casts.
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep {br i1 false}
; END.

define i32 @bug(i8 %inbuff) {
entry:
        %tmp = bitcast i8 %inbuff to i8         ; <i8> [#uses=1]
        %tmp.upgrd.1 = sext i8 %tmp to i32              ; <i32> [#uses=3]
        %tmp.upgrd.2 = icmp eq i32 %tmp.upgrd.1, 1              ; <i1> [#uses=1]
        br i1 %tmp.upgrd.2, label %cond_true, label %cond_next

cond_true:              ; preds = %entry
        br label %bb

cond_next:              ; preds = %entry
        %tmp3 = icmp eq i32 %tmp.upgrd.1, -1            ; <i1> [#uses=1]
        br i1 %tmp3, label %cond_true4, label %cond_next5

cond_true4:             ; preds = %cond_next
        br label %bb

cond_next5:             ; preds = %cond_next
        %tmp7 = icmp sgt i32 %tmp.upgrd.1, 1            ; <i1> [#uses=1]
        br i1 %tmp7, label %cond_true8, label %cond_false

cond_true8:             ; preds = %cond_next5
        br label %cond_next9

cond_false:             ; preds = %cond_next5
        br label %cond_next9

cond_next9:             ; preds = %cond_false, %cond_true8
        %iftmp.1.0 = phi i32 [ 42, %cond_true8 ], [ 23, %cond_false ]           ; <i32> [#uses=1]
        br label %return

bb:             ; preds = %cond_true4, %cond_true
        br label %return

return:         ; preds = %bb, %cond_next9
        %retval.0 = phi i32 [ 17, %bb ], [ %iftmp.1.0, %cond_next9 ]            ; <i32> [#uses=1]
        ret i32 %retval.0
}

