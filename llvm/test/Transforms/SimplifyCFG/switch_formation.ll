; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br
; END.

define i1 @_ZN4llvm11SetCondInst7classofEPKNS_11InstructionE({ i32, i32 }* %I) {
entry:
        %tmp.1.i = getelementptr { i32, i32 }* %I, i64 0, i32 1         ; <i32*> [#uses=1]
        %tmp.2.i = load i32* %tmp.1.i           ; <i32> [#uses=6]
        %tmp.2 = icmp eq i32 %tmp.2.i, 14               ; <i1> [#uses=1]
        br i1 %tmp.2, label %shortcirc_done.4, label %shortcirc_next.0
shortcirc_next.0:               ; preds = %entry
        %tmp.6 = icmp eq i32 %tmp.2.i, 15               ; <i1> [#uses=1]
        br i1 %tmp.6, label %shortcirc_done.4, label %shortcirc_next.1
shortcirc_next.1:               ; preds = %shortcirc_next.0
        %tmp.11 = icmp eq i32 %tmp.2.i, 16              ; <i1> [#uses=1]
        br i1 %tmp.11, label %shortcirc_done.4, label %shortcirc_next.2
shortcirc_next.2:               ; preds = %shortcirc_next.1
        %tmp.16 = icmp eq i32 %tmp.2.i, 17              ; <i1> [#uses=1]
        br i1 %tmp.16, label %shortcirc_done.4, label %shortcirc_next.3
shortcirc_next.3:               ; preds = %shortcirc_next.2
        %tmp.21 = icmp eq i32 %tmp.2.i, 18              ; <i1> [#uses=1]
        br i1 %tmp.21, label %shortcirc_done.4, label %shortcirc_next.4
shortcirc_next.4:               ; preds = %shortcirc_next.3
        %tmp.26 = icmp eq i32 %tmp.2.i, 19              ; <i1> [#uses=1]
        br label %UnifiedReturnBlock
shortcirc_done.4:               ; preds = %shortcirc_next.3, %shortcirc_next.2, %shortcirc_next.1, %shortcirc_next.0, %entry
        br label %UnifiedReturnBlock
UnifiedReturnBlock:             ; preds = %shortcirc_done.4, %shortcirc_next.4
        %UnifiedRetVal = phi i1 [ %tmp.26, %shortcirc_next.4 ], [ true, %shortcirc_done.4 ]             ; <i1> [#uses=1]
        ret i1 %UnifiedRetVal
}

