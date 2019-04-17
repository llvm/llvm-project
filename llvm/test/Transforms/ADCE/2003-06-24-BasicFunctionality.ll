; RUN: opt < %s -adce -S | FileCheck %s
; RUN: opt < %s -adce -adce-remove-loops -S | FileCheck %s

define void @dead_test8(i32* %data.1, i32 %idx.1) {
entry:
        %tmp.1 = load i32, i32* %data.1              ; <i32> [#uses=2]
        %tmp.41 = icmp sgt i32 %tmp.1, 0                ; <i1> [#uses=1]
        br i1 %tmp.41, label %no_exit.preheader, label %return

no_exit.preheader:              ; preds = %entry
        %tmp.11 = getelementptr i32, i32* %data.1, i64 1             ; <i32*> [#uses=1]
        %tmp.22-idxcast = sext i32 %idx.1 to i64                ; <i64> [#uses=1]
        %tmp.28 = getelementptr i32, i32* %data.1, i64 %tmp.22-idxcast               ; <i32*> [#uses=1]
        br label %no_exit

no_exit:                ; preds = %endif, %no_exit.preheader
        %k.1 = phi i32 [ %k.0, %endif ], [ 0, %no_exit.preheader ]              ; <i32> [#uses=3]
        %i.0 = phi i32 [ %inc.1, %endif ], [ 0, %no_exit.preheader ]            ; <i32> [#uses=1]
        %tmp.12 = load i32, i32* %tmp.11             ; <i32> [#uses=1]
        %tmp.14 = sub i32 0, %tmp.12            ; <i32> [#uses=1]
; CHECK-NOT: %tmp.161
        %tmp.161 = icmp ne i32 %k.1, %tmp.14            ; <i1> [#uses=1]
; CHECK: br label %then
        br i1 %tmp.161, label %then, label %else

then:           ; preds = %no_exit
        %inc.0 = add i32 %k.1, 1                ; <i32> [#uses=1]
        br label %endif

else:           ; preds = %no_exit
        %dec = add i32 %k.1, -1         ; <i32> [#uses=1]
        br label %endif

endif:          ; preds = %else, %then
        %k.0 = phi i32 [ %dec, %else ], [ %inc.0, %then ]               ; <i32> [#uses=1]
        store i32 2, i32* %tmp.28
        %inc.1 = add i32 %i.0, 1                ; <i32> [#uses=2]
        %tmp.4 = icmp slt i32 %inc.1, %tmp.1            ; <i1> [#uses=1]
        br i1 %tmp.4, label %no_exit, label %return

return:         ; preds = %endif, %entry
        ret void
}

