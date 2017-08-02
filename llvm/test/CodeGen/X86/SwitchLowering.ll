; RUN: llc < %s -mtriple=i686-- | grep cmp | count 1
; PR964

define i8* @FindChar(i8* %CurPtr) {
entry:
        br label %bb

bb:             ; preds = %bb, %entry
        %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]          ; <i32> [#uses=3]
        %CurPtr_addr.0.rec = bitcast i32 %indvar to i32         ; <i32> [#uses=1]
        %gep.upgrd.1 = zext i32 %indvar to i64          ; <i64> [#uses=1]
        %CurPtr_addr.0 = getelementptr i8, i8* %CurPtr, i64 %gep.upgrd.1            ; <i8*> [#uses=1]
        %tmp = load i8, i8* %CurPtr_addr.0          ; <i8> [#uses=3]
        %tmp2.rec = add i32 %CurPtr_addr.0.rec, 1               ; <i32> [#uses=1]
        %tmp2 = getelementptr i8, i8* %CurPtr, i32 %tmp2.rec                ; <i8*> [#uses=1]
        %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=1]
        switch i8 %tmp, label %bb [
                 i8 0, label %bb7
                 i8 120, label %bb7
        ]

bb7:            ; preds = %bb, %bb
        tail call void @foo( i8 %tmp )
        ret i8* %tmp2
}

declare void @foo(i8)

