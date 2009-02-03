; RUN: llvm-as < %s | llc -march=arm | grep {str r1, \\\[r.*, -r.*, lsl #2\}

define void @test(i32* %P, i32 %A, i32 %i) {
entry:
        icmp eq i32 %i, 0               ; <i1>:0 [#uses=1]
        br i1 %0, label %return, label %bb

bb:             ; preds = %bb, %entry
        %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]          ; <i32> [#uses=2]
        %i_addr.09.0 = sub i32 %i, %indvar              ; <i32> [#uses=1]
        %tmp2 = getelementptr i32* %P, i32 %i_addr.09.0         ; <i32*> [#uses=1]
        store i32 %A, i32* %tmp2
        %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
        icmp eq i32 %indvar.next, %i            ; <i1>:1 [#uses=1]
        br i1 %1, label %return, label %bb

return:         ; preds = %bb, %entry
        ret void
}

