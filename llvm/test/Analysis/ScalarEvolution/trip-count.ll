; RUN: opt < %s -analyze -scalar-evolution \
; RUN:   -scalar-evolution-max-iterations=0 | grep "backedge-taken count is 10000"
; PR1101

@A = weak global [1000 x i32] zeroinitializer, align 32         


define void @test(i32 %N) {
entry:
        br label %bb3

bb:             ; preds = %bb3
        %tmp = getelementptr [1000 x i32]* @A, i32 0, i32 %i.0          ; <i32*> [#uses=1]
        store i32 123, i32* %tmp
        %tmp2 = add i32 %i.0, 1         ; <i32> [#uses=1]
        br label %bb3

bb3:            ; preds = %bb, %entry
        %i.0 = phi i32 [ 0, %entry ], [ %tmp2, %bb ]            ; <i32> [#uses=3]
        %tmp3 = icmp sle i32 %i.0, 9999          ; <i1> [#uses=1]
        br i1 %tmp3, label %bb, label %bb5

bb5:            ; preds = %bb3
        br label %return

return:         ; preds = %bb5
        ret void
}
