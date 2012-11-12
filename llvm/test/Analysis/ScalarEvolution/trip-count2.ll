; RUN: opt < %s -analyze -scalar-evolution | \
; RUN:   grep "backedge-taken count is 4"
; PR1101

@A = weak global [1000 x i32] zeroinitializer, align 32         


define void @test(i32 %N) {
entry:
        br label %bb3

bb:             ; preds = %bb3
        %tmp = getelementptr [1000 x i32]* @A, i32 0, i32 %i.0          ; <i32*> [#uses=1]
        store i32 123, i32* %tmp
        %tmp4 = mul i32 %i.0, 4         ; <i32> [#uses=1]
        %tmp5 = or i32 %tmp4, 1
        %tmp61 = xor i32 %tmp5, -2147483648
        %tmp6 = trunc i32 %tmp61 to i16
        %tmp71 = shl i16 %tmp6, 2
        %tmp7 = zext i16 %tmp71 to i32
        %tmp2 = add i32 %tmp7, %i.0
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
