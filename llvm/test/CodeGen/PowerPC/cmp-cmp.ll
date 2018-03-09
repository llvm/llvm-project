; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep mfcr

define void @test(i64 %X) {
        %tmp1 = and i64 %X, 3           ; <i64> [#uses=1]
        %tmp = icmp sgt i64 %tmp1, 2            ; <i1> [#uses=1]
        br i1 %tmp, label %UnifiedReturnBlock, label %cond_true
cond_true:              ; preds = %0
        tail call void @test( i64 0 )
        ret void
UnifiedReturnBlock:             ; preds = %0
        ret void
}

