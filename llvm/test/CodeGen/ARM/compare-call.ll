; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2 | \
; RUN:   grep fcmpes

define void @test3(float* %glob, i32 %X) {
entry:
        %tmp = load float* %glob                ; <float> [#uses=1]
        %tmp2 = getelementptr float* %glob, i32 2               ; <float*> [#uses=1]
        %tmp3 = load float* %tmp2               ; <float> [#uses=1]
        %tmp.upgrd.1 = fcmp ogt float %tmp, %tmp3               ; <i1> [#uses=1]
        br i1 %tmp.upgrd.1, label %cond_true, label %UnifiedReturnBlock

cond_true:              ; preds = %entry
        %tmp.upgrd.2 = tail call i32 (...)* @bar( )             ; <i32> [#uses=0]
        ret void

UnifiedReturnBlock:             ; preds = %entry
        ret void
}

declare i32 @bar(...)
