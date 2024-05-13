; RUN: opt < %s -passes=instcombine,mem2reg,simplifycfg -simplifycfg-require-and-preserve-domtree=1 | \
; RUN:   llvm-dis | grep -v store | not grep "i32 1"

; Test to make sure that instcombine does not accidentally propagate the load
; into the PHI, which would break the program.

define i32 @test(i1 %C) {
entry:
        %X = alloca i32         ; <ptr> [#uses=3]
        %X2 = alloca i32                ; <ptr> [#uses=2]
        store i32 1, ptr %X
        store i32 2, ptr %X2
        br i1 %C, label %cond_true.i, label %cond_continue.i

cond_true.i:            ; preds = %entry
        br label %cond_continue.i

cond_continue.i:                ; preds = %cond_true.i, %entry
        %mem_tmp.i.0 = phi ptr [ %X, %cond_true.i ], [ %X2, %entry ]           ; <ptr> [#uses=1]
        store i32 3, ptr %X
        %tmp.3 = load i32, ptr %mem_tmp.i.0         ; <i32> [#uses=1]
        ret i32 %tmp.3
}


