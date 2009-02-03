; RUN: llvm-as < %s | opt -instcombine -mem2reg -simplifycfg | \
; RUN:   llvm-dis | grep -v store | not grep {i32 1}

; Test to make sure that instcombine does not accidentally propagate the load
; into the PHI, which would break the program.

define i32 @test(i1 %C) {
entry:
        %X = alloca i32         ; <i32*> [#uses=3]
        %X2 = alloca i32                ; <i32*> [#uses=2]
        store i32 1, i32* %X
        store i32 2, i32* %X2
        br i1 %C, label %cond_true.i, label %cond_continue.i

cond_true.i:            ; preds = %entry
        br label %cond_continue.i

cond_continue.i:                ; preds = %cond_true.i, %entry
        %mem_tmp.i.0 = phi i32* [ %X, %cond_true.i ], [ %X2, %entry ]           ; <i32*> [#uses=1]
        store i32 3, i32* %X
        %tmp.3 = load i32* %mem_tmp.i.0         ; <i32> [#uses=1]
        ret i32 %tmp.3
}


