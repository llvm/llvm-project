; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep IMPLICIT_DEF

define void @foo(i64 %X) {
entry:
        %tmp1 = and i64 %X, 3           ; <i64> [#uses=1]
        %tmp = icmp sgt i64 %tmp1, 2            ; <i1> [#uses=1]
        br i1 %tmp, label %UnifiedReturnBlock, label %cond_true
cond_true:              ; preds = %entry
        %tmp.upgrd.1 = tail call i32 (...) @bar( )             ; <i32> [#uses=0]
        ret void
UnifiedReturnBlock:             ; preds = %entry
        ret void
}

declare i32 @bar(...)

