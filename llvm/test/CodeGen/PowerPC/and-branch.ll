; RUN: llvm-as < %s | llc -march=ppc32 | not grep mfcr

define void @foo(i32 %X, i32 %Y, i32 %Z) {
entry:
        %tmp = icmp eq i32 %X, 0                ; <i1> [#uses=1]
        %tmp3 = icmp slt i32 %Y, 5              ; <i1> [#uses=1]
        %tmp4 = and i1 %tmp3, %tmp              ; <i1> [#uses=1]
        br i1 %tmp4, label %cond_true, label %UnifiedReturnBlock
cond_true:              ; preds = %entry
        %tmp5 = tail call i32 (...)* @bar( )            ; <i32> [#uses=0]
        ret void
UnifiedReturnBlock:             ; preds = %entry
        ret void
}

declare i32 @bar(...)

