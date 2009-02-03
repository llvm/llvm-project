; RUN: llvm-as < %s | opt -argpromotion | llvm-dis | \
; RUN:   grep {load i32\\* %A}

define internal i32 @callee(i1 %C, i32* %P) {
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 17

F:              ; preds = %0
        %X = load i32* %P               ; <i32> [#uses=1]
        ret i32 %X
}

define i32 @foo() {
        %A = alloca i32         ; <i32*> [#uses=2]
        store i32 17, i32* %A
        %X = call i32 @callee( i1 false, i32* %A )              ; <i32> [#uses=1]
        ret i32 %X
}

