; RUN: opt < %s -argpromotion -S | \
; RUN:    not grep "load i32* null"

define internal i32 @callee(i1 %C, i32* %P) {
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 17

F:              ; preds = %0
        %X = load i32* %P               ; <i32> [#uses=1]
        ret i32 %X
}

define i32 @foo() {
        %X = call i32 @callee( i1 true, i32* null )             ; <i32> [#uses=1]
        ret i32 %X
}

