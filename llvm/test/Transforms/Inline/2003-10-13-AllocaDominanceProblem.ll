; RUN: opt < %s -passes=inline -disable-output

define i32 @reload() {
reloadentry:
        br label %A

A:              ; preds = %reloadentry
        call void @callee( )
        ret i32 0
}

define internal void @callee() {
entry:
        %X = alloca i8, i32 0           ; <ptr> [#uses=0]
        %Y = bitcast i32 0 to i32               ; <i32> [#uses=1]
        %Z = alloca i8, i32 %Y          ; <ptr> [#uses=0]
        ret void
}

