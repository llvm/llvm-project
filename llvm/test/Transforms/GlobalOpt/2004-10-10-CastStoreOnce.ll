; RUN: opt < %s -passes=globalopt

@V = global float 1.200000e+01          ; <ptr> [#uses=1]
@G = internal global ptr null          ; <ptr> [#uses=2]

define i32 @user() {
        %P = load ptr, ptr @G              ; <ptr> [#uses=1]
        %Q = load i32, ptr %P               ; <i32> [#uses=1]
        ret i32 %Q
}

define void @setter() {
        store ptr @V, ptr @G
        ret void
}

