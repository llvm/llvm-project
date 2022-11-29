; RUN: opt < %s -passes=globalopt

@V = global float 1.200000e+01          ; <float*> [#uses=1]
@G = internal global i32* null          ; <i32**> [#uses=2]

define i32 @user() {
        %P = load i32*, i32** @G              ; <i32*> [#uses=1]
        %Q = load i32, i32* %P               ; <i32> [#uses=1]
        ret i32 %Q
}

define void @setter() {
        %Vi = bitcast float* @V to i32*         ; <i32*> [#uses=1]
        store i32* %Vi, i32** @G
        ret void
}

