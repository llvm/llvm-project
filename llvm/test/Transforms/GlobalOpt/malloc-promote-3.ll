; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep malloc

@G = internal global i32* null          ; <i32**> [#uses=4]

define void @init() {
        %P = malloc i32, i32 100                ; <i32*> [#uses=1]
        store i32* %P, i32** @G
        %GV = load i32** @G             ; <i32*> [#uses=1]
        %GVe = getelementptr i32* %GV, i32 40           ; <i32*> [#uses=1]
        store i32 20, i32* %GVe
        ret void
}

define i32 @get() {
        %GV = load i32** @G             ; <i32*> [#uses=1]
        %GVe = getelementptr i32* %GV, i32 40           ; <i32*> [#uses=1]
        %V = load i32* %GVe             ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @check() {
        %GV = load i32** @G             ; <i32*> [#uses=1]
        %V = icmp eq i32* %GV, null             ; <i1> [#uses=1]
        ret i1 %V
}

