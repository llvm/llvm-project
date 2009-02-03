; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep global

@G = internal global i32* null          ; <i32**> [#uses=3]

define void @init() {
        %P = malloc i32         ; <i32*> [#uses=1]
        store i32* %P, i32** @G
        %GV = load i32** @G             ; <i32*> [#uses=1]
        store i32 0, i32* %GV
        ret void
}

define i32 @get() {
        %GV = load i32** @G             ; <i32*> [#uses=1]
        %V = load i32* %GV              ; <i32> [#uses=1]
        ret i32 %V
}

