; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep G

@G = internal global i32 17             ; <i32*> [#uses=3]

define void @foo() {
        %V = load i32* @G               ; <i32> [#uses=1]
        store i32 %V, i32* @G
        ret void
}

define i32 @bar() {
        %X = load i32* @G               ; <i32> [#uses=1]
        ret i32 %X
}

