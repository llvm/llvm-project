; RUN: opt < %s -globalopt -S | not grep global

@X = internal global i32 4              ; <i32*> [#uses=1]

define i32 @foo() {
        %V = load i32* @X               ; <i32> [#uses=1]
        ret i32 %V
}

