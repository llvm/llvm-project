; RUN: llc < %s

@g = global i32 0               ; <ptr> [#uses=1]

define i32 @main() {
        %h = load i32, ptr @g               ; <i32> [#uses=1]
        ret i32 %h
}
