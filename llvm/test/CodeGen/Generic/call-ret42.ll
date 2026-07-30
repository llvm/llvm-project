; RUN: llc < %s

define i32 @foo(i32 %x) {
        ret i32 42
}

define i32 @main() {
        %r = call i32 @foo( i32 15 )            ; <i32> [#uses=1]
        ret i32 %r
}
