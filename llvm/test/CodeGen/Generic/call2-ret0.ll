; RUN: llvm-as < %s | llc

define i32 @bar(i32 %x) {
        ret i32 0
}

define i32 @foo(i32 %x) {
        %q = call i32 @bar( i32 1 )             ; <i32> [#uses=1]
        ret i32 %q
}

define i32 @main() {
        %r = call i32 @foo( i32 2 )             ; <i32> [#uses=1]
        ret i32 %r
}

