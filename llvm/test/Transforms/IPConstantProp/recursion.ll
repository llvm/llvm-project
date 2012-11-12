; RUN: opt < %s -ipconstprop -deadargelim -S | not grep %X
define internal i32 @foo(i32 %X) {
        %Y = call i32 @foo( i32 %X )            ; <i32> [#uses=1]
        %Z = add i32 %Y, 1              ; <i32> [#uses=1]
        ret i32 %Z
}

define void @bar() {
        call i32 @foo( i32 17 )         ; <i32>:1 [#uses=0]
        ret void
}

