; PR1553
; RUN: llvm-as < %s > /dev/null
define void @bar() {
        %t = call i8 @foo( i8 10 )
        zext i8 %t to i32
        ret void
}

declare i8 @foo(i8)
