; Test that upgrading zext/sext attributes to zeroext and signext
; works correctly.
; PR1553
; RUN: llvm-as < %s > /dev/null

define i32 @bar() {
        %t = call i8 @foo( i8 10 sext ) zext
        %x = zext i8 %t to i32
        ret i32 %x
}

declare i8 @foo(i8 signext ) zeroext
