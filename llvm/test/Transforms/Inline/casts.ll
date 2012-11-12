; RUN: opt < %s -inline -S | grep "ret i32 1"
; ModuleID = 'short.opt.bc'

define i32 @testBool(i1 %X) {
        %tmp = zext i1 %X to i32                ; <i32> [#uses=1]
        ret i32 %tmp
}

define i32 @testByte(i8 %X) {
        %tmp = icmp ne i8 %X, 0         ; <i1> [#uses=1]
        %tmp.i = zext i1 %tmp to i32            ; <i32> [#uses=1]
        ret i32 %tmp.i
}

define i32 @main() {
        %rslt = call i32 @testByte( i8 123 )            ; <i32> [#uses=1]
        ret i32 %rslt
}

