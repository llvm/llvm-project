; RUN: llvm-as < %s | llc

; Compiling this file produces:
; Sparc.cpp:91: failed assertion `(offset - OFFSET) % getStackFrameSizeAlignment() == 0'
;
declare i32 @SIM(i8*, i8*, i32, i32, i32, [256 x i32]*, i32, i32, i32)

define void @foo() {
bb0:
        %V = alloca [256 x i32], i32 256                ; <[256 x i32]*> [#uses=1]
        call i32 @SIM( i8* null, i8* null, i32 0, i32 0, i32 0, [256 x i32]* %V, i32 0, i32 0, i32 2 )          ; <i32>:0 [#uses=0]
        ret void
}

