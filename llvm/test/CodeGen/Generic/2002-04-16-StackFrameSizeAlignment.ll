; RUN: llc < %s

; Compiling this file produces:
; Sparc.cpp:91: failed assertion `(offset - OFFSET) % getStackFrameSizeAlignment() == 0'
;
declare i32 @SIM(ptr, ptr, i32, i32, i32, ptr, i32, i32, i32)

define void @foo() {
bb0:
        %V = alloca [256 x i32], i32 256                ; <ptr> [#uses=1]
        call i32 @SIM( ptr null, ptr null, i32 0, i32 0, i32 0, ptr %V, i32 0, i32 0, i32 2 )          ; <i32>:0 [#uses=0]
        ret void
}

