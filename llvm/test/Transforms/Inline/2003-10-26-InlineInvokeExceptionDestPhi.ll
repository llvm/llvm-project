; The inliner is breaking inlining invoke instructions where there is a PHI 
; node in the exception destination, and the inlined function contains an 
; unwind instruction.

; RUN: llvm-as < %s | opt -inline -disable-output

define linkonce void @foo() {
        unwind
}

define i32 @test() {
BB1:
        invoke void @foo( )
                        to label %Cont unwind label %Cont

Cont:           ; preds = %BB1, %BB1
        %A = phi i32 [ 0, %BB1 ], [ 0, %BB1 ]           ; <i32> [#uses=1]
        ret i32 %A
}

