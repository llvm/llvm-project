; This testcase caused the combiner to go into an infinite loop, moving the 
; cast back and forth, changing the seteq to operate on int vs uint and back.

; RUN: opt < %s -instcombine -disable-output

define i1 @test(i32 %A, i32 %B) {
        %C = sub i32 0, %A              ; <i32> [#uses=1]
        %Cc = bitcast i32 %C to i32             ; <i32> [#uses=1]
        %D = sub i32 0, %B              ; <i32> [#uses=1]
        %E = icmp eq i32 %Cc, %D                ; <i1> [#uses=1]
        ret i1 %E
}

