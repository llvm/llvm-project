; RUN: llvm-as < %s | llc -march=x86 | grep and | count 1

; The dag combiner should fold together (x&127)|(y&16711680) -> (x|y)&c1
; in this case.

define i32 @test6(i32 %x, i16 %y) {
        %tmp1 = zext i16 %y to i32              ; <i32> [#uses=1]
        %tmp2 = and i32 %tmp1, 127              ; <i32> [#uses=1]
        %tmp4 = shl i32 %x, 16          ; <i32> [#uses=1]
        %tmp5 = and i32 %tmp4, 16711680         ; <i32> [#uses=1]
        %tmp6 = or i32 %tmp2, %tmp5             ; <i32> [#uses=1]
        ret i32 %tmp6
}

