; RUN: llc < %s -march=x86 | grep lea
; RUN: llc < %s -march=x86 | not grep add

define i32 @test(i32 %X, i32 %Y) {
	; Push the shl through the mul to allow an LEA to be formed, instead
        ; of using a shift and add separately.
        %tmp.2 = shl i32 %X, 1          ; <i32> [#uses=1]
        %tmp.3 = mul i32 %tmp.2, %Y             ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.3, %Y             ; <i32> [#uses=1]
        ret i32 %tmp.5
}

