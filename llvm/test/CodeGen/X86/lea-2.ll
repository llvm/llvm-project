; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep "lea	EAX, DWORD PTR \[... + 4\*... - 5\]"
; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | \
; RUN:   not grep add

define i32 @test1(i32 %A, i32 %B) {
        %tmp1 = shl i32 %A, 2           ; <i32> [#uses=1]
        %tmp3 = add i32 %B, -5          ; <i32> [#uses=1]
        %tmp4 = add i32 %tmp3, %tmp1            ; <i32> [#uses=1]
        ret i32 %tmp4
}


