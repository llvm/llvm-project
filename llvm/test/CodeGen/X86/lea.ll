; RUN: llvm-as < %s | llc -march=x86
; RUN: llvm-as < %s | llc -march=x86 | not grep orl

define i32 @test(i32 %x) {
        %tmp1 = shl i32 %x, 3           ; <i32> [#uses=1]
        %tmp2 = add i32 %tmp1, 7                ; <i32> [#uses=1]
        ret i32 %tmp2
}

