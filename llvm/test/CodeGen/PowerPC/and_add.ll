; RUN: llvm-as < %s | llc -march=ppc32 -o %t -f
; RUN: grep slwi %t
; RUN: not grep addi %t
; RUN: not grep rlwinm %t

define i32 @test(i32 %A) {
        ;; shift
        %B = mul i32 %A, 8              ; <i32> [#uses=1]
        ;; dead, no demanded bits.
        %C = add i32 %B, 7              ; <i32> [#uses=1]
        ;; dead once add is gone.
        %D = and i32 %C, -8             ; <i32> [#uses=1]
        ret i32 %D
}

