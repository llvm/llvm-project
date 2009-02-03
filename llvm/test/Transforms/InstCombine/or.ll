; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v xor | not grep {or }
; END.

define i32 @test1(i32 %A) {
        %B = or i32 %A, 0               ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i32 %A) {
        %B = or i32 %A, -1              ; <i32> [#uses=1]
        ret i32 %B
}

define i8 @test2a(i8 %A) {
        %B = or i8 %A, -1               ; <i8> [#uses=1]
        ret i8 %B
}

define i1 @test3(i1 %A) {
        %B = or i1 %A, false            ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test4(i1 %A) {
        %B = or i1 %A, true             ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test5(i1 %A) {
        %B = or i1 %A, %A               ; <i1> [#uses=1]
        ret i1 %B
}

define i32 @test6(i32 %A) {
        %B = or i32 %A, %A              ; <i32> [#uses=1]
        ret i32 %B
}

; A | ~A == -1
define i32 @test7(i32 %A) {
        %NotA = xor i32 -1, %A          ; <i32> [#uses=1]
        %B = or i32 %A, %NotA           ; <i32> [#uses=1]
        ret i32 %B
}

define i8 @test8(i8 %A) {
        %B = or i8 %A, -2               ; <i8> [#uses=1]
        %C = or i8 %B, 1                ; <i8> [#uses=1]
        ret i8 %C
}

; Test that (A|c1)|(B|c2) == (A|B)|(c1|c2)
define i8 @test9(i8 %A, i8 %B) {
        %C = or i8 %A, 1                ; <i8> [#uses=1]
        %D = or i8 %B, -2               ; <i8> [#uses=1]
        %E = or i8 %C, %D               ; <i8> [#uses=1]
        ret i8 %E
}

define i8 @test10(i8 %A) {
        %B = or i8 %A, 1                ; <i8> [#uses=1]
        %C = and i8 %B, -2              ; <i8> [#uses=1]
        ; (X & C1) | C2 --> (X | C2) & (C1|C2)
        %D = or i8 %C, -2               ; <i8> [#uses=1]
        ret i8 %D
}

define i8 @test11(i8 %A) {
        %B = or i8 %A, -2               ; <i8> [#uses=1]
        %C = xor i8 %B, 13              ; <i8> [#uses=1]
        ; (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
        %D = or i8 %C, 1                ; <i8> [#uses=1]
        %E = xor i8 %D, 12              ; <i8> [#uses=1]
        ret i8 %E
}

define i32 @test12(i32 %A) {
        ; Should be eliminated
        %B = or i32 %A, 4               ; <i32> [#uses=1]
        %C = and i32 %B, 8              ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test13(i32 %A) {
        %B = or i32 %A, 12              ; <i32> [#uses=1]
        ; Always equal to 8
        %C = and i32 %B, 8              ; <i32> [#uses=1]
        ret i32 %C
}

define i1 @test14(i32 %A, i32 %B) {
        %C1 = icmp ult i32 %A, %B               ; <i1> [#uses=1]
        %C2 = icmp ugt i32 %A, %B               ; <i1> [#uses=1]
        ; (A < B) | (A > B) === A != B
        %D = or i1 %C1, %C2             ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test15(i32 %A, i32 %B) {
        %C1 = icmp ult i32 %A, %B               ; <i1> [#uses=1]
        %C2 = icmp eq i32 %A, %B                ; <i1> [#uses=1]
        ; (A < B) | (A == B) === A <= B
        %D = or i1 %C1, %C2             ; <i1> [#uses=1]
        ret i1 %D
}

define i32 @test16(i32 %A) {
        %B = and i32 %A, 1              ; <i32> [#uses=1]
        ; -2 = ~1
        %C = and i32 %A, -2             ; <i32> [#uses=1]
        ; %D = and int %B, -1 == %B
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test17(i32 %A) {
        %B = and i32 %A, 1              ; <i32> [#uses=1]
        %C = and i32 %A, 4              ; <i32> [#uses=1]
        ; %D = and int %B, 5
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i1 @test18(i32 %A) {
        %B = icmp sge i32 %A, 100               ; <i1> [#uses=1]
        %C = icmp slt i32 %A, 50                ; <i1> [#uses=1]
        ;; (A-50) >u 50
        %D = or i1 %B, %C               ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test19(i32 %A) {
        %B = icmp eq i32 %A, 50         ; <i1> [#uses=1]
        %C = icmp eq i32 %A, 51         ; <i1> [#uses=1]
        ;; (A-50) < 2
        %D = or i1 %B, %C               ; <i1> [#uses=1]
        ret i1 %D
}

define i32 @test20(i32 %x) {
        %y = and i32 %x, 123            ; <i32> [#uses=1]
        %z = or i32 %y, %x              ; <i32> [#uses=1]
        ret i32 %z
}

define i32 @test21(i32 %tmp.1) {
        %tmp.1.mask1 = add i32 %tmp.1, 2                ; <i32> [#uses=1]
        %tmp.3 = and i32 %tmp.1.mask1, -2               ; <i32> [#uses=1]
        %tmp.5 = and i32 %tmp.1, 1              ; <i32> [#uses=1]
        ;; add tmp.1, 2
        %tmp.6 = or i32 %tmp.5, %tmp.3          ; <i32> [#uses=1]
        ret i32 %tmp.6
}

define i32 @test22(i32 %B) {
        %ELIM41 = and i32 %B, 1         ; <i32> [#uses=1]
        %ELIM7 = and i32 %B, -2         ; <i32> [#uses=1]
        %ELIM5 = or i32 %ELIM41, %ELIM7         ; <i32> [#uses=1]
        ret i32 %ELIM5
}

define i16 @test23(i16 %A) {
        %B = lshr i16 %A, 1             ; <i16> [#uses=1]
        ;; fold or into xor
        %C = or i16 %B, -32768          ; <i16> [#uses=1]
        %D = xor i16 %C, 8193           ; <i16> [#uses=1]
        ret i16 %D
}
