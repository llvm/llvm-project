; This test makes sure that add instructions are properly eliminated.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v OK | not grep add

define i32 @test1(i32 %A) {
        %B = add i32 %A, 0              ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i32 %A) {
        %B = add i32 %A, 5              ; <i32> [#uses=1]
        %C = add i32 %B, -5             ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test3(i32 %A) {
        %B = add i32 %A, 5              ; <i32> [#uses=1]
        ;; This should get converted to an add
        %C = sub i32 %B, 5              ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test4(i32 %A, i32 %B) {
        %C = sub i32 0, %A              ; <i32> [#uses=1]
        ; D = B + -A = B - A
        %D = add i32 %B, %C             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test5(i32 %A, i32 %B) {
        %C = sub i32 0, %A              ; <i32> [#uses=1]
        ; D = -A + B = B - A
        %D = add i32 %C, %B             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test6(i32 %A) {
        %B = mul i32 7, %A              ; <i32> [#uses=1]
        ; C = 7*A+A == 8*A == A << 3
        %C = add i32 %B, %A             ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test7(i32 %A) {
        %B = mul i32 7, %A              ; <i32> [#uses=1]
        ; C = A+7*A == 8*A == A << 3
        %C = add i32 %A, %B             ; <i32> [#uses=1]
        ret i32 %C
}

; (A & C1)+(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
define i32 @test8(i32 %A, i32 %B) {
        %A1 = and i32 %A, 7             ; <i32> [#uses=1]
        %B1 = and i32 %B, 128           ; <i32> [#uses=1]
        %C = add i32 %A1, %B1           ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test9(i32 %A) {
        %B = shl i32 %A, 4              ; <i32> [#uses=2]
        ; === shl int %A, 5
        %C = add i32 %B, %B             ; <i32> [#uses=1]
        ret i32 %C
}

define i1 @test10(i8 %A, i8 %b) {
        %B = add i8 %A, %b              ; <i8> [#uses=1]
        ; === A != -b
        %c = icmp ne i8 %B, 0           ; <i1> [#uses=1]
        ret i1 %c
}

define i1 @test11(i8 %A) {
        %B = add i8 %A, -1              ; <i8> [#uses=1]
        ; === A != 1
        %c = icmp ne i8 %B, 0           ; <i1> [#uses=1]
        ret i1 %c
}

define i32 @test12(i32 %A, i32 %B) {
        ; Should be transformed into shl A, 1
         %C_OK = add i32 %B, %A          ; <i32> [#uses=1]
        br label %X

X:              ; preds = %0
        %D = add i32 %C_OK, %A          ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test13(i32 %A, i32 %B, i32 %C) {
        %D_OK = add i32 %A, %B          ; <i32> [#uses=1]
        %E_OK = add i32 %D_OK, %C               ; <i32> [#uses=1]
        ;; shl A, 1
        %F = add i32 %E_OK, %A          ; <i32> [#uses=1]
        ret i32 %F
}

define i32 @test14(i32 %offset, i32 %difference) {
        %tmp.2 = and i32 %difference, 3         ; <i32> [#uses=1]
        %tmp.3_OK = add i32 %tmp.2, %offset             ; <i32> [#uses=1]
        %tmp.5.mask = and i32 %difference, -4           ; <i32> [#uses=1]
        ; == add %offset, %difference
        %tmp.8 = add i32 %tmp.3_OK, %tmp.5.mask         ; <i32> [#uses=1]
        ret i32 %tmp.8
}

define i8 @test15(i8 %A) {
        ; Does not effect result
        %B = add i8 %A, -64             ; <i8> [#uses=1]
        ; Only one bit set
        %C = and i8 %B, 16              ; <i8> [#uses=1]
        ret i8 %C
}

define i8 @test16(i8 %A) {
        ; Turn this into a XOR
        %B = add i8 %A, 16              ; <i8> [#uses=1]
        ; Only one bit set
        %C = and i8 %B, 16              ; <i8> [#uses=1]
        ret i8 %C
}

define i32 @test17(i32 %A) {
        %B = xor i32 %A, -1             ; <i32> [#uses=1]
        ; == sub int 0, %A
        %C = add i32 %B, 1              ; <i32> [#uses=1]
        ret i32 %C
}

define i8 @test18(i8 %A) {
        %B = xor i8 %A, -1              ; <i8> [#uses=1]
        ; == sub ubyte 16, %A
        %C = add i8 %B, 17              ; <i8> [#uses=1]
        ret i8 %C
}

define i32 @test19(i1 %C) {
        %A = select i1 %C, i32 1000, i32 10             ; <i32> [#uses=1]
        %V = add i32 %A, 123            ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test20(i32 %x) {
        %tmp.2 = xor i32 %x, -2147483648                ; <i32> [#uses=1]
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add i32 %tmp.2, -2147483648            ; <i32> [#uses=1]
        ret i32 %tmp.4
}

define i1 @test21(i32 %x) {
        %t = add i32 %x, 4              ; <i32> [#uses=1]
        %y = icmp eq i32 %t, 123                ; <i1> [#uses=1]
        ret i1 %y
}

define i32 @test22(i32 %V) {
        %V2 = add i32 %V, 10            ; <i32> [#uses=1]
        switch i32 %V2, label %Default [
                 i32 20, label %Lab1
                 i32 30, label %Lab2
        ]

Default:                ; preds = %0
        ret i32 123

Lab1:           ; preds = %0
        ret i32 12312

Lab2:           ; preds = %0
        ret i32 1231231
}

define i32 @test23(i1 %C, i32 %a) {
entry:
        br i1 %C, label %endif, label %else

else:           ; preds = %entry
        br label %endif

endif:          ; preds = %else, %entry
        %b.0 = phi i32 [ 0, %entry ], [ 1, %else ]              ; <i32> [#uses=1]
        %tmp.4 = add i32 %b.0, 1                ; <i32> [#uses=1]
        ret i32 %tmp.4
}

define i32 @test24(i32 %A) {
        %B = add i32 %A, 1              ; <i32> [#uses=1]
        %C = shl i32 %B, 1              ; <i32> [#uses=1]
        %D = sub i32 %C, 2              ; <i32> [#uses=1]
        ret i32 %D
}

define i64 @test25(i64 %Y) {
        %tmp.4 = shl i64 %Y, 2          ; <i64> [#uses=1]
        %tmp.12 = shl i64 %Y, 2         ; <i64> [#uses=1]
        %tmp.8 = add i64 %tmp.4, %tmp.12                ; <i64> [#uses=1]
        ret i64 %tmp.8
}

define i32 @test26(i32 %A, i32 %B) {
        %C = add i32 %A, %B             ; <i32> [#uses=1]
        %D = sub i32 %C, %B             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test27(i1 %C, i32 %X, i32 %Y) {
        %A = add i32 %X, %Y             ; <i32> [#uses=1]
        %B = add i32 %Y, 123            ; <i32> [#uses=1]
        ;; Fold add through select.
        %C.upgrd.1 = select i1 %C, i32 %A, i32 %B               ; <i32> [#uses=1]
        %D = sub i32 %C.upgrd.1, %Y             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test28(i32 %X) {
        %Y = add i32 %X, 1234           ; <i32> [#uses=1]
        %Z = sub i32 42, %Y             ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test29(i32 %X, i32 %x) {
        %tmp.2 = sub i32 %X, %x         ; <i32> [#uses=2]
        %tmp.2.mask = and i32 %tmp.2, 63                ; <i32> [#uses=1]
        %tmp.6 = add i32 %tmp.2.mask, %x                ; <i32> [#uses=1]
        %tmp.7 = and i32 %tmp.6, 63             ; <i32> [#uses=1]
        %tmp.9 = and i32 %tmp.2, -64            ; <i32> [#uses=1]
        %tmp.10 = or i32 %tmp.7, %tmp.9         ; <i32> [#uses=1]
        ret i32 %tmp.10
}

define i64 @test30(i64 %x) {
        %tmp.2 = xor i64 %x, -9223372036854775808               ; <i64> [#uses=1]
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add i64 %tmp.2, -9223372036854775808           ; <i64> [#uses=1]
        ret i64 %tmp.4
}

define i32 @test31(i32 %A) {
        %B = add i32 %A, 4              ; <i32> [#uses=1]
        %C = mul i32 %B, 5              ; <i32> [#uses=1]
        %D = sub i32 %C, 20             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test32(i32 %A) {
        %B = add i32 %A, 4              ; <i32> [#uses=1]
        %C = shl i32 %B, 2              ; <i32> [#uses=1]
        %D = sub i32 %C, 16             ; <i32> [#uses=1]
        ret i32 %D
}

define i8 @test33(i8 %A) {
        %B = and i8 %A, -2              ; <i8> [#uses=1]
        %C = add i8 %B, 1               ; <i8> [#uses=1]
        ret i8 %C
}

define i8 @test34(i8 %A) {
        %B = add i8 %A, 64              ; <i8> [#uses=1]
        %C = and i8 %B, 12              ; <i8> [#uses=1]
        ret i8 %C
}

define i32 @test35(i32 %a) {
        %tmpnot = xor i32 %a, -1                ; <i32> [#uses=1]
        %tmp2 = add i32 %tmpnot, %a             ; <i32> [#uses=1]
        ret i32 %tmp2
}

define i32 @test36(i32 %a) {
	%x = and i32 %a, -2
	%y = and i32 %a, -126
	%z = add i32 %x, %y
	%q = and i32 %z, 1  ; always zero
	ret i32 %q
}
