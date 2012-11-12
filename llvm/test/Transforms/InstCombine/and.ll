; This test makes sure that these instructions are properly eliminated.
;

; RUN: opt < %s -instcombine -S | not grep and

define i32 @test1(i32 %A) {
        ; zero result
        %B = and i32 %A, 0              ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i32 %A) {
        ; noop
        %B = and i32 %A, -1             ; <i32> [#uses=1]
        ret i32 %B
}

define i1 @test3(i1 %A) {
        ; always = false
        %B = and i1 %A, false           ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test4(i1 %A) {
        ; noop
        %B = and i1 %A, true            ; <i1> [#uses=1]
        ret i1 %B
}

define i32 @test5(i32 %A) {
        %B = and i32 %A, %A             ; <i32> [#uses=1]
        ret i32 %B
}

define i1 @test6(i1 %A) {
        %B = and i1 %A, %A              ; <i1> [#uses=1]
        ret i1 %B
}

; A & ~A == 0
define i32 @test7(i32 %A) {
        %NotA = xor i32 %A, -1          ; <i32> [#uses=1]
        %B = and i32 %A, %NotA          ; <i32> [#uses=1]
        ret i32 %B
}

; AND associates
define i8 @test8(i8 %A) {
        %B = and i8 %A, 3               ; <i8> [#uses=1]
        %C = and i8 %B, 4               ; <i8> [#uses=1]
        ret i8 %C
}

define i1 @test9(i32 %A) {
        ; Test of sign bit, convert to setle %A, 0
        %B = and i32 %A, -2147483648            ; <i32> [#uses=1]
        %C = icmp ne i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test9a(i32 %A) {
        ; Test of sign bit, convert to setle %A, 0
        %B = and i32 %A, -2147483648            ; <i32> [#uses=1]
        %C = icmp ne i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
}

define i32 @test10(i32 %A) {
        %B = and i32 %A, 12             ; <i32> [#uses=1]
        %C = xor i32 %B, 15             ; <i32> [#uses=1]
        ; (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
        %D = and i32 %C, 1              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test11(i32 %A, i32* %P) {
        %B = or i32 %A, 3               ; <i32> [#uses=1]
        %C = xor i32 %B, 12             ; <i32> [#uses=2]
        ; additional use of C
        store i32 %C, i32* %P
        ; %C = and uint %B, 3 --> 3
        %D = and i32 %C, 3              ; <i32> [#uses=1]
        ret i32 %D
}

define i1 @test12(i32 %A, i32 %B) {
        %C1 = icmp ult i32 %A, %B               ; <i1> [#uses=1]
        %C2 = icmp ule i32 %A, %B               ; <i1> [#uses=1]
        ; (A < B) & (A <= B) === (A < B)
        %D = and i1 %C1, %C2            ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test13(i32 %A, i32 %B) {
        %C1 = icmp ult i32 %A, %B               ; <i1> [#uses=1]
        %C2 = icmp ugt i32 %A, %B               ; <i1> [#uses=1]
        ; (A < B) & (A > B) === false
        %D = and i1 %C1, %C2            ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test14(i8 %A) {
        %B = and i8 %A, -128            ; <i8> [#uses=1]
        %C = icmp ne i8 %B, 0           ; <i1> [#uses=1]
        ret i1 %C
}

define i8 @test15(i8 %A) {
        %B = lshr i8 %A, 7              ; <i8> [#uses=1]
        ; Always equals zero
        %C = and i8 %B, 2               ; <i8> [#uses=1]
        ret i8 %C
}

define i8 @test16(i8 %A) {
        %B = shl i8 %A, 2               ; <i8> [#uses=1]
        %C = and i8 %B, 3               ; <i8> [#uses=1]
        ret i8 %C
}

;; ~(~X & Y) --> (X | ~Y)
define i8 @test17(i8 %X, i8 %Y) {
        %B = xor i8 %X, -1              ; <i8> [#uses=1]
        %C = and i8 %B, %Y              ; <i8> [#uses=1]
        %D = xor i8 %C, -1              ; <i8> [#uses=1]
        ret i8 %D
}

define i1 @test18(i32 %A) {
        %B = and i32 %A, -128           ; <i32> [#uses=1]
        ;; C >= 128
        %C = icmp ne i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test18a(i8 %A) {
        %B = and i8 %A, -2              ; <i8> [#uses=1]
        %C = icmp eq i8 %B, 0           ; <i1> [#uses=1]
        ret i1 %C
}

define i32 @test19(i32 %A) {
        %B = shl i32 %A, 3              ; <i32> [#uses=1]
        ;; Clearing a zero bit
        %C = and i32 %B, -2             ; <i32> [#uses=1]
        ret i32 %C
}

define i8 @test20(i8 %A) {
        %C = lshr i8 %A, 7              ; <i8> [#uses=1]
        ;; Unneeded
        %D = and i8 %C, 1               ; <i8> [#uses=1]
        ret i8 %D
}

define i1 @test22(i32 %A) {
        %B = icmp eq i32 %A, 1          ; <i1> [#uses=1]
        %C = icmp sge i32 %A, 3         ; <i1> [#uses=1]
        ;; false
        %D = and i1 %B, %C              ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test23(i32 %A) {
        %B = icmp sgt i32 %A, 1         ; <i1> [#uses=1]
        %C = icmp sle i32 %A, 2         ; <i1> [#uses=1]
        ;; A == 2
        %D = and i1 %B, %C              ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test24(i32 %A) {
        %B = icmp sgt i32 %A, 1         ; <i1> [#uses=1]
        %C = icmp ne i32 %A, 2          ; <i1> [#uses=1]
        ;; A > 2
        %D = and i1 %B, %C              ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test25(i32 %A) {
        %B = icmp sge i32 %A, 50                ; <i1> [#uses=1]
        %C = icmp slt i32 %A, 100               ; <i1> [#uses=1]
        ;; (A-50) <u 50
        %D = and i1 %B, %C              ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test26(i32 %A) {
        %B = icmp ne i32 %A, 50         ; <i1> [#uses=1]
        %C = icmp ne i32 %A, 51         ; <i1> [#uses=1]
        ;; (A-50) > 1
        %D = and i1 %B, %C              ; <i1> [#uses=1]
        ret i1 %D
}

define i8 @test27(i8 %A) {
        %B = and i8 %A, 4               ; <i8> [#uses=1]
        %C = sub i8 %B, 16              ; <i8> [#uses=1]
        ;; 0xF0
        %D = and i8 %C, -16             ; <i8> [#uses=1]
        %E = add i8 %D, 16              ; <i8> [#uses=1]
        ret i8 %E
}

;; This is juse a zero extending shr.
define i32 @test28(i32 %X) {
        ;; Sign extend
        %Y = ashr i32 %X, 24            ; <i32> [#uses=1]
        ;; Mask out sign bits
        %Z = and i32 %Y, 255            ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test29(i8 %X) {
        %Y = zext i8 %X to i32          ; <i32> [#uses=1]
       ;; Zero extend makes this unneeded.
        %Z = and i32 %Y, 255            ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test30(i1 %X) {
        %Y = zext i1 %X to i32          ; <i32> [#uses=1]
        %Z = and i32 %Y, 1              ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test31(i1 %X) {
        %Y = zext i1 %X to i32          ; <i32> [#uses=1]
        %Z = shl i32 %Y, 4              ; <i32> [#uses=1]
        %A = and i32 %Z, 16             ; <i32> [#uses=1]
        ret i32 %A
}

define i32 @test32(i32 %In) {
        %Y = and i32 %In, 16            ; <i32> [#uses=1]
        %Z = lshr i32 %Y, 2             ; <i32> [#uses=1]
        %A = and i32 %Z, 1              ; <i32> [#uses=1]
        ret i32 %A
}

;; Code corresponding to one-bit bitfield ^1.
define i32 @test33(i32 %b) {
        %tmp.4.mask = and i32 %b, 1             ; <i32> [#uses=1]
        %tmp.10 = xor i32 %tmp.4.mask, 1                ; <i32> [#uses=1]
        %tmp.12 = and i32 %b, -2                ; <i32> [#uses=1]
        %tmp.13 = or i32 %tmp.12, %tmp.10               ; <i32> [#uses=1]
        ret i32 %tmp.13
}

define i32 @test34(i32 %A, i32 %B) {
        %tmp.2 = or i32 %B, %A          ; <i32> [#uses=1]
        %tmp.4 = and i32 %tmp.2, %B             ; <i32> [#uses=1]
        ret i32 %tmp.4
}

