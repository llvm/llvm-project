; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep sh
; END.

define i32 @test1(i32 %A) {
        %B = shl i32 %A, 0              ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i8 %A) {
        %shift.upgrd.1 = zext i8 %A to i32              ; <i32> [#uses=1]
        %B = shl i32 0, %shift.upgrd.1          ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test3(i32 %A) {
        %B = ashr i32 %A, 0             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test4(i8 %A) {
        %shift.upgrd.2 = zext i8 %A to i32              ; <i32> [#uses=1]
        %B = ashr i32 0, %shift.upgrd.2         ; <i32> [#uses=1]
        ret i32 %B
}


define i32 @test5(i32 %A) {
        %B = lshr i32 %A, 32  ;; shift all bits out 
        ret i32 %B
}

define i32 @test5a(i32 %A) {
        %B = shl i32 %A, 32     ;; shift all bits out 
        ret i32 %B
}

define i32 @test6(i32 %A) {
        %B = shl i32 %A, 1      ;; convert to an mul instruction 
        %C = mul i32 %B, 3             
        ret i32 %C
}

define i32 @test7(i8 %A) {
        %shift.upgrd.3 = zext i8 %A to i32 
        %B = ashr i32 -1, %shift.upgrd.3  ;; Always equal to -1
        ret i32 %B
}

;; (A << 5) << 3 === A << 8 == 0
define i8 @test8(i8 %A) {
        %B = shl i8 %A, 5               ; <i8> [#uses=1]
        %C = shl i8 %B, 3               ; <i8> [#uses=1]
        ret i8 %C
}

;; (A << 7) >> 7 === A & 1
define i8 @test9(i8 %A) {
        %B = shl i8 %A, 7               ; <i8> [#uses=1]
        %C = lshr i8 %B, 7              ; <i8> [#uses=1]
        ret i8 %C
}

;; (A >> 7) << 7 === A & 128
define i8 @test10(i8 %A) {
        %B = lshr i8 %A, 7              ; <i8> [#uses=1]
        %C = shl i8 %B, 7               ; <i8> [#uses=1]
        ret i8 %C
}

;; (A >> 3) << 4 === (A & 0x1F) << 1
define i8 @test11(i8 %A) {
        %a = mul i8 %A, 3               ; <i8> [#uses=1]
        %B = lshr i8 %a, 3              ; <i8> [#uses=1]
        %C = shl i8 %B, 4               ; <i8> [#uses=1]
        ret i8 %C
}

;; (A >> 8) << 8 === A & -256
define i32 @test12(i32 %A) {
        %B = ashr i32 %A, 8             ; <i32> [#uses=1]
        %C = shl i32 %B, 8              ; <i32> [#uses=1]
        ret i32 %C
}

;; (A >> 3) << 4 === (A & -8) * 2
define i8 @test13(i8 %A) {
        %a = mul i8 %A, 3               ; <i8> [#uses=1]
        %B = ashr i8 %a, 3              ; <i8> [#uses=1]
        %C = shl i8 %B, 4               ; <i8> [#uses=1]
        ret i8 %C
}

;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
define i32 @test14(i32 %A) {
        %B = lshr i32 %A, 4             ; <i32> [#uses=1]
        %C = or i32 %B, 1234            ; <i32> [#uses=1]
        %D = shl i32 %C, 4              ; <i32> [#uses=1]
        ret i32 %D
}

;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
define i32 @test14a(i32 %A) {
        %B = shl i32 %A, 4              ; <i32> [#uses=1]
        %C = and i32 %B, 1234           ; <i32> [#uses=1]
        %D = lshr i32 %C, 4             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test15(i1 %C) {
        %A = select i1 %C, i32 3, i32 1         ; <i32> [#uses=1]
        %V = shl i32 %A, 2              ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test15a(i1 %C) {
        %A = select i1 %C, i8 3, i8 1           ; <i8> [#uses=1]
        %shift.upgrd.4 = zext i8 %A to i32              ; <i32> [#uses=1]
        %V = shl i32 64, %shift.upgrd.4         ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @test16(i32 %X) {
        %tmp.3 = ashr i32 %X, 4         ; <i32> [#uses=1]
        %tmp.6 = and i32 %tmp.3, 1              ; <i32> [#uses=1]
        %tmp.7 = icmp ne i32 %tmp.6, 0          ; <i1> [#uses=1]
        ret i1 %tmp.7
}

define i1 @test17(i32 %A) {
        %B = lshr i32 %A, 3             ; <i32> [#uses=1]
        %C = icmp eq i32 %B, 1234               ; <i1> [#uses=1]
        ret i1 %C
}


define i1 @test18(i8 %A) {
        %B = lshr i8 %A, 7              ; <i8> [#uses=1]
        ;; false
        %C = icmp eq i8 %B, 123         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test19(i32 %A) {
        %B = ashr i32 %A, 2             ; <i32> [#uses=1]
        ;; (X & -4) == 0
        %C = icmp eq i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
}


define i1 @test19a(i32 %A) {
        %B = ashr i32 %A, 2             ; <i32> [#uses=1]
        ;; (X & -4) == -4
        %C = icmp eq i32 %B, -1         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test20(i8 %A) {
        %B = ashr i8 %A, 7              ; <i8> [#uses=1]
        ;; false
        %C = icmp eq i8 %B, 123         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test21(i8 %A) {
        %B = shl i8 %A, 4               ; <i8> [#uses=1]
        %C = icmp eq i8 %B, -128                ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test22(i8 %A) {
        %B = shl i8 %A, 4               ; <i8> [#uses=1]
        %C = icmp eq i8 %B, 0           ; <i1> [#uses=1]
        ret i1 %C
}

define i8 @test23(i32 %A) {
        ;; casts not needed
        %B = shl i32 %A, 24             ; <i32> [#uses=1]
        %C = ashr i32 %B, 24            ; <i32> [#uses=1]
        %D = trunc i32 %C to i8         ; <i8> [#uses=1]
        ret i8 %D
}

define i8 @test24(i8 %X) {
        %Y = and i8 %X, -5              ; <i8> [#uses=1]
        %Z = shl i8 %Y, 5               ; <i8> [#uses=1]
        %Q = ashr i8 %Z, 5              ; <i8> [#uses=1]
        ret i8 %Q
}

define i32 @test25(i32 %tmp.2, i32 %AA) {
        %x = lshr i32 %AA, 17           ; <i32> [#uses=1]
        %tmp.3 = lshr i32 %tmp.2, 17            ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.3, %x             ; <i32> [#uses=1]
        %tmp.6 = shl i32 %tmp.5, 17             ; <i32> [#uses=1]
        ret i32 %tmp.6
}

;; handle casts between shifts.
define i32 @test26(i32 %A) {
        %B = lshr i32 %A, 1             ; <i32> [#uses=1]
        %C = bitcast i32 %B to i32              ; <i32> [#uses=1]
        %D = shl i32 %C, 1              ; <i32> [#uses=1]
        ret i32 %D
}
 
