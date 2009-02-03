; This test makes sure that these instructions are properly eliminated.
; PR1822

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep select

define i32 @test1(i32 %A, i32 %B) {
        %C = select i1 false, i32 %A, i32 %B            ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test2(i32 %A, i32 %B) {
        %C = select i1 true, i32 %A, i32 %B             ; <i32> [#uses=1]
        ret i32 %C
}


define i32 @test3(i1 %C, i32 %I) {
        ; V = I
        %V = select i1 %C, i32 %I, i32 %I               ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @test4(i1 %C) {
        ; V = C
        %V = select i1 %C, i1 true, i1 false            ; <i1> [#uses=1]
        ret i1 %V
}

define i1 @test5(i1 %C) {
        ; V = !C
        %V = select i1 %C, i1 false, i1 true            ; <i1> [#uses=1]
        ret i1 %V
}

define i32 @test6(i1 %C) { 
        ; V = cast C to int
        %V = select i1 %C, i32 1, i32 0         ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @test7(i1 %C, i1 %X) {
        ; R = or C, X       
        %R = select i1 %C, i1 true, i1 %X               ; <i1> [#uses=1]
        ret i1 %R
}

define i1 @test8(i1 %C, i1 %X) {
        ; R = and C, X
        %R = select i1 %C, i1 %X, i1 false              ; <i1> [#uses=1]
        ret i1 %R
}

define i1 @test9(i1 %C, i1 %X) {
        ; R = and !C, X
        %R = select i1 %C, i1 false, i1 %X              ; <i1> [#uses=1]
        ret i1 %R
}

define i1 @test10(i1 %C, i1 %X) {
        ; R = or !C, X
        %R = select i1 %C, i1 %X, i1 true               ; <i1> [#uses=1]
        ret i1 %R
}

define i32 @test11(i32 %a) {
        %C = icmp eq i32 %a, 0          ; <i1> [#uses=1]
        %R = select i1 %C, i32 0, i32 1         ; <i32> [#uses=1]
        ret i32 %R
}

define i32 @test12(i1 %cond, i32 %a) {
        %b = or i32 %a, 1               ; <i32> [#uses=1]
        %c = select i1 %cond, i32 %b, i32 %a            ; <i32> [#uses=1]
        ret i32 %c
}

define i32 @test12a(i1 %cond, i32 %a) {
        %b = ashr i32 %a, 1             ; <i32> [#uses=1]
        %c = select i1 %cond, i32 %b, i32 %a            ; <i32> [#uses=1]
        ret i32 %c
}

define i32 @test12b(i1 %cond, i32 %a) {
        %b = ashr i32 %a, 1             ; <i32> [#uses=1]
        %c = select i1 %cond, i32 %a, i32 %b            ; <i32> [#uses=1]
        ret i32 %c
}

define i32 @test13(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b         ; <i1> [#uses=1]
        %V = select i1 %C, i32 %a, i32 %b               ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test13a(i32 %a, i32 %b) {
        %C = icmp ne i32 %a, %b         ; <i1> [#uses=1]
        %V = select i1 %C, i32 %a, i32 %b               ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test13b(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b         ; <i1> [#uses=1]
        %V = select i1 %C, i32 %b, i32 %a               ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @test14a(i1 %C, i32 %X) {
        %V = select i1 %C, i32 %X, i32 0                ; <i32> [#uses=1]
        ; (X < 1) | !C
        %R = icmp slt i32 %V, 1         ; <i1> [#uses=1]
        ret i1 %R
}

define i1 @test14b(i1 %C, i32 %X) {
        %V = select i1 %C, i32 0, i32 %X                ; <i32> [#uses=1]
        ; (X < 1) | C
        %R = icmp slt i32 %V, 1         ; <i1> [#uses=1]
        ret i1 %R
}

;; Code sequence for (X & 16) ? 16 : 0
define i32 @test15a(i32 %X) {
        %t1 = and i32 %X, 16            ; <i32> [#uses=1]
        %t2 = icmp eq i32 %t1, 0                ; <i1> [#uses=1]
        %t3 = select i1 %t2, i32 0, i32 16              ; <i32> [#uses=1]
        ret i32 %t3
}

;; Code sequence for (X & 32) ? 0 : 24
define i32 @test15b(i32 %X) {
        %t1 = and i32 %X, 32            ; <i32> [#uses=1]
        %t2 = icmp eq i32 %t1, 0                ; <i1> [#uses=1]
        %t3 = select i1 %t2, i32 32, i32 0              ; <i32> [#uses=1]
        ret i32 %t3
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15c(i32 %X) {
        %t1 = and i32 %X, 16            ; <i32> [#uses=1]
        %t2 = icmp eq i32 %t1, 16               ; <i1> [#uses=1]
        %t3 = select i1 %t2, i32 16, i32 0              ; <i32> [#uses=1]
        ret i32 %t3
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15d(i32 %X) {
        %t1 = and i32 %X, 16            ; <i32> [#uses=1]
        %t2 = icmp ne i32 %t1, 0                ; <i1> [#uses=1]
        %t3 = select i1 %t2, i32 16, i32 0              ; <i32> [#uses=1]
        ret i32 %t3
}

define i32 @test16(i1 %C, i32* %P) {
        %P2 = select i1 %C, i32* %P, i32* null          ; <i32*> [#uses=1]
        %V = load i32* %P2              ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @test17(i32* %X, i1 %C) {
        %R = select i1 %C, i32* %X, i32* null           ; <i32*> [#uses=1]
        %RV = icmp eq i32* %R, null             ; <i1> [#uses=1]
        ret i1 %RV
}

define i32 @test18(i32 %X, i32 %Y, i1 %C) {
        %R = select i1 %C, i32 %X, i32 0                ; <i32> [#uses=1]
        %V = sdiv i32 %Y, %R            ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test19(i32 %x) {
        %tmp = icmp ugt i32 %x, 2147483647              ; <i1> [#uses=1]
        %retval = select i1 %tmp, i32 -1, i32 0         ; <i32> [#uses=1]
        ret i32 %retval
}

define i32 @test20(i32 %x) {
        %tmp = icmp slt i32 %x, 0               ; <i1> [#uses=1]
        %retval = select i1 %tmp, i32 -1, i32 0         ; <i32> [#uses=1]
        ret i32 %retval
}

define i64 @test21(i32 %x) {
        %tmp = icmp slt i32 %x, 0               ; <i1> [#uses=1]
        %retval = select i1 %tmp, i64 -1, i64 0         ; <i64> [#uses=1]
        ret i64 %retval
}

define i16 @test22(i32 %x) {
        %tmp = icmp slt i32 %x, 0               ; <i1> [#uses=1]
        %retval = select i1 %tmp, i16 -1, i16 0         ; <i16> [#uses=1]
        ret i16 %retval
}

define i1 @test23(i1 %a, i1 %b) {
        %c = select i1 %a, i1 %b, i1 %a         ; <i1> [#uses=1]
        ret i1 %c
}

define i1 @test24(i1 %a, i1 %b) {
        %c = select i1 %a, i1 %a, i1 %b         ; <i1> [#uses=1]
        ret i1 %c
}

