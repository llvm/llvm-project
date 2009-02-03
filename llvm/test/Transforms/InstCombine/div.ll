; This test makes sure that div instructions are properly eliminated.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep div

define i32 @test1(i32 %A) {
        %B = sdiv i32 %A, 1             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i32 %A) {
        ; => Shift
        %B = udiv i32 %A, 8             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test3(i32 %A) {
        ; => 0, don't need to keep traps
        %B = sdiv i32 0, %A             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test4(i32 %A) {
        ; 0-A
        %B = sdiv i32 %A, -1            ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test5(i32 %A) {
        %B = udiv i32 %A, -16           ; <i32> [#uses=1]
        %C = udiv i32 %B, -4            ; <i32> [#uses=1]
        ret i32 %C
}

define i1 @test6(i32 %A) {
        %B = udiv i32 %A, 123           ; <i32> [#uses=1]
        ; A < 123
        %C = icmp eq i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test7(i32 %A) {
        %B = udiv i32 %A, 10            ; <i32> [#uses=1]
        ; A >= 20 && A < 30
        %C = icmp eq i32 %B, 2          ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test8(i8 %A) {
        %B = udiv i8 %A, 123            ; <i8> [#uses=1]
        ; A >= 246
        %C = icmp eq i8 %B, 2           ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test9(i8 %A) {
        %B = udiv i8 %A, 123            ; <i8> [#uses=1]
        ; A < 246
        %C = icmp ne i8 %B, 2           ; <i1> [#uses=1]
        ret i1 %C
}

define i32 @test10(i32 %X, i1 %C) {
        %V = select i1 %C, i32 64, i32 8                ; <i32> [#uses=1]
        %R = udiv i32 %X, %V            ; <i32> [#uses=1]
        ret i32 %R
}

define i32 @test11(i32 %X, i1 %C) {
        %A = select i1 %C, i32 1024, i32 32             ; <i32> [#uses=1]
        %B = udiv i32 %X, %A            ; <i32> [#uses=1]
        ret i32 %B
}

; PR2328
define i32 @test12(i32 %x) nounwind  {
	%tmp3 = udiv i32 %x, %x		; 1
	ret i32 %tmp3
}

define i32 @test13(i32 %x) nounwind  {
	%tmp3 = sdiv i32 %x, %x		; 1
	ret i32 %tmp3
}

