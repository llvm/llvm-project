; This test makes sure that mul instructions are properly eliminated.
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep mul

define i32 @test1(i32 %A) {
        %B = mul i32 %A, 1              ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i32 %A) {
        ; Should convert to an add instruction
        %B = mul i32 %A, 2              ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test3(i32 %A) {
        ; This should disappear entirely
        %B = mul i32 %A, 0              ; <i32> [#uses=1]
        ret i32 %B
}

define double @test4(double %A) {
        ; This is safe for FP
        %B = mul double 1.000000e+00, %A                ; <double> [#uses=1]
        ret double %B
}

define i32 @test5(i32 %A) {
        %B = mul i32 %A, 8              ; <i32> [#uses=1]
        ret i32 %B
}

define i8 @test6(i8 %A) {
        %B = mul i8 %A, 8               ; <i8> [#uses=1]
        %C = mul i8 %B, 8               ; <i8> [#uses=1]
        ret i8 %C
}

define i32 @test7(i32 %i) {
        %tmp = mul i32 %i, -1           ; <i32> [#uses=1]
        ret i32 %tmp
}

define i64 @test8(i64 %i) {
       ; tmp = sub 0, %i
        %j = mul i64 %i, -1             ; <i64> [#uses=1]
        ret i64 %j
}

define i32 @test9(i32 %i) {
        ; %j = sub 0, %i
        %j = mul i32 %i, -1             ; <i32> [#uses=1]
        ret i32 %j
}

define i32 @test10(i32 %a, i32 %b) {
        %c = icmp slt i32 %a, 0         ; <i1> [#uses=1]
        %d = zext i1 %c to i32          ; <i32> [#uses=1]
       ; e = b & (a >> 31)
        %e = mul i32 %d, %b             ; <i32> [#uses=1]
        ret i32 %e
}

define i32 @test11(i32 %a, i32 %b) {
        %c = icmp sle i32 %a, -1                ; <i1> [#uses=1]
        %d = zext i1 %c to i32          ; <i32> [#uses=1]
        ; e = b & (a >> 31)
        %e = mul i32 %d, %b             ; <i32> [#uses=1]
        ret i32 %e
}

define i32 @test12(i8 %a, i32 %b) {
        %c = icmp ugt i8 %a, 127                ; <i1> [#uses=1]
        %d = zext i1 %c to i32          ; <i32> [#uses=1]
        ; e = b & (a >> 31)
        %e = mul i32 %d, %b             ; <i32> [#uses=1]
        ret i32 %e
}

; PR2642
define internal void @test13(<4 x float>*) {
	load <4 x float>* %0, align 1
	mul <4 x float> %2, < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >
	store <4 x float> %3, <4 x float>* %0, align 1
	ret void
}

define internal void @test14(<4 x float>*) {
	load <4 x float>* %0, align 1
	mul <4 x float> %2, zeroinitializer
	store <4 x float> %3, <4 x float>* %0, align 1
	ret void
}

