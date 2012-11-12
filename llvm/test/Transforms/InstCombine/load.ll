; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | not grep load

@X = constant i32 42		; <i32*> [#uses=2]
@X2 = constant i32 47		; <i32*> [#uses=1]
@Y = constant [2 x { i32, float }] [ { i32, float } { i32 12, float 1.000000e+00 }, { i32, float } { i32 37, float 0x3FF3B2FEC0000000 } ]		; <[2 x { i32, float }]*> [#uses=2]
@Z = constant [2 x { i32, float }] zeroinitializer		; <[2 x { i32, float }]*> [#uses=1]

@GLOBAL = internal constant [4 x i32] zeroinitializer


define i32 @test1() {
	%B = load i32* @X		; <i32> [#uses=1]
	ret i32 %B
}

define float @test2() {
	%A = getelementptr [2 x { i32, float }]* @Y, i64 0, i64 1, i32 1		; <float*> [#uses=1]
	%B = load float* %A		; <float> [#uses=1]
	ret float %B
}

define i32 @test3() {
	%A = getelementptr [2 x { i32, float }]* @Y, i64 0, i64 0, i32 0		; <i32*> [#uses=1]
	%B = load i32* %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @test4() {
	%A = getelementptr [2 x { i32, float }]* @Z, i64 0, i64 1, i32 0		; <i32*> [#uses=1]
	%B = load i32* %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @test5(i1 %C) {
	%Y = select i1 %C, i32* @X, i32* @X2		; <i32*> [#uses=1]
	%Z = load i32* %Y		; <i32> [#uses=1]
	ret i32 %Z
}

define i32 @test7(i32 %X) {
	%V = getelementptr i32* null, i32 %X		; <i32*> [#uses=1]
	%R = load i32* %V		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @test8(i32* %P) {
	store i32 1, i32* %P
	%X = load i32* %P		; <i32> [#uses=1]
	ret i32 %X
}

define i32 @test9(i32* %P) {
	%X = load i32* %P		; <i32> [#uses=1]
	%Y = load i32* %P		; <i32> [#uses=1]
	%Z = sub i32 %X, %Y		; <i32> [#uses=1]
	ret i32 %Z
}

define i32 @test10(i1 %C.upgrd.1, i32* %P, i32* %Q) {
	br i1 %C.upgrd.1, label %T, label %F
T:		; preds = %0
	store i32 1, i32* %Q
	store i32 0, i32* %P
	br label %C
F:		; preds = %0
	store i32 0, i32* %P
	br label %C
C:		; preds = %F, %T
	%V = load i32* %P		; <i32> [#uses=1]
	ret i32 %V
}

define double @test11(double* %p) {
  %t0 = getelementptr double* %p, i32 1
  store double 2.0, double* %t0
  %t1 = getelementptr double* %p, i32 1
  %x = load double* %t1
  ret double %x
}

define i32 @test12(i32* %P) {
        %A = alloca i32
        store i32 123, i32* %A
        ; Cast the result of the load not the source
        %Q = bitcast i32* %A to i32*
        %V = load i32* %Q
        ret i32 %V
}

define <16 x i8> @test13(<2 x i64> %x) {
entry:
	%tmp = load <16 x i8> * bitcast ([4 x i32]* @GLOBAL to <16 x i8>*)
	ret <16 x i8> %tmp
}


