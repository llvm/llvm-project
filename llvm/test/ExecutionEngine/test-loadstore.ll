; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

define void @test(i8* %P, i16* %P.upgrd.1, i32* %P.upgrd.2, i64* %P.upgrd.3) {
	%V = load i8* %P		; <i8> [#uses=1]
	store i8 %V, i8* %P
	%V.upgrd.4 = load i16* %P.upgrd.1		; <i16> [#uses=1]
	store i16 %V.upgrd.4, i16* %P.upgrd.1
	%V.upgrd.5 = load i32* %P.upgrd.2		; <i32> [#uses=1]
	store i32 %V.upgrd.5, i32* %P.upgrd.2
	%V.upgrd.6 = load i64* %P.upgrd.3		; <i64> [#uses=1]
	store i64 %V.upgrd.6, i64* %P.upgrd.3
	ret void
}

define i32 @varalloca(i32 %Size) {
        ;; Variable sized alloca
	%X = alloca i32, i32 %Size		; <i32*> [#uses=2]
	store i32 %Size, i32* %X
	%Y = load i32* %X		; <i32> [#uses=1]
	ret i32 %Y
}

define i32 @main() {
	%A = alloca i8		; <i8*> [#uses=1]
	%B = alloca i16		; <i16*> [#uses=1]
	%C = alloca i32		; <i32*> [#uses=1]
	%D = alloca i64		; <i64*> [#uses=1]
	call void @test( i8* %A, i16* %B, i32* %C, i64* %D )
	call i32 @varalloca( i32 7 )		; <i32>:1 [#uses=0]
	ret i32 0
}
