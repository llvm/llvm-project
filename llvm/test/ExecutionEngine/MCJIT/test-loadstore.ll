; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

define void @test(ptr %P, ptr %P.upgrd.1, ptr %P.upgrd.2, ptr %P.upgrd.3) {
	%V = load i8, ptr %P		; <i8> [#uses=1]
	store i8 %V, ptr %P
	%V.upgrd.4 = load i16, ptr %P.upgrd.1		; <i16> [#uses=1]
	store i16 %V.upgrd.4, ptr %P.upgrd.1
	%V.upgrd.5 = load i32, ptr %P.upgrd.2		; <i32> [#uses=1]
	store i32 %V.upgrd.5, ptr %P.upgrd.2
	%V.upgrd.6 = load i64, ptr %P.upgrd.3		; <i64> [#uses=1]
	store i64 %V.upgrd.6, ptr %P.upgrd.3
	ret void
}

define i32 @varalloca(i32 %Size) {
        ;; Variable sized alloca
	%X = alloca i32, i32 %Size		; <ptr> [#uses=2]
	store i32 %Size, ptr %X
	%Y = load i32, ptr %X		; <i32> [#uses=1]
	ret i32 %Y
}

define i32 @main() {
	%A = alloca i8		; <ptr> [#uses=1]
	%B = alloca i16		; <ptr> [#uses=1]
	%C = alloca i32		; <ptr> [#uses=1]
	%D = alloca i64		; <ptr> [#uses=1]
	call void @test( ptr %A, ptr %B, ptr %C, ptr %D )
	call i32 @varalloca( i32 7 )		; <i32>:1 [#uses=0]
	ret i32 0
}
