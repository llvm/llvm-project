; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | \
; RUN:   not grep alloca

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.0.0"

define i64 @test1(i64 %X) {
	%A = alloca i64		; <i64*> [#uses=3]
	store i64 %X, i64* %A
	%B = bitcast i64* %A to i32*		; <i32*> [#uses=1]
	%C = bitcast i32* %B to i8*		; <i8*> [#uses=1]
	store i8 0, i8* %C
	%Y = load i64* %A		; <i64> [#uses=1]
	ret i64 %Y
}

define i8 @test2(i64 %X) {
	%X_addr = alloca i64		; <i64*> [#uses=2]
	store i64 %X, i64* %X_addr
	%tmp.0 = bitcast i64* %X_addr to i32*		; <i32*> [#uses=1]
	%tmp.1 = getelementptr i32* %tmp.0, i32 1		; <i32*> [#uses=1]
	%tmp.2 = bitcast i32* %tmp.1 to i8*		; <i8*> [#uses=1]
	%tmp.3 = getelementptr i8* %tmp.2, i32 3		; <i8*> [#uses=1]
	%tmp.2.upgrd.1 = load i8* %tmp.3		; <i8> [#uses=1]
	ret i8 %tmp.2.upgrd.1
}

define i16 @crafty(i64 %X) {
	%a = alloca { i64 }		; <{ i64 }*> [#uses=2]
	%tmp.0 = getelementptr { i64 }* %a, i32 0, i32 0		; <i64*> [#uses=1]
	store i64 %X, i64* %tmp.0
	%tmp.3 = bitcast { i64 }* %a to [4 x i16]*		; <[4 x i16]*> [#uses=2]
	%tmp.4 = getelementptr [4 x i16]* %tmp.3, i32 0, i32 3		; <i16*> [#uses=1]
	%tmp.5 = load i16* %tmp.4		; <i16> [#uses=1]
	%tmp.8 = getelementptr [4 x i16]* %tmp.3, i32 0, i32 2		; <i16*> [#uses=1]
	%tmp.9 = load i16* %tmp.8		; <i16> [#uses=1]
	%tmp.10 = or i16 %tmp.9, %tmp.5		; <i16> [#uses=1]
	ret i16 %tmp.10
}

define i16 @crafty2(i64 %X) {
	%a = alloca i64		; <i64*> [#uses=2]
	store i64 %X, i64* %a
	%tmp.3 = bitcast i64* %a to [4 x i16]*		; <[4 x i16]*> [#uses=2]
	%tmp.4 = getelementptr [4 x i16]* %tmp.3, i32 0, i32 3		; <i16*> [#uses=1]
	%tmp.5 = load i16* %tmp.4		; <i16> [#uses=1]
	%tmp.8 = getelementptr [4 x i16]* %tmp.3, i32 0, i32 2		; <i16*> [#uses=1]
	%tmp.9 = load i16* %tmp.8		; <i16> [#uses=1]
	%tmp.10 = or i16 %tmp.9, %tmp.5		; <i16> [#uses=1]
	ret i16 %tmp.10
}
