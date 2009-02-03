; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep bitcast
; RUN: llvm-as < %s | opt -instcombine -scalarrepl | llvm-dis | grep {ret i8. %v}
; PR1345

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-apple-darwin8.8.0"

define i8* @test(i8* %v) {
	%A = alloca [4 x i8*], align 16		; <[4 x i8*]*> [#uses=3]
	%B = getelementptr [4 x i8*]* %A, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* null, i8** %B
	%C = bitcast [4 x i8*]* %A to { [16 x i8] }*		; <{ [16 x i8] }*> [#uses=1]
	%D = getelementptr { [16 x i8] }* %C, i32 0, i32 0, i32 8		; <i8*> [#uses=1]
	%E = bitcast i8* %D to i8**		; <i8**> [#uses=1]
	store i8* %v, i8** %E
	%F = getelementptr [4 x i8*]* %A, i32 0, i32 2		; <i8**> [#uses=1]
	%G = load i8** %F		; <i8*> [#uses=1]
	ret i8* %G
}

; PR3290
%struct.Key = type { { i32, i32 } }
%struct.anon = type <{ i8, [3 x i8], i32 }>

define i32 *@test2(%struct.Key *%A) {
	%B = bitcast %struct.Key* %A to %struct.anon*
        %C = getelementptr %struct.anon* %B, i32 0, i32 2 
	ret i32 *%C
}

