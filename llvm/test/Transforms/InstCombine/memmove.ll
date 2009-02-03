; This test makes sure that memmove instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep {call void @llvm.memmove}

@S = internal constant [33 x i8] c"panic: restorelist inconsistency\00"		; <[33 x i8]*> [#uses=1]
@h = constant [2 x i8] c"h\00"		; <[2 x i8]*> [#uses=1]
@hel = constant [4 x i8] c"hel\00"		; <[4 x i8]*> [#uses=1]
@hello_u = constant [8 x i8] c"hello_u\00"		; <[8 x i8]*> [#uses=1]


declare void @llvm.memmove.i32(i8*, i8*, i32, i32)

define void @test1(i8* %A, i8* %B, i32 %N) {
	call void @llvm.memmove.i32( i8* %A, i8* %B, i32 0, i32 1 )
	ret void
}

define void @test2(i8* %A, i32 %N) {
        ;; dest can't alias source since we can't write to source!
	call void @llvm.memmove.i32( i8* %A, i8* getelementptr ([33 x i8]* @S, i32 0, i32 0), i32 %N, i32 1 )
	ret void
}

define i32 @test3() {
	%h_p = getelementptr [2 x i8]* @h, i32 0, i32 0		; <i8*> [#uses=1]
	%hel_p = getelementptr [4 x i8]* @hel, i32 0, i32 0		; <i8*> [#uses=1]
	%hello_u_p = getelementptr [8 x i8]* @hello_u, i32 0, i32 0		; <i8*> [#uses=1]
	%target = alloca [1024 x i8]		; <[1024 x i8]*> [#uses=1]
	%target_p = getelementptr [1024 x i8]* %target, i32 0, i32 0		; <i8*> [#uses=3]
	call void @llvm.memmove.i32( i8* %target_p, i8* %h_p, i32 2, i32 2 )
	call void @llvm.memmove.i32( i8* %target_p, i8* %hel_p, i32 4, i32 4 )
	call void @llvm.memmove.i32( i8* %target_p, i8* %hello_u_p, i32 8, i32 8 )
	ret i32 0
}

; PR2370
define void @test4(i8* %a) {
        tail call void @llvm.memmove.i32( i8* %a, i8* %a, i32 100, i32 1 )
        ret void
}

