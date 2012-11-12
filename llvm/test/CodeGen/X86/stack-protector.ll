; RUN: llc -mtriple=i386-pc-linux-gnu < %s -o - | grep %gs:
; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s -o - | grep %fs:
; RUN: llc -code-model=kernel -mtriple=x86_64-pc-linux-gnu < %s -o - | grep %gs:
; RUN: llc -mtriple=x86_64-apple-darwin < %s -o - | grep "__stack_chk_guard"
; RUN: llc -mtriple=x86_64-apple-darwin < %s -o - | grep "__stack_chk_fail"

@"\01LC" = internal constant [11 x i8] c"buf == %s\0A\00"		; <[11 x i8]*> [#uses=1]

define void @test(i8* %a) nounwind ssp {
entry:
	%a_addr = alloca i8*		; <i8**> [#uses=2]
	%buf = alloca [8 x i8]		; <[8 x i8]*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i8* %a, i8** %a_addr
	%buf1 = bitcast [8 x i8]* %buf to i8*		; <i8*> [#uses=1]
	%0 = load i8** %a_addr, align 4		; <i8*> [#uses=1]
	%1 = call i8* @strcpy(i8* %buf1, i8* %0) nounwind		; <i8*> [#uses=0]
	%buf2 = bitcast [8 x i8]* %buf to i8*		; <i8*> [#uses=1]
	%2 = call i32 (i8*, ...)* @printf(i8* getelementptr ([11 x i8]* @"\01LC", i32 0, i32 0), i8* %buf2) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare i8* @strcpy(i8*, i8*) nounwind

declare i32 @printf(i8*, ...) nounwind
