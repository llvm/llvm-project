; RUN: llc -verify-machineinstrs -mtriple=wasm32-unknown-unknown < %s | FileCheck -check-prefix=WASM32 %s

; WASM32: i32.load        28
; WASM32-NEXT: i32.sub
; WASM32-NEXT: br_if           0

; WASM32: __stack_chk_fail

@"\01LC" = internal constant [11 x i8] c"buf == %s\0A\00"		; <[11 x i8]*> [#uses=1]

define void @test(i8* %a) nounwind ssp {
entry:
	%a_addr = alloca i8*		; <i8**> [#uses=2]
	%buf = alloca [8 x i8]		; <[8 x i8]*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i8* %a, i8** %a_addr
	%buf1 = bitcast [8 x i8]* %buf to i8*		; <i8*> [#uses=1]
	%0 = load i8*, i8** %a_addr, align 4		; <i8*> [#uses=1]
	%1 = call i8* @strcpy(i8* %buf1, i8* %0) nounwind		; <i8*> [#uses=0]
  %buf2 = bitcast [8 x i8]* %buf to i8*		; <i8*> [#uses=1]
	%2 = call i32 (i8*, ...) @printf(i8* getelementptr ([11 x i8], [11 x i8]* @"\01LC", i32 0, i32 0), i8* %buf2) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare i8* @strcpy(i8*, i8*) nounwind

declare i32 @printf(i8*, ...) nounwind
