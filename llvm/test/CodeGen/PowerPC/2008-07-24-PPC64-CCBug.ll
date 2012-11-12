; RUN: llc < %s -mtriple=powerpc64-apple-darwin | grep lwz | grep 228

@"\01LC" = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

define void @llvm_static_func(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8, i32 %a9, i32 %a10, i32 %a11, i32 %a12, i32 %a13, i32 %a14, i32 %a15) nounwind  {
entry:
	tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @"\01LC", i32 0, i64 0), i32 %a8 ) nounwind 		; <i32>:0 [#uses=0]
	ret void
}

declare i32 @printf(i8*, ...) nounwind 
