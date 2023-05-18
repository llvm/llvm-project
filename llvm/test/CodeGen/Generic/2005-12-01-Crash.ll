; RUN: llc < %s
@str = external global [36 x i8]		; <ptr> [#uses=0]
@str.upgrd.1 = external global [29 x i8]		; <ptr> [#uses=0]
@str1 = external global [29 x i8]		; <ptr> [#uses=0]
@str2 = external global [29 x i8]		; <ptr> [#uses=1]
@str.upgrd.2 = external global [2 x i8]		; <ptr> [#uses=0]
@str3 = external global [2 x i8]		; <ptr> [#uses=0]
@str4 = external global [2 x i8]		; <ptr> [#uses=0]
@str5 = external global [2 x i8]		; <ptr> [#uses=0]

define void @printArgsNoRet(i32 %a1, float %a2, i8 %a3, double %a4, ptr %a5, i32 %a6, float %a7, i8 %a8, double %a9, ptr %a10, i32 %a11, float %a12, i8 %a13, double %a14, ptr %a15) {
entry:
	%tmp17 = sext i8 %a13 to i32		; <i32> [#uses=1]
	%tmp23 = call i32 (ptr, ...) @printf( ptr @str2, i32 %a11, double 0.000000e+00, i32 %tmp17, double %a14, i32 0 )		; <i32> [#uses=0]
	ret void
}

declare i32 @printf(ptr, ...)

declare i32 @main(i32, ptr)
