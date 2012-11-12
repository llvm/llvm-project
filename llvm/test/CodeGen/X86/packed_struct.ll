; RUN: llc < %s -march=x86 > %t
; RUN: grep foos+5 %t
; RUN: grep foos+1 %t
; RUN: grep foos+9 %t
; RUN: grep bara+19 %t
; RUN: grep bara+4 %t

; make sure we compute the correct offset for a packed structure

;Note: codegen for this could change rendering the above checks wrong

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.anon = type <{ i8, i32, i32, i32 }>
@foos = external global %struct.anon		; <%struct.anon*> [#uses=3]
@bara = weak global [4 x <{ i32, i8 }>] zeroinitializer		; <[4 x <{ i32, i8 }>]*> [#uses=2]

define i32 @foo() nounwind {
entry:
	%tmp = load i32* getelementptr (%struct.anon* @foos, i32 0, i32 1)		; <i32> [#uses=1]
	%tmp3 = load i32* getelementptr (%struct.anon* @foos, i32 0, i32 2)		; <i32> [#uses=1]
	%tmp6 = load i32* getelementptr (%struct.anon* @foos, i32 0, i32 3)		; <i32> [#uses=1]
	%tmp4 = add i32 %tmp3, %tmp		; <i32> [#uses=1]
	%tmp7 = add i32 %tmp4, %tmp6		; <i32> [#uses=1]
	ret i32 %tmp7
}

define i8 @bar() nounwind {
entry:
	%tmp = load i8* getelementptr ([4 x <{ i32, i8 }>]* @bara, i32 0, i32 0, i32 1)		; <i8> [#uses=1]
	%tmp4 = load i8* getelementptr ([4 x <{ i32, i8 }>]* @bara, i32 0, i32 3, i32 1)		; <i8> [#uses=1]
	%tmp5 = add i8 %tmp4, %tmp		; <i8> [#uses=1]
	ret i8 %tmp5
}
