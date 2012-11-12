; RUN: opt < %s -simplify-libcalls -S -o %t
; RUN: FileCheck < %t %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "xcore-xmos-elf"

@.str = internal constant [4 x i8] c"%f\0A\00"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

; Verify printf with no floating point arguments is transformed to iprintf
define i32 @f0(i32 %x) nounwind {
entry:
; CHECK: define i32 @f0
; CHECK: @iprintf
; CHECK: }
	%0 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([4 x i8]* @.str1, i32 0, i32 0), i32 %x)		; <i32> [#uses=0]
	ret i32 %0
}

; Verify we don't turn this into an iprintf call
define void @f1(double %x) nounwind {
entry:
; CHECK: define void @f1
; CHECK: @printf
; CHECK: }
	%0 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), double %x) nounwind		; <i32> [#uses=0]
	ret void
}

; Verify sprintf with no floating point arguments is transformed to siprintf
define i32 @f2(i8* %p, i32 %x) nounwind {
entry:
; CHECK: define i32 @f2
; CHECK: @siprintf
; CHECK: }
	%0 = tail call i32 (i8*, i8*, ...)* @sprintf(i8 *%p, i8* getelementptr ([4 x i8]* @.str1, i32 0, i32 0), i32 %x)
	ret i32 %0
}

; Verify we don't turn this into an siprintf call
define i32 @f3(i8* %p, double %x) nounwind {
entry:
; CHECK: define i32 @f3
; CHECK: @sprintf
; CHECK: }
	%0 = tail call i32 (i8*, i8*, ...)* @sprintf(i8 *%p, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), double %x)
	ret i32 %0
}

; Verify fprintf with no floating point arguments is transformed to fiprintf
define i32 @f4(i8* %p, i32 %x) nounwind {
entry:
; CHECK: define i32 @f4
; CHECK: @fiprintf
; CHECK: }
	%0 = tail call i32 (i8*, i8*, ...)* @fprintf(i8 *%p, i8* getelementptr ([4 x i8]* @.str1, i32 0, i32 0), i32 %x)
	ret i32 %0
}

; Verify we don't turn this into an fiprintf call
define i32 @f5(i8* %p, double %x) nounwind {
entry:
; CHECK: define i32 @f5
; CHECK: @fprintf
; CHECK: }
	%0 = tail call i32 (i8*, i8*, ...)* @fprintf(i8 *%p, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), double %x)
	ret i32 %0
}

declare i32 @printf(i8* nocapture, ...) nounwind
declare i32 @sprintf(i8* nocapture, i8* nocapture, ...) nounwind
declare i32 @fprintf(i8* nocapture, i8* nocapture, ...) nounwind
