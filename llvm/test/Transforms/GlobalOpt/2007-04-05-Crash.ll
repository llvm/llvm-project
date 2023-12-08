; RUN: opt < %s -passes=globalopt -disable-output

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumb-apple-darwin8"
@replacementUnichar = internal global i16 -3		; <ptr> [#uses=2]
@"L_OBJC_IMAGE_INFO" = internal global [2 x i32] zeroinitializer		; <ptr> [#uses=1]
@llvm.used = appending global [1 x ptr] [ ptr @"L_OBJC_IMAGE_INFO" ]		; <ptr> [#uses=0]

define zeroext i16 @__NSCharToUnicharCFWrapper(i8 zeroext  %ch)   {
entry:
	%iftmp.0.0.in.in = select i1 false, ptr @replacementUnichar, ptr null		; <ptr> [#uses=1]
	%iftmp.0.0.in = load i16, ptr %iftmp.0.0.in.in		; <i16> [#uses=1]
	ret i16 %iftmp.0.0.in
}

define void @__NSASCIICharToUnichar() {
entry:
	ret void
}

define void @_NSDefaultCStringEncoding() {
entry:
	call void @__NSSetCStringCharToUnichar( )
	br i1 false, label %cond_true6, label %cond_next8

cond_true6:		; preds = %entry
	store i16 -2, ptr @replacementUnichar
	ret void

cond_next8:		; preds = %entry
	ret void
}

declare void @__NSSetCStringCharToUnichar()
