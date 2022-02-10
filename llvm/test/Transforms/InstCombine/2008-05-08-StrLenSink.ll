; RUN: opt -S -passes=instcombine < %s | FileCheck %s
; PR2297
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define i32 @a() nounwind  {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp1 = call i8* @malloc( i32 10 ) nounwind 		; <i8*> [#uses=5]
	%tmp3 = getelementptr i8, i8* %tmp1, i32 1		; <i8*> [#uses=1]
	store i8 0, i8* %tmp3, align 1
	%tmp5 = getelementptr i8, i8* %tmp1, i32 0		; <i8*> [#uses=1]
	store i8 1, i8* %tmp5, align 1
; CHECK: store
; CHECK: store
; CHECK-NEXT: strlen
; CHECK-NEXT: store
	%tmp7 = call i32 @strlen( i8* %tmp1 ) nounwind readonly 		; <i32> [#uses=1]
	%tmp9 = getelementptr i8, i8* %tmp1, i32 0		; <i8*> [#uses=1]
	store i8 0, i8* %tmp9, align 1
	%tmp11 = call i32 (...) @b( i8* %tmp1 ) nounwind 		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret i32 %tmp7
}

declare i8* @malloc(i32) nounwind 

declare i32 @strlen(i8*) nounwind readonly 

declare i32 @b(...)
