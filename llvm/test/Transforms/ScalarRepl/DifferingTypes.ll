; This is a feature test.  Hopefully one day this will be implemented.  The 
; generated code should perform the appropriate masking operations required 
; depending on the endianness of the target...
; RUN: opt < %s -scalarrepl -S | \
; RUN:   not grep alloca
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define i32 @testfunc(i32 %i, i8 %j) {
	%I = alloca i32		; <i32*> [#uses=3]
	store i32 %i, i32* %I
	%P = bitcast i32* %I to i8*		; <i8*> [#uses=1]
	store i8 %j, i8* %P
	%t = load i32* %I		; <i32> [#uses=1]
	ret i32 %t
}

