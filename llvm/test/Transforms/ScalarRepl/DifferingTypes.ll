; This is a feature test.  Hopefully one day this will be implemented.  The 
; generated code should perform the appropriate masking operations required 
; depending on the endianness of the target...
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | \
; RUN:   not grep alloca

define i32 @testfunc(i32 %i, i8 %j) {
	%I = alloca i32		; <i32*> [#uses=3]
	store i32 %i, i32* %I
	%P = bitcast i32* %I to i8*		; <i8*> [#uses=1]
	store i8 %j, i8* %P
	%t = load i32* %I		; <i32> [#uses=1]
	ret i32 %t
}

