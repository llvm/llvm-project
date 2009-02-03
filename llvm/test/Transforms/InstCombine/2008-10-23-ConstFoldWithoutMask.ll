; RUN: llvm-as < %s | opt -instcombine
; PR2940

define i32 @tstid() {
	%var0 = inttoptr i32 1 to i8*		; <i8*> [#uses=1]
	%var2 = ptrtoint i8* %var0 to i32		; <i32> [#uses=1]
	ret i32 %var2
}
