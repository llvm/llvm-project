; RUN: opt < %s -passes=instcombine
; PR2940

define i32 @tstid() {
	%var0 = inttoptr i32 1 to ptr		; <ptr> [#uses=1]
	%var2 = ptrtoint ptr %var0 to i32		; <i32> [#uses=1]
	ret i32 %var2
}
