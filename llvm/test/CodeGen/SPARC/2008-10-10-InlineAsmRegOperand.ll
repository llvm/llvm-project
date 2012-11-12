; RUN: llc < %s -march=sparc
; PR 1557

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f128:128:128"
module asm "\09.section\09\22.ctors\22,#alloc,#write"
module asm "\09.section\09\22.dtors\22,#alloc,#write"

define void @frame_dummy() nounwind {
entry:
	%asmtmp = tail call void (i8*)* (void (i8*)*)* asm "", "=r,0"(void (i8*)* @_Jv_RegisterClasses) nounwind		; <void (i8*)*> [#uses=0]
	unreachable
}

declare void @_Jv_RegisterClasses(i8*)
