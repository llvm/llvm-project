; RUN: llc < %s -mtriple=sparc
; PR 1557

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f128:128:128"
module asm "\09.section\09.ctors,\22aw\22"
module asm "\09.section\09.dtors,\22aw\22"

define void @frame_dummy() nounwind {
entry:
	%asmtmp = tail call ptr (ptr) asm "", "=r,0"(ptr @_Jv_RegisterClasses) nounwind		; <void (i8*)*> [#uses=0]
	unreachable
}

declare void @_Jv_RegisterClasses(ptr)
