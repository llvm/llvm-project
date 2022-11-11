; RUN: llc -verify-machineinstrs -no-integrated-as < %s
; PR1382

;target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-unknown-linux-gnu"
@x = global [2 x i32] [ i32 1, i32 2 ]		; <ptr> [#uses=1]

define void @foo() {
entry:
	tail call void asm sideeffect "$0 $1", "s,i"( ptr getelementptr ([2 x i32], ptr @x, i32 0, i32 1), ptr getelementptr ([2 x i32], ptr @x, i32 0, i32 1) )
	ret void
}
