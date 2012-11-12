; RUN: llc < %s -march=arm
; PR2589

define void @main({ i32 }*) {
entry:
	%sret1 = alloca { i32 }		; <{ i32 }*> [#uses=1]
	load { i32 }* %sret1		; <{ i32 }>:1 [#uses=0]
	ret void
}
