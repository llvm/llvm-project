; RUN: llc -mtriple=arm-eabi %s -o /dev/null
; PR2589

define void @main(ptr) {
entry:
	%sret1 = alloca { i32 }		; <ptr> [#uses=1]
	load { i32 }, ptr %sret1		; <{ i32 }>:1 [#uses=0]
	ret void
}
