; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1
; RUN: grep "movl	a(%rip)," %t1
; RUN: not grep GOTPCREL %t1

@a = hidden global i32 0

define i32 @get_a() {
entry:
	%tmp1 = load i32* @a, align 4
	ret i32 %tmp1
}
