; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1
; RUN: grep "leaq	f(%rip)," %t1
; RUN: not grep GOTPCREL %t1

define void ()* @g() {
entry:
	ret void ()* @f
}

declare hidden void @f()
