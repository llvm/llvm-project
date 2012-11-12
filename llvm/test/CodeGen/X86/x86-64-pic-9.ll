; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1
; RUN: grep "leaq	f(%rip)," %t1
; RUN: not grep GOTPCREL %t1

define void ()* @g() nounwind {
entry:
	ret void ()* @f
}

define internal void @f() nounwind {
entry:
	ret void
}
