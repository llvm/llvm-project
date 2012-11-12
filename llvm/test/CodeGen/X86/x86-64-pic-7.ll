; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1
; RUN: grep "movq	f@GOTPCREL(%rip)," %t1

define void ()* @g() nounwind {
entry:
	ret void ()* @f
}

declare void @f()
