; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1 -f
; RUN: grep {leaq	f(%rip),} %t1
; RUN: not grep GOTPCREL %t1

define void ()* @g() {
entry:
	ret void ()* @f
}

define internal void @f() {
entry:
	ret void
}
