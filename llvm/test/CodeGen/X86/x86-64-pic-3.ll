; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1
; RUN: grep "callq	f" %t1
; RUN: not grep "callq	f@PLT" %t1

define void @g() {
entry:
	call void @f( )
	ret void
}

define internal void @f() {
entry:
	ret void
}
