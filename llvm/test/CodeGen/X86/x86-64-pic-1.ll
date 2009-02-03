; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1 -f
; RUN: grep {call	f@PLT} %t1

define void @g() {
entry:
	call void @f( )
	ret void
}

declare void @f()
