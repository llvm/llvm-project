; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1 -f
; RUN: grep {movq	f@GOTPCREL(%rip),} %t1

define void ()* @g() {
entry:
	ret void ()* @f
}

declare void @f()
