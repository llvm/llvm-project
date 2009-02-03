; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-unknown-linux-gnu | grep ^.L_Z1fv.eh
; RUN: llvm-as < %s | llc  -march=ppc32 -mtriple=powerpc-apple-darwin9 | grep ^__Z1fv.eh

define void @_Z1fv() {
entry:
	br label %return

return:
	ret void
}
