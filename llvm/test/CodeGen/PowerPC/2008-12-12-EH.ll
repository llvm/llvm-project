; RUN: llc < %s -disable-cfi -march=ppc32 -mtriple=powerpc-apple-darwin9 | grep ^__Z1fv.eh

define void @_Z1fv() {
entry:
	br label %return

return:
	ret void
}
