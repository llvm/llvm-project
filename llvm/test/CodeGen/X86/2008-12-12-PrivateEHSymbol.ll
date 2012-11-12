; RUN: llc < %s -disable-cfi -march=x86-64 -mtriple=x86_64-apple-darwin9 | grep ^__Z1fv.eh
; RUN: llc < %s -disable-cfi -march=x86    -mtriple=i386-apple-darwin9 | grep ^__Z1fv.eh

define void @_Z1fv() {
entry:
	br label %return

return:
	ret void
}
