; RUN: llvm-as < %s | llc -march=x86-64 -mtriple=x86_64-unknown-linux-gnu | grep ^.L_Z1fv.eh
; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-unknown-linux-gnu | grep ^.L_Z1fv.eh
; RUN: llvm-as < %s | llc -march=x86-64 -mtriple=-mtriple=x86_64-apple-darwin9 | grep ^__Z1fv.eh
; RUN: llvm-as < %s | llc -march=x86 -mtriple=-mtriple=i386-apple-darwin9 | grep ^__Z1fv.eh

define void @_Z1fv() {
entry:
	br label %return

return:
	ret void
}
