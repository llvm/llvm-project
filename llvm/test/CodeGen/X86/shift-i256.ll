; RUN: llc < %s -march=x86
; RUN: llc < %s -march=x86-64

define void @t(i256 %x, i256 %a, i256* nocapture %r) nounwind readnone {
entry:
	%0 = ashr i256 %x, %a
	store i256 %0, i256* %r
        ret void
}
