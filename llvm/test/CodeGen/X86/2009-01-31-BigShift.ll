; RUN: llc < %s -mtriple=i686-- | not grep and
; PR3401

define void @x(i288 %i) nounwind {
	call void @add(i288 %i)
	ret void
}

declare void @add(i288)
