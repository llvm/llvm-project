; RUN: llc < %s -march=x86-64 | grep lea | count 3
; RUN: llc < %s -march=x86-64 | grep shl | count 1
; RUN: llc < %s -march=x86-64 | not grep imul

define i64 @t1(i64 %a) nounwind readnone {
entry:
	%0 = mul i64 %a, 81		; <i64> [#uses=1]
	ret i64 %0
}

define i64 @t2(i64 %a) nounwind readnone {
entry:
	%0 = mul i64 %a, 40		; <i64> [#uses=1]
	ret i64 %0
}
