; RUN: opt < %s -ipsccp -S | grep -v "ret i101 0" | \
; RUN:    grep -v "ret i101 undef" | not grep ret


define internal i101 @bar(i101 %A) {
	%x = icmp eq i101 %A, 0
	br i1 %x, label %T, label %F
T:
	%B = call i101 @bar(i101 0)
	ret i101 0
F:      ; unreachable
	%C = call i101 @bar(i101 1)
	ret i101 %C
}

define i101 @foo() {
	%X = call i101 @bar(i101 0)
	ret i101 %X
}
