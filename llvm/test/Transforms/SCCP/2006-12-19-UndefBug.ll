; RUN: opt < %s -sccp -S | \
; RUN:   grep "ret i1 false"

define i1 @foo() {
	%X = and i1 false, undef		; <i1> [#uses=1]
	ret i1 %X
}

