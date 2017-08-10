; RUN: llc < %s -mtriple=i686-- | not grep fstp
; RUN: llc < %s -mtriple=i686-- -mcpu=yonah | not grep movsd

declare double @foo()

define double @bar() {
entry:
	%tmp5 = tail call double @foo()
	ret double %tmp5
}

