; RUN: %lli %s > /dev/null

; Testcase distilled from 256.bzip2.

target datalayout = "e-p:32:32"

define i32 @main() {
entry:
	br label %loopentry.0
loopentry.0:		; preds = %loopentry.0, %entry
	%h.0 = phi i32 [ %tmp.2, %loopentry.0 ], [ -1, %entry ]		; <i32> [#uses=1]
	%tmp.2 = add i32 %h.0, 1		; <i32> [#uses=3]
	%tmp.4 = icmp ne i32 %tmp.2, 0		; <i1> [#uses=1]
	br i1 %tmp.4, label %loopentry.0, label %loopentry.1
loopentry.1:		; preds = %loopentry.0
	%h.1 = phi i32 [ %tmp.2, %loopentry.0 ]		; <i32> [#uses=1]
	ret i32 %h.1
}

