;RUN: opt < %s -analyze -dot-cfg-only 2>/dev/null
;RUN: opt < %s -analyze -passes=dot-cfg-only 2>/dev/null
;PR 1497

define void @foo() {
entry:
	br label %return

return:		; preds = %entry
	ret void
}

