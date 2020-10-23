;RUN: opt < %s -analyze -dot-cfg-only -enable-new-pm=0 2>/dev/null
;RUN: opt < %s -passes=dot-cfg-only 2>/dev/null
;RUN: opt < %s -analyze -dot-cfg-only \
;RUN:          -cfg-heat-colors=true -cfg-weights=true -enable-new-pm=0 2>/dev/null
;RUN: opt < %s -analyze -dot-cfg-only \
;RUN:          -cfg-heat-colors=false -cfg-weights=false -enable-new-pm=0 2>/dev/null
;RUN: opt < %s -analyze -dot-cfg \
;RUN:          -cfg-heat-colors=true -cfg-weights=true -enable-new-pm=0 2>/dev/null
;RUN: opt < %s -analyze -dot-cfg \
;RUN:          -cfg-heat-colors=false -cfg-weights=false -enable-new-pm=0 2>/dev/null
;PR 1497

define void @foo() {
entry:
	br label %return

return:		; preds = %entry
	ret void
}

