; RUN: opt < %s -prune-eh -S | not grep nounwind

define weak void @f() {
entry:
        ret void
}

define void @g() {
entry:
	call void @f()
	ret void
}
