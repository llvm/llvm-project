; RUN: opt < %s -tailcallelim -S | \
; RUN:    grep "tail call void @foo"


declare void @foo()

define void @bar() {
	call void @foo( )
	ret void
}

