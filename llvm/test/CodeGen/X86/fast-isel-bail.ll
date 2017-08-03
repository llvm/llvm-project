; RUN: llc < %s -march=x86 -O0

; This file is for regression tests for cases where FastISel needs
; to gracefully bail out and let SelectionDAGISel take over.

	%0 = type { i64, i8* }		; type %0

declare void @bar(%0)

define fastcc void @foo() nounwind {
entry:
	call void @bar(%0 zeroinitializer)
	unreachable
}
