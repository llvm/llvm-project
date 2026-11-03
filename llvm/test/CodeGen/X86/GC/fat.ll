; RUN: llvm-as < %s > /dev/null

; The contents of a gc root are an opaque blob of arbitrary type and size, so
; an aggregate root is accepted.

declare void @llvm.gcroot(ptr, ptr) nounwind

define void @f() gc "x" {
	%st = alloca { ptr, i1 }		; <ptr> [#uses=1]
	call void @llvm.gcroot(ptr %st, ptr null)
	ret void
}
