; RUN: not llvm-as < %s > /dev/null 2>&1

declare void @llvm.gcroot(ptr, ptr) nounwind

define void @f() gc "x" {
	%st = alloca { ptr, i1 }		; <ptr> [#uses=1]
	call void @llvm.gcroot(ptr %st, ptr null)
	ret void
}
