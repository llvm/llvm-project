; RUN: llvm-as < %s | llvm-dis | grep byval
; RUN: verify-uselistorder %s

	%struct = type {  }

declare void @foo(...)

define void @bar() {
	call void (...) @foo(%struct* byval(%struct) null )
	ret void
}
