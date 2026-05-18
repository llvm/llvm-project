; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

%struct = type {  }

; CHECK: declare void @foo(...)
declare void @foo(...)

; CHECK: call void (...) @foo(ptr byval(%struct) null)
define void @bar() {
	call void (...) @foo(ptr byval(%struct) null )
	ret void
}
