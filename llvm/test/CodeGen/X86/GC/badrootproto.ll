; RUN: not llvm-as < %s > /dev/null 2>&1

	%list = type { i32, ptr }
	%meta = type opaque

; This usage is invalid now; instead, objects must be bitcast to ptr for input
; to the gc intrinsics.
declare void @llvm.gcroot(ptr, ptr)

define void @root() gc "example" {
	%x.var = alloca ptr
	call void @llvm.gcroot(ptr %x.var, ptr null)
}
