; RUN: not llvm-as < %s > /dev/null 2>&1

	%list = type { i32, %list* }
	%meta = type opaque

; This usage is invalid now; instead, objects must be bitcast to i8* for input
; to the gc intrinsics.
declare void @llvm.gcroot(%list*, %meta*)

define void @root() gc "example" {
	%x.var = alloca i8*
	call void @llvm.gcroot(i8** %x.var, %meta* null)
}
