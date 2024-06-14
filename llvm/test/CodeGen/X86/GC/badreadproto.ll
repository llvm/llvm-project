; RUN: not llvm-as < %s > /dev/null 2>&1

	%list = type { i32, ptr }

; This usage is invalid now; instead, objects must be bitcast to ptr for input
; to the gc intrinsics.
declare ptr @llvm.gcread(ptr, ptr)

define ptr @tl(ptr %l) gc "example" {
	%hd = call ptr @llvm.gcread(ptr %l, ptr %l)
	ret i32 %tmp
}
