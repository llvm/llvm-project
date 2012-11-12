; RUN: not llvm-as < %s >& /dev/null

	%list = type { i32, %list* }

; This usage is invalid now; instead, objects must be bitcast to i8* for input
; to the gc intrinsics.
declare %list* @llvm.gcread(%list*, %list**)

define %list* @tl(%list* %l) gc "example" {
	%hd.ptr = getelementptr %list* %l, i32 0, i32 0
	%hd = call %list* @llvm.gcread(%list* %l, %list** %hd.ptr)
	ret i32 %tmp
}
