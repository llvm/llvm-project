; RUN: not llvm-as < %s > /dev/null 2>&1

	%list = type { i32, ptr }

; This usage is invalid now; instead, objects must be bitcast to ptr for input
; to the gc intrinsics.
declare void @llvm.gcwrite(ptr, ptr, ptr)

define ptr @cons(i32 %hd, ptr %tl) gc "example" {
	%tmp = call ptr @gcalloc(i32 bitcast(ptr getelementptr(%list, ptr null, i32 1) to i32))
	
	store i32 %hd, ptr %tmp
	
	call void @llvm.gcwrite(ptr %tl, ptr %tmp, ptr %tmp)
	
	ret %cell.2
}

declare ptr @gcalloc(i32)
