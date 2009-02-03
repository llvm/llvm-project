; RUN: not llvm-as < %s >& /dev/null

	%list = type { i32, %list* }

; This usage is invalid now; instead, objects must be bitcast to i8* for input
; to the gc intrinsics.
declare void @llvm.gcwrite(%list*, %list*, %list**)

define %list* @cons(i32 %hd, %list* %tl) gc "example" {
	%tmp = call i8* @gcalloc(i32 bitcast(%list* getelementptr(%list* null, i32 1) to i32))
	%cell = bitcast i8* %tmp to %list*
	
	%hd.ptr = getelementptr %list* %cell, i32 0, i32 0
	store i32 %hd, i32* %hd.ptr
	
	%tl.ptr = getelementptr %list* %cell, i32 0, i32 0
	call void @llvm.gcwrite(%list* %tl, %list* %cell, %list** %tl.ptr)
	
	ret %cell.2
}

declare i8* @gcalloc(i32)
