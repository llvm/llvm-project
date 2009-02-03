; Test that the ExitInMainOptimization pass works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:    grep {ret i32 3} | count 1
; END.

declare void @exit(i32)

declare void @exitonly(i32)

define i32 @main() {
	call void @exitonly( i32 3 )
	call void @exit( i32 3 )
	ret i32 0
}

