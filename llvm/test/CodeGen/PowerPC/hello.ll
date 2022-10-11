; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--
; PR1399

@.str = internal constant [13 x i8] c"Hello World!\00"

define i32 @main() {
	%tmp2 = tail call i32 @puts( ptr @.str )
	ret i32 0
}

declare i32 @puts(ptr)
