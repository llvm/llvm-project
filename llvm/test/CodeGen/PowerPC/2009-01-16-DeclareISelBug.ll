; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu.5
; rdar://6499616

@"\01LC" = internal constant [13 x i8] c"conftest.val\00"		; <ptr> [#uses=1]

define i32 @main() nounwind {
entry:
	%0 = call ptr @fopen(ptr @"\01LC", ptr null) nounwind		; <ptr> [#uses=0]
	unreachable
}

declare ptr @fopen(ptr, ptr)
