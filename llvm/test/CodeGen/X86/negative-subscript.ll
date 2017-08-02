; RUN: llc < %s -mtriple=i686--
; rdar://6559995

@a = external global [255 x i8*], align 32

define i32 @main() nounwind {
entry:
	store i8* bitcast (i8** getelementptr ([255 x i8*], [255 x i8*]* @a, i32 0, i32 -2147483624) to i8*), i8** getelementptr ([255 x i8*], [255 x i8*]* @a, i32 0, i32 16), align 32
	ret i32 0
}
