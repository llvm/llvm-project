; RUN: llc < %s -mtriple=i686--
; rdar://6559995

@a = external dso_local global [255 x ptr], align 32

define i32 @main() nounwind {
entry:
	store ptr getelementptr ([255 x ptr], ptr @a, i32 0, i32 -2147483624), ptr getelementptr ([255 x ptr], ptr @a, i32 0, i32 16), align 32
	ret i32 0
}
