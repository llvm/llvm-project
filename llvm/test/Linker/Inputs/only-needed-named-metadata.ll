@X = external global i32

declare i32 @foo()

define void @bar() {
	load i32, ptr @X
	call i32 @foo()
	ret void
}
