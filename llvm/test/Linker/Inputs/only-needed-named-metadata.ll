@X = external global i32

declare i32 @foo()

declare void @c1_a()

define void @bar() {
	load i32, i32* @X
	call i32 @foo()
	call void @c1_a()
	ret void
}
