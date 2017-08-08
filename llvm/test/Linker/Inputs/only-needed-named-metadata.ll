@X = external global i32

declare i32 @foo()

define void @bar() {
	load i32, i32* @X
	call i32 @foo()
	ret void
}
