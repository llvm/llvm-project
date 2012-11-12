declare i32* @foo(...)
define i32* @bar() {
	%ret = call i32* (...)* @foo( i32 123 )
	ret i32* %ret
}
@baz = global i32 0
