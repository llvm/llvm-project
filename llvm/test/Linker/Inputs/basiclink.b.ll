declare ptr @foo(...)
define ptr @bar() {
	%ret = call ptr (...) @foo( i32 123 )
	ret ptr %ret
}
@baz = global i32 0
