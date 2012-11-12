; RUN: opt < %s -globalopt -S | grep internal | count 2

global i32 0
define i32* @1() {
	ret i32* @0
}
define i32* @f() {
entry:
	call i32* @1()
	ret i32* %0
}
