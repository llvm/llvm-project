; Test linking two functions with different prototypes and two globals 
; in different modules. This is for PR411
; RUN: llvm-as %s -o %t.bar.bc -f
; RUN: echo {define i32* @foo(i32 %x) \{ ret i32* @baz \} \
; RUN:   @baz = external global i32 } | llvm-as -o %t.foo.bc -f
; RUN: llvm-link %t.bar.bc %t.foo.bc -o %t.bc -f
; RUN: llvm-link %t.foo.bc %t.bar.bc -o %t.bc -f
declare i32* @foo(...)
define i32* @bar() {
	%ret = call i32* (...)* @foo( i32 123 )
	ret i32* %ret
}
@baz = global i32 0
