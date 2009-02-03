; Test linking of a bc file to an archive via llvm-ld. 
; PR1434
; RUN: llvm-as %s -o %t.bar.bc -f
; RUN: echo {define i32* @foo(i32 %x) \{ ret i32* @baz \} \
; RUN:   @baz = external global i32 } | llvm-as -o %t.foo.bc -f
; RUN: llvm-ar rcf %t.foo.a %t.foo.bc
; RUN: llvm-ar rcf %t.bar.a %t.bar.bc
; RUN: llvm-ld -disable-opt %t.bar.bc %t.foo.a -o %t.bc 
; RUN: llvm-ld -disable-opt %t.foo.bc %t.bar.a -o %t.bc
declare i32* @foo(...)
define i32* @bar() {
	%ret = call i32* (...)* @foo( i32 123 )
	ret i32* %ret
}
@baz = global i32 0
