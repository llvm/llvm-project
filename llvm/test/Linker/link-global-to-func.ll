; RUN: llvm-as %s -o %t1.bc
; RUN: echo "declare void @__eprintf(ptr, ptr, i32, ptr) noreturn     define void @foo() {      tail call void @__eprintf( ptr undef, ptr undef, i32 4, ptr null ) noreturn nounwind       unreachable }" | llvm-as -o %t2.bc
; RUN: llvm-link %t2.bc %t1.bc -S | FileCheck %s
; RUN: llvm-link %t1.bc %t2.bc -S | FileCheck %s
; CHECK: __eprintf

; rdar://6072702

@__eprintf = external global ptr		; <ptr> [#uses=1]

define ptr @test() {
	%A = load ptr, ptr @__eprintf		; <ptr> [#uses=1]
	ret ptr %A
}
