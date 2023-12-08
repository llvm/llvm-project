; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: invalid type for alloca
; PR2113

define void @test() {
	%A = alloca void()
	ret void
}

