; RUN: llvm-as %s -o - | opt -inline | llvm-dis | grep nounwind
; RUN: llvm-as %s -o - | opt -inline | llvm-dis | grep unreachable

declare i1 @extern()

define internal i32 @test() {
entry:
	%n = call i1 @extern( )
	br i1 %n, label %r, label %u
r:
	ret i32 0;
u:
	unwind
}

define i32 @caller() {
	%X = call i32 @test( ) nounwind
	ret i32 %X
}
