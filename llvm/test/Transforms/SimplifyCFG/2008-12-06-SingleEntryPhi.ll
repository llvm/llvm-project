; RUN: opt < %s -simplifycfg | llvm-dis
define i32 @test() {
entry:
	br label %T
T:
	%C = phi i1 [false, %entry] 
	br i1 %C, label %X, label %Y
X:
	ret i32 2
Y:
	add i32 1, 2
	ret i32 1
}
