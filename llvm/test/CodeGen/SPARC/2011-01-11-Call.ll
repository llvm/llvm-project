; RUN: llc -march=sparc -O0 <%s

define void @test() nounwind {
entry:
 %0 = tail call i32 (...)* @foo() nounwind
 tail call void (...)* @bar() nounwind
 ret void
}

declare i32 @foo(...)

declare void @bar(...)

