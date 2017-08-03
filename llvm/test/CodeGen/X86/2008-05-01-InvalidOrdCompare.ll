; RUN: llc < %s -enable-unsafe-fp-math -march=x86 | grep jp
; rdar://5902801

declare void @test2()

define i32 @test(double %p) nounwind {
	%tmp5 = fcmp uno double %p, 0.000000e+00
	br i1 %tmp5, label %bb, label %UnifiedReturnBlock
bb:
	call void @test2()
	ret i32 17
UnifiedReturnBlock:
	ret i32 42
}

