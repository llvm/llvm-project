; RUN: llc < %s -march=x86-64

define i64 @foo() nounwind {
entry:
	%t0 = load i32, i32* null, align 8
	switch i32 %t0, label %bb65 [
		i32 16, label %bb
		i32 12, label %bb56
	]

bb:
	br label %bb65

bb56:
	unreachable

bb65:
	%a = phi i64 [ 0, %bb ], [ 0, %entry ]
	tail call void asm "", "{cx}"(i64 %a) nounwind
	%t15 = and i64 %a, 4294967295
	ret i64 %t15
}

define i64 @bar(i64 %t0) nounwind {
	call void asm "", "{cx}"(i64 0) nounwind
	%t1 = sub i64 0, %t0
	%t2 = and i64 %t1, 4294967295
	ret i64 %t2
}
