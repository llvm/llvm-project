; RUN: llc < %s -march=x86-64 | grep orb | count 1
; RUN: llc < %s -march=x86-64 | grep orb | grep 1
; RUN: llc < %s -march=x86-64 | grep orl | count 1
; RUN: llc < %s -march=x86-64 | grep orl | grep 16842752

	%struct.bf = type { i64, i16, i16, i32 }
@bfi = common global %struct.bf zeroinitializer, align 16

define void @t1() nounwind optsize ssp {
entry:
	%0 = load i32* bitcast (i16* getelementptr (%struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	%1 = or i32 %0, 65536
	store i32 %1, i32* bitcast (i16* getelementptr (%struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	ret void
}

define void @t2() nounwind optsize ssp {
entry:
	%0 = load i32* bitcast (i16* getelementptr (%struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	%1 = or i32 %0, 16842752
	store i32 %1, i32* bitcast (i16* getelementptr (%struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	ret void
}
