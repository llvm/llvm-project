; RUN: llc < %s -march=x86-64 | FileCheck %s

	%struct.bf = type { i64, i16, i16, i32 }
@bfi = common global %struct.bf zeroinitializer, align 16

define void @t1() nounwind optsize ssp {
entry:
	%0 = load i32, i32* bitcast (i16* getelementptr (%struct.bf, %struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	%1 = or i32 %0, 65536
	store i32 %1, i32* bitcast (i16* getelementptr (%struct.bf, %struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	ret void

; CHECK-LABEL: t1:
; CHECK: orb $1
; CHECK-NEXT: ret
}

define void @t2() nounwind optsize ssp {
entry:
	%0 = load i32, i32* bitcast (i16* getelementptr (%struct.bf, %struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	%1 = or i32 %0, 16842752
	store i32 %1, i32* bitcast (i16* getelementptr (%struct.bf, %struct.bf* @bfi, i32 0, i32 1) to i32*), align 8
	ret void

; CHECK-LABEL: t2:
; CHECK: orl $16842752
; CHECK-NEXT: ret
}
