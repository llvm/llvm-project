; RUN: llc < %s -mtriple=x86_64-apple-darwin11 -mattr=+mmx,+sse2 | FileCheck %s
; rdar://6602459

@g_v1di = external global <1 x i64>

define void @t1() nounwind {
entry:
	%call = call <1 x i64> @return_v1di()		; <<1 x i64>> [#uses=0]
	store <1 x i64> %call, <1 x i64>* @g_v1di
        ret void
; CHECK: t1:
; CHECK: callq
; CHECK-NEXT: movq	_g_v1di
; CHECK-NEXT: movq	%rax,
}

declare <1 x i64> @return_v1di()

define <1 x i64> @t2() nounwind {
	ret <1 x i64> <i64 1>
; CHECK: t2:
; CHECK: movl	$1
; CHECK-NEXT: ret
}

define <2 x i32> @t3() nounwind {
	ret <2 x i32> <i32 1, i32 0>
; CHECK: t3:
; CHECK: movl $1
; CHECK: movd {{.*}}, %xmm0
}

define double @t4() nounwind {
	ret double bitcast (<2 x i32> <i32 1, i32 0> to double)
; CHECK: t4:
; CHECK: movl $1
; CHECK: movd {{.*}}, %xmm0
}

