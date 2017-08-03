; RUN: llc < %s -march=x86 -mcpu=yonah -mtriple=i386-apple-darwin | FileCheck %s


define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test1:
; CHECK: pcmpgtd
; CHECK: ret

	%C = icmp sgt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test2(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test2:
; CHECK: pcmp
; CHECK: pcmp
; CHECK: pxor
; CHECK: ret
	%C = icmp sge <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test3(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test3:
; CHECK: pcmpgtd
; CHECK: movdqa
; CHECK: ret
	%C = icmp slt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test4(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test4:
; CHECK: movdqa
; CHECK: pcmpgtd
; CHECK: ret
	%C = icmp ugt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <2 x i64> @test5(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test5:
; CHECK: pcmpeqd
; CHECK: pshufd $177
; CHECK: pand
; CHECK: ret
	%C = icmp eq <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test6(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test6:
; CHECK: pcmpeqd
; CHECK: pshufd $177
; CHECK: pand
; CHECK: pcmpeqd
; CHECK: pxor
; CHECK: ret
	%C = icmp ne <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test7(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK: [[CONSTSEG:[A-Z0-9_]*]]:
; CHECK:      .long	2147483648
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	2147483648
; CHECK-NEXT: .long	0
; CHECK-LABEL: test7:
; CHECK: movdqa [[CONSTSEG]], [[CONSTREG:%xmm[0-9]*]]
; CHECK: pxor [[CONSTREG]]
; CHECK: pxor [[CONSTREG]]
; CHECK: pcmpgtd %xmm1
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: ret
	%C = icmp sgt <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test8(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test8:
; CHECK: pxor
; CHECK: pxor
; CHECK: pcmpgtd %xmm0
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: ret
	%C = icmp slt <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test9(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test9:
; CHECK: pxor
; CHECK: pxor
; CHECK: pcmpgtd %xmm0
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: pcmpeqd
; CHECK: pxor
; CHECK: ret
	%C = icmp sge <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test10(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test10:
; CHECK: pxor
; CHECK: pxor
; CHECK: pcmpgtd %xmm1
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: pcmpeqd
; CHECK: pxor
; CHECK: ret
	%C = icmp sle <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test11(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK: [[CONSTSEG:[A-Z0-9_]*]]:
; CHECK:      .long	2147483648
; CHECK-NEXT: .long	2147483648
; CHECK-NEXT: .long	2147483648
; CHECK-NEXT: .long	2147483648
; CHECK-LABEL: test11:
; CHECK: movdqa [[CONSTSEG]], [[CONSTREG:%xmm[0-9]*]]
; CHECK: pxor [[CONSTREG]]
; CHECK: pxor [[CONSTREG]]
; CHECK: pcmpgtd %xmm1
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: ret
	%C = icmp ugt <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test12(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test12:
; CHECK: pxor
; CHECK: pxor
; CHECK: pcmpgtd %xmm0
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: ret
	%C = icmp ult <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test13(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test13:
; CHECK: pxor
; CHECK: pxor
; CHECK: pcmpgtd %xmm0
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: pcmpeqd
; CHECK: pxor
; CHECK: ret
	%C = icmp uge <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test14(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: test14:
; CHECK: pxor
; CHECK: pxor
; CHECK: pcmpgtd %xmm1
; CHECK: pshufd $160
; CHECK: pcmpeqd
; CHECK: pshufd $245
; CHECK: pand
; CHECK: pshufd $245
; CHECK: por
; CHECK: pcmpeqd
; CHECK: pxor
; CHECK: ret
	%C = icmp ule <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}
