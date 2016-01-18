; RUN: llc < %s -march=x86 -mattr=-sse3,+sse2 | FileCheck %s -check-prefix=SSE2
; RUN: llc < %s -march=x86 -mattr=-sse4.2,+sse4.1 | FileCheck %s -check-prefix=SSE41
; RUN: llc < %s -march=x86 -mattr=+sse4.2 | FileCheck %s -check-prefix=SSE42

define <2 x i64> @test1(<2 x i64> %A, <2 x i64> %B) nounwind {
; SSE42-LABEL: test1:
; SSE42: pcmpgtq
; SSE42: ret
; SSE41-LABEL: test1:
; SSE41-NOT: pcmpgtq
; SSE41: ret
; SSE2-LABEL: test1:
; SSE2-NOT: pcmpgtq
; SSE2: ret

	%C = icmp sgt <2 x i64> %A, %B
  %D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test2(<2 x i64> %A, <2 x i64> %B) nounwind {
; SSE42-LABEL: test2:
; SSE42: pcmpeqq
; SSE42: ret
; SSE41-LABEL: test2:
; SSE41: pcmpeqq
; SSE41: ret
; SSE2-LABEL: test2:
; SSE2-NOT: pcmpeqq
; SSE2: ret

	%C = icmp eq <2 x i64> %A, %B
  %D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}
