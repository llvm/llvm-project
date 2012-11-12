; RUN: llc < %s -march=x86 -mcpu=yonah -mtriple=i386-apple-darwin | FileCheck %s


define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test1:
; CHECK: pcmpgtd
; CHECK: ret

	%C = icmp sgt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test2(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test2:
; CHECK: pcmp
; CHECK: pcmp
; CHECK: pxor
; CHECK: ret
	%C = icmp sge <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test3(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test3:
; CHECK: pcmpgtd
; CHECK: movdqa
; CHECK: ret
	%C = icmp slt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test4(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test4:
; CHECK: movdqa
; CHECK: pcmpgtd
; CHECK: ret
	%C = icmp ugt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}
