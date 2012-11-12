; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep movaps
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep esp | count 2

; These should both generate something like this:
;_test3:
;	movl	$1234567, %eax
;	andl	4(%esp), %eax
;	movd	%eax, %xmm0
;	ret

define <2 x i64> @test3(i64 %arg) nounwind {
entry:
        %A = and i64 %arg, 1234567
        %B = insertelement <2 x i64> zeroinitializer, i64 %A, i32 0
        ret <2 x i64> %B
}

define <2 x i64> @test2(i64 %arg) nounwind {
entry:
	%A = and i64 %arg, 1234567
	%B = insertelement <2 x i64> undef, i64 %A, i32 0
	ret <2 x i64> %B
}

