; RUN: llc < %s -march=x86 -mattr=+sse2,-mmx | FileCheck %s

; Test that turning off mmx doesn't turn off sse

define void @test1(<2 x double>* %r, <2 x double>* %A, double %B) nounwind  {
; CHECK-LABEL: test1:
; CHECK:       # BB#0:
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movapd (%ecx), %xmm0
; CHECK-NEXT:    movlpd {{[0-9]+}}(%esp), %xmm0
; CHECK-NEXT:    movapd %xmm0, (%eax)
; CHECK-NEXT:    retl
	%tmp3 = load <2 x double>, <2 x double>* %A, align 16
	%tmp7 = insertelement <2 x double> undef, double %B, i32 0
	%tmp9 = shufflevector <2 x double> %tmp3, <2 x double> %tmp7, <2 x i32> < i32 2, i32 1 >
	store <2 x double> %tmp9, <2 x double>* %r, align 16
	ret void
}
