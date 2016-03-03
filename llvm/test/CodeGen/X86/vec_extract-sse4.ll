; RUN: llc < %s -mcpu=corei7 -march=x86 -mattr=+sse4.1 | FileCheck %s

define void @t1(float* %R, <4 x float>* %P1) nounwind {
; CHECK-LABEL: t1:
; CHECK:       # BB#0:
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movss 12(%ecx), %xmm0
; CHECK-NEXT:    movss %xmm0, (%eax)
; CHECK-NEXT:    retl

	%X = load <4 x float>, <4 x float>* %P1
	%tmp = extractelement <4 x float> %X, i32 3
	store float %tmp, float* %R
	ret void
}

define float @t2(<4 x float>* %P1) nounwind {
; CHECK-LABEL: t2:
; CHECK:       # BB#0:
; CHECK-NEXT:    pushl %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movddup {{.*#+}} xmm0 = mem[0,0]
; CHECK-NEXT:    movss %xmm0, (%esp)
; CHECK-NEXT:    flds (%esp)
; CHECK-NEXT:    popl %eax
; CHECK-NEXT:    retl

	%X = load <4 x float>, <4 x float>* %P1
	%tmp = extractelement <4 x float> %X, i32 2
	ret float %tmp
}

define void @t3(i32* %R, <4 x i32>* %P1) nounwind {
; CHECK-LABEL: t3:
; CHECK:       # BB#0:
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movl 12(%ecx), %ecx
; CHECK-NEXT:    movl %ecx, (%eax)
; CHECK-NEXT:    retl

	%X = load <4 x i32>, <4 x i32>* %P1
	%tmp = extractelement <4 x i32> %X, i32 3
	store i32 %tmp, i32* %R
	ret void
}

define i32 @t4(<4 x i32>* %P1) nounwind {
; CHECK-LABEL: t4:
; CHECK:       # BB#0:
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl 12(%eax), %eax
; CHECK-NEXT:    retl

	%X = load <4 x i32>, <4 x i32>* %P1
	%tmp = extractelement <4 x i32> %X, i32 3
	ret i32 %tmp
}
