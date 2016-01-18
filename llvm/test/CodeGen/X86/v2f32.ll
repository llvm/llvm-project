; RUN: llc < %s -mtriple=x86_64-linux -mcpu=penryn -o - | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mcpu=yonah -march=x86 -mtriple=i386-linux-gnu -o - | FileCheck %s --check-prefix=X32

; PR7518
define void @test1(<2 x float> %Q, float *%P2) nounwind {
; X64-LABEL: test1:
; X64:       # BB#0:
; X64-NEXT:    movshdup {{.*#+}} xmm1 = xmm0[1,1,3,3]
; X64-NEXT:    addss %xmm0, %xmm1
; X64-NEXT:    movss %xmm1, (%rdi)
; X64-NEXT:    retq
;
; X32-LABEL: test1:
; X32:       # BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movshdup {{.*#+}} xmm1 = xmm0[1,1,3,3]
; X32-NEXT:    addss %xmm0, %xmm1
; X32-NEXT:    movss %xmm1, (%eax)
; X32-NEXT:    retl
  %a = extractelement <2 x float> %Q, i32 0
  %b = extractelement <2 x float> %Q, i32 1
  %c = fadd float %a, %b
  store float %c, float* %P2
  ret void
}

define <2 x float> @test2(<2 x float> %Q, <2 x float> %R, <2 x float> *%P) nounwind {
; X64-LABEL: test2:
; X64:       # BB#0:
; X64-NEXT:    addps %xmm1, %xmm0
; X64-NEXT:    retq
;
; X32-LABEL: test2:
; X32:       # BB#0:
; X32-NEXT:    addps %xmm1, %xmm0
; X32-NEXT:    retl
  %Z = fadd <2 x float> %Q, %R
  ret <2 x float> %Z
}

define <2 x float> @test3(<4 x float> %A) nounwind {
; X64-LABEL: test3:
; X64:       # BB#0:
; X64-NEXT:    addps %xmm0, %xmm0
; X64-NEXT:    retq
;
; X32-LABEL: test3:
; X32:       # BB#0:
; X32-NEXT:    addps %xmm0, %xmm0
; X32-NEXT:    retl
	%B = shufflevector <4 x float> %A, <4 x float> undef, <2 x i32> <i32 0, i32 1>
	%C = fadd <2 x float> %B, %B
	ret <2 x float> %C
}

define <2 x float> @test4(<2 x float> %A) nounwind {
; X64-LABEL: test4:
; X64:       # BB#0:
; X64-NEXT:    addps %xmm0, %xmm0
; X64-NEXT:    retq
;
; X32-LABEL: test4:
; X32:       # BB#0:
; X32-NEXT:    addps %xmm0, %xmm0
; X32-NEXT:    retl
	%C = fadd <2 x float> %A, %A
	ret <2 x float> %C
}

define <4 x float> @test5(<4 x float> %A) nounwind {
; X64-LABEL: test5:
; X64:       # BB#0:
; X64-NEXT:    addps %xmm0, %xmm0
; X64-NEXT:    addps %xmm0, %xmm0
; X64-NEXT:    retq
;
; X32-LABEL: test5:
; X32:       # BB#0:
; X32-NEXT:    addps %xmm0, %xmm0
; X32-NEXT:    addps %xmm0, %xmm0
; X32-NEXT:    retl
	%B = shufflevector <4 x float> %A, <4 x float> undef, <2 x i32> <i32 0, i32 1>
	%C = fadd <2 x float> %B, %B
  br label %BB

BB:
  %D = fadd <2 x float> %C, %C
	%E = shufflevector <2 x float> %D, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
	ret <4 x float> %E
}


