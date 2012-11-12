; RUN: llc < %s -mtriple=x86_64-linux -mcpu=penryn -asm-verbose=0 -o - | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=penryn -asm-verbose=0 -o - | FileCheck %s -check-prefix=W64
; RUN: llc < %s -mcpu=yonah -march=x86 -mtriple=i386-linux-gnu -asm-verbose=0 -o - | FileCheck %s -check-prefix=X32

; PR7518
define void @test1(<2 x float> %Q, float *%P2) nounwind {
  %a = extractelement <2 x float> %Q, i32 0
  %b = extractelement <2 x float> %Q, i32 1
  %c = fadd float %a, %b

  store float %c, float* %P2
  ret void
; X64: test1:
; X64-NEXT: pshufd	$1, %xmm0, %xmm1
; X64-NEXT: addss	%xmm0, %xmm1
; X64-NEXT: movss	%xmm1, (%rdi)
; X64-NEXT: ret

; W64: test1:
; W64-NEXT: movdqa  (%rcx), %xmm0
; W64-NEXT: pshufd  $1, %xmm0, %xmm1
; W64-NEXT: addss   %xmm0, %xmm1
; W64-NEXT: movss   %xmm1, (%rdx)
; W64-NEXT: ret

; X32: test1:
; X32-NEXT: pshufd	$1, %xmm0, %xmm1
; X32-NEXT: addss	%xmm0, %xmm1
; X32-NEXT: movl	4(%esp), %eax
; X32-NEXT: movss	%xmm1, (%eax)
; X32-NEXT: ret
}


define <2 x float> @test2(<2 x float> %Q, <2 x float> %R, <2 x float> *%P) nounwind {
  %Z = fadd <2 x float> %Q, %R
  ret <2 x float> %Z
  
; X64: test2:
; X64-NEXT: addps	%xmm1, %xmm0
; X64-NEXT: ret

; W64: test2:
; W64-NEXT: movaps  (%rcx), %xmm0
; W64-NEXT: addps   (%rdx), %xmm0
; W64-NEXT: ret

; X32: test2:
; X32:      addps	%xmm1, %xmm0
}


define <2 x float> @test3(<4 x float> %A) nounwind {
	%B = shufflevector <4 x float> %A, <4 x float> undef, <2 x i32> <i32 0, i32 1>
	%C = fadd <2 x float> %B, %B
	ret <2 x float> %C
; X64: test3:
; X64-NEXT: addps	%xmm0, %xmm0
; X64-NEXT: ret

; W64: test3:
; W64-NEXT: movaps  (%rcx), %xmm0
; W64-NEXT: addps   %xmm0, %xmm0
; W64-NEXT: ret

; X32: test3:
; X32-NEXT: addps	%xmm0, %xmm0
; X32-NEXT: ret
}

define <2 x float> @test4(<2 x float> %A) nounwind {
	%C = fadd <2 x float> %A, %A
	ret <2 x float> %C
; X64: test4:
; X64-NEXT: addps	%xmm0, %xmm0
; X64-NEXT: ret

; W64: test4:
; W64-NEXT: movaps  (%rcx), %xmm0
; W64-NEXT: addps   %xmm0, %xmm0
; W64-NEXT: ret

; X32: test4:
; X32-NEXT: addps	%xmm0, %xmm0
; X32-NEXT: ret
}

define <4 x float> @test5(<4 x float> %A) nounwind {
	%B = shufflevector <4 x float> %A, <4 x float> undef, <2 x i32> <i32 0, i32 1>
	%C = fadd <2 x float> %B, %B
        br label %BB
        
BB:
        %D = fadd <2 x float> %C, %C
	%E = shufflevector <2 x float> %D, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
	ret <4 x float> %E
        
; X64: test5:
; X64-NEXT: addps	%xmm0, %xmm0
; X64-NEXT: addps	%xmm0, %xmm0
; X64-NEXT: ret

; W64: test5:
; W64-NEXT: movaps  (%rcx), %xmm0
; W64-NEXT: addps   %xmm0, %xmm0
; W64-NEXT: addps   %xmm0, %xmm0
; W64-NEXT: ret

; X32: test5:
; X32-NEXT: addps	%xmm0, %xmm0
; X32-NEXT: addps	%xmm0, %xmm0
; X32-NEXT: ret
}


