; RUN: llc < %s -mattr=+sse2      -mtriple=i686-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=SSE2-Darwin
; RUN: llc < %s -mattr=+sse2      -mtriple=i686-pc-mingw32 -mcpu=core2 | FileCheck %s -check-prefix=SSE2-Mingw32
; RUN: llc < %s -mattr=+sse,-sse2 -mtriple=i686-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=SSE1
; RUN: llc < %s -mattr=-sse       -mtriple=i686-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=NOSSE
; RUN: llc < %s                 -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=X86-64
; RUN: llc < %s                 -mtriple=x86_64-apple-darwin -mcpu=nehalem | FileCheck %s -check-prefix=NHM_64


@.str = internal constant [25 x i8] c"image\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
@.str2 = internal constant [30 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 4

define void @t1(i32 %argc, ptr %argv) nounwind  {
entry:
; SSE2-Darwin-LABEL: t1:
; SSE2-Darwin: movsd _.str+16, %xmm0
; SSE2-Darwin: movsd %xmm0, 16(%esp)
; SSE2-Darwin: movaps _.str, %xmm0
; SSE2-Darwin: movaps %xmm0
; SSE2-Darwin: movb $0, 24(%esp)

; SSE2-Mingw32-LABEL: t1:
; SSE2-Mingw32: movsd _.str+16, %xmm0
; SSE2-Mingw32: movsd %xmm0, 16(%esp)
; SSE2-Mingw32: movaps _.str, %xmm0
; SSE2-Mingw32: movups %xmm0
; SSE2-Mingw32: movb $0, 24(%esp)

; SSE1-LABEL: t1:
; SSE1: movaps _.str, %xmm0
; SSE1: movaps %xmm0
; SSE1: movb $0, 24(%esp)
; SSE1: movl $0, 20(%esp)
; SSE1: movl $0, 16(%esp)

; NOSSE-LABEL: t1:
; NOSSE: movb $0
; NOSSE: movl $0
; NOSSE: movl $0
; NOSSE: movl $0
; NOSSE: movl $0
; NOSSE: movl $101
; NOSSE: movl $1734438249

; X86-64-LABEL: t1:
; X86-64: movaps _.str(%rip), %xmm0
; X86-64: movaps %xmm0
; X86-64: movb $0
; X86-64: movq $0
  %tmp1 = alloca [25 x i8]
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %tmp1, ptr align 1 @.str, i32 25, i1 false)
  unreachable
}

;rdar://7774704
%struct.s0 = type { [2 x double] }

define void @t2(ptr nocapture %a, ptr nocapture %b) nounwind ssp {
entry:
; SSE2-Darwin-LABEL: t2:
; SSE2-Darwin: movaps (%ecx), %xmm0
; SSE2-Darwin: movaps %xmm0, (%eax)

; SSE2-Mingw32-LABEL: t2:
; SSE2-Mingw32: movaps (%ecx), %xmm0
; SSE2-Mingw32: movaps %xmm0, (%eax)

; SSE1-LABEL: t2:
; SSE1: movaps (%ecx), %xmm0
; SSE1: movaps %xmm0, (%eax)

; NOSSE-LABEL: t2:
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl

; X86-64-LABEL: t2:
; X86-64: movaps (%rsi), %xmm0
; X86-64: movaps %xmm0, (%rdi)
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 16 %a, ptr align 16 %b, i32 16, i1 false)
  ret void
}

define void @t3(ptr nocapture %a, ptr nocapture %b) nounwind ssp {
entry:
; SSE2-Darwin-LABEL: t3:
; SSE2-Darwin: movsd (%ecx), %xmm0
; SSE2-Darwin: movsd 8(%ecx), %xmm1
; SSE2-Darwin: movsd %xmm1, 8(%eax)
; SSE2-Darwin: movsd %xmm0, (%eax)

; SSE2-Mingw32-LABEL: t3:
; SSE2-Mingw32: movsd (%ecx), %xmm0
; SSE2-Mingw32: movsd 8(%ecx), %xmm1
; SSE2-Mingw32: movsd %xmm1, 8(%eax)
; SSE2-Mingw32: movsd %xmm0, (%eax)

; SSE1-LABEL: t3:
; SSE1: movl
; SSE1: movl
; SSE1: movl
; SSE1: movl
; SSE1: movl
; SSE1: movl
; SSE1: movl
; SSE1: movl
; SSE1: movl
; SSE1: movl

; NOSSE-LABEL: t3:
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl

; X86-64-LABEL: t3:
; X86-64: movq (%rsi), %rax
; X86-64: movq 8(%rsi), %rcx
; X86-64: movq %rcx, 8(%rdi)
; X86-64: movq %rax, (%rdi)
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 8 %a, ptr align 8 %b, i32 16, i1 false)
  ret void
}

define void @t4() nounwind {
entry:
; SSE2-Darwin-LABEL: t4:
; SSE2-Darwin: movw $120
; SSE2-Darwin: movl $2021161080
; SSE2-Darwin: movl $2021161080
; SSE2-Darwin: movl $2021161080
; SSE2-Darwin: movl $2021161080
; SSE2-Darwin: movl $2021161080
; SSE2-Darwin: movl $2021161080
; SSE2-Darwin: movl $2021161080

; SSE2-Mingw32-LABEL: t4:
; SSE2-Mingw32: movw $120
; SSE2-Mingw32: movl $2021161080
; SSE2-Mingw32: movl $2021161080
; SSE2-Mingw32: movl $2021161080
; SSE2-Mingw32: movl $2021161080
; SSE2-Mingw32: movl $2021161080
; SSE2-Mingw32: movl $2021161080
; SSE2-Mingw32: movl $2021161080

; SSE1-LABEL: t4:
; SSE1: movw $120
; SSE1: movl $2021161080
; SSE1: movl $2021161080
; SSE1: movl $2021161080
; SSE1: movl $2021161080
; SSE1: movl $2021161080
; SSE1: movl $2021161080
; SSE1: movl $2021161080

; NOSSE-LABEL: t4:
; NOSSE: movw $120
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080

;;; TODO: (1) Some of the loads and stores are certainly unaligned and (2) the first load and first
;;; store overlap with the second load and second store respectively.
;;;
;;; Is either of the sequences ideal?

; X86-64-LABEL: t4:
; X86-64: movabsq  $33909456017848440, %rax ## imm = 0x78787878787878
; X86-64: movq     %rax, -10(%rsp)
; X86-64: movabsq  $8680820740569200760, %rax ## imm = 0x7878787878787878
; X86-64: movq     %rax, -16(%rsp)
; X86-64: movq     %rax, -24(%rsp)
; X86-64: movq     %rax, -32(%rsp)

; NHM_64-LABEL: t4:
; NHM_64: movups   _.str2+14(%rip), %xmm0
; NHM_64: movups   %xmm0, -26(%rsp)
; NHM_64: movups   _.str2(%rip), %xmm0
; NHM_64: movaps   %xmm0, -40(%rsp)

  %tmp1 = alloca [30 x i8]
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %tmp1, ptr align 1 @.str2, i32 30, i1 false)
  unreachable
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
