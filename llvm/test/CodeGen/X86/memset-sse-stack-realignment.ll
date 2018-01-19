; Make sure that we realign the stack. Mingw32 uses 4 byte stack alignment, we
; need 16 bytes for SSE and 32 bytes for AVX.

; RUN: llc < %s -mtriple=i386-pc-mingw32 -mcpu=pentium2 | FileCheck %s -check-prefix=NOSSE
; RUN: llc < %s -mtriple=i386-pc-mingw32 -mcpu=pentium3 | FileCheck %s -check-prefix=SSE1
; RUN: llc < %s -mtriple=i386-pc-mingw32 -mcpu=yonah | FileCheck %s -check-prefix=SSE2
; RUN: llc < %s -mtriple=i386-pc-mingw32 -mcpu=corei7-avx | FileCheck %s -check-prefix=AVX1
; RUN: llc < %s -mtriple=i386-pc-mingw32 -mcpu=core-avx2 | FileCheck %s -check-prefix=AVX2

define void @test1(i32 %t) nounwind {
  %tmp1210 = alloca i8, i32 32, align 4
  call void @llvm.memset.p0i8.i64(i8* align 4 %tmp1210, i8 0, i64 32, i1 false)
  %x = alloca i8, i32 %t
  call void @dummy(i8* %x)
  ret void

; NOSSE-LABEL: test1:
; NOSSE-NOT: and
; NOSSE: movl $0

; SSE1-LABEL: test1:
; SSE1: andl $-16
; SSE1: movl %esp, %esi
; SSE1: movaps

; SSE2-LABEL: test1:
; SSE2: andl $-16
; SSE2: movl %esp, %esi
; SSE2: movaps

; AVX1-LABEL: test1:
; AVX1: andl $-32
; AVX1: movl %esp, %esi
; AVX1: vmovaps %ymm

; AVX2-LABEL: test1:
; AVX2: andl $-32
; AVX2: movl %esp, %esi
; AVX2: vmovaps %ymm

}

define void @test2(i32 %t) nounwind {
  %tmp1210 = alloca i8, i32 16, align 4
  call void @llvm.memset.p0i8.i64(i8* align 4 %tmp1210, i8 0, i64 16, i1 false)
  %x = alloca i8, i32 %t
  call void @dummy(i8* %x)
  ret void

; NOSSE-LABEL: test2:
; NOSSE-NOT: and
; NOSSE: movl $0

; SSE1-LABEL: test2:
; SSE1: andl $-16
; SSE1: movl %esp, %esi
; SSE1: movaps

; SSE2-LABEL: test2:
; SSE2: andl $-16
; SSE2: movl %esp, %esi
; SSE2: movaps

; AVX1-LABEL: test2:
; AVX1: andl $-16
; AVX1: movl %esp, %esi
; AVX1: vmovaps %xmm

; AVX2-LABEL: test2:
; AVX2: andl $-16
; AVX2: movl %esp, %esi
; AVX2: vmovaps %xmm
}

declare void @dummy(i8*)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
