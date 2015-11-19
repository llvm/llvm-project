; RUN: llc < %s -march=x86 -mcpu=pentium2 -mtriple=i686-apple-darwin8.8.0 | FileCheck %s --check-prefix=X86
; RUN: llc < %s -march=x86 -mcpu=pentium3 -mtriple=i686-apple-darwin8.8.0 | FileCheck %s --check-prefix=XMM
; RUN: llc < %s -march=x86 -mcpu=bdver1   -mtriple=i686-apple-darwin8.8.0 | FileCheck %s --check-prefix=YMM

	%struct.x = type { i16, i16 }

define void @t() nounwind  {
entry:
	%up_mvd = alloca [8 x %struct.x]		; <[8 x %struct.x]*> [#uses=2]
	%up_mvd116 = getelementptr [8 x %struct.x], [8 x %struct.x]* %up_mvd, i32 0, i32 0		; <%struct.x*> [#uses=1]
	%tmp110117 = bitcast [8 x %struct.x]* %up_mvd to i8*		; <i8*> [#uses=1]

	call void @llvm.memset.p0i8.i64(i8* %tmp110117, i8 0, i64 32, i1 false)
; X86: movl $0,
; X86: movl $0,
; X86: movl $0,
; X86: movl $0,
; X86: movl $0,
; X86: movl $0,
; X86: movl $0,
; X86: movl $0,
; X86-NOT: movl $0,
; X86: ret

; XMM: xorps %xmm{{[0-9]+}}, [[Z:%xmm[0-9]+]]
; XMM: movaps [[Z]],
; XMM: movaps [[Z]],
; XMM-NOT: movaps
; XMM: ret

; YMM: vxorps %ymm{{[0-9]+}}, %ymm{{[0-9]+}}, [[Z:%ymm[0-9]+]]
; YMM: vmovaps [[Z]],
; YMM-NOT: movaps
; YMM: ret

	call void @foo( %struct.x* %up_mvd116 ) nounwind 
	ret void
}

declare void @foo(%struct.x*)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

define void @PR15348(i8* %a) {
; Ensure that alignment of '0' in an @llvm.memset intrinsic results in
; unaligned loads and stores.
; XMM: PR15348
; XMM: movb $0,
; XMM: movl $0,
; XMM: movl $0,
; XMM: movl $0,
; XMM: movl $0,
  call void @llvm.memset.p0i8.i64(i8* %a, i8 0, i64 17, i1 false)
  ret void
}
