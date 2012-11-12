; RUN: llc < %s -mtriple=i686-apple-darwin9 -mattr=sse41 -mcpu=penryn | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-apple-darwin9 -mattr=sse41 -mcpu=penryn | FileCheck %s -check-prefix=X64

@g16 = external global i16

define <4 x i32> @pinsrd_1(i32 %s, <4 x i32> %tmp) nounwind {
        %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 1
        ret <4 x i32> %tmp1
; X32: pinsrd_1:
; X32:    pinsrd $1, 4(%esp), %xmm0

; X64: pinsrd_1:
; X64:    pinsrd $1, %edi, %xmm0
}

define <16 x i8> @pinsrb_1(i8 %s, <16 x i8> %tmp) nounwind {
        %tmp1 = insertelement <16 x i8> %tmp, i8 %s, i32 1
        ret <16 x i8> %tmp1
; X32: pinsrb_1:
; X32:    pinsrb $1, 4(%esp), %xmm0

; X64: pinsrb_1:
; X64:    pinsrb $1, %edi, %xmm0
}


define <2 x i64> @pmovsxbd_1(i32* %p) nounwind {
entry:
	%0 = load i32* %p, align 4
	%1 = insertelement <4 x i32> undef, i32 %0, i32 0
	%2 = insertelement <4 x i32> %1, i32 0, i32 1
	%3 = insertelement <4 x i32> %2, i32 0, i32 2
	%4 = insertelement <4 x i32> %3, i32 0, i32 3
	%5 = bitcast <4 x i32> %4 to <16 x i8>
	%6 = tail call <4 x i32> @llvm.x86.sse41.pmovsxbd(<16 x i8> %5) nounwind readnone
	%7 = bitcast <4 x i32> %6 to <2 x i64>
	ret <2 x i64> %7
        
; X32: _pmovsxbd_1:
; X32:   movl      4(%esp), %eax
; X32:   pmovsxbd   (%eax), %xmm0

; X64: _pmovsxbd_1:
; X64:   pmovsxbd   (%rdi), %xmm0
}

define <2 x i64> @pmovsxwd_1(i64* %p) nounwind readonly {
entry:
	%0 = load i64* %p		; <i64> [#uses=1]
	%tmp2 = insertelement <2 x i64> zeroinitializer, i64 %0, i32 0		; <<2 x i64>> [#uses=1]
	%1 = bitcast <2 x i64> %tmp2 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%2 = tail call <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16> %1) nounwind readnone		; <<4 x i32>> [#uses=1]
	%3 = bitcast <4 x i32> %2 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %3
        
; X32: _pmovsxwd_1:
; X32:   movl 4(%esp), %eax
; X32:   pmovsxwd (%eax), %xmm0

; X64: _pmovsxwd_1:
; X64:   pmovsxwd (%rdi), %xmm0
}




define <2 x i64> @pmovzxbq_1() nounwind {
entry:
	%0 = load i16* @g16, align 2		; <i16> [#uses=1]
	%1 = insertelement <8 x i16> undef, i16 %0, i32 0		; <<8 x i16>> [#uses=1]
	%2 = bitcast <8 x i16> %1 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%3 = tail call <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8> %2) nounwind readnone		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %3

; X32: _pmovzxbq_1:
; X32:   movl	L_g16$non_lazy_ptr, %eax
; X32:   pmovzxbq	(%eax), %xmm0

; X64: _pmovzxbq_1:
; X64:   movq	_g16@GOTPCREL(%rip), %rax
; X64:   pmovzxbq	(%rax), %xmm0
}

declare <4 x i32> @llvm.x86.sse41.pmovsxbd(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8>) nounwind readnone




define i32 @extractps_1(<4 x float> %v) nounwind {
  %s = extractelement <4 x float> %v, i32 3
  %i = bitcast float %s to i32
  ret i32 %i

; X32: _extractps_1:  
; X32:	  extractps	$3, %xmm0, %eax

; X64: _extractps_1:  
; X64:	  extractps	$3, %xmm0, %eax
}
define i32 @extractps_2(<4 x float> %v) nounwind {
  %t = bitcast <4 x float> %v to <4 x i32>
  %s = extractelement <4 x i32> %t, i32 3
  ret i32 %s

; X32: _extractps_2:
; X32:	  extractps	$3, %xmm0, %eax

; X64: _extractps_2:
; X64:	  extractps	$3, %xmm0, %eax
}


; The non-store form of extractps puts its result into a GPR.
; This makes it suitable for an extract from a <4 x float> that
; is bitcasted to i32, but unsuitable for much of anything else.

define float @ext_1(<4 x float> %v) nounwind {
  %s = extractelement <4 x float> %v, i32 3
  %t = fadd float %s, 1.0
  ret float %t

; X32: _ext_1:
; X32:	  pshufd	$3, %xmm0, %xmm0
; X32:	  addss	LCPI7_0, %xmm0

; X64: _ext_1:
; X64:	  pshufd	$3, %xmm0, %xmm0
; X64:	  addss	LCPI7_0(%rip), %xmm0
}
define float @ext_2(<4 x float> %v) nounwind {
  %s = extractelement <4 x float> %v, i32 3
  ret float %s

; X32: _ext_2:
; X32:	  pshufd	$3, %xmm0, %xmm0

; X64: _ext_2:
; X64:	  pshufd	$3, %xmm0, %xmm0
}
define i32 @ext_3(<4 x i32> %v) nounwind {
  %i = extractelement <4 x i32> %v, i32 3
  ret i32 %i

; X32: _ext_3:
; X32:	  pextrd	$3, %xmm0, %eax

; X64: _ext_3:
; X64:	  pextrd	$3, %xmm0, %eax
}

define <4 x float> @insertps_1(<4 x float> %t1, <4 x float> %t2) nounwind {
        %tmp1 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %t1, <4 x float> %t2, i32 1) nounwind readnone
        ret <4 x float> %tmp1
; X32: _insertps_1:
; X32:    insertps  $1, %xmm1, %xmm0

; X64: _insertps_1:
; X64:    insertps  $1, %xmm1, %xmm0
}

declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i32) nounwind readnone

define <4 x float> @insertps_2(<4 x float> %t1, float %t2) nounwind {
        %tmp1 = insertelement <4 x float> %t1, float %t2, i32 0
        ret <4 x float> %tmp1
; X32: _insertps_2:
; X32:    insertps  $0, 4(%esp), %xmm0

; X64: _insertps_2:
; X64:    insertps  $0, %xmm1, %xmm0        
}

define <4 x float> @insertps_3(<4 x float> %t1, <4 x float> %t2) nounwind {
        %tmp2 = extractelement <4 x float> %t2, i32 0
        %tmp1 = insertelement <4 x float> %t1, float %tmp2, i32 0
        ret <4 x float> %tmp1
; X32: _insertps_3:
; X32:    insertps  $0, %xmm1, %xmm0        

; X64: _insertps_3:
; X64:    insertps  $0, %xmm1, %xmm0        
}

define i32 @ptestz_1(<2 x i64> %t1, <2 x i64> %t2) nounwind {
        %tmp1 = call i32 @llvm.x86.sse41.ptestz(<2 x i64> %t1, <2 x i64> %t2) nounwind readnone
        ret i32 %tmp1
; X32: _ptestz_1:
; X32:    ptest 	%xmm1, %xmm0
; X32:    sete	%al

; X64: _ptestz_1:
; X64:    ptest 	%xmm1, %xmm0
; X64:    sete	%al
}

define i32 @ptestz_2(<2 x i64> %t1, <2 x i64> %t2) nounwind {
        %tmp1 = call i32 @llvm.x86.sse41.ptestc(<2 x i64> %t1, <2 x i64> %t2) nounwind readnone
        ret i32 %tmp1
; X32: _ptestz_2:
; X32:    ptest 	%xmm1, %xmm0
; X32:    sbbl	%eax

; X64: _ptestz_2:
; X64:    ptest 	%xmm1, %xmm0
; X64:    sbbl	%eax
}

define i32 @ptestz_3(<2 x i64> %t1, <2 x i64> %t2) nounwind {
        %tmp1 = call i32 @llvm.x86.sse41.ptestnzc(<2 x i64> %t1, <2 x i64> %t2) nounwind readnone
        ret i32 %tmp1
; X32: _ptestz_3:
; X32:    ptest 	%xmm1, %xmm0
; X32:    seta	%al

; X64: _ptestz_3:
; X64:    ptest 	%xmm1, %xmm0
; X64:    seta	%al
}


declare i32 @llvm.x86.sse41.ptestz(<2 x i64>, <2 x i64>) nounwind readnone
declare i32 @llvm.x86.sse41.ptestc(<2 x i64>, <2 x i64>) nounwind readnone
declare i32 @llvm.x86.sse41.ptestnzc(<2 x i64>, <2 x i64>) nounwind readnone

; This used to compile to insertps $0  + insertps $16.  insertps $0 is always
; pointless.
define <2 x float> @buildvector(<2 x float> %A, <2 x float> %B) nounwind  {
entry:
  %tmp7 = extractelement <2 x float> %A, i32 0
  %tmp5 = extractelement <2 x float> %A, i32 1
  %tmp3 = extractelement <2 x float> %B, i32 0
  %tmp1 = extractelement <2 x float> %B, i32 1
  %add.r = fadd float %tmp7, %tmp3
  %add.i = fadd float %tmp5, %tmp1
  %tmp11 = insertelement <2 x float> undef, float %add.r, i32 0
  %tmp9 = insertelement <2 x float> %tmp11, float %add.i, i32 1
  ret <2 x float> %tmp9
; X32: buildvector:
; X32-NOT: insertps $0
; X32: insertps $16
; X32-NOT: insertps $0
; X32: ret
; X64: buildvector:
; X64-NOT: insertps $0
; X64: insertps $16
; X64-NOT: insertps $0
; X64: ret
}

