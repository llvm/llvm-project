; RUN: llc < %s -mtriple=i686-apple-darwin9 -mattr=sse4.1 -mcpu=penryn | FileCheck %s -check-prefix=X32 --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-apple-darwin9 -mattr=sse4.1 -mcpu=penryn | FileCheck %s -check-prefix=X64 --check-prefix=CHECK

@g16 = external global i16

define <4 x i32> @pinsrd_1(i32 %s, <4 x i32> %tmp) nounwind {
        %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 1
        ret <4 x i32> %tmp1
; X32-LABEL: pinsrd_1:
; X32:    pinsrd $1, 4(%esp), %xmm0

; X64-LABEL: pinsrd_1:
; X64:    pinsrd $1, %edi, %xmm0
}

define <16 x i8> @pinsrb_1(i8 %s, <16 x i8> %tmp) nounwind {
        %tmp1 = insertelement <16 x i8> %tmp, i8 %s, i32 1
        ret <16 x i8> %tmp1
; X32-LABEL: pinsrb_1:
; X32:    pinsrb $1, 4(%esp), %xmm0

; X64-LABEL: pinsrb_1:
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
; X32-LABEL: buildvector:
; X32-NOT: insertps $0
; X32: insertps $16
; X32-NOT: insertps $0
; X32: ret
; X64-LABEL: buildvector:
; X64-NOT: insertps $0
; X64: insertps $16
; X64-NOT: insertps $0
; X64: ret
}

define <4 x float> @insertps_from_shufflevector_1(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
entry:
  %0 = load <4 x float>* %pb, align 16
  %vecinit6 = shufflevector <4 x float> %a, <4 x float> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecinit6
; CHECK-LABEL: insertps_from_shufflevector_1:
; CHECK-NOT: movss
; CHECK-NOT: shufps
; CHECK: insertps    $48,
; CHECK: ret
}

define <4 x float> @insertps_from_shufflevector_2(<4 x float> %a, <4 x float> %b) {
entry:
  %vecinit6 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x float> %vecinit6
; CHECK-LABEL: insertps_from_shufflevector_2:
; CHECK-NOT: shufps
; CHECK: insertps    $96,
; CHECK: ret
}

; For loading an i32 from memory into an xmm register we use pinsrd
; instead of insertps
define <4 x i32> @pinsrd_from_shufflevector_i32(<4 x i32> %a, <4 x i32>* nocapture readonly %pb) {
entry:
  %0 = load <4 x i32>* %pb, align 16
  %vecinit6 = shufflevector <4 x i32> %a, <4 x i32> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecinit6
; CHECK-LABEL: pinsrd_from_shufflevector_i32:
; CHECK-NOT: movss
; CHECK-NOT: shufps
; CHECK: pinsrd  $3,
; CHECK: ret
}

define <4 x i32> @insertps_from_shufflevector_i32_2(<4 x i32> %a, <4 x i32> %b) {
entry:
  %vecinit6 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 7, i32 2, i32 3>
  ret <4 x i32> %vecinit6
; CHECK-LABEL: insertps_from_shufflevector_i32_2:
; CHECK-NOT: shufps
; CHECK-NOT: movaps
; CHECK: insertps    $208,
; CHECK: ret
}

define <4 x float> @insertps_from_load_ins_elt_undef(<4 x float> %a, float* %b) {
; CHECK-LABEL: insertps_from_load_ins_elt_undef:
; CHECK-NOT: movss
; CHECK-NOT: shufps
; CHECK: insertps    $16,
; CHECK: ret
  %1 = load float* %b, align 4
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %result = shufflevector <4 x float> %a, <4 x float> %2, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
  ret <4 x float> %result
}

define <4 x i32> @insertps_from_load_ins_elt_undef_i32(<4 x i32> %a, i32* %b) {
; CHECK-LABEL: insertps_from_load_ins_elt_undef_i32:
; TODO: Like on pinsrd_from_shufflevector_i32, remove this mov instr
;; aCHECK-NOT: movd
; CHECK-NOT: shufps
; CHECK: insertps    $32,
; CHECK: ret
  %1 = load i32* %b, align 4
  %2 = insertelement <4 x i32> undef, i32 %1, i32 0
  %result = shufflevector <4 x i32> %a, <4 x i32> %2, <4 x i32> <i32 0, i32 1, i32 4, i32 3>
  ret <4 x i32> %result
}

;;;;;; Shuffles optimizable with a single insertps instruction
define <4 x float> @shuf_XYZ0(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_XYZ0:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $8
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %x, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecext3 = extractelement <4 x float> %x, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit2, float %vecext3, i32 2
  %vecinit5 = insertelement <4 x float> %vecinit4, float 0.0, i32 3
  ret <4 x float> %vecinit5
}

define <4 x float> @shuf_XY00(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_XY00:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $12
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %x, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.0, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float 0.0, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_XYY0(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_XYY0:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $104
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %x, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecinit4 = insertelement <4 x float> %vecinit2, float %vecext1, i32 2
  %vecinit5 = insertelement <4 x float> %vecinit4, float 0.0, i32 3
  ret <4 x float> %vecinit5
}

define <4 x float> @shuf_XYW0(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_XYW0:
; CHECK: insertps    $232
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %x, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecext2 = extractelement <4 x float> %x, i32 3
  %vecinit3 = insertelement <4 x float> %vecinit2, float %vecext2, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float 0.0, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_W00W(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_W00W:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $198
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 3
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit2 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.0, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float %vecext, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_X00A(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_X00A:
; CHECK-NOT: movaps
; CHECK-NOT: shufps
; CHECK: insertps    $48
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit1, float 0.0, i32 2
  %vecinit4 = shufflevector <4 x float> %vecinit2, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_X00X(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_X00X:
; CHECK-NOT: movaps
; CHECK-NOT: shufps
; CHECK: insertps    $48
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit1, float 0.0, i32 2
  %vecinit4 = shufflevector <4 x float> %vecinit2, <4 x float> %x, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_X0YC(<4 x float> %x, <4 x float> %a) {
; CHECK-LABEL: shuf_X0YC:
; CHECK: shufps
; CHECK-NOT: movhlps
; CHECK-NOT: shufps
; CHECK: insertps    $176
; CHECK: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit3 = shufflevector <4 x float> %vecinit1, <4 x float> %x, <4 x i32> <i32 0, i32 1, i32 5, i32 undef>
  %vecinit5 = shufflevector <4 x float> %vecinit3, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x float> %vecinit5
}

define <4 x i32> @i32_shuf_XYZ0(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_XYZ0:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $8
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <4 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecext3 = extractelement <4 x i32> %x, i32 2
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %vecext3, i32 2
  %vecinit5 = insertelement <4 x i32> %vecinit4, i32 0, i32 3
  ret <4 x i32> %vecinit5
}

define <4 x i32> @i32_shuf_XY00(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_XY00:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $12
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <4 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 0, i32 2
  %vecinit4 = insertelement <4 x i32> %vecinit3, i32 0, i32 3
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_XYY0(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_XYY0:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $104
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <4 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %vecext1, i32 2
  %vecinit5 = insertelement <4 x i32> %vecinit4, i32 0, i32 3
  ret <4 x i32> %vecinit5
}

define <4 x i32> @i32_shuf_XYW0(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_XYW0:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $232
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <4 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecext2 = extractelement <4 x i32> %x, i32 3
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 %vecext2, i32 2
  %vecinit4 = insertelement <4 x i32> %vecinit3, i32 0, i32 3
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_W00W(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_W00W:
; CHECK-NOT: pextrd
; CHECK-NOT: punpckldq
; CHECK: insertps    $198
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 3
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 0, i32 2
  %vecinit4 = insertelement <4 x i32> %vecinit3, i32 %vecext, i32 3
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_X00A(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_X00A:
; CHECK-NOT: movaps
; CHECK-NOT: shufps
; CHECK: insertps    $48
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 0, i32 2
  %vecinit4 = shufflevector <4 x i32> %vecinit2, <4 x i32> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_X00X(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_X00X:
; CHECK-NOT: movaps
; CHECK-NOT: shufps
; CHECK: insertps    $48
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 0, i32 2
  %vecinit4 = shufflevector <4 x i32> %vecinit2, <4 x i32> %x, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_X0YC(<4 x i32> %x, <4 x i32> %a) {
; CHECK-LABEL: i32_shuf_X0YC:
; CHECK: shufps
; CHECK-NOT: movhlps
; CHECK-NOT: shufps
; CHECK: insertps    $176
; CHECK: ret
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit3 = shufflevector <4 x i32> %vecinit1, <4 x i32> %x, <4 x i32> <i32 0, i32 1, i32 5, i32 undef>
  %vecinit5 = shufflevector <4 x i32> %vecinit3, <4 x i32> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x i32> %vecinit5
}

;; Test for a bug in the first implementation of LowerBuildVectorv4x32
define < 4 x float> @test_insertps_no_undef(<4 x float> %x) {
; CHECK-LABEL: test_insertps_no_undef:
; CHECK: movaps  %xmm0, %xmm1
; CHECK-NEXT: insertps        $8, %xmm1, %xmm1
; CHECK-NEXT: maxps   %xmm1, %xmm0
; CHECK-NEXT: ret
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %x, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecext3 = extractelement <4 x float> %x, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit2, float %vecext3, i32 2
  %vecinit5 = insertelement <4 x float> %vecinit4, float 0.0, i32 3
  %mask = fcmp olt <4 x float> %vecinit5, %x
  %res = select  <4 x i1> %mask, <4 x float> %x, <4 x float>%vecinit5
  ret <4 x float> %res
}

define <8 x i16> @blendvb_fallback(<8 x i1> %mask, <8 x i16> %x, <8 x i16> %y) {
; CHECK-LABEL: blendvb_fallback
; CHECK: blendvb
; CHECK: ret
  %ret = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %ret
}

define <4 x float> @insertps_from_vector_load(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
; CHECK-LABEL: insertps_from_vector_load:
; On X32, account for the argument's move to registers
; X32: movl    4(%esp), %eax
; CHECK-NOT: mov
; CHECK: insertps    $48
; CHECK-NEXT: ret
  %1 = load <4 x float>* %pb, align 16
  %2 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %1, i32 48)
  ret <4 x float> %2
}

;; Use a non-zero CountS for insertps
define <4 x float> @insertps_from_vector_load_offset(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
; CHECK-LABEL: insertps_from_vector_load_offset:
; On X32, account for the argument's move to registers
; X32: movl    4(%esp), %eax
; CHECK-NOT: mov
;; Try to match a bit more of the instr, since we need the load's offset.
; CHECK: insertps    $96, 4(%{{...}}), %
; CHECK-NEXT: ret
  %1 = load <4 x float>* %pb, align 16
  %2 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %1, i32 96)
  ret <4 x float> %2
}

define <4 x float> @insertps_from_vector_load_offset_2(<4 x float> %a, <4 x float>* nocapture readonly %pb, i64 %index) {
; CHECK-LABEL: insertps_from_vector_load_offset_2:
; On X32, account for the argument's move to registers
; X32: movl    4(%esp), %eax
; X32: movl    8(%esp), %ecx
; CHECK-NOT: mov
;; Try to match a bit more of the instr, since we need the load's offset.
; CHECK: insertps    $192, 12(%{{...}},%{{...}}), %
; CHECK-NEXT: ret
  %1 = getelementptr inbounds <4 x float>* %pb, i64 %index
  %2 = load <4 x float>* %1, align 16
  %3 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %2, i32 192)
  ret <4 x float> %3
}

define <4 x float> @insertps_from_broadcast_loadf32(<4 x float> %a, float* nocapture readonly %fb, i64 %index) {
; CHECK-LABEL: insertps_from_broadcast_loadf32:
; On X32, account for the arguments' move to registers
; X32: movl    8(%esp), %eax
; X32: movl    4(%esp), %ecx
; CHECK-NOT: mov
; CHECK: insertps    $48
; CHECK-NEXT: ret
  %1 = getelementptr inbounds float* %fb, i64 %index
  %2 = load float* %1, align 4
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = insertelement <4 x float> %3, float %2, i32 1
  %5 = insertelement <4 x float> %4, float %2, i32 2
  %6 = insertelement <4 x float> %5, float %2, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %6, i32 48)
  ret <4 x float> %7
}

define <4 x float> @insertps_from_broadcast_loadv4f32(<4 x float> %a, <4 x float>* nocapture readonly %b) {
; CHECK-LABEL: insertps_from_broadcast_loadv4f32:
; On X32, account for the arguments' move to registers
; X32: movl    4(%esp), %{{...}}
; CHECK-NOT: mov
; CHECK: insertps    $48
; CHECK-NEXT: ret
  %1 = load <4 x float>* %b, align 4
  %2 = extractelement <4 x float> %1, i32 0
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = insertelement <4 x float> %3, float %2, i32 1
  %5 = insertelement <4 x float> %4, float %2, i32 2
  %6 = insertelement <4 x float> %5, float %2, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %6, i32 48)
  ret <4 x float> %7
}

;; FIXME: We're emitting an extraneous pshufd/vbroadcast.
define <4 x float> @insertps_from_broadcast_multiple_use(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, float* nocapture readonly %fb, i64 %index) {
; CHECK-LABEL: insertps_from_broadcast_multiple_use:
; On X32, account for the arguments' move to registers
; X32: movl    8(%esp), %eax
; X32: movl    4(%esp), %ecx
; CHECK: movss
; CHECK-NOT: mov
; CHECK: insertps    $48
; CHECK: insertps    $48
; CHECK: insertps    $48
; CHECK: insertps    $48
; CHECK: addps
; CHECK: addps
; CHECK: addps
; CHECK-NEXT: ret
  %1 = getelementptr inbounds float* %fb, i64 %index
  %2 = load float* %1, align 4
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = insertelement <4 x float> %3, float %2, i32 1
  %5 = insertelement <4 x float> %4, float %2, i32 2
  %6 = insertelement <4 x float> %5, float %2, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %6, i32 48)
  %8 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %b, <4 x float> %6, i32 48)
  %9 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %c, <4 x float> %6, i32 48)
  %10 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %d, <4 x float> %6, i32 48)
  %11 = fadd <4 x float> %7, %8
  %12 = fadd <4 x float> %9, %10
  %13 = fadd <4 x float> %11, %12
  ret <4 x float> %13
}

define <4 x float> @insertps_with_undefs(<4 x float> %a, float* %b) {
; CHECK-LABEL: insertps_with_undefs:
; CHECK-NOT: shufps
; CHECK: insertps    $32, %xmm0
; CHECK: ret
  %1 = load float* %b, align 4
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %result = shufflevector <4 x float> %a, <4 x float> %2, <4 x i32> <i32 4, i32 undef, i32 0, i32 7>
  ret <4 x float> %result
}

; Test for a bug in X86ISelLowering.cpp:getINSERTPS where we were using
; the destination index to change the load, instead of the source index.
define <4 x float> @pr20087(<4 x float> %a, <4 x float> *%ptr) {
; CHECK-LABEL: pr20087:
; CHECK: insertps  $48
; CHECK: ret
  %load = load <4 x float> *%ptr
  %ret = shufflevector <4 x float> %load, <4 x float> %a, <4 x i32> <i32 4, i32 undef, i32 6, i32 2>
  ret <4 x float> %ret
}

; Edge case for insertps where we end up with a shuffle with mask=<0, 7, -1, -1>
define void @insertps_pr20411(i32* noalias nocapture %RET) #1 {
; CHECK-LABEL: insertps_pr20411:
; CHECK: movaps  {{[^,]*}}, %[[REG1:xmm.]]
; CHECK: pshufd  {{.*}} ## [[REG2:xmm.]] = mem[3,0,0,0]
; CHECK: insertps  {{.*}} ## xmm1 = [[REG2]][0],[[REG1]][3]{{.*}}

  %gather_load = shufflevector <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, <8 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %shuffle109 = shufflevector <4 x i32> <i32 4, i32 5, i32 6, i32 7>, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>  ; 4 5 6 7

  %shuffle116 = shufflevector <8 x i32> %gather_load, <8 x i32> undef, <4 x i32> <i32 3, i32 undef, i32 undef, i32 undef> ; 3 x x x
  %shuffle117 = shufflevector <4 x i32> %shuffle109, <4 x i32> %shuffle116, <4 x i32> <i32 4, i32 3, i32 undef, i32 undef> ; 3 7 x x

  %ptrcast = bitcast i32* %RET to <4 x i32>*
  store <4 x i32> %shuffle117, <4 x i32>* %ptrcast, align 4
  ret void
}
