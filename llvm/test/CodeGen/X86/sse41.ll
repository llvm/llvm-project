; RUN: llc < %s -mtriple=i686-apple-darwin9 -mattr=sse4.1 -mcpu=penryn | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-apple-darwin9 -mattr=sse4.1 -mcpu=penryn | FileCheck %s --check-prefix=X64

@g16 = external global i16

define <4 x i32> @pinsrd_1(i32 %s, <4 x i32> %tmp) nounwind {
; X32-LABEL: pinsrd_1:
; X32:       ## BB#0:
; X32-NEXT:    pinsrd $1, {{[0-9]+}}(%esp), %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: pinsrd_1:
; X64:       ## BB#0:
; X64-NEXT:    pinsrd $1, %edi, %xmm0
; X64-NEXT:    retq
  %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 1
  ret <4 x i32> %tmp1
}

define <16 x i8> @pinsrb_1(i8 %s, <16 x i8> %tmp) nounwind {
; X32-LABEL: pinsrb_1:
; X32:       ## BB#0:
; X32-NEXT:    pinsrb $1, {{[0-9]+}}(%esp), %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: pinsrb_1:
; X64:       ## BB#0:
; X64-NEXT:    pinsrb $1, %edi, %xmm0
; X64-NEXT:    retq
  %tmp1 = insertelement <16 x i8> %tmp, i8 %s, i32 1
  ret <16 x i8> %tmp1
}

define <2 x i64> @pmovzxbq_1() nounwind {
; X32-LABEL: pmovzxbq_1:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    movl L_g16$non_lazy_ptr, %eax
; X32-NEXT:    pmovzxbq {{.*#+}} xmm0 = mem[0],zero,zero,zero,zero,zero,zero,zero,mem[1],zero,zero,zero,zero,zero,zero,zero
; X32-NEXT:    retl
;
; X64-LABEL: pmovzxbq_1:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    movq _g16@{{.*}}(%rip), %rax
; X64-NEXT:    pmovzxbq {{.*#+}} xmm0 = mem[0],zero,zero,zero,zero,zero,zero,zero,mem[1],zero,zero,zero,zero,zero,zero,zero
; X64-NEXT:    retq
entry:
	%0 = load i16, i16* @g16, align 2		; <i16> [#uses=1]
	%1 = insertelement <8 x i16> undef, i16 %0, i32 0		; <<8 x i16>> [#uses=1]
	%2 = bitcast <8 x i16> %1 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%3 = tail call <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8> %2) nounwind readnone		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %3
}

declare <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8>) nounwind readnone

define i32 @extractps_1(<4 x float> %v) nounwind {
; X32-LABEL: extractps_1:
; X32:       ## BB#0:
; X32-NEXT:    extractps $3, %xmm0, %eax
; X32-NEXT:    retl
;
; X64-LABEL: extractps_1:
; X64:       ## BB#0:
; X64-NEXT:    extractps $3, %xmm0, %eax
; X64-NEXT:    retq
  %s = extractelement <4 x float> %v, i32 3
  %i = bitcast float %s to i32
  ret i32 %i
}
define i32 @extractps_2(<4 x float> %v) nounwind {
; X32-LABEL: extractps_2:
; X32:       ## BB#0:
; X32-NEXT:    extractps $3, %xmm0, %eax
; X32-NEXT:    retl
;
; X64-LABEL: extractps_2:
; X64:       ## BB#0:
; X64-NEXT:    extractps $3, %xmm0, %eax
; X64-NEXT:    retq
  %t = bitcast <4 x float> %v to <4 x i32>
  %s = extractelement <4 x i32> %t, i32 3
  ret i32 %s
}


; The non-store form of extractps puts its result into a GPR.
; This makes it suitable for an extract from a <4 x float> that
; is bitcasted to i32, but unsuitable for much of anything else.

define float @ext_1(<4 x float> %v) nounwind {
; X32-LABEL: ext_1:
; X32:       ## BB#0:
; X32-NEXT:    pushl %eax
; X32-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; X32-NEXT:    addss LCPI5_0, %xmm0
; X32-NEXT:    movss %xmm0, (%esp)
; X32-NEXT:    flds (%esp)
; X32-NEXT:    popl %eax
; X32-NEXT:    retl
;
; X64-LABEL: ext_1:
; X64:       ## BB#0:
; X64-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; X64-NEXT:    addss {{.*}}(%rip), %xmm0
; X64-NEXT:    retq
  %s = extractelement <4 x float> %v, i32 3
  %t = fadd float %s, 1.0
  ret float %t
}
define float @ext_2(<4 x float> %v) nounwind {
; X32-LABEL: ext_2:
; X32:       ## BB#0:
; X32-NEXT:    pushl %eax
; X32-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; X32-NEXT:    movss %xmm0, (%esp)
; X32-NEXT:    flds (%esp)
; X32-NEXT:    popl %eax
; X32-NEXT:    retl
;
; X64-LABEL: ext_2:
; X64:       ## BB#0:
; X64-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; X64-NEXT:    retq
  %s = extractelement <4 x float> %v, i32 3
  ret float %s
}
define i32 @ext_3(<4 x i32> %v) nounwind {
; X32-LABEL: ext_3:
; X32:       ## BB#0:
; X32-NEXT:    pextrd $3, %xmm0, %eax
; X32-NEXT:    retl
;
; X64-LABEL: ext_3:
; X64:       ## BB#0:
; X64-NEXT:    pextrd $3, %xmm0, %eax
; X64-NEXT:    retq
  %i = extractelement <4 x i32> %v, i32 3
  ret i32 %i
}

define <4 x float> @insertps_1(<4 x float> %t1, <4 x float> %t2) nounwind {
; X32-LABEL: insertps_1:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = zero,xmm1[0],zero,xmm0[3]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_1:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = zero,xmm1[0],zero,xmm0[3]
; X64-NEXT:    retq
  %tmp1 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %t1, <4 x float> %t2, i32 21) nounwind readnone
  ret <4 x float> %tmp1
}

declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i32) nounwind readnone

; When optimizing for speed, prefer blendps over insertps even if it means we have to
; generate a separate movss to load the scalar operand.
define <4 x float> @blendps_not_insertps_1(<4 x float> %t1, float %t2) nounwind {
; X32-LABEL: blendps_not_insertps_1:
; X32:       ## BB#0:
; X32-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X32-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3]
; X32-NEXT:    retl
;
; X64-LABEL: blendps_not_insertps_1:
; X64:       ## BB#0:
; X64-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3]
; X64-NEXT:    retq
  %tmp1 = insertelement <4 x float> %t1, float %t2, i32 0
  ret <4 x float> %tmp1
}

; When optimizing for size, generate an insertps if there's a load fold opportunity.
; The difference between i386 and x86-64 ABIs for the float operand means we should
; generate an insertps for X32 but not for X64!
define <4 x float> @insertps_or_blendps(<4 x float> %t1, float %t2) minsize nounwind {
; X32-LABEL: insertps_or_blendps:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = mem[0],xmm0[1,2,3]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_or_blendps:
; X64:       ## BB#0:
; X64-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3]
; X64-NEXT:    retq
  %tmp1 = insertelement <4 x float> %t1, float %t2, i32 0
  ret <4 x float> %tmp1
}

; An insert into the low 32-bits of a vector from the low 32-bits of another vector
; is always just a blendps because blendps is never more expensive than insertps.
define <4 x float> @blendps_not_insertps_2(<4 x float> %t1, <4 x float> %t2) nounwind {
; X32-LABEL: blendps_not_insertps_2:
; X32:       ## BB#0:
; X32-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3]
; X32-NEXT:    retl
;
; X64-LABEL: blendps_not_insertps_2:
; X64:       ## BB#0:
; X64-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3]
; X64-NEXT:    retq
  %tmp2 = extractelement <4 x float> %t2, i32 0
  %tmp1 = insertelement <4 x float> %t1, float %tmp2, i32 0
  ret <4 x float> %tmp1
}

define i32 @ptestz_1(<2 x i64> %t1, <2 x i64> %t2) nounwind {
; X32-LABEL: ptestz_1:
; X32:       ## BB#0:
; X32-NEXT:    ptest %xmm1, %xmm0
; X32-NEXT:    sete %al
; X32-NEXT:    movzbl %al, %eax
; X32-NEXT:    retl
;
; X64-LABEL: ptestz_1:
; X64:       ## BB#0:
; X64-NEXT:    ptest %xmm1, %xmm0
; X64-NEXT:    sete %al
; X64-NEXT:    movzbl %al, %eax
; X64-NEXT:    retq
  %tmp1 = call i32 @llvm.x86.sse41.ptestz(<2 x i64> %t1, <2 x i64> %t2) nounwind readnone
  ret i32 %tmp1
}

define i32 @ptestz_2(<2 x i64> %t1, <2 x i64> %t2) nounwind {
; X32-LABEL: ptestz_2:
; X32:       ## BB#0:
; X32-NEXT:    ptest %xmm1, %xmm0
; X32-NEXT:    sbbl %eax, %eax
; X32-NEXT:    andl $1, %eax
; X32-NEXT:    retl
;
; X64-LABEL: ptestz_2:
; X64:       ## BB#0:
; X64-NEXT:    ptest %xmm1, %xmm0
; X64-NEXT:    sbbl %eax, %eax
; X64-NEXT:    andl $1, %eax
; X64-NEXT:    retq
  %tmp1 = call i32 @llvm.x86.sse41.ptestc(<2 x i64> %t1, <2 x i64> %t2) nounwind readnone
  ret i32 %tmp1
}

define i32 @ptestz_3(<2 x i64> %t1, <2 x i64> %t2) nounwind {
; X32-LABEL: ptestz_3:
; X32:       ## BB#0:
; X32-NEXT:    ptest %xmm1, %xmm0
; X32-NEXT:    seta %al
; X32-NEXT:    movzbl %al, %eax
; X32-NEXT:    retl
;
; X64-LABEL: ptestz_3:
; X64:       ## BB#0:
; X64-NEXT:    ptest %xmm1, %xmm0
; X64-NEXT:    seta %al
; X64-NEXT:    movzbl %al, %eax
; X64-NEXT:    retq
  %tmp1 = call i32 @llvm.x86.sse41.ptestnzc(<2 x i64> %t1, <2 x i64> %t2) nounwind readnone
  ret i32 %tmp1
}


declare i32 @llvm.x86.sse41.ptestz(<2 x i64>, <2 x i64>) nounwind readnone
declare i32 @llvm.x86.sse41.ptestc(<2 x i64>, <2 x i64>) nounwind readnone
declare i32 @llvm.x86.sse41.ptestnzc(<2 x i64>, <2 x i64>) nounwind readnone

; This used to compile to insertps $0  + insertps $16.  insertps $0 is always
; pointless.
define <2 x float> @buildvector(<2 x float> %A, <2 x float> %B) nounwind  {
; X32-LABEL: buildvector:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    movshdup {{.*#+}} xmm2 = xmm0[1,1,3,3]
; X32-NEXT:    movshdup {{.*#+}} xmm3 = xmm1[1,1,3,3]
; X32-NEXT:    addss %xmm1, %xmm0
; X32-NEXT:    addss %xmm2, %xmm3
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[2,3]
; X32-NEXT:    retl
;
; X64-LABEL: buildvector:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    movshdup {{.*#+}} xmm2 = xmm0[1,1,3,3]
; X64-NEXT:    movshdup {{.*#+}} xmm3 = xmm1[1,1,3,3]
; X64-NEXT:    addss %xmm1, %xmm0
; X64-NEXT:    addss %xmm2, %xmm3
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[2,3]
; X64-NEXT:    retq
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
}

define <4 x float> @insertps_from_shufflevector_1(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
; X32-LABEL: insertps_from_shufflevector_1:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],mem[0]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_shufflevector_1:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],mem[0]
; X64-NEXT:    retq
entry:
  %0 = load <4 x float>, <4 x float>* %pb, align 16
  %vecinit6 = shufflevector <4 x float> %a, <4 x float> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecinit6
}

define <4 x float> @insertps_from_shufflevector_2(<4 x float> %a, <4 x float> %b) {
; X32-LABEL: insertps_from_shufflevector_2:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1],xmm1[1],xmm0[3]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_shufflevector_2:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1],xmm1[1],xmm0[3]
; X64-NEXT:    retq
entry:
  %vecinit6 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x float> %vecinit6
}

; For loading an i32 from memory into an xmm register we use pinsrd
; instead of insertps
define <4 x i32> @pinsrd_from_shufflevector_i32(<4 x i32> %a, <4 x i32>* nocapture readonly %pb) {
; X32-LABEL: pinsrd_from_shufflevector_i32:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    pshufd {{.*#+}} xmm1 = mem[0,1,2,0]
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: pinsrd_from_shufflevector_i32:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    pshufd {{.*#+}} xmm1 = mem[0,1,2,0]
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; X64-NEXT:    retq
entry:
  %0 = load <4 x i32>, <4 x i32>* %pb, align 16
  %vecinit6 = shufflevector <4 x i32> %a, <4 x i32> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecinit6
}

define <4 x i32> @insertps_from_shufflevector_i32_2(<4 x i32> %a, <4 x i32> %b) {
; X32-LABEL: insertps_from_shufflevector_i32_2:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_shufflevector_i32_2:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; X64-NEXT:    retq
entry:
  %vecinit6 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 7, i32 2, i32 3>
  ret <4 x i32> %vecinit6
}

define <4 x float> @insertps_from_load_ins_elt_undef(<4 x float> %a, float* %b) {
; X32-LABEL: insertps_from_load_ins_elt_undef:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],mem[0],xmm0[2,3]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_load_ins_elt_undef:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],mem[0],xmm0[2,3]
; X64-NEXT:    retq
  %1 = load float, float* %b, align 4
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %result = shufflevector <4 x float> %a, <4 x float> %2, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
  ret <4 x float> %result
}

; TODO: Like on pinsrd_from_shufflevector_i32, remove this mov instr
define <4 x i32> @insertps_from_load_ins_elt_undef_i32(<4 x i32> %a, i32* %b) {
; X32-LABEL: insertps_from_load_ins_elt_undef_i32:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movd {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,1,0,1]
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5],xmm0[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_load_ins_elt_undef_i32:
; X64:       ## BB#0:
; X64-NEXT:    movd {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,1,0,1]
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5],xmm0[6,7]
; X64-NEXT:    retq
  %1 = load i32, i32* %b, align 4
  %2 = insertelement <4 x i32> undef, i32 %1, i32 0
  %result = shufflevector <4 x i32> %a, <4 x i32> %2, <4 x i32> <i32 0, i32 1, i32 4, i32 3>
  ret <4 x i32> %result
}

;;;;;; Shuffles optimizable with a single insertps or blend instruction
define <4 x float> @shuf_XYZ0(<4 x float> %x, <4 x float> %a) {
; X32-LABEL: shuf_XYZ0:
; X32:       ## BB#0:
; X32-NEXT:    xorps %xmm1, %xmm1
; X32-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3]
; X32-NEXT:    retl
;
; X64-LABEL: shuf_XYZ0:
; X64:       ## BB#0:
; X64-NEXT:    xorps %xmm1, %xmm1
; X64-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3]
; X64-NEXT:    retq
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
; X32-LABEL: shuf_XY00:
; X32:       ## BB#0:
; X32-NEXT:    movq {{.*#+}} xmm0 = xmm0[0],zero
; X32-NEXT:    retl
;
; X64-LABEL: shuf_XY00:
; X64:       ## BB#0:
; X64-NEXT:    movq {{.*#+}} xmm0 = xmm0[0],zero
; X64-NEXT:    retq
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %x, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.0, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float 0.0, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_XYY0(<4 x float> %x, <4 x float> %a) {
; X32-LABEL: shuf_XYY0:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,1],zero
; X32-NEXT:    retl
;
; X64-LABEL: shuf_XYY0:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,1],zero
; X64-NEXT:    retq
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %x, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecinit4 = insertelement <4 x float> %vecinit2, float %vecext1, i32 2
  %vecinit5 = insertelement <4 x float> %vecinit4, float 0.0, i32 3
  ret <4 x float> %vecinit5
}

define <4 x float> @shuf_XYW0(<4 x float> %x, <4 x float> %a) {
; X32-LABEL: shuf_XYW0:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,3],zero
; X32-NEXT:    retl
;
; X64-LABEL: shuf_XYW0:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,3],zero
; X64-NEXT:    retq
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
; X32-LABEL: shuf_W00W:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[3],zero,zero,xmm0[3]
; X32-NEXT:    retl
;
; X64-LABEL: shuf_W00W:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[3],zero,zero,xmm0[3]
; X64-NEXT:    retq
  %vecext = extractelement <4 x float> %x, i32 3
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit2 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.0, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float %vecext, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_X00A(<4 x float> %x, <4 x float> %a) {
; X32-LABEL: shuf_X00A:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,zero,xmm1[0]
; X32-NEXT:    retl
;
; X64-LABEL: shuf_X00A:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,zero,xmm1[0]
; X64-NEXT:    retq
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit1, float 0.0, i32 2
  %vecinit4 = shufflevector <4 x float> %vecinit2, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_X00X(<4 x float> %x, <4 x float> %a) {
; X32-LABEL: shuf_X00X:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,zero,xmm0[0]
; X32-NEXT:    retl
;
; X64-LABEL: shuf_X00X:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,zero,xmm0[0]
; X64-NEXT:    retq
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit1, float 0.0, i32 2
  %vecinit4 = shufflevector <4 x float> %vecinit2, <4 x float> %x, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecinit4
}

define <4 x float> @shuf_X0YC(<4 x float> %x, <4 x float> %a) {
; X32-LABEL: shuf_X0YC:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[2]
; X32-NEXT:    retl
;
; X64-LABEL: shuf_X0YC:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[2]
; X64-NEXT:    retq
  %vecext = extractelement <4 x float> %x, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 1
  %vecinit3 = shufflevector <4 x float> %vecinit1, <4 x float> %x, <4 x i32> <i32 0, i32 1, i32 5, i32 undef>
  %vecinit5 = shufflevector <4 x float> %vecinit3, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x float> %vecinit5
}

define <4 x i32> @i32_shuf_XYZ0(<4 x i32> %x, <4 x i32> %a) {
; X32-LABEL: i32_shuf_XYZ0:
; X32:       ## BB#0:
; X32-NEXT:    pxor %xmm1, %xmm1
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_XYZ0:
; X64:       ## BB#0:
; X64-NEXT:    pxor %xmm1, %xmm1
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; X64-NEXT:    retq
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
; X32-LABEL: i32_shuf_XY00:
; X32:       ## BB#0:
; X32-NEXT:    movq {{.*#+}} xmm0 = xmm0[0],zero
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_XY00:
; X64:       ## BB#0:
; X64-NEXT:    movq {{.*#+}} xmm0 = xmm0[0],zero
; X64-NEXT:    retq
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <4 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 0, i32 2
  %vecinit4 = insertelement <4 x i32> %vecinit3, i32 0, i32 3
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_XYY0(<4 x i32> %x, <4 x i32> %a) {
; X32-LABEL: i32_shuf_XYY0:
; X32:       ## BB#0:
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,1,1,3]
; X32-NEXT:    pxor %xmm0, %xmm0
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5],xmm0[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_XYY0:
; X64:       ## BB#0:
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,1,1,3]
; X64-NEXT:    pxor %xmm0, %xmm0
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5],xmm0[6,7]
; X64-NEXT:    retq
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <4 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %vecext1, i32 2
  %vecinit5 = insertelement <4 x i32> %vecinit4, i32 0, i32 3
  ret <4 x i32> %vecinit5
}

define <4 x i32> @i32_shuf_XYW0(<4 x i32> %x, <4 x i32> %a) {
; X32-LABEL: i32_shuf_XYW0:
; X32:       ## BB#0:
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,1,3,3]
; X32-NEXT:    pxor %xmm0, %xmm0
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5],xmm0[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_XYW0:
; X64:       ## BB#0:
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,1,3,3]
; X64-NEXT:    pxor %xmm0, %xmm0
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5],xmm0[6,7]
; X64-NEXT:    retq
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
; X32-LABEL: i32_shuf_W00W:
; X32:       ## BB#0:
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; X32-NEXT:    pxor %xmm0, %xmm0
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5],xmm1[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_W00W:
; X64:       ## BB#0:
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; X64-NEXT:    pxor %xmm0, %xmm0
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5],xmm1[6,7]
; X64-NEXT:    retq
  %vecext = extractelement <4 x i32> %x, i32 3
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 0, i32 2
  %vecinit4 = insertelement <4 x i32> %vecinit3, i32 %vecext, i32 3
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_X00A(<4 x i32> %x, <4 x i32> %a) {
; X32-LABEL: i32_shuf_X00A:
; X32:       ## BB#0:
; X32-NEXT:    pxor %xmm2, %xmm2
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm2[2,3,4,5,6,7]
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,1,2,0]
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_X00A:
; X64:       ## BB#0:
; X64-NEXT:    pxor %xmm2, %xmm2
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm2[2,3,4,5,6,7]
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,1,2,0]
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; X64-NEXT:    retq
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 0, i32 2
  %vecinit4 = shufflevector <4 x i32> %vecinit2, <4 x i32> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_X00X(<4 x i32> %x, <4 x i32> %a) {
; X32-LABEL: i32_shuf_X00X:
; X32:       ## BB#0:
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,1,2,0]
; X32-NEXT:    pxor %xmm0, %xmm0
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5],xmm1[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_X00X:
; X64:       ## BB#0:
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,1,2,0]
; X64-NEXT:    pxor %xmm0, %xmm0
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5],xmm1[6,7]
; X64-NEXT:    retq
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 0, i32 2
  %vecinit4 = shufflevector <4 x i32> %vecinit2, <4 x i32> %x, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecinit4
}

define <4 x i32> @i32_shuf_X0YC(<4 x i32> %x, <4 x i32> %a) {
; X32-LABEL: i32_shuf_X0YC:
; X32:       ## BB#0:
; X32-NEXT:    pmovzxdq {{.*#+}} xmm2 = xmm0[0],zero,xmm0[1],zero
; X32-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,1,2,2]
; X32-NEXT:    pblendw {{.*#+}} xmm0 = xmm2[0,1,2,3,4,5],xmm0[6,7]
; X32-NEXT:    retl
;
; X64-LABEL: i32_shuf_X0YC:
; X64:       ## BB#0:
; X64-NEXT:    pmovzxdq {{.*#+}} xmm2 = xmm0[0],zero,xmm0[1],zero
; X64-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,1,2,2]
; X64-NEXT:    pblendw {{.*#+}} xmm0 = xmm2[0,1,2,3,4,5],xmm0[6,7]
; X64-NEXT:    retq
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit3 = shufflevector <4 x i32> %vecinit1, <4 x i32> %x, <4 x i32> <i32 0, i32 1, i32 5, i32 undef>
  %vecinit5 = shufflevector <4 x i32> %vecinit3, <4 x i32> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x i32> %vecinit5
}

;; Test for a bug in the first implementation of LowerBuildVectorv4x32
define < 4 x float> @test_insertps_no_undef(<4 x float> %x) {
; X32-LABEL: test_insertps_no_undef:
; X32:       ## BB#0:
; X32-NEXT:    xorps %xmm1, %xmm1
; X32-NEXT:    blendps {{.*#+}} xmm1 = xmm0[0,1,2],xmm1[3]
; X32-NEXT:    maxps %xmm1, %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: test_insertps_no_undef:
; X64:       ## BB#0:
; X64-NEXT:    xorps %xmm1, %xmm1
; X64-NEXT:    blendps {{.*#+}} xmm1 = xmm0[0,1,2],xmm1[3]
; X64-NEXT:    maxps %xmm1, %xmm0
; X64-NEXT:    retq
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
; X32-LABEL: blendvb_fallback:
; X32:       ## BB#0:
; X32-NEXT:    psllw $15, %xmm0
; X32-NEXT:    psraw $15, %xmm0
; X32-NEXT:    pblendvb %xmm1, %xmm2
; X32-NEXT:    movdqa %xmm2, %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: blendvb_fallback:
; X64:       ## BB#0:
; X64-NEXT:    psllw $15, %xmm0
; X64-NEXT:    psraw $15, %xmm0
; X64-NEXT:    pblendvb %xmm1, %xmm2
; X64-NEXT:    movdqa %xmm2, %xmm0
; X64-NEXT:    retq
  %ret = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %ret
}

; On X32, account for the argument's move to registers
define <4 x float> @insertps_from_vector_load(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
; X32-LABEL: insertps_from_vector_load:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    insertps    $48, (%{{...}}), {{.*#+}} xmm0 = xmm0[0,1,2],mem[0]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_vector_load:
; X64:       ## BB#0:
; X64-NEXT:    insertps    $48, (%{{...}}), {{.*#+}} xmm0 = xmm0[0,1,2],mem[0]
; X64-NEXT:    retq
  %1 = load <4 x float>, <4 x float>* %pb, align 16
  %2 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %1, i32 48)
  ret <4 x float> %2
}

;; Use a non-zero CountS for insertps
;; Try to match a bit more of the instr, since we need the load's offset.
define <4 x float> @insertps_from_vector_load_offset(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
; X32-LABEL: insertps_from_vector_load_offset:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    insertps    $32, 4(%{{...}}), {{.*#+}} xmm0 = xmm0[0,1],mem[0],xmm0[3]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_vector_load_offset:
; X64:       ## BB#0:
; X64-NEXT:    insertps    $32, 4(%{{...}}), {{.*#+}} xmm0 = xmm0[0,1],mem[0],xmm0[3]
; X64-NEXT:    retq
  %1 = load <4 x float>, <4 x float>* %pb, align 16
  %2 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %1, i32 96)
  ret <4 x float> %2
}

;; Try to match a bit more of the instr, since we need the load's offset.
define <4 x float> @insertps_from_vector_load_offset_2(<4 x float> %a, <4 x float>* nocapture readonly %pb, i64 %index) {
; X32-LABEL: insertps_from_vector_load_offset_2:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X32-NEXT:    shll $4, %ecx
; X32-NEXT:    insertps    $0, 12(%{{...}},%{{...}}), {{.*#+}} xmm0 = mem[0],xmm0[1,2,3]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_vector_load_offset_2:
; X64:       ## BB#0:
; X64-NEXT:    shlq $4, %rsi
; X64-NEXT:    insertps    $0, 12(%{{...}},%{{...}}), {{.*#+}} xmm0 = mem[0],xmm0[1,2,3]
; X64-NEXT:    retq
  %1 = getelementptr inbounds <4 x float>, <4 x float>* %pb, i64 %index
  %2 = load <4 x float>, <4 x float>* %1, align 16
  %3 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %2, i32 192)
  ret <4 x float> %3
}

define <4 x float> @insertps_from_broadcast_loadf32(<4 x float> %a, float* nocapture readonly %fb, i64 %index) {
; X32-LABEL: insertps_from_broadcast_loadf32:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X32-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X32-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,0,0,0]
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_broadcast_loadf32:
; X64:       ## BB#0:
; X64-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X64-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,0,0,0]
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; X64-NEXT:    retq
  %1 = getelementptr inbounds float, float* %fb, i64 %index
  %2 = load float, float* %1, align 4
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = insertelement <4 x float> %3, float %2, i32 1
  %5 = insertelement <4 x float> %4, float %2, i32 2
  %6 = insertelement <4 x float> %5, float %2, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %6, i32 48)
  ret <4 x float> %7
}

define <4 x float> @insertps_from_broadcast_loadv4f32(<4 x float> %a, <4 x float>* nocapture readonly %b) {
; X32-LABEL: insertps_from_broadcast_loadv4f32:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movups (%eax), %xmm1
; X32-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,0,0,0]
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_broadcast_loadv4f32:
; X64:       ## BB#0:
; X64-NEXT:    movups (%rdi), %xmm1
; X64-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,0,0,0]
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; X64-NEXT:    retq
  %1 = load <4 x float>, <4 x float>* %b, align 4
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
; X32-LABEL: insertps_from_broadcast_multiple_use:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X32-NEXT:    movss {{.*#+}} xmm4 = mem[0],zero,zero,zero
; X32-NEXT:    shufps {{.*#+}} xmm4 = xmm4[0,0,0,0]
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm4[0]
; X32-NEXT:    insertps {{.*#+}} xmm1 = xmm1[0,1,2],xmm4[0]
; X32-NEXT:    insertps {{.*#+}} xmm2 = xmm2[0,1,2],xmm4[0]
; X32-NEXT:    insertps {{.*#+}} xmm3 = xmm3[0,1,2],xmm4[0]
; X32-NEXT:    addps %xmm1, %xmm0
; X32-NEXT:    addps %xmm2, %xmm3
; X32-NEXT:    addps %xmm3, %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: insertps_from_broadcast_multiple_use:
; X64:       ## BB#0:
; X64-NEXT:    movss {{.*#+}} xmm4 = mem[0],zero,zero,zero
; X64-NEXT:    shufps {{.*#+}} xmm4 = xmm4[0,0,0,0]
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm4[0]
; X64-NEXT:    insertps {{.*#+}} xmm1 = xmm1[0,1,2],xmm4[0]
; X64-NEXT:    insertps {{.*#+}} xmm2 = xmm2[0,1,2],xmm4[0]
; X64-NEXT:    insertps {{.*#+}} xmm3 = xmm3[0,1,2],xmm4[0]
; X64-NEXT:    addps %xmm1, %xmm0
; X64-NEXT:    addps %xmm2, %xmm3
; X64-NEXT:    addps %xmm3, %xmm0
; X64-NEXT:    retq
  %1 = getelementptr inbounds float, float* %fb, i64 %index
  %2 = load float, float* %1, align 4
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
; X32-LABEL: insertps_with_undefs:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X32-NEXT:    unpcklpd {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; X32-NEXT:    movapd %xmm1, %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: insertps_with_undefs:
; X64:       ## BB#0:
; X64-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X64-NEXT:    unpcklpd {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; X64-NEXT:    movapd %xmm1, %xmm0
; X64-NEXT:    retq
  %1 = load float, float* %b, align 4
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %result = shufflevector <4 x float> %a, <4 x float> %2, <4 x i32> <i32 4, i32 undef, i32 0, i32 7>
  ret <4 x float> %result
}

; Test for a bug in X86ISelLowering.cpp:getINSERTPS where we were using
; the destination index to change the load, instead of the source index.
define <4 x float> @pr20087(<4 x float> %a, <4 x float> *%ptr) {
; X32-LABEL: pr20087:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm0[2],mem[0]
; X32-NEXT:    retl
;
; X64-LABEL: pr20087:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm0[2],mem[0]
; X64-NEXT:    retq
  %load = load <4 x float> , <4 x float> *%ptr
  %ret = shufflevector <4 x float> %load, <4 x float> %a, <4 x i32> <i32 4, i32 undef, i32 6, i32 2>
  ret <4 x float> %ret
}

; Edge case for insertps where we end up with a shuffle with mask=<0, 7, -1, -1>
define void @insertps_pr20411(<4 x i32> %shuffle109, <4 x i32> %shuffle116, i32* noalias nocapture %RET) #1 {
; X32-LABEL: insertps_pr20411:
; X32:       ## BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; X32-NEXT:    pblendw {{.*#+}} xmm1 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; X32-NEXT:    movdqu %xmm1, (%eax)
; X32-NEXT:    retl
;
; X64-LABEL: insertps_pr20411:
; X64:       ## BB#0:
; X64-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; X64-NEXT:    pblendw {{.*#+}} xmm1 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; X64-NEXT:    movdqu %xmm1, (%rdi)
; X64-NEXT:    retq
  %shuffle117 = shufflevector <4 x i32> %shuffle109, <4 x i32> %shuffle116, <4 x i32> <i32 0, i32 7, i32 undef, i32 undef>
  %ptrcast = bitcast i32* %RET to <4 x i32>*
  store <4 x i32> %shuffle117, <4 x i32>* %ptrcast, align 4
  ret void
}

define <4 x float> @insertps_4(<4 x float> %A, <4 x float> %B) {
; X32-LABEL: insertps_4:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm1[2],zero
; X32-NEXT:    retl
;
; X64-LABEL: insertps_4:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm1[2],zero
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.000000e+00, i32 1
  %vecext2 = extractelement <4 x float> %B, i32 2
  %vecinit3 = insertelement <4 x float> %vecinit1, float %vecext2, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @insertps_5(<4 x float> %A, <4 x float> %B) {
; X32-LABEL: insertps_5:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],xmm1[1],zero,zero
; X32-NEXT:    retl
;
; X64-LABEL: insertps_5:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],xmm1[1],zero,zero
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %B, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.000000e+00, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @insertps_6(<4 x float> %A, <4 x float> %B) {
; X32-LABEL: insertps_6:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    insertps {{.*#+}} xmm0 = zero,xmm0[1],xmm1[2],zero
; X32-NEXT:    retl
;
; X64-LABEL: insertps_6:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm0 = zero,xmm0[1],xmm1[2],zero
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 1
  %vecinit = insertelement <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, float %vecext, i32 1
  %vecext1 = extractelement <4 x float> %B, i32 2
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 2
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit3
}

define <4 x float> @insertps_7(<4 x float> %A, <4 x float> %B) {
; X32-LABEL: insertps_7:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm1[1],zero
; X32-NEXT:    retl
;
; X64-LABEL: insertps_7:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm1[1],zero
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.000000e+00, i32 1
  %vecext2 = extractelement <4 x float> %B, i32 1
  %vecinit3 = insertelement <4 x float> %vecinit1, float %vecext2, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @insertps_8(<4 x float> %A, <4 x float> %B) {
; X32-LABEL: insertps_8:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],xmm1[0],zero,zero
; X32-NEXT:    retl
;
; X64-LABEL: insertps_8:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],xmm1[0],zero,zero
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 0
  %vecinit = insertelement <4 x float> undef, float %vecext, i32 0
  %vecext1 = extractelement <4 x float> %B, i32 0
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 1
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.000000e+00, i32 2
  %vecinit4 = insertelement <4 x float> %vecinit3, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit4
}

define <4 x float> @insertps_9(<4 x float> %A, <4 x float> %B) {
; X32-LABEL: insertps_9:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    insertps {{.*#+}} xmm1 = zero,xmm0[0],xmm1[2],zero
; X32-NEXT:    movaps %xmm1, %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: insertps_9:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    insertps {{.*#+}} xmm1 = zero,xmm0[0],xmm1[2],zero
; X64-NEXT:    movaps %xmm1, %xmm0
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 0
  %vecinit = insertelement <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, float %vecext, i32 1
  %vecext1 = extractelement <4 x float> %B, i32 2
  %vecinit2 = insertelement <4 x float> %vecinit, float %vecext1, i32 2
  %vecinit3 = insertelement <4 x float> %vecinit2, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit3
}

define <4 x float> @insertps_10(<4 x float> %A)
; X32-LABEL: insertps_10:
; X32:       ## BB#0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm0[0],zero
; X32-NEXT:    retl
;
; X64-LABEL: insertps_10:
; X64:       ## BB#0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0],zero,xmm0[0],zero
; X64-NEXT:    retq
{
  %vecext = extractelement <4 x float> %A, i32 0
  %vecbuild1 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %vecext, i32 0
  %vecbuild2 = insertelement <4 x float> %vecbuild1, float %vecext, i32 2
  ret <4 x float> %vecbuild2
}

define <4 x float> @build_vector_to_shuffle_1(<4 x float> %A) {
; X32-LABEL: build_vector_to_shuffle_1:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    xorps %xmm1, %xmm1
; X32-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3]
; X32-NEXT:    retl
;
; X64-LABEL: build_vector_to_shuffle_1:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    xorps %xmm1, %xmm1
; X64-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3]
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 1
  %vecinit = insertelement <4 x float> zeroinitializer, float %vecext, i32 1
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 2
  %vecinit3 = shufflevector <4 x float> %vecinit1, <4 x float> %A, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %vecinit3
}

define <4 x float> @build_vector_to_shuffle_2(<4 x float> %A) {
; X32-LABEL: build_vector_to_shuffle_2:
; X32:       ## BB#0: ## %entry
; X32-NEXT:    xorps %xmm1, %xmm1
; X32-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2,3]
; X32-NEXT:    retl
;
; X64-LABEL: build_vector_to_shuffle_2:
; X64:       ## BB#0: ## %entry
; X64-NEXT:    xorps %xmm1, %xmm1
; X64-NEXT:    blendps {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2,3]
; X64-NEXT:    retq
entry:
  %vecext = extractelement <4 x float> %A, i32 1
  %vecinit = insertelement <4 x float> zeroinitializer, float %vecext, i32 1
  %vecinit1 = insertelement <4 x float> %vecinit, float 0.0, i32 2
  ret <4 x float> %vecinit1
}
