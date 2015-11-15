; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Check that a <4 x float> compare is generated and that we are
; not stuck in an endless loop.

define void @cmp_2_floats() {
; CHECK-LABEL: cmp_2_floats:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    cmpordps %xmm0, %xmm0
; CHECK-NEXT:    pmovzxdq {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; CHECK-NEXT:    psllq $32, %xmm0
; CHECK-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[1,1,3,3]
; CHECK-NEXT:    psrad $31, %xmm0
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3],xmm1[4,5],xmm0[6,7]
; CHECK-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; CHECK-NEXT:    pslld $31, %xmm0
; CHECK-NEXT:    blendvps %xmm0, %xmm0
; CHECK-NEXT:    movlps %xmm0, (%rax)
; CHECK-NEXT:    retq
entry:
  %0 = fcmp oeq <2 x float> undef, undef
  %1 = select <2 x i1> %0, <2 x float> undef, <2 x float> undef
  store <2 x float> %1, <2 x float>* undef
  ret void
}

define void @cmp_2_doubles() {
; CHECK-LABEL: cmp_2_doubles:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    cmpordpd %xmm0, %xmm0
; CHECK-NEXT:    blendvpd %xmm0, %xmm0
; CHECK-NEXT:    movapd %xmm0, (%rax)
; CHECK-NEXT:    retq
entry:
  %0 = fcmp oeq <2 x double> undef, undef
  %1 = select <2 x i1> %0, <2 x double> undef, <2 x double> undef
  store <2 x double> %1, <2 x double>* undef
  ret void
}

define void @mp_11193(<8 x float> * nocapture %aFOO, <8 x float>* nocapture %RET) nounwind {
; CHECK-LABEL: mp_11193:
; CHECK:       # BB#0: # %allocas
; CHECK-NEXT:    movaps {{.*#+}} xmm0 = [3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00]
; CHECK-NEXT:    movaps {{.*#+}} xmm1 = [9.000000e+00,1.000000e+00,9.000000e+00,1.000000e+00]
; CHECK-NEXT:    cmpltps %xmm0, %xmm1
; CHECK-NEXT:    movdqa {{.*#+}} xmm2 = [0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; CHECK-NEXT:    pshufb %xmm2, %xmm1
; CHECK-NEXT:    movaps {{.*#+}} xmm3 = [1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00]
; CHECK-NEXT:    cmpltps %xmm0, %xmm3
; CHECK-NEXT:    pshufb %xmm2, %xmm3
; CHECK-NEXT:    punpcklqdq {{.*#+}} xmm3 = xmm3[0],xmm1[0]
; CHECK-NEXT:    psllw $15, %xmm3
; CHECK-NEXT:    psraw $15, %xmm3
; CHECK-NEXT:    pextrb $0, %xmm3, %eax
; CHECK-NEXT:    shlb $7, %al
; CHECK-NEXT:    sarb $7, %al
; CHECK-NEXT:    movsbl %al, %eax
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    cvtsi2ssl %eax, %xmm0
; CHECK-NEXT:    movss %xmm0, (%rsi)
; CHECK-NEXT:    retq
allocas:
  %bincmp = fcmp olt <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 9.000000e+00, float 1.000000e+00, float 9.000000e+00, float 1.000000e+00> , <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %t = extractelement <8 x i1> %bincmp, i32 0
  %ft = sitofp i1 %t to float
  %pp = bitcast <8 x float>* %RET to float*
  store float %ft, float* %pp
  ret void
}

