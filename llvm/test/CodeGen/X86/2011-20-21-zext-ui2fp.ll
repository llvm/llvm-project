; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Check that the booleans are converted using zext and not via sext.
; 0x1 means that we only look at the first bit.

define void @ui_to_fp_conv(<8 x float> * nocapture %aFOO, <8 x float>* nocapture %RET) nounwind {
; CHECK: 0x1
; CHECK-LABEL: ui_to_fp_conv:
; CHECK:       # BB#0: # %allocas
; CHECK-NEXT:    movaps {{.*#+}} xmm0 = [1.000000e+00,1.000000e+00,3.000000e+00,3.000000e+00]
; CHECK-NEXT:    cmpltps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    pshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-NEXT:    psllw $15, %xmm0
; CHECK-NEXT:    psraw $15, %xmm0
; CHECK-NEXT:    pmovzxwd {{.*#+}} xmm1 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; CHECK-NEXT:    movdqa {{.*#+}} xmm2 = [1,1,1,1]
; CHECK-NEXT:    pand %xmm2, %xmm1
; CHECK-NEXT:    movdqa {{.*#+}} xmm3 = [1258291200,1258291200,1258291200,1258291200]
; CHECK-NEXT:    movdqa %xmm1, %xmm4
; CHECK-NEXT:    pblendw {{.*#+}} xmm4 = xmm4[0],xmm3[1],xmm4[2],xmm3[3],xmm4[4],xmm3[5],xmm4[6],xmm3[7]
; CHECK-NEXT:    psrld $16, %xmm1
; CHECK-NEXT:    movdqa {{.*#+}} xmm5 = [1392508928,1392508928,1392508928,1392508928]
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0],xmm5[1],xmm1[2],xmm5[3],xmm1[4],xmm5[5],xmm1[6],xmm5[7]
; CHECK-NEXT:    movaps {{.*#+}} xmm6 = [-5.497642e+11,-5.497642e+11,-5.497642e+11,-5.497642e+11]
; CHECK-NEXT:    addps %xmm6, %xmm1
; CHECK-NEXT:    addps %xmm4, %xmm1
; CHECK-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; CHECK-NEXT:    pand %xmm2, %xmm0
; CHECK-NEXT:    pblendw {{.*#+}} xmm3 = xmm0[0],xmm3[1],xmm0[2],xmm3[3],xmm0[4],xmm3[5],xmm0[6],xmm3[7]
; CHECK-NEXT:    psrld $16, %xmm0
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0],xmm5[1],xmm0[2],xmm5[3],xmm0[4],xmm5[5],xmm0[6],xmm5[7]
; CHECK-NEXT:    addps %xmm6, %xmm0
; CHECK-NEXT:    addps %xmm3, %xmm0
; CHECK-NEXT:    movups %xmm0, 16(%rsi)
; CHECK-NEXT:    movups %xmm1, (%rsi)
; CHECK-NEXT:    retq
allocas:
  %bincmp = fcmp olt <8 x float> <float 1.000000e+00, float 1.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00> , <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %bool2float = uitofp <8 x i1> %bincmp to <8 x float>
  store <8 x float> %bool2float, <8 x float>* %RET, align 4
  ret void
}



