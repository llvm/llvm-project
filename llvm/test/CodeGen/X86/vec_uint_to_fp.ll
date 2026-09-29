; RUN: llc < %s -mtriple=x86_64-apple-macosx | FileCheck --check-prefix=CHECK --check-prefix=SSE --check-prefix=CST %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx -mattr=+sse4.1 | FileCheck --check-prefix=CHECK --check-prefix=SSE41 --check-prefix=CST  %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx -mattr=+avx | FileCheck --check-prefix=CHECK --check-prefix=AVX --check-prefix=CST %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx -mattr=+avx2 | FileCheck --check-prefix=CHECK --check-prefix=AVX2 %s

; Check that the constant used in the vectors are the right ones.
; SSE: [[MASKCSTADDR:LCPI0_[0-9]+]]:
; SSE-NEXT: .long	65535                   ## 0xffff
; SSE-NEXT: .long	65535                   ## 0xffff
; SSE-NEXT: .long	65535                   ## 0xffff
; SSE-NEXT: .long	65535                   ## 0xffff

; CST: [[LOWCSTADDR:LCPI0_[0-9]+]]:
; CST-NEXT: .long	1258291200              ## 0x4b000000
; CST-NEXT: .long	1258291200              ## 0x4b000000
; CST-NEXT: .long	1258291200              ## 0x4b000000
; CST-NEXT: .long	1258291200              ## 0x4b000000

; CST: [[HIGHCSTADDR:LCPI0_[0-9]+]]:
; CST-NEXT: .long	1392508928              ## 0x53000000
; CST-NEXT: .long	1392508928              ## 0x53000000
; CST-NEXT: .long	1392508928              ## 0x53000000
; CST-NEXT: .long	1392508928              ## 0x53000000

; CST: [[MAGICCSTADDR:LCPI0_[0-9]+]]:
; CST-NEXT: .long	0x53000080              ## float 5.49764202E+11
; CST-NEXT: .long	0x53000080              ## float 5.49764202E+11
; CST-NEXT: .long	0x53000080              ## float 5.49764202E+11
; CST-NEXT: .long	0x53000080              ## float 5.49764202E+11

; AVX2: [[LOWCSTADDR:LCPI0_[0-9]+]]:
; AVX2-NEXT: .long	1258291200              ## 0x4b000000

; AVX2: [[HIGHCSTADDR:LCPI0_[0-9]+]]:
; AVX2-NEXT: .long	1392508928              ## 0x53000000

; AVX2: [[MAGICCSTADDR:LCPI0_[0-9]+]]:
; AVX2-NEXT: .long	0x53000080              ## float 5.49764202E+11

define <4 x float> @test1(<4 x i32> %A) nounwind {
; CHECK-LABEL: test1:
;
; SSE: movdqa [[MASKCSTADDR]](%rip), [[MASK:%xmm[0-9]+]]
; SSE-NEXT: pand %xmm0, [[MASK]]
; After this instruction, MASK will have the value of the low parts
; of the vector.
; SSE-NEXT: por [[LOWCSTADDR]](%rip), [[MASK]]
; SSE-NEXT: psrld $16, %xmm0
; SSE-NEXT: por [[HIGHCSTADDR]](%rip), %xmm0
; SSE-NEXT: subps [[MAGICCSTADDR]](%rip), %xmm0
; SSE-NEXT: addps [[MASK]], %xmm0
; SSE-NEXT: retq
;
; Currently we commute the arguments of the first blend, but this could be
; improved to match the lowering of the second blend.
; SSE41: movdqa [[LOWCSTADDR]](%rip), [[LOWVEC:%xmm[0-9]+]]
; SSE41-NEXT: pblendw $85, %xmm0, [[LOWVEC]]
; SSE41-NEXT: psrld $16, %xmm0
; SSE41-NEXT: pblendw $170, [[HIGHCSTADDR]](%rip), %xmm0
; SSE41-NEXT: subps [[MAGICCSTADDR]](%rip), %xmm0
; SSE41-NEXT: addps [[LOWVEC]], %xmm0
; SSE41-NEXT: retq
;
; AVX: vpblendw $170, [[LOWCSTADDR]](%rip), %xmm0, [[LOWVEC:%xmm[0-9]+]]
; AVX-NEXT: vpsrld $16, %xmm0, [[SHIFTVEC:%xmm[0-9]+]]
; AVX-NEXT: vpblendw $170, [[HIGHCSTADDR]](%rip), [[SHIFTVEC]], [[HIGHVEC:%xmm[0-9]+]]
; AVX-NEXT: vsubps [[MAGICCSTADDR]](%rip), [[HIGHVEC]], [[TMP:%xmm[0-9]+]]
; AVX-NEXT: vaddps [[TMP]], [[LOWVEC]], %xmm0
; AVX-NEXT: retq
;
; The lowering for AVX2 is a bit messy, because we select broadcast
; instructions, instead of folding the constant loads.
; AVX2: vpbroadcastd [[LOWCSTADDR]](%rip), [[LOWCST:%xmm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[LOWCST]], %xmm0, [[LOWVEC:%xmm[0-9]+]]
; AVX2-NEXT: vpsrld $16, %xmm0, [[SHIFTVEC:%xmm[0-9]+]]
; AVX2-NEXT: vpbroadcastd [[HIGHCSTADDR]](%rip), [[HIGHCST:%xmm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[HIGHCST]], [[SHIFTVEC]], [[HIGHVEC:%xmm[0-9]+]]
; AVX2-NEXT: vbroadcastss [[MAGICCSTADDR]](%rip), [[MAGICCST:%xmm[0-9]+]]
; AVX2-NEXT: vsubps [[MAGICCST]], [[HIGHVEC]], [[TMP:%xmm[0-9]+]]
; AVX2-NEXT: vaddps [[TMP]], [[LOWVEC]], %xmm0
; AVX2-NEXT: retq
  %C = uitofp <4 x i32> %A to <4 x float>
  ret <4 x float> %C
}

; Match the AVX2 constants used in the next function
; AVX2: [[LOWCSTADDR:LCPI1_[0-9]+]]:
; AVX2-NEXT: .long	1258291200              ## 0x4b000000

; AVX2: [[HIGHCSTADDR:LCPI1_[0-9]+]]:
; AVX2-NEXT: .long	1392508928              ## 0x53000000

; AVX2: [[MAGICCSTADDR:LCPI1_[0-9]+]]:
; AVX2-NEXT: .long	0x53000080              ## float 5.49764202E+11

define <8 x float> @test2(<8 x i32> %A) nounwind {
; CHECK-LABEL: test2:
; Legalization will break the thing is 2 x <4 x i32> on anthing prior AVX.
; The constant used for in the vector instruction are shared between the
; two sequences of instructions.
;
; SSE: movdqa {{.*#+}} [[MASK:xmm[0-9]+]] = [65535,65535,65535,65535]
; SSE-NEXT: movdqa %xmm0, [[VECLOW:%xmm[0-9]+]]
; SSE-NEXT: pand %[[MASK]], [[VECLOW]]
; SSE-NEXT: movdqa {{.*#+}} [[LOWCST:xmm[0-9]+]] = [1258291200,1258291200,1258291200,1258291200]
; SSE-NEXT: por %[[LOWCST]], [[VECLOW]]
; SSE-NEXT: psrld $16, %xmm0
; SSE-NEXT: movdqa {{.*#+}} [[HIGHCST:xmm[0-9]+]] = [1392508928,1392508928,1392508928,1392508928]
; SSE-NEXT: por %[[HIGHCST]], %xmm0
; SSE-NEXT: movaps {{.*#+}} [[MAGICCST:xmm[0-9]+]] = [5.49764202E+11,5.49764202E+11,5.49764202E+11,5.49764202E+11]
; SSE-NEXT: subps %[[MAGICCST]], %xmm0
; SSE-NEXT: addps [[VECLOW]], %xmm0
; MASK is the low vector of the second part after this point.
; SSE-NEXT: pand %xmm1, %[[MASK]]
; SSE-NEXT: por %[[LOWCST]], %[[MASK]]
; SSE-NEXT: psrld $16, %xmm1
; SSE-NEXT: por %[[HIGHCST]], %xmm1
; SSE-NEXT: subps %[[MAGICCST]], %xmm1
; SSE-NEXT: addps %[[MASK]], %xmm1
; SSE-NEXT: retq
;
; SSE41: movdqa {{.*#+}} [[LOWCST:xmm[0-9]+]] = [1258291200,1258291200,1258291200,1258291200]
; SSE41-NEXT: movdqa %xmm0, [[VECLOW:%xmm[0-9]+]]
; SSE41-NEXT: pblendw $170, %[[LOWCST]], [[VECLOW]]
; SSE41-NEXT: psrld $16, %xmm0
; SSE41-NEXT: movdqa {{.*#+}} [[HIGHCST:xmm[0-9]+]] = [1392508928,1392508928,1392508928,1392508928]
; SSE41-NEXT: pblendw $170, %[[HIGHCST]], %xmm0
; SSE41-NEXT: movaps {{.*#+}} [[MAGICCST:xmm[0-9]+]] = [5.49764202E+11,5.49764202E+11,5.49764202E+11,5.49764202E+11]
; SSE41-NEXT: subps %[[MAGICCST]], %xmm0
; SSE41-NEXT: addps [[VECLOW]], %xmm0
; LOWCST is the low vector of the second part after this point.
; The operands of the blend are inverted because we reuse xmm1
; in the next shift.
; SSE41-NEXT: pblendw $85, %xmm1, %[[LOWCST]]
; SSE41-NEXT: psrld $16, %xmm1
; SSE41-NEXT: pblendw $170, %[[HIGHCST]], %xmm1
; SSE41-NEXT: subps %[[MAGICCST]], %xmm1
; SSE41-NEXT: addps %[[LOWCST]], %xmm1
; SSE41-NEXT: retq
;
; Test that we are not lowering uinttofp to scalars
; AVX-NOT: cvtsd2ss
; AVX: retq
;
; AVX2: vpbroadcastd [[LOWCSTADDR]](%rip), [[LOWCST:%ymm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[LOWCST]], %ymm0, [[LOWVEC:%ymm[0-9]+]]
; AVX2-NEXT: vpsrld $16, %ymm0, [[SHIFTVEC:%ymm[0-9]+]]
; AVX2-NEXT: vpbroadcastd [[HIGHCSTADDR]](%rip), [[HIGHCST:%ymm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[HIGHCST]], [[SHIFTVEC]], [[HIGHVEC:%ymm[0-9]+]]
; AVX2-NEXT: vbroadcastss [[MAGICCSTADDR]](%rip), [[MAGICCST:%ymm[0-9]+]]
; AVX2-NEXT: vsubps [[MAGICCST]], [[HIGHVEC]], [[TMP:%ymm[0-9]+]]
; AVX2-NEXT: vaddps [[TMP]], [[LOWVEC]], %ymm0
; AVX2-NEXT: retq
  %C = uitofp <8 x i32> %A to <8 x float>
  ret <8 x float> %C
}

define <4 x double> @test3(<4 x i32> %arg) {
; CHECK-LABEL: test3:
; This test used to crash because we were custom lowering it as if it was
; a conversion between <4 x i32> and <4 x float>.
; AVX: vsubpd
; AVX2: vsubpd
; CHECK: retq
  %tmp = uitofp <4 x i32> %arg to <4 x double>
  ret <4 x double> %tmp
}
