; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math | FileCheck %s --check-prefix=CST --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+sse4.1 | FileCheck %s --check-prefix=CST --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx | FileCheck %s --check-prefix=CST --check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx2 | FileCheck %s --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx512f | FileCheck %s --check-prefix=AVX512F
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx512vl | FileCheck %s --check-prefix=AVX512VL

; Check that the constant used in the vectors are the right ones.
; SSE2: [[MASKCSTADDR:.LCPI[0-9_]+]]:
; SSE2-NEXT: .long 65535 # 0xffff
; SSE2-NEXT: .long 65535 # 0xffff
; SSE2-NEXT: .long 65535 # 0xffff
; SSE2-NEXT: .long 65535 # 0xffff

; CST: [[LOWCSTADDR:.LCPI[0-9_]+]]:
; CST-NEXT: .long 1258291200 # 0x4b000000
; CST-NEXT: .long 1258291200 # 0x4b000000
; CST-NEXT: .long 1258291200 # 0x4b000000
; CST-NEXT: .long 1258291200 # 0x4b000000

; CST: [[HIGHCSTADDR:.LCPI[0-9_]+]]:
; CST-NEXT: .long 1392508928 # 0x53000000
; CST-NEXT: .long 1392508928 # 0x53000000
; CST-NEXT: .long 1392508928 # 0x53000000
; CST-NEXT: .long 1392508928 # 0x53000000

; CST: [[MAGICCSTADDR:.LCPI[0-9_]+]]:
; CST-NEXT: .long 0x53000080 # float 5.49764202E+11
; CST-NEXT: .long 0x53000080 # float 5.49764202E+11
; CST-NEXT: .long 0x53000080 # float 5.49764202E+11
; CST-NEXT: .long 0x53000080 # float 5.49764202E+11

; AVX2: [[LOWCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1258291200 # 0x4b000000

; AVX2: [[HIGHCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1392508928 # 0x53000000

; AVX2: [[MAGICCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 0x53000080 # float 5.49764202E+11

define <4 x float> @test_uitofp_v4i32_to_v4f32(<4 x i32> %arg) {
; SSE2-LABEL: test_uitofp_v4i32_to_v4f32:
; SSE2: movdqa [[MASKCSTADDR]](%rip), [[MASK:%xmm[0-9]+]]
; SSE2-NEXT: pand %xmm0, [[MASK]]
; After this instruction, MASK will have the value of the low parts
; of the vector.
; SSE2-NEXT: por [[LOWCSTADDR]](%rip), [[MASK]]
; SSE2-NEXT: psrld $16, %xmm0
; SSE2-NEXT: por [[HIGHCSTADDR]](%rip), %xmm0
; SSE2-NEXT: subps [[MAGICCSTADDR]](%rip), %xmm0
; SSE2-NEXT: addps [[MASK]], %xmm0
; SSE2-NEXT: retq
;
; Currently we commute the arguments of the first blend, but this could be
; improved to match the lowering of the second blend.
; SSE41-LABEL: test_uitofp_v4i32_to_v4f32:
; SSE41: movdqa [[LOWCSTADDR]](%rip), [[LOWVEC:%xmm[0-9]+]]
; SSE41-NEXT: pblendw $85, %xmm0, [[LOWVEC]]
; SSE41-NEXT: psrld $16, %xmm0
; SSE41-NEXT: pblendw $170, [[HIGHCSTADDR]](%rip), %xmm0
; SSE41-NEXT: subps [[MAGICCSTADDR]](%rip), %xmm0
; SSE41-NEXT: addps [[LOWVEC]], %xmm0
; SSE41-NEXT: retq
;
; AVX-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX: vpblendw $170, [[LOWCSTADDR]](%rip), %xmm0, [[LOWVEC:%xmm[0-9]+]]
; AVX-NEXT: vpsrld $16, %xmm0, [[SHIFTVEC:%xmm[0-9]+]]
; AVX-NEXT: vpblendw $170, [[HIGHCSTADDR]](%rip), [[SHIFTVEC]], [[HIGHVEC:%xmm[0-9]+]]
; AVX-NEXT: vsubps [[MAGICCSTADDR]](%rip), [[HIGHVEC]], [[TMP:%xmm[0-9]+]]
; AVX-NEXT: vaddps [[TMP]], [[LOWVEC]], %xmm0
; AVX-NEXT: retq
;
; The lowering for AVX2 is a bit messy, because we select broadcast
; instructions, instead of folding the constant loads.
; AVX2-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX2: vpbroadcastd [[LOWCSTADDR]](%rip), [[LOWCST:%xmm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[LOWCST]], %xmm0, [[LOWVEC:%xmm[0-9]+]]
; AVX2-NEXT: vpsrld $16, %xmm0, [[SHIFTVEC:%xmm[0-9]+]]
; AVX2-NEXT: vpbroadcastd [[HIGHCSTADDR]](%rip), [[HIGHCST:%xmm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[HIGHCST]], [[SHIFTVEC]], [[HIGHVEC:%xmm[0-9]+]]
; AVX2-NEXT: vbroadcastss [[MAGICCSTADDR]](%rip), [[MAGICCST:%xmm[0-9]+]]
; AVX2-NEXT: vsubps [[MAGICCST]], [[HIGHVEC]], [[TMP:%xmm[0-9]+]]
; AVX2-NEXT: vaddps [[TMP]], [[LOWVEC]], %xmm0
; AVX2-NEXT: retq
;
; AVX512F-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    # kill: def $xmm0 killed $xmm0 def $zmm0
; AVX512F-NEXT:    vcvtudq2ps %zmm0, %zmm0
; AVX512F-NEXT:    # kill: def $xmm0 killed $xmm0 killed $zmm0
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512VL-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX512VL:       # %bb.0:
; AVX512VL-NEXT:    vcvtudq2ps %xmm0, %xmm0
; AVX512VL-NEXT:    retq
  %tmp = uitofp <4 x i32> %arg to <4 x float>
  ret <4 x float> %tmp
}

; Match the AVX2 constants used in the next function
; AVX2: [[LOWCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1258291200 # 0x4b000000

; AVX2: [[HIGHCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1392508928 # 0x53000000

; AVX2: [[MAGICCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 0x53000080 # float 5.49764202E+11

define <8 x float> @test_uitofp_v8i32_to_v8f32(<8 x i32> %arg) {
; Legalization will break the thing is 2 x <4 x i32> on anthing prior AVX.
; The constant used for in the vector instruction are shared between the
; two sequences of instructions.
;
; SSE2-LABEL: test_uitofp_v8i32_to_v8f32:
; SSE2: movdqa {{.*#+}} [[MASK:xmm[0-9]+]] = [65535,65535,65535,65535]
; SSE2-NEXT: movdqa %xmm0, [[VECLOW:%xmm[0-9]+]]
; SSE2-NEXT: pand %[[MASK]], [[VECLOW]]
; SSE2-NEXT: movdqa {{.*#+}} [[LOWCST:xmm[0-9]+]] = [1258291200,1258291200,1258291200,1258291200]
; SSE2-NEXT: por %[[LOWCST]], [[VECLOW]]
; SSE2-NEXT: psrld $16, %xmm0
; SSE2-NEXT: movdqa {{.*#+}} [[HIGHCST:xmm[0-9]+]] = [1392508928,1392508928,1392508928,1392508928]
; SSE2-NEXT: por %[[HIGHCST]], %xmm0
; SSE2-NEXT: movaps {{.*#+}} [[MAGICCST:xmm[0-9]+]] = [5.49764202E+11,5.49764202E+11,5.49764202E+11,5.49764202E+11]
; SSE2-NEXT: subps %[[MAGICCST]], %xmm0
; SSE2-NEXT: addps [[VECLOW]], %xmm0
; MASK is the low vector of the second part after this point.
; SSE2-NEXT: pand %xmm1, %[[MASK]]
; SSE2-NEXT: por %[[LOWCST]], %[[MASK]]
; SSE2-NEXT: psrld $16, %xmm1
; SSE2-NEXT: por %[[HIGHCST]], %xmm1
; SSE2-NEXT: subps %[[MAGICCST]], %xmm1
; SSE2-NEXT: addps %[[MASK]], %xmm1
; SSE2-NEXT: retq
;
; SSE41-LABEL: test_uitofp_v8i32_to_v8f32:
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
; AVX2-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX2: vpbroadcastd [[LOWCSTADDR]](%rip), [[LOWCST:%ymm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[LOWCST]], %ymm0, [[LOWVEC:%ymm[0-9]+]]
; AVX2-NEXT: vpsrld $16, %ymm0, [[SHIFTVEC:%ymm[0-9]+]]
; AVX2-NEXT: vpbroadcastd [[HIGHCSTADDR]](%rip), [[HIGHCST:%ymm[0-9]+]]
; AVX2-NEXT: vpblendw $170, [[HIGHCST]], [[SHIFTVEC]], [[HIGHVEC:%ymm[0-9]+]]
; AVX2-NEXT: vbroadcastss [[MAGICCSTADDR]](%rip), [[MAGICCST:%ymm[0-9]+]]
; AVX2-NEXT: vsubps [[MAGICCST]], [[HIGHVEC]], [[TMP:%ymm[0-9]+]]
; AVX2-NEXT: vaddps [[TMP]], [[LOWVEC]], %ymm0
; AVX2-NEXT: retq
;
; AVX512F-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    # kill
; AVX512F-NEXT:    vcvtudq2ps %zmm0, %zmm0
; AVX512F-NEXT:    # kill
; AVX512F-NEXT:    retq
;
; AVX512VL-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX512VL:       # %bb.0:
; AVX512VL-NEXT:    vcvtudq2ps %ymm0, %ymm0
; AVX512VL-NEXT:    retq
  %tmp = uitofp <8 x i32> %arg to <8 x float>
  ret <8 x float> %tmp
}
