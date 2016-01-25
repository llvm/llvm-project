; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+ifma | FileCheck %s

declare <8 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_mask_vpmadd52h_uq_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpmadd52h_uq_512:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm3 {%k1}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm4
; CHECK: vpxord %zmm2, %zmm2, %zmm2
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm0 {%k1}
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm2 {%k1} {z}
; CHECK: vpaddq %zmm0, %zmm3, %zmm0
; CHECK: vpaddq %zmm2, %zmm4, %zmm1
; CHECK: vpaddq %zmm0, %zmm1, %zmm0

  %res = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res2 = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.512(<8 x i64> zeroinitializer, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res3 = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res4 = add <8 x i64> %res, %res1
  %res5 = add <8 x i64> %res3, %res2
  %res6 = add <8 x i64> %res5, %res4
  ret <8 x i64> %res6
}

declare <8 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_maskz_vpmadd52h_uq_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vpmadd52h_uq_512:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm3 {%k1} {z}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm4
; CHECK: vpxord %zmm2, %zmm2, %zmm2
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm0 {%k1} {z}
; CHECK: vpmadd52huq %zmm2, %zmm1, %zmm2 {%k1} {z}
; CHECK: vpaddq %zmm0, %zmm3, %zmm0
; CHECK: vpaddq %zmm2, %zmm4, %zmm1
; CHECK: vpaddq %zmm0, %zmm1, %zmm0

  %res = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res2 = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.512(<8 x i64> zeroinitializer, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res3 = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res4 = add <8 x i64> %res, %res1
  %res5 = add <8 x i64> %res3, %res2
  %res6 = add <8 x i64> %res5, %res4
  ret <8 x i64> %res6
}

declare <8 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_mask_vpmadd52l_uq_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpmadd52l_uq_512:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm3 {%k1}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm4
; CHECK: vpxord %zmm2, %zmm2, %zmm2
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm0 {%k1}
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm2 {%k1} {z}
; CHECK: vpaddq %zmm0, %zmm3, %zmm0
; CHECK: vpaddq %zmm2, %zmm4, %zmm1
; CHECK: vpaddq %zmm0, %zmm1, %zmm0

  %res = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res2 = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.512(<8 x i64> zeroinitializer, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res3 = call <8 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res4 = add <8 x i64> %res, %res1
  %res5 = add <8 x i64> %res3, %res2
  %res6 = add <8 x i64> %res5, %res4
  ret <8 x i64> %res6
}

declare <8 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_maskz_vpmadd52l_uq_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vpmadd52l_uq_512:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm3 {%k1} {z}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm4
; CHECK: vpxord %zmm2, %zmm2, %zmm2
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm0 {%k1} {z}
; CHECK: vpmadd52luq %zmm2, %zmm1, %zmm2 {%k1} {z}
; CHECK: vpaddq %zmm0, %zmm3, %zmm0
; CHECK: vpaddq %zmm2, %zmm4, %zmm1
; CHECK: vpaddq %zmm0, %zmm1, %zmm0

  %res = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res2 = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.512(<8 x i64> zeroinitializer, <8 x i64> %x1, <8 x i64> zeroinitializer, i8 %x3)
  %res3 = call <8 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res4 = add <8 x i64> %res, %res1
  %res5 = add <8 x i64> %res3, %res2
  %res6 = add <8 x i64> %res5, %res4
  ret <8 x i64> %res6
}
