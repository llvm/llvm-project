; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vl -mattr=+avx512ifma | FileCheck %s

declare <2 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_vpmadd52h_uq_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpmadd52h_uq_128:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm3 {%k1}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm4
; CHECK: vxorps %xmm2, %xmm2, %xmm2
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm0 {%k1}
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm2 {%k1} {z}
; CHECK: vpaddq %xmm0, %xmm3, %xmm0
; CHECK: vpaddq %xmm2, %xmm4, %xmm1
; CHECK: vpaddq %xmm0, %xmm1, %xmm0

  %res = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.128(<2 x i64> zeroinitializer, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res3 = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res4 = add <2 x i64> %res, %res1
  %res5 = add <2 x i64> %res3, %res2
  %res6 = add <2 x i64> %res5, %res4
  ret <2 x i64> %res6
}

declare <4 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_vpmadd52h_uq_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpmadd52h_uq_256:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm3 {%k1}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm4
; CHECK: vxorps %ymm2, %ymm2, %ymm2
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm0 {%k1}
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm2 {%k1} {z}
; CHECK: vpaddq %ymm0, %ymm3, %ymm0
; CHECK: vpaddq %ymm2, %ymm4, %ymm1
; CHECK: vpaddq %ymm0, %ymm1, %ymm0

  %res = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.256(<4 x i64> zeroinitializer, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res3 = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52h.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res4 = add <4 x i64> %res, %res1
  %res5 = add <4 x i64> %res3, %res2
  %res6 = add <4 x i64> %res5, %res4
  ret <4 x i64> %res6
}

declare <2 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_maskz_vpmadd52h_uq_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vpmadd52h_uq_128:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm4
; CHECK: vxorps %xmm2, %xmm2, %xmm2
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm0 {%k1} {z}
; CHECK: vpmadd52huq %xmm2, %xmm1, %xmm2 {%k1} {z}
; CHECK: vpaddq %xmm0, %xmm3, %xmm0
; CHECK: vpaddq %xmm2, %xmm4, %xmm1
; CHECK: vpaddq %xmm0, %xmm1, %xmm0

  %res = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.128(<2 x i64> zeroinitializer, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res3 = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res4 = add <2 x i64> %res, %res1
  %res5 = add <2 x i64> %res3, %res2
  %res6 = add <2 x i64> %res5, %res4
  ret <2 x i64> %res6
}

declare <4 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_maskz_vpmadd52h_uq_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vpmadd52h_uq_256:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm4
; CHECK: vxorps %ymm2, %ymm2, %ymm2
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm0 {%k1} {z}
; CHECK: vpmadd52huq %ymm2, %ymm1, %ymm2 {%k1} {z}
; CHECK: vpaddq %ymm0, %ymm3, %ymm0
; CHECK: vpaddq %ymm2, %ymm4, %ymm1
; CHECK: vpaddq %ymm0, %ymm1, %ymm0

  %res = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.256(<4 x i64> zeroinitializer, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res3 = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52h.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res4 = add <4 x i64> %res, %res1
  %res5 = add <4 x i64> %res3, %res2
  %res6 = add <4 x i64> %res5, %res4
  ret <4 x i64> %res6
}

declare <2 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_vpmadd52l_uq_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpmadd52l_uq_128:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm3 {%k1}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm4
; CHECK: vxorps %xmm2, %xmm2, %xmm2
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm0 {%k1}
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm2 {%k1} {z}
; CHECK: vpaddq %xmm0, %xmm3, %xmm0
; CHECK: vpaddq %xmm2, %xmm4, %xmm1
; CHECK: vpaddq %xmm0, %xmm1, %xmm0

  %res = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.128(<2 x i64> zeroinitializer, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res3 = call <2 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res4 = add <2 x i64> %res, %res1
  %res5 = add <2 x i64> %res3, %res2
  %res6 = add <2 x i64> %res5, %res4
  ret <2 x i64> %res6
}

declare <4 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_vpmadd52l_uq_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpmadd52l_uq_256:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm3 {%k1}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm4
; CHECK: vxorps %ymm2, %ymm2, %ymm2
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm0 {%k1}
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm2 {%k1} {z}
; CHECK: vpaddq %ymm0, %ymm3, %ymm0
; CHECK: vpaddq %ymm2, %ymm4, %ymm1
; CHECK: vpaddq %ymm0, %ymm1, %ymm0

  %res = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.256(<4 x i64> zeroinitializer, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res3 = call <4 x i64> @llvm.x86.avx512.mask.vpmadd52l.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res4 = add <4 x i64> %res, %res1
  %res5 = add <4 x i64> %res3, %res2
  %res6 = add <4 x i64> %res5, %res4
  ret <4 x i64> %res6
}

declare <2 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_maskz_vpmadd52l_uq_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vpmadd52l_uq_128:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm4
; CHECK: vxorps %xmm2, %xmm2, %xmm2
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm0 {%k1} {z}
; CHECK: vpmadd52luq %xmm2, %xmm1, %xmm2 {%k1} {z}
; CHECK: vpaddq %xmm0, %xmm3, %xmm0
; CHECK: vpaddq %xmm2, %xmm4, %xmm1
; CHECK: vpaddq %xmm0, %xmm1, %xmm0

  %res = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.128(<2 x i64> zeroinitializer, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res3 = call <2 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res4 = add <2 x i64> %res, %res1
  %res5 = add <2 x i64> %res3, %res2
  %res6 = add <2 x i64> %res5, %res4
  ret <2 x i64> %res6
}

declare <4 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_maskz_vpmadd52l_uq_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vpmadd52l_uq_256:
; CHECK: kmovw %edi, %k1
; CHECK: vmovaps %zmm0, %zmm3
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK: vmovaps %zmm0, %zmm4
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm4
; CHECK: vxorps %ymm2, %ymm2, %ymm2
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm0 {%k1} {z}
; CHECK: vpmadd52luq %ymm2, %ymm1, %ymm2 {%k1} {z}
; CHECK: vpaddq %ymm0, %ymm3, %ymm0
; CHECK: vpaddq %ymm2, %ymm4, %ymm1
; CHECK: vpaddq %ymm0, %ymm1, %ymm0

  %res = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.256(<4 x i64> zeroinitializer, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res3 = call <4 x i64> @llvm.x86.avx512.maskz.vpmadd52l.uq.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res4 = add <4 x i64> %res, %res1
  %res5 = add <4 x i64> %res3, %res2
  %res6 = add <4 x i64> %res5, %res4
  ret <4 x i64> %res6
}
