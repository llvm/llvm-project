; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vnni | FileCheck %s

declare <16 x i32> @llvm.x86.avx512.mask.vpdpbusd.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.maskz.vpdpbusd.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_vpdpbusd_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32>* %x2p, <16 x i32> %x4, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpbusd_512:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpdpbusd (%rdi), %zmm1, %zmm3 {%k1}
; CHECK-NEXT:    vmovaps %zmm0, %zmm4
; CHECK-NEXT:    vpdpbusd %zmm2, %zmm1, %zmm4
; CHECK-NEXT:    vpdpbusd %zmm2, %zmm1, %zmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %zmm0, %zmm4, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %x2 = load <16 x i32>, <16 x i32>* %x2p
  %res = call <16 x i32> @llvm.x86.avx512.mask.vpdpbusd.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.vpdpbusd.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16 -1)
  %res2 = call <16 x i32> @llvm.x86.avx512.maskz.vpdpbusd.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16  %x3)
  %res3 = add <16 x i32> %res, %res1
  %res4 = add <16 x i32> %res2, %res3
  ret <16 x i32> %res4
}

declare <16 x i32> @llvm.x86.avx512.mask.vpdpbusds.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.maskz.vpdpbusds.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_vpdpbusds_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32>* %x2p, <16 x i32> %x4, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpbusds_512:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpdpbusds (%rdi), %zmm1, %zmm3 {%k1}
; CHECK-NEXT:    vmovaps %zmm0, %zmm4
; CHECK-NEXT:    vpdpbusds %zmm2, %zmm1, %zmm4
; CHECK-NEXT:    vpdpbusds %zmm2, %zmm1, %zmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %zmm0, %zmm4, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %x2 = load <16 x i32>, <16 x i32>* %x2p
  %res = call <16 x i32> @llvm.x86.avx512.mask.vpdpbusds.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.vpdpbusds.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16 -1)
  %res2 = call <16 x i32> @llvm.x86.avx512.maskz.vpdpbusds.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16  %x3)
  %res3 = add <16 x i32> %res, %res1
  %res4 = add <16 x i32> %res2, %res3
  ret <16 x i32> %res4
}

declare <16 x i32> @llvm.x86.avx512.mask.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.maskz.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_vpdpwssd_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32>* %x2p, <16 x i32> %x4, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpwssd_512:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpdpwssd (%rdi), %zmm1, %zmm3 {%k1}
; CHECK-NEXT:    vmovaps %zmm0, %zmm4
; CHECK-NEXT:    vpdpwssd %zmm2, %zmm1, %zmm4
; CHECK-NEXT:    vpdpwssd %zmm2, %zmm1, %zmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %zmm0, %zmm4, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %x2 = load <16 x i32>, <16 x i32>* %x2p
  %res = call <16 x i32> @llvm.x86.avx512.mask.vpdpwssd.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.vpdpwssd.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16 -1)
  %res2 = call <16 x i32> @llvm.x86.avx512.maskz.vpdpwssd.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16  %x3)
  %res3 = add <16 x i32> %res, %res1
  %res4 = add <16 x i32> %res2, %res3
  ret <16 x i32> %res4
}

declare <16 x i32> @llvm.x86.avx512.mask.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.maskz.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_vpdpwssds_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32>* %x2p, <16 x i32> %x4, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpwssds_512:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpdpwssds (%rdi), %zmm1, %zmm3 {%k1}
; CHECK-NEXT:    vmovaps %zmm0, %zmm4
; CHECK-NEXT:    vpdpwssds %zmm2, %zmm1, %zmm4
; CHECK-NEXT:    vpdpwssds %zmm2, %zmm1, %zmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %zmm0, %zmm4, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %x2 = load <16 x i32>, <16 x i32>* %x2p
  %res = call <16 x i32> @llvm.x86.avx512.mask.vpdpwssds.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.vpdpwssds.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16 -1)
  %res2 = call <16 x i32> @llvm.x86.avx512.maskz.vpdpwssds.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x4, i16  %x3)
  %res3 = add <16 x i32> %res, %res1
  %res4 = add <16 x i32> %res2, %res3
  ret <16 x i32> %res4
}

