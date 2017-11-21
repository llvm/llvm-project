; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vnni,+avx512vl| FileCheck %s

declare <8 x i32> @llvm.x86.avx512.mask.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpbusd_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpbusd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %ymm0, %ymm3
; CHECK-NEXT:    vpdpbusd (%rdi), %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vmovaps %ymm0, %ymm4
; CHECK-NEXT:    vpdpbusd %ymm2, %ymm1, %ymm4
; CHECK-NEXT:    vpdpbusd %ymm2, %ymm1, %ymm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %ymm0, %ymm4, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpbusd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpbusd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpbusd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpbusd_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpbusd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %xmm0, %xmm3
; CHECK-NEXT:    vpdpbusd (%rdi), %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vmovaps %xmm0, %xmm4
; CHECK-NEXT:    vpdpbusd %xmm2, %xmm1, %xmm4
; CHECK-NEXT:    vpdpbusd %xmm2, %xmm1, %xmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %xmm0, %xmm4, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpbusd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpbusds_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpbusds_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %ymm0, %ymm3
; CHECK-NEXT:    vpdpbusds (%rdi), %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vmovaps %ymm0, %ymm4
; CHECK-NEXT:    vpdpbusds %ymm2, %ymm1, %ymm4
; CHECK-NEXT:    vpdpbusds %ymm2, %ymm1, %ymm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %ymm0, %ymm4, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpbusds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpbusds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpbusds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpbusds_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpbusds_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %xmm0, %xmm3
; CHECK-NEXT:    vpdpbusds (%rdi), %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vmovaps %xmm0, %xmm4
; CHECK-NEXT:    vpdpbusds %xmm2, %xmm1, %xmm4
; CHECK-NEXT:    vpdpbusds %xmm2, %xmm1, %xmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %xmm0, %xmm4, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpbusds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpwssd_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpwssd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %ymm0, %ymm3
; CHECK-NEXT:    vpdpwssd (%rdi), %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vmovaps %ymm0, %ymm4
; CHECK-NEXT:    vpdpwssd %ymm2, %ymm1, %ymm4
; CHECK-NEXT:    vpdpwssd %ymm2, %ymm1, %ymm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %ymm0, %ymm4, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpwssd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpwssd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpwssd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpwssd_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpwssd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %xmm0, %xmm3
; CHECK-NEXT:    vpdpwssd (%rdi), %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vmovaps %xmm0, %xmm4
; CHECK-NEXT:    vpdpwssd %xmm2, %xmm1, %xmm4
; CHECK-NEXT:    vpdpwssd %xmm2, %xmm1, %xmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %xmm0, %xmm4, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpwssd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}


declare <8 x i32> @llvm.x86.avx512.mask.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpwssds_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpwssds_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %ymm0, %ymm3
; CHECK-NEXT:    vpdpwssds (%rdi), %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vmovaps %ymm0, %ymm4
; CHECK-NEXT:    vpdpwssds %ymm2, %ymm1, %ymm4
; CHECK-NEXT:    vpdpwssds %ymm2, %ymm1, %ymm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %ymm0, %ymm4, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpwssds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpwssds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpwssds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpwssds_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpdpwssds_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %xmm0, %xmm3
; CHECK-NEXT:    vpdpwssds (%rdi), %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vmovaps %xmm0, %xmm4
; CHECK-NEXT:    vpdpwssds %xmm2, %xmm1, %xmm4
; CHECK-NEXT:    vpdpwssds %xmm2, %xmm1, %xmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %xmm0, %xmm4, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpwssds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

