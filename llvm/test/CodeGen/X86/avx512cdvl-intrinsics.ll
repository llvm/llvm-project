; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512cd -mattr=+avx512vl| FileCheck %s

declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>, i1) nounwind readonly

declare <4 x i32> @llvm.x86.avx512.mask.lzcnt.d.128(<4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vplzcnt_d_128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vplzcnt_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vplzcntd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vplzcntd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vplzcntd %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.lzcnt.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.lzcnt.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 -1)
  %res3 = call <4 x i32> @llvm.x86.avx512.mask.lzcnt.d.128(<4 x i32> %x0, <4 x i32> zeroinitializer, i8 %x2)
  %res2 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.lzcnt.d.256(<8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vplzcnt_d_256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vplzcnt_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vplzcntd %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vplzcntd %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.lzcnt.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.lzcnt.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.lzcnt.q.128(<2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_vplzcnt_q_128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vplzcnt_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vplzcntq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vplzcntq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.lzcnt.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.lzcnt.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.lzcnt.q.256(<4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_vplzcnt_q_256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vplzcnt_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vplzcntq %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vplzcntq %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.lzcnt.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.lzcnt.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.conflict.d.128(<4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpconflict_d_128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpconflict_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpconflictd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpconflictd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpconflictd %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.conflict.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.conflict.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 -1)
  %res3 = call <4 x i32> @llvm.x86.avx512.mask.conflict.d.128(<4 x i32> %x0, <4 x i32> zeroinitializer, i8 %x2)
  %res2 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.conflict.d.256(<8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpconflict_d_256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpconflict_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpconflictd %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpconflictd %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.conflict.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.conflict.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.conflict.q.128(<2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_vpconflict_q_128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpconflict_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpconflictq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpconflictq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.conflict.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.conflict.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.conflict.q.256(<4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_vpconflict_q_256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpconflict_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpconflictq %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpconflictq %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.conflict.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.conflict.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

