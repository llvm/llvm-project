; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw -mattr=+avx512vl --show-mc-encoding| FileCheck %s

; 256-bit

define i32 @test_pcmpeq_b_256(<32 x i8> %a, <32 x i8> %b) {
; CHECK-LABEL: test_pcmpeq_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.b.256(<32 x i8> %a, <32 x i8> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpeq_b_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.b.256(<32 x i8> %a, <32 x i8> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpeq.b.256(<32 x i8>, <32 x i8>, i32)

define i16 @test_pcmpeq_w_256(<16 x i16> %a, <16 x i16> %b) {
; CHECK-LABEL: test_pcmpeq_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.w.256(<16 x i16> %a, <16 x i16> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpeq_w_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.w.256(<16 x i16> %a, <16 x i16> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpeq.w.256(<16 x i16>, <16 x i16>, i16)

define i32 @test_pcmpgt_b_256(<32 x i8> %a, <32 x i8> %b) {
; CHECK-LABEL: test_pcmpgt_b_256
; CHECK: vpcmpgtb %ymm1, %ymm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.b.256(<32 x i8> %a, <32 x i8> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpgt_b_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_b_256
; CHECK: vpcmpgtb %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.b.256(<32 x i8> %a, <32 x i8> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpgt.b.256(<32 x i8>, <32 x i8>, i32)

define i16 @test_pcmpgt_w_256(<16 x i16> %a, <16 x i16> %b) {
; CHECK-LABEL: test_pcmpgt_w_256
; CHECK: vpcmpgtw %ymm1, %ymm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.w.256(<16 x i16> %a, <16 x i16> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpgt_w_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_w_256
; CHECK: vpcmpgtw %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.w.256(<16 x i16> %a, <16 x i16> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpgt.w.256(<16 x i16>, <16 x i16>, i16)

define <8 x i32> @test_cmp_b_256(<32 x i8> %a0, <32 x i8> %a1) {
; CHECK_LABEL: test_cmp_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltb %ymm1, %ymm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleb %ymm1, %ymm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordb %ymm1, %ymm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqb %ymm1, %ymm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltb %ymm1, %ymm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleb %ymm1, %ymm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordb %ymm1, %ymm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_cmp_b_256(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_cmp_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltb %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleb %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordb %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqb %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltb %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleb %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordb %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8>, <32 x i8>, i32, i32) nounwind readnone

define <8 x i32> @test_ucmp_b_256(<32 x i8> %a0, <32 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_256
; CHECK: vpcmpequb %ymm1, %ymm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltub %ymm1, %ymm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleub %ymm1, %ymm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordub %ymm1, %ymm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequb %ymm1, %ymm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltub %ymm1, %ymm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleub %ymm1, %ymm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordub %ymm1, %ymm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_ucmp_b_256(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_256
; CHECK: vpcmpequb %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltub %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleub %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordub %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequb %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltub %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleub %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordub %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8>, <32 x i8>, i32, i32) nounwind readnone

define <8 x i16> @test_cmp_w_256(<16 x i16> %a0, <16 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltw %ymm1, %ymm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmplew %ymm1, %ymm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordw %ymm1, %ymm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqw %ymm1, %ymm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltw %ymm1, %ymm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnlew %ymm1, %ymm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordw %ymm1, %ymm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_w_256(<16 x i16> %a0, <16 x i16> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltw %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmplew %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordw %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqw %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltw %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnlew %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordw %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16>, <16 x i16>, i32, i16) nounwind readnone

define <8 x i16> @test_ucmp_w_256(<16 x i16> %a0, <16 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_256
; CHECK: vpcmpequw %ymm1, %ymm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltuw %ymm1, %ymm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleuw %ymm1, %ymm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunorduw %ymm1, %ymm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequw %ymm1, %ymm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltuw %ymm1, %ymm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleuw %ymm1, %ymm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmporduw %ymm1, %ymm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_w_256(<16 x i16> %a0, <16 x i16> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_256
; CHECK: vpcmpequw %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltuw %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleuw %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunorduw %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequw %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltuw %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleuw %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmporduw %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16>, <16 x i16>, i32, i16) nounwind readnone

; 128-bit

define i16 @test_pcmpeq_b_128(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_pcmpeq_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.b.128(<16 x i8> %a, <16 x i8> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpeq_b_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.b.128(<16 x i8> %a, <16 x i8> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpeq.b.128(<16 x i8>, <16 x i8>, i16)

define i8 @test_pcmpeq_w_128(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_pcmpeq_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.w.128(<8 x i16> %a, <8 x i16> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_w_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.w.128(<8 x i16> %a, <8 x i16> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.w.128(<8 x i16>, <8 x i16>, i8)

define i16 @test_pcmpgt_b_128(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_pcmpgt_b_128
; CHECK: vpcmpgtb %xmm1, %xmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.b.128(<16 x i8> %a, <16 x i8> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpgt_b_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_b_128
; CHECK: vpcmpgtb %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.b.128(<16 x i8> %a, <16 x i8> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpgt.b.128(<16 x i8>, <16 x i8>, i16)

define i8 @test_pcmpgt_w_128(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_pcmpgt_w_128
; CHECK: vpcmpgtw %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.w.128(<8 x i16> %a, <8 x i16> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_w_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_w_128
; CHECK: vpcmpgtw %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.w.128(<8 x i16> %a, <8 x i16> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.w.128(<8 x i16>, <8 x i16>, i8)

define <8 x i16> @test_cmp_b_128(<16 x i8> %a0, <16 x i8> %a1) {
; CHECK_LABEL: test_cmp_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltb %xmm1, %xmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleb %xmm1, %xmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordb %xmm1, %xmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqb %xmm1, %xmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltb %xmm1, %xmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleb %xmm1, %xmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordb %xmm1, %xmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_b_128(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltb %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleb %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordb %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqb %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltb %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleb %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordb %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8>, <16 x i8>, i32, i16) nounwind readnone

define <8 x i16> @test_ucmp_b_128(<16 x i8> %a0, <16 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_128
; CHECK: vpcmpequb %xmm1, %xmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltub %xmm1, %xmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleub %xmm1, %xmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordub %xmm1, %xmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequb %xmm1, %xmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltub %xmm1, %xmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleub %xmm1, %xmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordub %xmm1, %xmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_b_128(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_128
; CHECK: vpcmpequb %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltub %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleub %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordub %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequb %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltub %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleub %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordub %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8>, <16 x i8>, i32, i16) nounwind readnone

define <8 x i8> @test_cmp_w_128(<8 x i16> %a0, <8 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltw %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmplew %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordw %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqw %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltw %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnlew %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordw %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_w_128(<8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_cmp_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltw %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmplew %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordw %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqw %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltw %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnlew %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordw %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16>, <8 x i16>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_w_128(<8 x i16> %a0, <8 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_128
; CHECK: vpcmpequw %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuw %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuw %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduw %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequw %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuw %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuw %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduw %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_w_128(<8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_128
; CHECK: vpcmpequw %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuw %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuw %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduw %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequw %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuw %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuw %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduw %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16>, <8 x i16>, i32, i8) nounwind readnone

declare <8 x float> @llvm.x86.avx512.mask.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfmadd256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd256_ps
  ; CHECK: vfmadd213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xa8,0xc2]
  %res = call <8 x float> @llvm.x86.avx512.mask.vfmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfmadd128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps
  ; CHECK: vfmadd213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0xc2]
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double> @test_mask_fmadd256_pd(<4 x double> %a, <4 x double> %b, <4 x double> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmadd256_pd:
; CHECK: vfmadd213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa8,0xc2]
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c, i8 %mask)
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double> @test_mask_fmadd128_pd(<2 x double> %a, <2 x double> %b, <2 x double> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmadd128_pd:
; CHECK: vfmadd213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa8,0xc2]
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double> %a, <2 x double> %b, <2 x double> %c, i8 %mask)
  ret <2 x double> %res
}

define <2 x double>@test_int_x86_avx512_mask_vfmadd_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmadd_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213pd %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfmadd213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <2 x double> @llvm.x86.avx512.mask3.vfmadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask3_vfmadd_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmadd_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmadd231pd %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmadd213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask3.vfmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask3.vfmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <2 x double> @llvm.x86.avx512.maskz.vfmadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_maskz_vfmadd_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmadd_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213pd %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK-NEXT:    vfmadd213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.maskz.vfmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.maskz.vfmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

define <4 x double>@test_int_x86_avx512_mask_vfmadd_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmadd_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask3.vfmadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask3_vfmadd_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmadd_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmadd231pd %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask3.vfmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask3.vfmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.maskz.vfmadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_maskz_vfmadd_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmadd_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.maskz.vfmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.maskz.vfmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

define <4 x float>@test_int_x86_avx512_mask_vfmadd_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmadd_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213ps %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfmadd213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <4 x float> @llvm.x86.avx512.mask3.vfmadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask3_vfmadd_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmadd_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmadd231ps %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmadd213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask3.vfmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask3.vfmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <4 x float> @llvm.x86.avx512.maskz.vfmadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_maskz_vfmadd_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmadd_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213ps %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK-NEXT:    vfmadd213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.maskz.vfmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.maskz.vfmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

define <8 x float>@test_int_x86_avx512_mask_vfmadd_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmadd_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213ps %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfmadd213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.vfmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.vfmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask3.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask3_vfmadd_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmadd_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmadd231ps %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmadd213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask3.vfmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask3.vfmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.maskz.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_maskz_vfmadd_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmadd_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmadd213ps %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK-NEXT:    vfmadd213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.maskz.vfmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.maskz.vfmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}


declare <2 x double> @llvm.x86.avx512.mask3.vfmsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask3_vfmsub_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsub_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsub231pd %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmsub213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask3.vfmsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask3.vfmsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}


declare <4 x double> @llvm.x86.avx512.mask3.vfmsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask3_vfmsub_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsub_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsub231pd %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmsub213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask3.vfmsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask3.vfmsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask3.vfmsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask3_vfmsub_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsub_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsub231ps %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask3.vfmsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask3.vfmsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask3.vfmsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask3_vfmsub_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsub_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsub231ps %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask3.vfmsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask3.vfmsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.vfnmadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfnmadd256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd256_ps
  ; CHECK: vfnmadd213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xac,0xc2]
  %res = call <8 x float> @llvm.x86.avx512.mask.vfnmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.vfnmadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfnmadd128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd128_ps
  ; CHECK: vfnmadd213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xac,0xc2]
  %res = call <4 x float> @llvm.x86.avx512.mask.vfnmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.avx512.mask.vfnmadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfnmadd256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd256_pd
  ; CHECK: vfnmadd213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xac,0xc2]
  %res = call <4 x double> @llvm.x86.avx512.mask.vfnmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.avx512.mask.vfnmadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfnmadd128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd128_pd
  ; CHECK: vfnmadd213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xac,0xc2]
  %res = call <2 x double> @llvm.x86.avx512.mask.vfnmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

declare <8 x float> @llvm.x86.avx512.mask.vfnmsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfnmsub256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub256_ps
  ; CHECK: vfnmsub213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xae,0xc2]
  %res = call <8 x float> @llvm.x86.avx512.mask.vfnmsub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.vfnmsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfnmsub128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub128_ps
  ; CHECK: vfnmsub213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xae,0xc2]
  %res = call <4 x float> @llvm.x86.avx512.mask.vfnmsub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.avx512.mask.vfnmsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfnmsub256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub256_pd
  ; CHECK: vfnmsub213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xae,0xc2]
  %res = call <4 x double> @llvm.x86.avx512.mask.vfnmsub.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.avx512.mask.vfnmsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfnmsub128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub128_pd
  ; CHECK: vfnmsub213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xae,0xc2]
  %res = call <2 x double> @llvm.x86.avx512.mask.vfnmsub.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}


define <2 x double>@test_int_x86_avx512_mask_vfnmsub_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmsub_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmsub213pd %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfnmsub213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.vfnmsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.vfnmsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <2 x double> @llvm.x86.avx512.mask3.vfnmsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask3_vfnmsub_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfnmsub_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfnmsub231pd %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfnmsub213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask3.vfnmsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask3.vfnmsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

define <4 x double>@test_int_x86_avx512_mask_vfnmsub_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmsub_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmsub213pd %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfnmsub213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.vfnmsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.vfnmsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask3.vfnmsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask3_vfnmsub_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfnmsub_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfnmsub231pd %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfnmsub213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask3.vfnmsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask3.vfnmsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

define <4 x float>@test_int_x86_avx512_mask_vfnmsub_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmsub_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmsub213ps %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfnmsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.vfnmsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vfnmsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <4 x float> @llvm.x86.avx512.mask3.vfnmsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask3_vfnmsub_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfnmsub_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfnmsub231ps %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfnmsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask3.vfnmsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask3.vfnmsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

define <8 x float>@test_int_x86_avx512_mask_vfnmsub_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmsub_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmsub213ps %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfnmsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.vfnmsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.vfnmsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask3.vfnmsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask3_vfnmsub_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfnmsub_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfnmsub231ps %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfnmsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask3.vfnmsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask3.vfnmsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

define <2 x double>@test_int_x86_avx512_mask_vfnmadd_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmadd_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmadd213pd %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfnmadd213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.vfnmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.vfnmadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

define <4 x double>@test_int_x86_avx512_mask_vfnmadd_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmadd_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmadd213pd %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfnmadd213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.vfnmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.vfnmadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

define <4 x float>@test_int_x86_avx512_mask_vfnmadd_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmadd_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmadd213ps %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfnmadd213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.vfnmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vfnmadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

define <8 x float>@test_int_x86_avx512_mask_vfnmadd_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfnmadd_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfnmadd213ps %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfnmadd213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.vfnmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.vfnmadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_fmaddsub256_ps(<8 x float> %a, <8 x float> %b, <8 x float> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmaddsub256_ps:
; CHECK: vfmaddsub213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xa6,0xc2]
  %res = call <8 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c, i8 %mask)
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_fmaddsub128_ps(<4 x float> %a, <4 x float> %b, <4 x float> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmaddsub128_ps:
; CHECK: vfmaddsub213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa6,0xc2]
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.128(<4 x float> %a, <4 x float> %b, <4 x float> %c, i8 %mask)
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfmaddsub256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmaddsub256_pd
  ; CHECK: vfmaddsub213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa6,0xc2]
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfmaddsub128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmaddsub128_pd
  ; CHECK: vfmaddsub213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa6,0xc2]
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

define <2 x double>@test_int_x86_avx512_mask_vfmaddsub_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmaddsub_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213pd %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfmaddsub213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <2 x double> @llvm.x86.avx512.mask3.vfmaddsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask3_vfmaddsub_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmaddsub_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmaddsub231pd %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmaddsub213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask3.vfmaddsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask3.vfmaddsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <2 x double> @llvm.x86.avx512.maskz.vfmaddsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_maskz_vfmaddsub_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmaddsub_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213pd %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK-NEXT:    vfmaddsub213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.maskz.vfmaddsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.maskz.vfmaddsub.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

define <4 x double>@test_int_x86_avx512_mask_vfmaddsub_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmaddsub_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213pd %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfmaddsub213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.vfmaddsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask3.vfmaddsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask3_vfmaddsub_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmaddsub_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmaddsub231pd %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmaddsub213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask3.vfmaddsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask3.vfmaddsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.maskz.vfmaddsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_maskz_vfmaddsub_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmaddsub_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213pd %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK-NEXT:    vfmaddsub213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.maskz.vfmaddsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.maskz.vfmaddsub.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

define <4 x float>@test_int_x86_avx512_mask_vfmaddsub_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmaddsub_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213ps %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vfmaddsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <4 x float> @llvm.x86.avx512.mask3.vfmaddsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask3_vfmaddsub_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmaddsub_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmaddsub231ps %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmaddsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask3.vfmaddsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask3.vfmaddsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <4 x float> @llvm.x86.avx512.maskz.vfmaddsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_maskz_vfmaddsub_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmaddsub_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213ps %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK-NEXT:    vfmaddsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.maskz.vfmaddsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.maskz.vfmaddsub.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

define <8 x float>@test_int_x86_avx512_mask_vfmaddsub_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vfmaddsub_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213ps %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vfmaddsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.vfmaddsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask3.vfmaddsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask3_vfmaddsub_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmaddsub_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmaddsub231ps %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmaddsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask3.vfmaddsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask3.vfmaddsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.maskz.vfmaddsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_maskz_vfmaddsub_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_maskz_vfmaddsub_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vfmaddsub213ps %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK-NEXT:    vfmaddsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.maskz.vfmaddsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.maskz.vfmaddsub.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask3.vfmsubadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask3_vfmsubadd_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsubadd_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsubadd231pd %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmsubadd213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask3.vfmsubadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask3.vfmsubadd.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2=fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask3.vfmsubadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask3_vfmsubadd_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsubadd_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsubadd231pd %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmsubadd213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask3.vfmsubadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask3.vfmsubadd.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2=fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask3.vfmsubadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask3_vfmsubadd_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsubadd_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsubadd231ps %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vfmsubadd213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask3.vfmsubadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask3.vfmsubadd.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2=fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask3.vfmsubadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask3_vfmsubadd_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask3_vfmsubadd_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vfmsubadd231ps %ymm1, %ymm0, %ymm3 {%k1}
; CHECK-NEXT:    vfmsubadd213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask3.vfmsubadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask3.vfmsubadd.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2=fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}


define <4 x float> @test_mask_vfmadd128_ps_r(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_r
  ; CHECK: vfmadd213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0xc2]
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rz(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rz
  ; CHECK: vfmadd213ps %xmm2, %xmm1, %xmm0 ## encoding: [0x62,0xf2,0x75,0x08,0xa8,0xc2]
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmk(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmk
  ; CHECK: vfmadd213ps	(%rdi), %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmka(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmka
  ; CHECK: vfmadd213ps     (%rdi), %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2, align 8
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmkz(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmkz
  ; CHECK: vfmadd213ps	(%rdi), %xmm1, %xmm0 ## encoding: [0xc4,0xe2,0x71,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmkza(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmkza
  ; CHECK: vfmadd213ps	(%rdi), %xmm1, %xmm0 ## encoding: [0xc4,0xe2,0x71,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2, align 4
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmb(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmb
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x19,0xa8,0x07]
  %q = load float, float* %ptr_a2
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmba(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmba
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x19,0xa8,0x07]
  %q = load float, float* %ptr_a2, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmbz(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmbz
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0  ## encoding: [0x62,0xf2,0x75,0x18,0xa8,0x07]
  %q = load float, float* %ptr_a2
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmbza(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmbza
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0  ## encoding: [0x62,0xf2,0x75,0x18,0xa8,0x07]
  %q = load float, float* %ptr_a2, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.avx512.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 -1) nounwind
  ret <4 x float> %res
}

define <2 x double> @test_mask_vfmadd128_pd_r(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_r
  ; CHECK: vfmadd213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa8,0xc2]
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_mask_vfmadd128_pd_rz(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_rz
  ; CHECK: vfmadd213pd %xmm2, %xmm1, %xmm0 ## encoding: [0x62,0xf2,0xf5,0x08,0xa8,0xc2]
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 -1) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_mask_vfmadd128_pd_rmk(<2 x double> %a0, <2 x double> %a1, <2 x double>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_rmk
  ; CHECK: vfmadd213pd	(%rdi), %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa8,0x07]
  %a2 = load <2 x double>, <2 x double>* %ptr_a2
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_mask_vfmadd128_pd_rmkz(<2 x double> %a0, <2 x double> %a1, <2 x double>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_rmkz
  ; CHECK: vfmadd213pd	(%rdi), %xmm1, %xmm0 ## encoding: [0xc4,0xe2,0xf1,0xa8,0x07]
  %a2 = load <2 x double>, <2 x double>* %ptr_a2
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 -1) nounwind
  ret <2 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_r(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_r
  ; CHECK: vfmadd213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa8,0xc2]
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_rz(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_rz
  ; CHECK: vfmadd213pd %ymm2, %ymm1, %ymm0 ## encoding: [0x62,0xf2,0xf5,0x28,0xa8,0xc2]
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 -1) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_rmk(<4 x double> %a0, <4 x double> %a1, <4 x double>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_rmk
  ; CHECK: vfmadd213pd	(%rdi), %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa8,0x07]
  %a2 = load <4 x double>, <4 x double>* %ptr_a2
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_rmkz(<4 x double> %a0, <4 x double> %a1, <4 x double>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_rmkz
  ; CHECK: vfmadd213pd	(%rdi), %ymm1, %ymm0 ## encoding: [0xc4,0xe2,0xf5,0xa8,0x07]
  %a2 = load <4 x double>, <4 x double>* %ptr_a2
  %res = call <4 x double> @llvm.x86.avx512.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 -1) nounwind
  ret <4 x double> %res
}
define <8 x i16> @test_mask_add_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_add_epi16_rr_128
  ;CHECK: vpaddw %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xfd,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrk_128
  ;CHECK: vpaddw %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfd,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrkz_128
  ;CHECK: vpaddw %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfd,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi16_rm_128
  ;CHECK: vpaddw (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xfd,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmk_128
  ;CHECK: vpaddw (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfd,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmkz_128
  ;CHECK: vpaddw (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfd,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_add_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_add_epi16_rr_256
  ;CHECK: vpaddw %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xfd,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrk_256
  ;CHECK: vpaddw %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfd,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrkz_256
  ;CHECK: vpaddw %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfd,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi16_rm_256
  ;CHECK: vpaddw (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xfd,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmk_256
  ;CHECK: vpaddw (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfd,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmkz_256
  ;CHECK: vpaddw (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfd,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_sub_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rr_128
  ;CHECK: vpsubw %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xf9,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrk_128
  ;CHECK: vpsubw %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xf9,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrkz_128
  ;CHECK: vpsubw %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xf9,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rm_128
  ;CHECK: vpsubw (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xf9,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmk_128
  ;CHECK: vpsubw (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xf9,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmkz_128
  ;CHECK: vpsubw (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xf9,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_sub_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rr_256
  ;CHECK: vpsubw %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xf9,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrk_256
  ;CHECK: vpsubw %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xf9,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrkz_256
  ;CHECK: vpsubw %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xf9,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rm_256
  ;CHECK: vpsubw (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xf9,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmk_256
  ;CHECK: vpsubw (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xf9,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmkz_256
  ;CHECK: vpsubw (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xf9,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <32 x i16> @test_mask_add_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_add_epi16_rr_512
  ;CHECK: vpaddw %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xfd,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrk_512
  ;CHECK: vpaddw %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfd,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrkz_512
  ;CHECK: vpaddw %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfd,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi16_rm_512
  ;CHECK: vpaddw (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xfd,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmk_512
  ;CHECK: vpaddw (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfd,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmkz_512
  ;CHECK: vpaddw (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfd,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_sub_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rr_512
  ;CHECK: vpsubw %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xf9,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrk_512
  ;CHECK: vpsubw %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xf9,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrkz_512
  ;CHECK: vpsubw %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xf9,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rm_512
  ;CHECK: vpsubw (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xf9,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmk_512
  ;CHECK: vpsubw (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xf9,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmkz_512
  ;CHECK: vpsubw (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xf9,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_mullo_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rr_512
  ;CHECK: vpmullw %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xd5,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrk_512
  ;CHECK: vpmullw %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xd5,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrkz_512
  ;CHECK: vpmullw %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xd5,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rm_512
  ;CHECK: vpmullw (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xd5,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmk_512
  ;CHECK: vpmullw (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xd5,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmkz_512
  ;CHECK: vpmullw (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xd5,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <8 x i16> @test_mask_mullo_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rr_128
  ;CHECK: vpmullw %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xd5,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrk_128
  ;CHECK: vpmullw %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xd5,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrkz_128
  ;CHECK: vpmullw %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xd5,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rm_128
  ;CHECK: vpmullw (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xd5,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmk_128
  ;CHECK: vpmullw (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xd5,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmkz_128
  ;CHECK: vpmullw (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xd5,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_mullo_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rr_256
  ;CHECK: vpmullw %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xd5,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrk_256
  ;CHECK: vpmullw %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xd5,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrkz_256
  ;CHECK: vpmullw %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xd5,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rm_256
  ;CHECK: vpmullw (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xd5,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmk_256
  ;CHECK: vpmullw (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xd5,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmkz_256
  ;CHECK: vpmullw (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xd5,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)


define <8 x i16> @test_mask_packs_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rr_128
  ;CHECK: vpackssdw       %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x6b,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrk_128
  ;CHECK: vpackssdw       %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0x6b,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrkz_128
  ;CHECK: vpackssdw       %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0x6b,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rm_128
  ;CHECK: vpackssdw       (%rdi), %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x6b,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmk_128
  ;CHECK: vpackssdw       (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0x6b,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmkz_128
  ;CHECK: vpackssdw       (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0x6b,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmb_128
  ;CHECK: vpackssdw       (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbk_128
  ;CHECK: vpackssdw       (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0x6b,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbkz_128
  ;CHECK: vpackssdw       (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32>, <4 x i32>, <8 x i16>, i8)

define <16 x i16> @test_mask_packs_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rr_256
  ;CHECK: vpackssdw       %ymm1, %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x6b,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrk_256
  ;CHECK: vpackssdw       %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0x6b,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrkz_256
  ;CHECK: vpackssdw       %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0x6b,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rm_256
  ;CHECK: vpackssdw       (%rdi), %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x6b,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmk_256
  ;CHECK: vpackssdw       (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0x6b,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmkz_256
  ;CHECK: vpackssdw       (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0x6b,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmb_256
  ;CHECK: vpackssdw       (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbk_256
  ;CHECK: vpackssdw       (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0x6b,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbkz_256
  ;CHECK: vpackssdw       (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0x6b,0x07]  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32>, <8 x i32>, <16 x i16>, i16)

define <16 x i8> @test_mask_packs_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rr_128
  ;CHECK: vpacksswb       %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x63,0xc1]
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrk_128
  ;CHECK: vpacksswb       %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0x63,0xd1]
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrkz_128
  ;CHECK: vpacksswb       %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0x63,0xc1]
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rm_128
  ;CHECK: vpacksswb       (%rdi), %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x63,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmk_128
  ;CHECK: vpacksswb       (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0x63,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmkz_128
  ;CHECK: vpacksswb       (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0x63,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16>, <8 x i16>, <16 x i8>, i16)

define <32 x i8> @test_mask_packs_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rr_256
  ;CHECK: vpacksswb       %ymm1, %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x63,0xc1]
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrk_256
  ;CHECK: vpacksswb       %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0x63,0xd1]
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrkz_256
  ;CHECK: vpacksswb       %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0x63,0xc1]
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rm_256
  ;CHECK: vpacksswb       (%rdi), %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x63,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmk_256
  ;CHECK: vpacksswb       (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0x63,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmkz_256
  ;CHECK: vpacksswb       (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0x63,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16>, <16 x i16>, <32 x i8>, i32)


define <8 x i16> @test_mask_packus_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rr_128
  ;CHECK: vpackusdw       %xmm1, %xmm0, %xmm0 
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrk_128
  ;CHECK: vpackusdw       %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrkz_128
  ;CHECK: vpackusdw       %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rm_128
  ;CHECK: vpackusdw       (%rdi), %xmm0, %xmm0 
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmk_128
  ;CHECK: vpackusdw       (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmkz_128
  ;CHECK: vpackusdw       (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmb_128
  ;CHECK: vpackusdw       (%rdi){1to4}, %xmm0, %xmm0  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbk_128
  ;CHECK: vpackusdw       (%rdi){1to4}, %xmm0, %xmm1 {%k1} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbkz_128
  ;CHECK: vpackusdw       (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32>, <4 x i32>, <8 x i16>, i8)

define <16 x i16> @test_mask_packus_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rr_256
  ;CHECK: vpackusdw       %ymm1, %ymm0, %ymm0 
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrk_256
  ;CHECK: vpackusdw       %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrkz_256
  ;CHECK: vpackusdw       %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rm_256
  ;CHECK: vpackusdw       (%rdi), %ymm0, %ymm0 
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmk_256
  ;CHECK: vpackusdw       (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmkz_256
  ;CHECK: vpackusdw       (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmb_256
  ;CHECK: vpackusdw       (%rdi){1to8}, %ymm0, %ymm0  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbk_256
  ;CHECK: vpackusdw       (%rdi){1to8}, %ymm0, %ymm1 {%k1} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbkz_256
  ;CHECK: vpackusdw       (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32>, <8 x i32>, <16 x i16>, i16)

define <16 x i8> @test_mask_packus_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rr_128
  ;CHECK: vpackuswb       %xmm1, %xmm0, %xmm0 
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrk_128
  ;CHECK: vpackuswb       %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrkz_128
  ;CHECK: vpackuswb       %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rm_128
  ;CHECK: vpackuswb       (%rdi), %xmm0, %xmm0 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmk_128
  ;CHECK: vpackuswb       (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmkz_128
  ;CHECK: vpackuswb       (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16>, <8 x i16>, <16 x i8>, i16)

define <32 x i8> @test_mask_packus_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rr_256
  ;CHECK: vpackuswb       %ymm1, %ymm0, %ymm0 
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrk_256
  ;CHECK: vpackuswb       %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrkz_256
  ;CHECK: vpackuswb       %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rm_256
  ;CHECK: vpackuswb       (%rdi), %ymm0, %ymm0 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmk_256
  ;CHECK: vpackuswb       (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmkz_256
  ;CHECK: vpackuswb       (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16>, <16 x i16>, <32 x i8>, i32)

define <8 x i16> @test_mask_adds_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rr_128
  ;CHECK: vpaddsw %xmm1, %xmm0, %xmm0 
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrk_128
  ;CHECK: vpaddsw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrkz_128
  ;CHECK: vpaddsw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rm_128
  ;CHECK: vpaddsw (%rdi), %xmm0, %xmm0 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmk_128
  ;CHECK: vpaddsw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmkz_128
  ;CHECK: vpaddsw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_adds_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rr_256
  ;CHECK: vpaddsw %ymm1, %ymm0, %ymm0 
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrk_256
  ;CHECK: vpaddsw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrkz_256
  ;CHECK: vpaddsw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rm_256
  ;CHECK: vpaddsw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmk_256
  ;CHECK: vpaddsw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmkz_256
  ;CHECK: vpaddsw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_subs_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rr_128
  ;CHECK: vpsubsw %xmm1, %xmm0, %xmm0     
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrk_128
  ;CHECK: vpsubsw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrkz_128
  ;CHECK: vpsubsw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rm_128
  ;CHECK: vpsubsw (%rdi), %xmm0, %xmm0
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmk_128
  ;CHECK: vpsubsw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmkz_128
  ;CHECK: vpsubsw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_subs_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rr_256
  ;CHECK: vpsubsw %ymm1, %ymm0, %ymm0     
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrk_256
  ;CHECK: vpsubsw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrkz_256
  ;CHECK: vpsubsw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rm_256
  ;CHECK: vpsubsw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmk_256
  ;CHECK: vpsubsw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmkz_256
  ;CHECK: vpsubsw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_adds_epu16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rr_128
  ;CHECK: vpaddusw %xmm1, %xmm0, %xmm0 
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrk_128
  ;CHECK: vpaddusw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrkz_128
  ;CHECK: vpaddusw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rm_128
  ;CHECK: vpaddusw (%rdi), %xmm0, %xmm0 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmk_128
  ;CHECK: vpaddusw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmkz_128
  ;CHECK: vpaddusw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_adds_epu16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rr_256
  ;CHECK: vpaddusw %ymm1, %ymm0, %ymm0 
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrk_256
  ;CHECK: vpaddusw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrkz_256
  ;CHECK: vpaddusw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rm_256
  ;CHECK: vpaddusw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmk_256
  ;CHECK: vpaddusw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmkz_256
  ;CHECK: vpaddusw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_subs_epu16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rr_128
  ;CHECK: vpsubusw %xmm1, %xmm0, %xmm0     
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrk_128
  ;CHECK: vpsubusw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrkz_128
  ;CHECK: vpsubusw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rm_128
  ;CHECK: vpsubusw (%rdi), %xmm0, %xmm0
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmk_128
  ;CHECK: vpsubusw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmkz_128
  ;CHECK: vpsubusw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_subs_epu16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rr_256
  ;CHECK: vpsubusw %ymm1, %ymm0, %ymm0     
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrk_256
  ;CHECK: vpsubusw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrkz_256
  ;CHECK: vpsubusw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rm_256
  ;CHECK: vpsubusw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmk_256
  ;CHECK: vpsubusw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmkz_256
  ;CHECK: vpsubusw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i8> @test_mask_adds_epi8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rr_128
  ;CHECK: vpaddsb %xmm1, %xmm0, %xmm0 
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrk_128
  ;CHECK: vpaddsb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrkz_128
  ;CHECK: vpaddsb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rm_128
  ;CHECK: vpaddsb (%rdi), %xmm0, %xmm0 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmk_128
  ;CHECK: vpaddsb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmkz_128
  ;CHECK: vpaddsb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_adds_epi8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rr_256
  ;CHECK: vpaddsb %ymm1, %ymm0, %ymm0 
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrk_256
  ;CHECK: vpaddsb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrkz_256
  ;CHECK: vpaddsb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rm_256
  ;CHECK: vpaddsb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmk_256
  ;CHECK: vpaddsb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmkz_256
  ;CHECK: vpaddsb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <16 x i8> @test_mask_subs_epi8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rr_128
  ;CHECK: vpsubsb %xmm1, %xmm0, %xmm0     
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrk_128
  ;CHECK: vpsubsb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrkz_128
  ;CHECK: vpsubsb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rm_128
  ;CHECK: vpsubsb (%rdi), %xmm0, %xmm0
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmk_128
  ;CHECK: vpsubsb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmkz_128
  ;CHECK: vpsubsb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_subs_epi8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rr_256
  ;CHECK: vpsubsb %ymm1, %ymm0, %ymm0     
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrk_256
  ;CHECK: vpsubsb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrkz_256
  ;CHECK: vpsubsb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rm_256
  ;CHECK: vpsubsb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmk_256
  ;CHECK: vpsubsb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmkz_256
  ;CHECK: vpsubsb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <16 x i8> @test_mask_adds_epu8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rr_128
  ;CHECK: vpaddusb %xmm1, %xmm0, %xmm0 
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrk_128
  ;CHECK: vpaddusb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrkz_128
  ;CHECK: vpaddusb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rm_128
  ;CHECK: vpaddusb (%rdi), %xmm0, %xmm0 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmk_128
  ;CHECK: vpaddusb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmkz_128
  ;CHECK: vpaddusb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_adds_epu8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rr_256
  ;CHECK: vpaddusb %ymm1, %ymm0, %ymm0 
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrk_256
  ;CHECK: vpaddusb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrkz_256
  ;CHECK: vpaddusb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rm_256
  ;CHECK: vpaddusb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmk_256
  ;CHECK: vpaddusb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmkz_256
  ;CHECK: vpaddusb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <16 x i8> @test_mask_subs_epu8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rr_128
  ;CHECK: vpsubusb %xmm1, %xmm0, %xmm0     
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrk_128
  ;CHECK: vpsubusb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrkz_128
  ;CHECK: vpsubusb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rm_128
  ;CHECK: vpsubusb (%rdi), %xmm0, %xmm0
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmk_128
  ;CHECK: vpsubusb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmkz_128
  ;CHECK: vpsubusb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_subs_epu8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rr_256
  ;CHECK: vpsubusb %ymm1, %ymm0, %ymm0     
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrk_256
  ;CHECK: vpsubusb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrkz_256
  ;CHECK: vpsubusb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rm_256
  ;CHECK: vpsubusb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmk_256
  ;CHECK: vpsubusb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmkz_256
  ;CHECK: vpsubusb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

declare <16 x i8> @llvm.x86.avx512.mask.pmaxs.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_b_128
; CHECK-NOT: call 
; CHECK: vpmaxsb %xmm
; CHECK: {%k1} 
define <16 x i8>@test_int_x86_avx512_mask_pmaxs_b_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %mask) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.pmaxs.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2 ,i16 %mask)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmaxs.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> zeroinitializer, i16 %mask)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pmaxs.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_b_256
; CHECK-NOT: call 
; CHECK: vpmaxsb %ymm
; CHECK: {%k1} 
define <32 x i8>@test_int_x86_avx512_mask_pmaxs_b_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.pmaxs.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.pmaxs.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pmaxs.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_w_128
; CHECK-NOT: call 
; CHECK: vpmaxsw %xmm
; CHECK: {%k1} 
define <8 x i16>@test_int_x86_avx512_mask_pmaxs_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmaxs.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmaxs.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pmaxs.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_w_256
; CHECK-NOT: call 
; CHECK: vpmaxsw %ymm
; CHECK: {%k1} 
define <16 x i16>@test_int_x86_avx512_mask_pmaxs_w_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmaxs.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmaxs.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %mask)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pmaxu.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_b_128
; CHECK-NOT: call 
; CHECK: vpmaxub %xmm
; CHECK: {%k1} 
define <16 x i8>@test_int_x86_avx512_mask_pmaxu_b_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2,i16 %mask) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.pmaxu.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %mask)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmaxu.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> zeroinitializer, i16 %mask)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pmaxu.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_b_256
; CHECK-NOT: call 
; CHECK: vpmaxub %ymm
; CHECK: {%k1} 
define <32 x i8>@test_int_x86_avx512_mask_pmaxu_b_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.pmaxu.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.pmaxu.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pmaxu.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_w_128
; CHECK-NOT: call 
; CHECK: vpmaxuw %xmm
; CHECK: {%k1} 
define <8 x i16>@test_int_x86_avx512_mask_pmaxu_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmaxu.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmaxu.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pmaxu.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_w_256
; CHECK-NOT: call 
; CHECK: vpmaxuw %ymm
; CHECK: {%k1} 
define <16 x i16>@test_int_x86_avx512_mask_pmaxu_w_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmaxu.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmaxu.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %mask)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pmins.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_b_128
; CHECK-NOT: call 
; CHECK: vpminsb %xmm
; CHECK: {%k1} 
define <16 x i8>@test_int_x86_avx512_mask_pmins_b_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %mask) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.pmins.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %mask)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmins.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> zeroinitializer, i16 %mask)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pmins.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_b_256
; CHECK-NOT: call 
; CHECK: vpminsb %ymm
; CHECK: {%k1} 
define <32 x i8>@test_int_x86_avx512_mask_pmins_b_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.pmins.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.pmins.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pmins.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_w_128
; CHECK-NOT: call 
; CHECK: vpminsw %xmm
; CHECK: {%k1} 
define <8 x i16>@test_int_x86_avx512_mask_pmins_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmins.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmins.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pmins.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_w_256
; CHECK-NOT: call 
; CHECK: vpminsw %ymm
; CHECK: {%k1} 
define <16 x i16>@test_int_x86_avx512_mask_pmins_w_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmins.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmins.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %mask)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pminu.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_b_128
; CHECK-NOT: call 
; CHECK: vpminub %xmm
; CHECK: {%k1} 
define <16 x i8>@test_int_x86_avx512_mask_pminu_b_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %mask) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.pminu.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %mask)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.pminu.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> zeroinitializer, i16 %mask)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pminu.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_b_256
; CHECK-NOT: call 
; CHECK: vpminub %ymm
; CHECK: {%k1} 
define <32 x i8>@test_int_x86_avx512_mask_pminu_b_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.pminu.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.pminu.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pminu.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_w_128
; CHECK-NOT: call 
; CHECK: vpminuw %xmm
; CHECK: {%k1} 
define <8 x i16>@test_int_x86_avx512_mask_pminu_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pminu.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pminu.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pminu.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_w_256
; CHECK-NOT: call 
; CHECK: vpminuw %ymm
; CHECK: {%k1} 
define <16 x i16>@test_int_x86_avx512_mask_pminu_w_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pminu.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %mask)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pminu.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %mask)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermt2var_hi_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2w %xmm{{.*}}{%k1} 
; CHECK-NOT: {z}
define <8 x i16>@test_int_x86_avx512_mask_vpermt2var_hi_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <8 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_hi_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2w %xmm{{.*}}{%k1} {z}
define <8 x i16>@test_int_x86_avx512_maskz_vpermt2var_hi_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermt2var_hi_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2w %ymm{{.*}}{%k1} 
define <16 x i16>@test_int_x86_avx512_mask_vpermt2var_hi_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_hi_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2w %ymm{{.*}}{%k1} {z}
define <16 x i16>@test_int_x86_avx512_maskz_vpermt2var_hi_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
  %res = call <16 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_hi_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermi2w %xmm{{.*}}{%k1} 
define <8 x i16>@test_int_x86_avx512_mask_vpermi2var_hi_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_hi_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermi2w %ymm{{.*}}{%k1} 
define <16 x i16>@test_int_x86_avx512_mask_vpermi2var_hi_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pavg.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pavg_b_128
; CHECK-NOT: call 
; CHECK: vpavgb %xmm
; CHECK: {%k1} 
define <16 x i8>@test_int_x86_avx512_mask_pavg_b_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.pavg.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.pavg.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pavg.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pavg_b_256
; CHECK-NOT: call 
; CHECK: vpavgb %ymm
; CHECK: {%k1} 
define <32 x i8>@test_int_x86_avx512_mask_pavg_b_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.pavg.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.pavg.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pavg.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pavg_w_128
; CHECK-NOT: call 
; CHECK: vpavgw %xmm
; CHECK: {%k1} 
define <8 x i16>@test_int_x86_avx512_mask_pavg_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pavg.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pavg.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pavg.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pavg_w_256
; CHECK-NOT: call 
; CHECK: vpavgw %ymm
; CHECK: {%k1} 
define <16 x i16>@test_int_x86_avx512_mask_pavg_w_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pavg.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pavg.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pshuf.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pshuf_b_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpshufb %xmm{{.*}}{%k1} 
define <16 x i8>@test_int_x86_avx512_mask_pshuf_b_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.pshuf.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.pshuf.b.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pshuf.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pshuf_b_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpshufb %ymm{{.*}}{%k1} 
define <32 x i8>@test_int_x86_avx512_mask_pshuf_b_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.pshuf.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.pshuf.b.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pabs.b.128(<16 x i8>, <16 x i8>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_b_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsb{{.*}}{%k1} 
define <16 x i8>@test_int_x86_avx512_mask_pabs_b_128(<16 x i8> %x0, <16 x i8> %x1, i16 %x2) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.pabs.b.128(<16 x i8> %x0, <16 x i8> %x1, i16 %x2)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.pabs.b.128(<16 x i8> %x0, <16 x i8> %x1, i16 -1)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pabs.b.256(<32 x i8>, <32 x i8>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_b_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsb{{.*}}{%k1} 
define <32 x i8>@test_int_x86_avx512_mask_pabs_b_256(<32 x i8> %x0, <32 x i8> %x1, i32 %x2) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.pabs.b.256(<32 x i8> %x0, <32 x i8> %x1, i32 %x2)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.pabs.b.256(<32 x i8> %x0, <32 x i8> %x1, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pabs.w.128(<8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_w_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsw{{.*}}{%k1} 
define <8 x i16>@test_int_x86_avx512_mask_pabs_w_128(<8 x i16> %x0, <8 x i16> %x1, i8 %x2) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pabs.w.128(<8 x i16> %x0, <8 x i16> %x1, i8 %x2)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pabs.w.128(<8 x i16> %x0, <8 x i16> %x1, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pabs.w.256(<16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_w_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsw{{.*}}{%k1} 
define <16 x i16>@test_int_x86_avx512_mask_pabs_w_256(<16 x i16> %x0, <16 x i16> %x1, i16 %x2) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pabs.w.256(<16 x i16> %x0, <16 x i16> %x1, i16 %x2)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pabs.w.256(<16 x i16> %x0, <16 x i16> %x1, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

; CHECK-LABEL: test_x86_mask_blend_b_256
; CHECK: vpblendmb
define <32 x i8> @test_x86_mask_blend_b_256(i32 %a0, <32 x i8> %a1, <32 x i8> %a2) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.blend.b.256(<32 x i8> %a1, <32 x i8> %a2, i32 %a0) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx512.mask.blend.b.256(<32 x i8>, <32 x i8>, i32) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_w_256
define <16 x i16> @test_x86_mask_blend_w_256(i16 %mask, <16 x i16> %a1, <16 x i16> %a2) {
  ; CHECK: vpblendmw
  %res = call <16 x i16> @llvm.x86.avx512.mask.blend.w.256(<16 x i16> %a1, <16 x i16> %a2, i16 %mask) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx512.mask.blend.w.256(<16 x i16>, <16 x i16>, i16) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_b_128
; CHECK: vpblendmb
define <16 x i8> @test_x86_mask_blend_b_128(i16 %a0, <16 x i8> %a1, <16 x i8> %a2) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.blend.b.128(<16 x i8> %a1, <16 x i8> %a2, i16 %a0) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.avx512.mask.blend.b.128(<16 x i8>, <16 x i8>, i16) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_w_128
define <8 x i16> @test_x86_mask_blend_w_128(i8 %mask, <8 x i16> %a1, <8 x i16> %a2) {
  ; CHECK: vpblendmw
  %res = call <8 x i16> @llvm.x86.avx512.mask.blend.w.128(<8 x i16> %a1, <8 x i16> %a2, i8 %mask) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.avx512.mask.blend.w.128(<8 x i16>, <8 x i16>, i8) nounwind readonly

declare <8 x i16> @llvm.x86.avx512.mask.pmulhu.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmulhu_w_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhuw {{.*}}encoding: [0x62
define <8 x i16>@test_int_x86_avx512_mask_pmulhu_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmulhu.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmulhu.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pmulhu.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmulhu_w_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhuw {{.*}}encoding: [0x62
define <16 x i16>@test_int_x86_avx512_mask_pmulhu_w_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmulhu.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmulhu.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pmulh.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmulh_w_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhw {{.*}}encoding: [0x62
define <8 x i16>@test_int_x86_avx512_mask_pmulh_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmulh.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmulh.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pmulh.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)
; CHECK-LABEL: @test_int_x86_avx512_mask_pmulh_w_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhw {{.*}}encoding: [0x62
define <16 x i16>@test_int_x86_avx512_mask_pmulh_w_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmulh.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmulh.w.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)
; CHECK-LABEL: @test_int_x86_avx512_mask_pmulhr_sw_128
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhrsw {{.*}}encoding: [0x62
define <8 x i16>@test_int_x86_avx512_mask_pmulhr_sw_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)
; CHECK-LABEL: @test_int_x86_avx512_mask_pmulhr_sw_256
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhrsw {{.*}}encoding: [0x62
define <16 x i16>@test_int_x86_avx512_mask_pmulhr_sw_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.wb.128(<8 x i16>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmov_wb_128(<8 x i16> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_wb_128:
; CHECK:       vpmovwb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovwb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovwb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.wb.128(<8 x i16> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.wb.128(<8 x i16> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.wb.128(<8 x i16> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.wb.mem.128(i8* %ptr, <8 x i16>, i8)

define void @test_int_x86_avx512_mask_pmov_wb_mem_128(i8* %ptr, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_wb_mem_128:
; CHECK:  vpmovwb %xmm0, (%rdi)
; CHECK:  vpmovwb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.wb.mem.128(i8* %ptr, <8 x i16> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.wb.mem.128(i8* %ptr, <8 x i16> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.128(<8 x i16>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_wb_128(<8 x i16> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_wb_128:
; CHECK:       vpmovswb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovswb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovswb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.128(<8 x i16> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.128(<8 x i16> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.128(<8 x i16> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.wb.mem.128(i8* %ptr, <8 x i16>, i8)

define void @test_int_x86_avx512_mask_pmovs_wb_mem_128(i8* %ptr, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_wb_mem_128:
; CHECK:  vpmovswb %xmm0, (%rdi)
; CHECK:  vpmovswb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.wb.mem.128(i8* %ptr, <8 x i16> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.wb.mem.128(i8* %ptr, <8 x i16> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.128(<8 x i16>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_wb_128(<8 x i16> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_wb_128:
; CHECK:       vpmovuswb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovuswb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovuswb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.128(<8 x i16> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.128(<8 x i16> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.128(<8 x i16> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.wb.mem.128(i8* %ptr, <8 x i16>, i8)

define void @test_int_x86_avx512_mask_pmovus_wb_mem_128(i8* %ptr, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_wb_mem_128:
; CHECK:  vpmovuswb %xmm0, (%rdi)
; CHECK:  vpmovuswb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.wb.mem.128(i8* %ptr, <8 x i16> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.wb.mem.128(i8* %ptr, <8 x i16> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.wb.256(<16 x i16>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_pmov_wb_256(<16 x i16> %x0, <16 x i8> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_wb_256:
; CHECK:       vpmovwb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovwb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovwb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.wb.256(<16 x i16> %x0, <16 x i8> %x1, i16 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.wb.256(<16 x i16> %x0, <16 x i8> %x1, i16 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.wb.256(<16 x i16> %x0, <16 x i8> zeroinitializer, i16 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.wb.mem.256(i8* %ptr, <16 x i16>, i16)

define void @test_int_x86_avx512_mask_pmov_wb_mem_256(i8* %ptr, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_wb_mem_256:
; CHECK:  vpmovwb %ymm0, (%rdi)
; CHECK:  vpmovwb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.wb.mem.256(i8* %ptr, <16 x i16> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmov.wb.mem.256(i8* %ptr, <16 x i16> %x1, i16 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.256(<16 x i16>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_wb_256(<16 x i16> %x0, <16 x i8> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_wb_256:
; CHECK:       vpmovswb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovswb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovswb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.256(<16 x i16> %x0, <16 x i8> %x1, i16 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.256(<16 x i16> %x0, <16 x i8> %x1, i16 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.256(<16 x i16> %x0, <16 x i8> zeroinitializer, i16 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.wb.mem.256(i8* %ptr, <16 x i16>, i16)

define void @test_int_x86_avx512_mask_pmovs_wb_mem_256(i8* %ptr, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_wb_mem_256:
; CHECK:  vpmovswb %ymm0, (%rdi)
; CHECK:  vpmovswb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.wb.mem.256(i8* %ptr, <16 x i16> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmovs.wb.mem.256(i8* %ptr, <16 x i16> %x1, i16 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.256(<16 x i16>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_wb_256(<16 x i16> %x0, <16 x i8> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_wb_256:
; CHECK:       vpmovuswb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovuswb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovuswb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.256(<16 x i16> %x0, <16 x i8> %x1, i16 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.256(<16 x i16> %x0, <16 x i8> %x1, i16 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.256(<16 x i16> %x0, <16 x i8> zeroinitializer, i16 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.wb.mem.256(i8* %ptr, <16 x i16>, i16)

define void @test_int_x86_avx512_mask_pmovus_wb_mem_256(i8* %ptr, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_wb_mem_256:
; CHECK:  vpmovuswb %ymm0, (%rdi)
; CHECK:  vpmovuswb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.wb.mem.256(i8* %ptr, <16 x i16> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmovus.wb.mem.256(i8* %ptr, <16 x i16> %x1, i16 %x2)
    ret void
}

declare <4 x i32> @llvm.x86.avx512.mask.pmaddw.d.128(<8 x i16>, <8 x i16>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmaddw_d_128(<8 x i16> %x0, <8 x i16> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmaddw_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpmaddwd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpmaddwd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmaddw.d.128(<8 x i16> %x0, <8 x i16> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmaddw.d.128(<8 x i16> %x0, <8 x i16> %x1, <4 x i32> %x2, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.pmaddw.d.256(<16 x i16>, <16 x i16>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmaddw_d_256(<16 x i16> %x0, <16 x i16> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmaddw_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpmaddwd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpmaddwd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmaddw.d.256(<16 x i16> %x0, <16 x i16> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmaddw.d.256(<16 x i16> %x0, <16 x i16> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.pmaddubs.w.128(<16 x i8>, <16 x i8>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmaddubs_w_128(<16 x i8> %x0, <16 x i8> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmaddubs_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpmaddubsw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpmaddubsw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmaddubs.w.128(<16 x i8> %x0, <16 x i8> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmaddubs.w.128(<16 x i8> %x0, <16 x i8> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.pmaddubs.w.256(<32 x i8>, <32 x i8>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pmaddubs_w_256(<32 x i8> %x0, <32 x i8> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmaddubs_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmaddubsw %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpmaddubsw %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmaddubs.w.256(<32 x i8> %x0, <32 x i8> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmaddubs.w.256(<32 x i8> %x0, <32 x i8> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.punpckhb.w.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_punpckhb_w_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhb_w_128:
; CHECK:         vpunpckhbw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[8],k1[8],xmm2[9],k1[9],xmm2[10],k1[10],xmm2[11],k1[11],xmm2[12],k1[12],xmm2[13],k1[13],xmm2[14],k1[14],xmm2[15],k1[15]
; CHECK-NEXT:    vpunpckhbw %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7d,0x08,0x68,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[8],xmm1[8],xmm0[9],xmm1[9],xmm0[10],xmm1[10],xmm0[11],xmm1[11],xmm0[12],xmm1[12],xmm0[13],xmm1[13],xmm0[14],xmm1[14],xmm0[15],xmm1[15]
  %res = call <16 x i8> @llvm.x86.avx512.mask.punpckhb.w.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.punpckhb.w.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.punpcklb.w.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_punpcklb_w_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklb_w_128:
; CHECK:         vpunpcklbw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[0],k1[0],xmm2[1],k1[1],xmm2[2],k1[2],xmm2[3],k1[3],xmm2[4],k1[4],xmm2[5],k1[5],xmm2[6],k1[6],xmm2[7],k1[7]
; CHECK-NEXT:    vpunpcklbw %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7d,0x08,0x60,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
  %res = call <16 x i8> @llvm.x86.avx512.mask.punpcklb.w.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 %x3)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.punpcklb.w.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  %res2 = add <16 x i8> %res, %res1
  ret <16 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.punpckhb.w.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <32 x i8>@test_int_x86_avx512_mask_punpckhb_w_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhb_w_256:
; CHECK:         vpunpckhbw %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[8],k1[8],ymm2[9],k1[9],ymm2[10],k1[10],ymm2[11],k1[11],ymm2[12],k1[12],ymm2[13],k1[13],ymm2[14],k1[14],ymm2[15],k1[15],ymm2[24],k1[24],ymm2[25],k1[25],ymm2[26],k1[26],ymm2[27],k1[27],ymm2[28],k1[28],ymm2[29],k1[29],ymm2[30],k1[30],ymm2[31],k1[31]
; CHECK-NEXT:    vpunpckhbw %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7d,0x28,0x68,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[8],ymm1[8],ymm0[9],ymm1[9],ymm0[10],ymm1[10],ymm0[11],ymm1[11],ymm0[12],ymm1[12],ymm0[13],ymm1[13],ymm0[14],ymm1[14],ymm0[15],ymm1[15],ymm0[24],ymm1[24],ymm0[25],ymm1[25],ymm0[26],ymm1[26],ymm0[27],ymm1[27],ymm0[28],ymm1[28],ymm0[29],ymm1[29],ymm0[30],ymm1[30],ymm0[31],ymm1[31]
  %res = call <32 x i8> @llvm.x86.avx512.mask.punpckhb.w.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.punpckhb.w.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.punpcklb.w.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <32 x i8>@test_int_x86_avx512_mask_punpcklb_w_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklb_w_256:
; CHECK:         vpunpcklbw %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[0],k1[0],ymm2[1],k1[1],ymm2[2],k1[2],ymm2[3],k1[3],ymm2[4],k1[4],ymm2[5],k1[5],ymm2[6],k1[6],ymm2[7],k1[7],ymm2[16],k1[16],ymm2[17],k1[17],ymm2[18],k1[18],ymm2[19],k1[19],ymm2[20],k1[20],ymm2[21],k1[21],ymm2[22],k1[22],ymm2[23],k1[23]
; CHECK-NEXT:    vpunpcklbw %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7d,0x28,0x60,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[4],ymm1[4],ymm0[5],ymm1[5],ymm0[6],ymm1[6],ymm0[7],ymm1[7],ymm0[16],ymm1[16],ymm0[17],ymm1[17],ymm0[18],ymm1[18],ymm0[19],ymm1[19],ymm0[20],ymm1[20],ymm0[21],ymm1[21],ymm0[22],ymm1[22],ymm0[23],ymm1[23]
  %res = call <32 x i8> @llvm.x86.avx512.mask.punpcklb.w.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 %x3)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.punpcklb.w.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  %res2 = add <32 x i8> %res, %res1
  ret <32 x i8> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.punpcklw.d.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_punpcklw_d_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklw_d_128:
; CHECK:         vpunpcklwd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[0],k1[0],xmm2[1],k1[1],xmm2[2],k1[2],xmm2[3],k1[3]
; CHECK-NEXT:    vpunpcklwd %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7d,0x08,0x61,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
  %res = call <8 x i16> @llvm.x86.avx512.mask.punpcklw.d.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.punpcklw.d.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <8 x i16> @llvm.x86.avx512.mask.punpckhw.d.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_punpckhw_d_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhw_d_128:
; CHECK:         vpunpckhwd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[4],k1[4],xmm2[5],k1[5],xmm2[6],k1[6],xmm2[7],k1[7]
; CHECK-NEXT:    vpunpckhwd %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7d,0x08,0x69,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
  %res = call <8 x i16> @llvm.x86.avx512.mask.punpckhw.d.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.punpckhw.d.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = add <8 x i16> %res, %res1
  ret <8 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.punpcklw.d.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_punpcklw_d_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklw_d_256:
; CHECK:         vpunpcklwd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[0],k1[0],ymm2[1],k1[1],ymm2[2],k1[2],ymm2[3],k1[3],ymm2[8],k1[8],ymm2[9],k1[9],ymm2[10],k1[10],ymm2[11],k1[11]
; CHECK-NEXT:    vpunpcklwd %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7d,0x28,0x61,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[8],ymm1[8],ymm0[9],ymm1[9],ymm0[10],ymm1[10],ymm0[11],ymm1[11]
  %res = call <16 x i16> @llvm.x86.avx512.mask.punpcklw.d.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.punpcklw.d.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i16> @llvm.x86.avx512.mask.punpckhw.d.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_punpckhw_d_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhw_d_256:
; CHECK:         vpunpckhwd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[4],k1[4],ymm2[5],k1[5],ymm2[6],k1[6],ymm2[7],k1[7],ymm2[12],k1[12],ymm2[13],k1[13],ymm2[14],k1[14],ymm2[15],k1[15]
; CHECK-NEXT:    vpunpckhwd %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7d,0x28,0x69,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[4],ymm1[4],ymm0[5],ymm1[5],ymm0[6],ymm1[6],ymm0[7],ymm1[7],ymm0[12],ymm1[12],ymm0[13],ymm1[13],ymm0[14],ymm1[14],ymm0[15],ymm1[15]
  %res = call <16 x i16> @llvm.x86.avx512.mask.punpckhw.d.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.punpckhw.d.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = add <16 x i16> %res, %res1
  ret <16 x i16> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.palignr.128(<16 x i8>, <16 x i8>, i32, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_palignr_128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x3, i16 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_palignr_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpalignr $2, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpalignr $2, %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpalignr $2, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddb %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <16 x i8> @llvm.x86.avx512.mask.palignr.128(<16 x i8> %x0, <16 x i8> %x1, i32 2, <16 x i8> %x3, i16 %x4)
  %res1 = call <16 x i8> @llvm.x86.avx512.mask.palignr.128(<16 x i8> %x0, <16 x i8> %x1, i32 2, <16 x i8> zeroinitializer, i16 %x4)
  %res2 = call <16 x i8> @llvm.x86.avx512.mask.palignr.128(<16 x i8> %x0, <16 x i8> %x1, i32 2, <16 x i8> %x3, i16 -1)
  %res3 = add <16 x i8> %res, %res1
  %res4 = add <16 x i8> %res3, %res2
  ret <16 x i8> %res4
}

declare <32 x i8> @llvm.x86.avx512.mask.palignr.256(<32 x i8>, <32 x i8>, i32, <32 x i8>, i32)

define <32 x i8>@test_int_x86_avx512_mask_palignr_256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x3, i32 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_palignr_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpalignr $2, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpalignr $2, %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpalignr $2, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddb %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddb %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <32 x i8> @llvm.x86.avx512.mask.palignr.256(<32 x i8> %x0, <32 x i8> %x1, i32 2, <32 x i8> %x3, i32 %x4)
  %res1 = call <32 x i8> @llvm.x86.avx512.mask.palignr.256(<32 x i8> %x0, <32 x i8> %x1, i32 2, <32 x i8> zeroinitializer, i32 %x4)
  %res2 = call <32 x i8> @llvm.x86.avx512.mask.palignr.256(<32 x i8> %x0, <32 x i8> %x1, i32 2, <32 x i8> %x3, i32 -1)
  %res3 = add <32 x i8> %res, %res1
  %res4 = add <32 x i8> %res3, %res2
  ret <32 x i8> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.dbpsadbw.128(<16 x i8>, <16 x i8>, i32, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_dbpsadbw_128(<16 x i8> %x0, <16 x i8> %x1, <8 x i16> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_dbpsadbw_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vdbpsadbw $2, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vdbpsadbw $2, %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vdbpsadbw $2, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.dbpsadbw.128(<16 x i8> %x0, <16 x i8> %x1, i32 2, <8 x i16> %x3, i8 %x4)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.dbpsadbw.128(<16 x i8> %x0, <16 x i8> %x1, i32 2, <8 x i16> zeroinitializer, i8 %x4)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.dbpsadbw.128(<16 x i8> %x0, <16 x i8> %x1, i32 2, <8 x i16> %x3, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res2, %res3
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.dbpsadbw.256(<32 x i8>, <32 x i8>, i32, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_dbpsadbw_256(<32 x i8> %x0, <32 x i8> %x1, <16 x i16> %x3, i16 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_dbpsadbw_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vdbpsadbw $2, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vdbpsadbw $2, %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vdbpsadbw $2, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.dbpsadbw.256(<32 x i8> %x0, <32 x i8> %x1, i32 2, <16 x i16> %x3, i16 %x4)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.dbpsadbw.256(<32 x i8> %x0, <32 x i8> %x1, i32 2, <16 x i16> zeroinitializer, i16 %x4)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.dbpsadbw.256(<32 x i8> %x0, <32 x i8> %x1, i32 2, <16 x i16> %x3, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <32 x i8> @llvm.x86.avx512.pbroadcastb.256(<16 x i8>, <32 x i8>, i32)

define <32 x i8>@test_int_x86_avx512_pbroadcastb_256(<16 x i8> %x0, <32 x i8> %x1, i32 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastb_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpbroadcastb %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpbroadcastb %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastb %xmm0, %ymm0
; CHECK-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddb %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <32 x i8> @llvm.x86.avx512.pbroadcastb.256(<16 x i8> %x0, <32 x i8> %x1, i32 -1)
  %res1 = call <32 x i8> @llvm.x86.avx512.pbroadcastb.256(<16 x i8> %x0, <32 x i8> %x1, i32 %mask)
  %res2 = call <32 x i8> @llvm.x86.avx512.pbroadcastb.256(<16 x i8> %x0, <32 x i8> zeroinitializer, i32 %mask)
  %res3 = add <32 x i8> %res, %res1
  %res4 = add <32 x i8> %res2, %res3
  ret <32 x i8> %res4
}

declare <16 x i8> @llvm.x86.avx512.pbroadcastb.128(<16 x i8>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_pbroadcastb_128(<16 x i8> %x0, <16 x i8> %x1, i16 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastb_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpbroadcastb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpbroadcastb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastb %xmm0, %xmm0
; CHECK-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddb %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <16 x i8> @llvm.x86.avx512.pbroadcastb.128(<16 x i8> %x0, <16 x i8> %x1, i16 -1)
  %res1 = call <16 x i8> @llvm.x86.avx512.pbroadcastb.128(<16 x i8> %x0, <16 x i8> %x1, i16 %mask)
  %res2 = call <16 x i8> @llvm.x86.avx512.pbroadcastb.128(<16 x i8> %x0, <16 x i8> zeroinitializer, i16 %mask)
  %res3 = add <16 x i8> %res, %res1
  %res4 = add <16 x i8> %res2, %res3
  ret <16 x i8> %res4
}

declare <16 x i16> @llvm.x86.avx512.pbroadcastw.256(<8 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_pbroadcastw_256(<8 x i16> %x0, <16 x i16> %x1, i16 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastw_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpbroadcastw %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpbroadcastw %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastw %xmm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.pbroadcastw.256(<8 x i16> %x0, <16 x i16> %x1, i16 -1)
  %res1 = call <16 x i16> @llvm.x86.avx512.pbroadcastw.256(<8 x i16> %x0, <16 x i16> %x1, i16 %mask)
  %res2 = call <16 x i16> @llvm.x86.avx512.pbroadcastw.256(<8 x i16> %x0, <16 x i16> zeroinitializer, i16 %mask)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res2, %res3
  ret <16 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.pbroadcastw.128(<8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_pbroadcastw_128(<8 x i16> %x0, <8 x i16> %x1, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastw_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpbroadcastw %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpbroadcastw %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastw %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.pbroadcastw.128(<8 x i16> %x0, <8 x i16> %x1, i8 -1)
  %res1 = call <8 x i16> @llvm.x86.avx512.pbroadcastw.128(<8 x i16> %x0, <8 x i16> %x1, i8 %mask)
  %res2 = call <8 x i16> @llvm.x86.avx512.pbroadcastw.128(<8 x i16> %x0, <8 x i16> zeroinitializer, i8 %mask)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res2, %res3
  ret <8 x i16> %res4
}

declare <64 x i8> @llvm.x86.avx512.pbroadcastb.512(<16 x i8>, <64 x i8>, i64)

define <64 x i8>@test_int_x86_avx512_pbroadcastb_512(<16 x i8> %x0, <64 x i8> %x1, i64 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastb_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovq %rdi, %k1 ## encoding: [0xc4,0xe1,0xfb,0x92,0xcf]
; CHECK-NEXT:    vpbroadcastb %xmm0, %zmm1 {%k1} ## encoding: [0x62,0xf2,0x7d,0x49,0x78,0xc8]
; CHECK-NEXT:    vpbroadcastb %xmm0, %zmm2 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0xc9,0x78,0xd0]
; CHECK-NEXT:    vpbroadcastb %xmm0, %zmm0 ## encoding: [0x62,0xf2,0x7d,0x48,0x78,0xc0]
; CHECK-NEXT:    vpaddb %zmm1, %zmm0, %zmm0 ## encoding: [0x62,0xf1,0x7d,0x48,0xfc,0xc1]
; CHECK-NEXT:    vpaddb %zmm0, %zmm2, %zmm0 ## encoding: [0x62,0xf1,0x6d,0x48,0xfc,0xc0]
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call <64 x i8> @llvm.x86.avx512.pbroadcastb.512(<16 x i8> %x0, <64 x i8> %x1, i64 -1)
  %res1 = call <64 x i8> @llvm.x86.avx512.pbroadcastb.512(<16 x i8> %x0, <64 x i8> %x1, i64 %mask)
  %res2 = call <64 x i8> @llvm.x86.avx512.pbroadcastb.512(<16 x i8> %x0, <64 x i8> zeroinitializer, i64 %mask)
  %res3 = add <64 x i8> %res, %res1
  %res4 = add <64 x i8> %res2, %res3
  ret <64 x i8> %res4
}

declare <32 x i16> @llvm.x86.avx512.pbroadcastw.512(<8 x i16>, <32 x i16>, i32)

define <32 x i16>@test_int_x86_avx512_pbroadcastw_512(<8 x i16> %x0, <32 x i16> %x1, i32 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastw_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k1 ## encoding: [0xc5,0xfb,0x92,0xcf]
; CHECK-NEXT:    vpbroadcastw %xmm0, %zmm1 {%k1} ## encoding: [0x62,0xf2,0x7d,0x49,0x79,0xc8]
; CHECK-NEXT:    vpbroadcastw %xmm0, %zmm2 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0xc9,0x79,0xd0]
; CHECK-NEXT:    vpbroadcastw %xmm0, %zmm0 ## encoding: [0x62,0xf2,0x7d,0x48,0x79,0xc0]
; CHECK-NEXT:    vpaddw %zmm1, %zmm0, %zmm0 ## encoding: [0x62,0xf1,0x7d,0x48,0xfd,0xc1]
; CHECK-NEXT:    vpaddw %zmm0, %zmm2, %zmm0 ## encoding: [0x62,0xf1,0x6d,0x48,0xfd,0xc0]
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call <32 x i16> @llvm.x86.avx512.pbroadcastw.512(<8 x i16> %x0, <32 x i16> %x1, i32 -1)
  %res1 = call <32 x i16> @llvm.x86.avx512.pbroadcastw.512(<8 x i16> %x0, <32 x i16> %x1, i32 %mask)
  %res2 = call <32 x i16> @llvm.x86.avx512.pbroadcastw.512(<8 x i16> %x0, <32 x i16> zeroinitializer, i32 %mask)
  %res3 = add <32 x i16> %res, %res1
  %res4 = add <32 x i16> %res2, %res3
  ret <32 x i16> %res4
}

declare i16 @llvm.x86.avx512.cvtb2mask.128(<16 x i8>)

define i16@test_int_x86_avx512_cvtb2mask_128(<16 x i8> %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtb2mask_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovb2m %xmm0, %k0
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    retq
    %res = call i16 @llvm.x86.avx512.cvtb2mask.128(<16 x i8> %x0)
    ret i16 %res
}

declare i32 @llvm.x86.avx512.cvtb2mask.256(<32 x i8>)

define i32@test_int_x86_avx512_cvtb2mask_256(<32 x i8> %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtb2mask_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovb2m %ymm0, %k0
; CHECK-NEXT:    kmovd %k0, %eax
; CHECK-NEXT:    retq
    %res = call i32 @llvm.x86.avx512.cvtb2mask.256(<32 x i8> %x0)
    ret i32 %res
}

declare i8 @llvm.x86.avx512.cvtw2mask.128(<8 x i16>)

define i8@test_int_x86_avx512_cvtw2mask_128(<8 x i16> %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtw2mask_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovw2m %xmm0, %k0
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    retq
    %res = call i8 @llvm.x86.avx512.cvtw2mask.128(<8 x i16> %x0)
    ret i8 %res
}

declare i16 @llvm.x86.avx512.cvtw2mask.256(<16 x i16>)

define i16@test_int_x86_avx512_cvtw2mask_256(<16 x i16> %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtw2mask_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovw2m %ymm0, %k0
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    retq
    %res = call i16 @llvm.x86.avx512.cvtw2mask.256(<16 x i16> %x0)
    ret i16 %res
}

declare <16 x i8> @llvm.x86.avx512.cvtmask2b.128(i16)

define <16 x i8>@test_int_x86_avx512_cvtmask2b_128(i16 %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtmask2b_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k0
; CHECK-NEXT:    vpmovm2b %k0, %xmm0
; CHECK-NEXT:    retq
  %res = call <16 x i8> @llvm.x86.avx512.cvtmask2b.128(i16 %x0)
  ret <16 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.cvtmask2b.256(i32)

define <32 x i8>@test_int_x86_avx512_cvtmask2b_256(i32 %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtmask2b_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k0
; CHECK-NEXT:    vpmovm2b %k0, %ymm0
; CHECK-NEXT:    retq
  %res = call <32 x i8> @llvm.x86.avx512.cvtmask2b.256(i32 %x0)
  ret <32 x i8> %res
}

declare <8 x i16> @llvm.x86.avx512.cvtmask2w.128(i8)

define <8 x i16>@test_int_x86_avx512_cvtmask2w_128(i8 %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtmask2w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k0
; CHECK-NEXT:    vpmovm2w %k0, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.cvtmask2w.128(i8 %x0)
  ret <8 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.cvtmask2w.256(i16)

define <16 x i16>@test_int_x86_avx512_cvtmask2w_256(i16 %x0) {
; CHECK-LABEL: test_int_x86_avx512_cvtmask2w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k0
; CHECK-NEXT:    vpmovm2w %k0, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.cvtmask2w.256(i16 %x0)
  ret <16 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.psrl.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psrl_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsrlw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsrlw %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsrlw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    vpaddw %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psrl.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psrl.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psrl.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> zeroinitializer, i8 %x3)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res2, %res3
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psrl.w.256(<16 x i16>, <8 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psrl_w_256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlw %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsrlw %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsrlw %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    vpaddw %ymm3, %ymm0, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psrl.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psrl.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psrl.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> zeroinitializer, i16 %x3)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.psrl.wi.128(<8 x i16>, i8, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psrl_wi_128(<8 x i16> %x0, i8 %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_wi_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %sil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsrlw $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsrlw $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsrlw $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vpaddw %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psrl.wi.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psrl.wi.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 -1)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psrl.wi.128(<8 x i16> %x0, i8 3, <8 x i16> zeroinitializer, i8 %x3)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res2, %res3
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psrl.wi.256(<16 x i16>, i8, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psrl_wi_256(<16 x i16> %x0, i8 %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_wi_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrlw $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsrlw $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsrlw $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    vpaddw %ymm2, %ymm0, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psrl.wi.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psrl.wi.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 -1)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psrl.wi.256(<16 x i16> %x0, i8 3, <16 x i16> zeroinitializer, i16 %x3)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psrlv16.hi(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psrlv16_hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrlv16_hi:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlvw %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsrlvw %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsrlvw %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psrlv16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psrlv16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psrlv16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.psrlv8.hi(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psrlv8_hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrlv8_hi:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsrlvw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsrlvw %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsrlvw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psrlv8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psrlv8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psrlv8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.psra.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psra_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsraw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsraw %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsraw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psra.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psra.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psra.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.psra.wi.128(<8 x i16>, i8, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psra_wi_128(<8 x i16> %x0, i8 %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_wi_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %sil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsraw $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsraw $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsraw $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psra.wi.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psra.wi.128(<8 x i16> %x0, i8 3, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psra.wi.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psra.w.256(<16 x i16>, <8 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psra_w_256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsraw %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsraw %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsraw %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psra.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psra.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psra.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psra.wi.256(<16 x i16>, i8, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psra_wi_256(<16 x i16> %x0, i8 %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_wi_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsraw $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsraw $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsraw $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psra.wi.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psra.wi.256(<16 x i16> %x0, i8 3, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psra.wi.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.pshuf.d.128(<4 x i32>, i16, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pshuf_d_128(<4 x i32> %x0, i16 %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pshuf_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %sil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpshufd $3, %xmm0, %xmm1 {%k1} 
; CHECK-NEXT:    vpshufd $3, %xmm0, %xmm2 {%k1} {z} 
; CHECK-NEXT:    vpshufd $3, %xmm0, %xmm0 
; CHECK-NEXT:    ## xmm0 = xmm0[3,0,0,0]
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1 
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0 
; CHECK-NEXT:    retq 
	%res = call <4 x i32> @llvm.x86.avx512.mask.pshuf.d.128(<4 x i32> %x0, i16 3, <4 x i32> %x2, i8 %x3)
	%res1 = call <4 x i32> @llvm.x86.avx512.mask.pshuf.d.128(<4 x i32> %x0, i16 3, <4 x i32> zeroinitializer, i8 %x3)
	%res2 = call <4 x i32> @llvm.x86.avx512.mask.pshuf.d.128(<4 x i32> %x0, i16 3, <4 x i32> %x2, i8 -1)
	%res3 = add <4 x i32> %res, %res1
	%res4 = add <4 x i32> %res3, %res2
	ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.pshuf.d.256(<8 x i32>, i16, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pshuf_d_256(<8 x i32> %x0, i16 %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pshuf_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %sil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpshufd $3, %ymm0, %ymm1 {%k1} 
; CHECK-NEXT:    vpshufd $3, %ymm0, %ymm2 {%k1} {z} 
; CHECK-NEXT:    vpshufd $3, %ymm0, %ymm0 
; CHECK-NEXT:    ## ymm0 = ymm0[3,0,0,0,7,4,4,4]
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1 
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0 
; CHECK-NEXT:    retq 
	%res = call <8 x i32> @llvm.x86.avx512.mask.pshuf.d.256(<8 x i32> %x0, i16 3, <8 x i32> %x2, i8 %x3)
	%res1 = call <8 x i32> @llvm.x86.avx512.mask.pshuf.d.256(<8 x i32> %x0, i16 3, <8 x i32> zeroinitializer, i8 %x3)
	%res2 = call <8 x i32> @llvm.x86.avx512.mask.pshuf.d.256(<8 x i32> %x0, i16 3, <8 x i32> %x2, i8 -1)
	%res3 = add <8 x i32> %res, %res1
	%res4 = add <8 x i32> %res3, %res2
	ret <8 x i32> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.pshufh.w.128(<8 x i16>, i8, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pshufh_w_128(<8 x i16> %x0, i8 %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pshufh_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %sil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpshufhw $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpshufhw $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpshufhw $3, %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[0,1,2,3,7,4,4,4]
; CHECK-NEXT:    vpaddw %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.pshufh.w.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pshufh.w.128(<8 x i16> %x0, i8 3, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.pshufh.w.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.pshufh.w.256(<16 x i16>, i8, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pshufh_w_256(<16 x i16> %x0, i8 %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pshufh_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpshufhw $3, %ymm0, %ymm1 {%k1} 
; CHECK-NEXT:    vpshufhw $3, %ymm0, %ymm2 {%k1} {z} 
; CHECK-NEXT:    vpshufhw $3, %ymm0, %ymm0 
; CHECK-NEXT:    ## ymm0 = ymm0[0,1,2,3,7,4,4,4,8,9,10,11,15,12,12,12]
; CHECK-NEXT:    vpaddw %ymm2, %ymm1, %ymm1 
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0 
; CHECK-NEXT:    retq 
  %res = call <16 x i16> @llvm.x86.avx512.mask.pshufh.w.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pshufh.w.256(<16 x i16> %x0, i8 3, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.pshufh.w.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.pshufl.w.128(<8 x i16>, i8, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pshufl_w_128(<8 x i16> %x0, i8 %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pshufl_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %sil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpshuflw $3, %xmm0, %xmm1 {%k1} 
; CHECK-NEXT:    vpshuflw $3, %xmm0, %xmm2 {%k1} {z} 
; CHECK-NEXT:    vpshuflw $3, %xmm0, %xmm0 
; CHECK-NEXT:    ## xmm0 = xmm0[3,0,0,0,4,5,6,7]
; CHECK-NEXT:    vpaddw %xmm2, %xmm1, %xmm1 
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <8 x i16> @llvm.x86.avx512.mask.pshufl.w.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pshufl.w.128(<8 x i16> %x0, i8 3, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.pshufl.w.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.pshufl.w.256(<16 x i16>, i8, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pshufl_w_256(<16 x i16> %x0, i8 %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pshufl_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1 
; CHECK-NEXT:    vpshuflw $3, %ymm0, %ymm1 {%k1} 
; CHECK-NEXT:    vpshuflw $3, %ymm0, %ymm2 {%k1} {z} 
; CHECK-NEXT:    vpshuflw $3, %ymm0, %ymm0 
; CHECK-NEXT:    ## ymm0 = ymm0[3,0,0,0,4,5,6,7,11,8,8,8,12,13,14,15]
; CHECK-NEXT:    vpaddw %ymm2, %ymm1, %ymm1 
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0 
; CHECK-NEXT:    retq 
  %res = call <16 x i16> @llvm.x86.avx512.mask.pshufl.w.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pshufl.w.256(<16 x i16> %x0, i8 3, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.pshufl.w.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psrav16.hi(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psrav16_hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrav16_hi:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsravw %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsravw %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsravw %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psrav16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psrav16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psrav16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.psrav8.hi(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psrav8_hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrav8_hi:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsravw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsravw %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsravw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psrav8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psrav8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psrav8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}


declare <8 x i16> @llvm.x86.avx512.mask.psll.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psll_w_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsllw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsllw %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsllw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psll.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psll.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psll.w.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psll.w.256(<16 x i16>, <8 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psll_w_256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsllw %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsllw %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsllw %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psll.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psll.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psll.w.256(<16 x i16> %x0, <8 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.psll.wi.128(<8 x i16>, i8, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psll_wi_128(<8 x i16> %x0, i8 %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_wi_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %sil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsllw $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsllw $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsllw $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psll.wi.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psll.wi.128(<8 x i16> %x0, i8 3, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psll.wi.128(<8 x i16> %x0, i8 3, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psll.wi.256(<16 x i16>, i8, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psll_wi_256(<16 x i16> %x0, i8 %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_wi_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsllw $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsllw $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsllw $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psll.wi.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psll.wi.256(<16 x i16> %x0, i8 3, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psll.wi.256(<16 x i16> %x0, i8 3, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.psllv16.hi(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_psllv16_hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psllv16_hi:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsllvw %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsllvw %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsllvw %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddw %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.x86.avx512.mask.psllv16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.psllv16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.psllv16.hi(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.psllv8.hi(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_psllv8_hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psllv8_hi:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpsllvw %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsllvw %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsllvw %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddw %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.x86.avx512.mask.psllv8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.psllv8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.psllv8.hi(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovzxb.w.128(<16 x i8>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovzxb_w_128(<16 x i8> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxb_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpmovzxbw %xmm0, %xmm1 {%k1} 
; CHECK-NEXT:    vpmovzxbw %xmm0, %xmm2 {%k1} {z} 
; CHECK-NEXT:    vpmovzxbw %xmm0, %xmm0 
; CHECK-NEXT:    vpaddw %xmm2, %xmm1, %xmm1 
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmovzxb.w.128(<16 x i8> %x0, <8 x i16> %x1, i8 %x2)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovzxb.w.128(<16 x i8> %x0, <8 x i16> zeroinitializer, i8 %x2)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovzxb.w.128(<16 x i8> %x0, <8 x i16> %x1, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.pmovzxb.w.256(<16 x i8>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pmovzxb_w_256(<16 x i8> %x0, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxb_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vpmovzxbw %xmm0, %ymm1 {%k1} 
; CHECK-NEXT:    vpmovzxbw %xmm0, %ymm2 {%k1} {z} 
; CHECK-NEXT:    vpmovzxbw %xmm0, %ymm0 
; CHECK-NEXT:    vpaddw %ymm2, %ymm1, %ymm1 
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0 
; CHECK-NEXT:    retq 
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmovzxb.w.256(<16 x i8> %x0, <16 x i16> %x1, i16 %x2)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmovzxb.w.256(<16 x i8> %x0, <16 x i16> zeroinitializer, i16 %x2)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.pmovzxb.w.256(<16 x i8> %x0, <16 x i16> %x1, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}


declare <8 x i16> @llvm.x86.avx512.mask.pmovsxb.w.128(<16 x i8>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovsxb_w_128(<16 x i8> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxb_w_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpmovsxbw %xmm0, %xmm1 {%k1} 
; CHECK-NEXT:    vpmovsxbw %xmm0, %xmm2 {%k1} {z} 
; CHECK-NEXT:    vpmovsxbw %xmm0, %xmm0 
; CHECK-NEXT:    vpaddw %xmm2, %xmm1, %xmm1 
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmovsxb.w.128(<16 x i8> %x0, <8 x i16> %x1, i8 %x2)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovsxb.w.128(<16 x i8> %x0, <8 x i16> zeroinitializer, i8 %x2)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovsxb.w.128(<16 x i8> %x0, <8 x i16> %x1, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.pmovsxb.w.256(<16 x i8>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pmovsxb_w_256(<16 x i8> %x0, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxb_w_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vpmovsxbw %xmm0, %ymm1 {%k1} 
; CHECK-NEXT:    vpmovsxbw %xmm0, %ymm2 {%k1} {z} 
; CHECK-NEXT:    vpmovsxbw %xmm0, %ymm0 
; CHECK-NEXT:    vpaddw %ymm2, %ymm1, %ymm1 
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0 
; CHECK-NEXT:    retq 
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmovsxb.w.256(<16 x i8> %x0, <16 x i16> %x1, i16 %x2)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmovsxb.w.256(<16 x i8> %x0, <16 x i16> zeroinitializer, i16 %x2)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.pmovsxb.w.256(<16 x i8> %x0, <16 x i16> %x1, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.pmovsxd.q.128(<4 x i32>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_pmovsxd_q_128(<4 x i32> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxd_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpmovsxdq %xmm0, %xmm1 {%k1} 
; CHECK-NEXT:    vpmovsxdq %xmm0, %xmm2 {%k1} {z} 
; CHECK-NEXT:    vpmovsxdq %xmm0, %xmm0 
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1 
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmovsxd.q.128(<4 x i32> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmovsxd.q.128(<4 x i32> %x0, <2 x i64> zeroinitializer, i8 %x2)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.pmovsxd.q.128(<4 x i32> %x0, <2 x i64> %x1, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.pmovsxd.q.256(<4 x i32>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_pmovsxd_q_256(<4 x i32> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxd_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpmovsxdq %xmm0, %ymm1 {%k1} 
; CHECK-NEXT:    vpmovsxdq %xmm0, %ymm2 {%k1} {z} 
; CHECK-NEXT:    vpmovsxdq %xmm0, %ymm0 
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1 
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0 
; CHECK-NEXT:    retq 
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmovsxd.q.256(<4 x i32> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmovsxd.q.256(<4 x i32> %x0, <4 x i64> zeroinitializer, i8 %x2)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.pmovsxd.q.256(<4 x i32> %x0, <4 x i64> %x1, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <8 x i16> @llvm.x86.avx512.mask.permvar.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_permvar_hi_128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_permvar_hi_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax 
; CHECK-NEXT:    kmovw %eax, %k1 
; CHECK-NEXT:    vpermw %xmm1, %xmm0, %xmm2 {%k1} 
; CHECK-NEXT:    vpermw %xmm1, %xmm0, %xmm3 {%k1} {z} 
; CHECK-NEXT:    vpermw %xmm1, %xmm0, %xmm0 
; CHECK-NEXT:    vpaddw %xmm3, %xmm2, %xmm1 
; CHECK-NEXT:    vpaddw %xmm0, %xmm1, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <8 x i16> @llvm.x86.avx512.mask.permvar.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 %x3)
  %res1 = call <8 x i16> @llvm.x86.avx512.mask.permvar.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> zeroinitializer, i8 %x3)
  %res2 = call <8 x i16> @llvm.x86.avx512.mask.permvar.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  %res3 = add <8 x i16> %res, %res1
  %res4 = add <8 x i16> %res3, %res2
  ret <8 x i16> %res4
}

declare <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_permvar_hi_256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_permvar_hi_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vpermw %ymm1, %ymm0, %ymm2 {%k1} 
; CHECK-NEXT:    vpermw %ymm1, %ymm0, %ymm3 {%k1} {z} 
; CHECK-NEXT:    vpermw %ymm1, %ymm0, %ymm0 
; CHECK-NEXT:    vpaddw %ymm3, %ymm2, %ymm1 
; CHECK-NEXT:    vpaddw %ymm0, %ymm1, %ymm0 
; CHECK-NEXT:    retq 
  %res = call <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 %x3)
  %res1 = call <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> zeroinitializer, i16 %x3)
  %res2 = call <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  %res3 = add <16 x i16> %res, %res1
  %res4 = add <16 x i16> %res3, %res2
  ret <16 x i16> %res4
}
