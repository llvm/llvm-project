; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw --show-mc-encoding| FileCheck %s

define i64 @test_pcmpeq_b(<64 x i8> %a, <64 x i8> %b) {
; CHECK-LABEL: test_pcmpeq_b
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpeq.b.512(<64 x i8> %a, <64 x i8> %b, i64 -1)
  ret i64 %res
}

define i64 @test_mask_pcmpeq_b(<64 x i8> %a, <64 x i8> %b, i64 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_b
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpeq.b.512(<64 x i8> %a, <64 x i8> %b, i64 %mask)
  ret i64 %res
}

declare i64 @llvm.x86.avx512.mask.pcmpeq.b.512(<64 x i8>, <64 x i8>, i64)

define i32 @test_pcmpeq_w(<32 x i16> %a, <32 x i16> %b) {
; CHECK-LABEL: test_pcmpeq_w
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.w.512(<32 x i16> %a, <32 x i16> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpeq_w(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_w
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.w.512(<32 x i16> %a, <32 x i16> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpeq.w.512(<32 x i16>, <32 x i16>, i32)

define i64 @test_pcmpgt_b(<64 x i8> %a, <64 x i8> %b) {
; CHECK-LABEL: test_pcmpgt_b
; CHECK: vpcmpgtb %zmm1, %zmm0, %k0 ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpgt.b.512(<64 x i8> %a, <64 x i8> %b, i64 -1)
  ret i64 %res
}

define i64 @test_mask_pcmpgt_b(<64 x i8> %a, <64 x i8> %b, i64 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_b
; CHECK: vpcmpgtb %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpgt.b.512(<64 x i8> %a, <64 x i8> %b, i64 %mask)
  ret i64 %res
}

declare i64 @llvm.x86.avx512.mask.pcmpgt.b.512(<64 x i8>, <64 x i8>, i64)

define i32 @test_pcmpgt_w(<32 x i16> %a, <32 x i16> %b) {
; CHECK-LABEL: test_pcmpgt_w
; CHECK: vpcmpgtw %zmm1, %zmm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.w.512(<32 x i16> %a, <32 x i16> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpgt_w(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_w
; CHECK: vpcmpgtw %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.w.512(<32 x i16> %a, <32 x i16> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpgt.w.512(<32 x i16>, <32 x i16>, i32)

define <8 x i64> @test_cmp_b_512(<64 x i8> %a0, <64 x i8> %a1) {
; CHECK_LABEL: test_cmp_b_512
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 ##
  %res0 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 -1)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltb %zmm1, %zmm0, %k0 ##
  %res1 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 -1)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleb %zmm1, %zmm0, %k0 ##
  %res2 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 -1)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordb %zmm1, %zmm0, %k0 ##
  %res3 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 -1)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpneqb %zmm1, %zmm0, %k0 ##
  %res4 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 -1)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltb %zmm1, %zmm0, %k0 ##
  %res5 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 -1)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleb %zmm1, %zmm0, %k0 ##
  %res6 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 -1)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordb %zmm1, %zmm0, %k0 ##
  %res7 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 -1)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

define <8 x i64> @test_mask_cmp_b_512(<64 x i8> %a0, <64 x i8> %a1, i64 %mask) {
; CHECK_LABEL: test_mask_cmp_b_512
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 %mask)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltb %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 %mask)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleb %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 %mask)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordb %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 %mask)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpneqb %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 %mask)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltb %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 %mask)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleb %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 %mask)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordb %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 %mask)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

declare i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8>, <64 x i8>, i32, i64) nounwind readnone

define <8 x i64> @test_ucmp_b_512(<64 x i8> %a0, <64 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_512
; CHECK: vpcmpequb %zmm1, %zmm0, %k0 ##
  %res0 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 -1)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltub %zmm1, %zmm0, %k0 ##
  %res1 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 -1)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleub %zmm1, %zmm0, %k0 ##
  %res2 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 -1)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordub %zmm1, %zmm0, %k0 ##
  %res3 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 -1)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpnequb %zmm1, %zmm0, %k0 ##
  %res4 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 -1)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltub %zmm1, %zmm0, %k0 ##
  %res5 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 -1)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleub %zmm1, %zmm0, %k0 ##
  %res6 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 -1)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordub %zmm1, %zmm0, %k0 ##
  %res7 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 -1)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

define <8 x i64> @test_mask_x86_avx512_ucmp_b_512(<64 x i8> %a0, <64 x i8> %a1, i64 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_512
; CHECK: vpcmpequb %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 %mask)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltub %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 %mask)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleub %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 %mask)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordub %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 %mask)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpnequb %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 %mask)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltub %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 %mask)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleub %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 %mask)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordub %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 %mask)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

declare i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8>, <64 x i8>, i32, i64) nounwind readnone

define <8 x i32> @test_cmp_w_512(<32 x i16> %a0, <32 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_512
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltw %zmm1, %zmm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmplew %zmm1, %zmm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordw %zmm1, %zmm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqw %zmm1, %zmm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltw %zmm1, %zmm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnlew %zmm1, %zmm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordw %zmm1, %zmm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_cmp_w_512(<32 x i16> %a0, <32 x i16> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_cmp_w_512
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltw %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmplew %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordw %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqw %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltw %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnlew %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordw %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16>, <32 x i16>, i32, i32) nounwind readnone

define <8 x i32> @test_ucmp_w_512(<32 x i16> %a0, <32 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_512
; CHECK: vpcmpequw %zmm1, %zmm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltuw %zmm1, %zmm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleuw %zmm1, %zmm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunorduw %zmm1, %zmm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequw %zmm1, %zmm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltuw %zmm1, %zmm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleuw %zmm1, %zmm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmporduw %zmm1, %zmm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_ucmp_w_512(<32 x i16> %a0, <32 x i16> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_512
; CHECK: vpcmpequw %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltuw %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleuw %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunorduw %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequw %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltuw %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleuw %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmporduw %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16>, <32 x i16>, i32, i32) nounwind readnone

declare <64 x i8> @llvm.x86.avx512.mask.blend.b.512(<64 x i8>, <64 x i8>, i64) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_w_512
define <32 x i16> @test_x86_mask_blend_w_512(i32 %mask, <32 x i16> %a1, <32 x i16> %a2) {
  ; CHECK: vpblendmw
  %res = call <32 x i16> @llvm.x86.avx512.mask.blend.w.512(<32 x i16> %a1, <32 x i16> %a2, i32 %mask) ; <<32 x i16>> [#uses=1]
  ret <32 x i16> %res
}
declare <32 x i16> @llvm.x86.avx512.mask.blend.w.512(<32 x i16>, <32 x i16>, i32) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_b_512
; CHECK: vpblendmb
define <64 x i8> @test_x86_mask_blend_b_512(i64 %a0, <64 x i8> %a1, <64 x i8> %a2) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.blend.b.512(<64 x i8> %a1, <64 x i8> %a2, i64 %a0) ; <<64 x i8>> [#uses=1]
  ret <64 x i8> %res
}

define <32 x i16> @test_mask_packs_epi32_rr_512(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rr_512
  ;CHECK: vpackssdw       %zmm1, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x48,0x6b,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rrk_512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrk_512
  ;CHECK: vpackssdw       %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0x6b,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rrkz_512(<16 x i32> %a, <16 x i32> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrkz_512
  ;CHECK: vpackssdw       %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0x6b,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rm_512(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rm_512
  ;CHECK: vpackssdw       (%rdi), %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x48,0x6b,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmk_512(<16 x i32> %a, <16 x i32>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmk_512
  ;CHECK: vpackssdw       (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0x6b,0x0f]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmkz_512(<16 x i32> %a, <16 x i32>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmkz_512
  ;CHECK: vpackssdw       (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0x6b,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmb_512(<16 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmb_512
  ;CHECK: vpackssdw       (%rdi){1to16}, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x58,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmbk_512(<16 x i32> %a, i32* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbk_512
  ;CHECK: vpackssdw       (%rdi){1to16}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x59,0x6b,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmbkz_512(<16 x i32> %a, i32* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbkz_512
  ;CHECK: vpackssdw       (%rdi){1to16}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xd9,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32>, <16 x i32>, <32 x i16>, i32)

define <64 x i8> @test_mask_packs_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rr_512
  ;CHECK: vpacksswb       %zmm1, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x48,0x63,0xc1]
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrk_512
  ;CHECK: vpacksswb       %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0x63,0xd1]
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrkz_512
  ;CHECK: vpacksswb       %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0x63,0xc1]
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rm_512
  ;CHECK: vpacksswb       (%rdi), %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x48,0x63,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmk_512
  ;CHECK: vpacksswb       (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0x63,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmkz_512
  ;CHECK: vpacksswb       (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0x63,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

declare <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16>, <32 x i16>, <64 x i8>, i64)


define <32 x i16> @test_mask_packus_epi32_rr_512(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rr_512
  ;CHECK: vpackusdw       %zmm1, %zmm0, %zmm0  
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rrk_512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrk_512
  ;CHECK: vpackusdw       %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rrkz_512(<16 x i32> %a, <16 x i32> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrkz_512
  ;CHECK: vpackusdw       %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rm_512(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rm_512
  ;CHECK: vpackusdw       (%rdi), %zmm0, %zmm0  
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmk_512(<16 x i32> %a, <16 x i32>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmk_512
  ;CHECK: vpackusdw       (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmkz_512(<16 x i32> %a, <16 x i32>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmkz_512
  ;CHECK: vpackusdw       (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmb_512(<16 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmb_512
  ;CHECK: vpackusdw       (%rdi){1to16}, %zmm0, %zmm0  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmbk_512(<16 x i32> %a, i32* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbk_512
  ;CHECK: vpackusdw       (%rdi){1to16}, %zmm0, %zmm1 {%k1} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmbkz_512(<16 x i32> %a, i32* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbkz_512
  ;CHECK: vpackusdw       (%rdi){1to16}, %zmm0, %zmm0 {%k1} {z} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32>, <16 x i32>, <32 x i16>, i32)

define <64 x i8> @test_mask_packus_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rr_512
  ;CHECK: vpackuswb       %zmm1, %zmm0, %zmm0  
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrk_512
  ;CHECK: vpackuswb       %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrkz_512
  ;CHECK: vpackuswb       %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rm_512
  ;CHECK: vpackuswb       (%rdi), %zmm0, %zmm0  
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmk_512
  ;CHECK: vpackuswb       (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmkz_512
  ;CHECK: vpackuswb       (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

declare <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16>, <32 x i16>, <64 x i8>, i64)

define <32 x i16> @test_mask_adds_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rr_512
  ;CHECK: vpaddsw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrk_512
  ;CHECK: vpaddsw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrkz_512
  ;CHECK: vpaddsw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rm_512
  ;CHECK: vpaddsw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmk_512
  ;CHECK: vpaddsw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmkz_512
  ;CHECK: vpaddsw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_subs_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rr_512
  ;CHECK: vpsubsw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrk_512
  ;CHECK: vpsubsw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrkz_512
  ;CHECK: vpsubsw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rm_512
  ;CHECK: vpsubsw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmk_512
  ;CHECK: vpsubsw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmkz_512
  ;CHECK: vpsubsw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_adds_epu16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rr_512
  ;CHECK: vpaddusw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrk_512
  ;CHECK: vpaddusw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrkz_512
  ;CHECK: vpaddusw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rm_512
  ;CHECK: vpaddusw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmk_512
  ;CHECK: vpaddusw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmkz_512
  ;CHECK: vpaddusw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_subs_epu16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rr_512
  ;CHECK: vpsubusw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrk_512
  ;CHECK: vpsubusw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrkz_512
  ;CHECK: vpsubusw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rm_512
  ;CHECK: vpsubusw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmk_512
  ;CHECK: vpsubusw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmkz_512
  ;CHECK: vpsubusw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

declare <64 x i8> @llvm.x86.avx512.mask.pmaxs.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_b_512
; CHECK-NOT: call 
; CHECK: vpmaxsb %zmm
; CHECK: {%k1} 
define <64 x i8>@test_int_x86_avx512_mask_pmaxs_b_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.pmaxs.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.pmaxs.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pmaxs.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_w_512
; CHECK-NOT: call 
; CHECK: vpmaxsw %zmm
; CHECK: {%k1} 
define <32 x i16>@test_int_x86_avx512_mask_pmaxs_w_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmaxs.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pmaxs.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.pmaxu.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_b_512
; CHECK-NOT: call 
; CHECK: vpmaxub %zmm
; CHECK: {%k1} 
define <64 x i8>@test_int_x86_avx512_mask_pmaxu_b_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.pmaxu.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.pmaxu.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pmaxu.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_w_512
; CHECK-NOT: call 
; CHECK: vpmaxuw %zmm
; CHECK: {%k1} 
define <32 x i16>@test_int_x86_avx512_mask_pmaxu_w_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmaxu.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pmaxu.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.pmins.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_b_512
; CHECK-NOT: call 
; CHECK: vpminsb %zmm
; CHECK: {%k1} 
define <64 x i8>@test_int_x86_avx512_mask_pmins_b_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.pmins.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.pmins.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pmins.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_w_512
; CHECK-NOT: call 
; CHECK: vpminsw %zmm
; CHECK: {%k1} 
define <32 x i16>@test_int_x86_avx512_mask_pmins_w_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmins.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pmins.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.pminu.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_b_512
; CHECK-NOT: call 
; CHECK: vpminub %zmm
; CHECK: {%k1} 
define <64 x i8>@test_int_x86_avx512_mask_pminu_b_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.pminu.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.pminu.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pminu.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_w_512
; CHECK-NOT: call 
; CHECK: vpminuw %zmm
; CHECK: {%k1} 
define <32 x i16>@test_int_x86_avx512_mask_pminu_w_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pminu.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pminu.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermt2var_hi_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2w %zmm{{.*}}{%k1} 
define <32 x i16>@test_int_x86_avx512_mask_vpermt2var_hi_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <32 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_hi_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2w %zmm{{.*}}{%k1} {z}
define <32 x i16>@test_int_x86_avx512_maskz_vpermt2var_hi_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.maskz.vpermt2var.hi.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_hi_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermi2w %zmm{{.*}}{%k1} 
define <32 x i16>@test_int_x86_avx512_mask_vpermi2var_hi_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.pavg.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

; CHECK-LABEL: @test_int_x86_avx512_mask_pavg_b_512
; CHECK-NOT: call 
; CHECK: vpavgb %zmm
; CHECK: {%k1} 
define <64 x i8>@test_int_x86_avx512_mask_pavg_b_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.pavg.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.pavg.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pavg.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pavg_w_512
; CHECK-NOT: call 
; CHECK: vpavgw %zmm
; CHECK: {%k1} 
define <32 x i16>@test_int_x86_avx512_mask_pavg_w_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pavg.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pavg.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.pshuf.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

; CHECK-LABEL: @test_int_x86_avx512_mask_pshuf_b_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpshufb %zmm{{.*}}{%k1} 
define <64 x i8>@test_int_x86_avx512_mask_pshuf_b_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.pshuf.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.pshuf.b.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pabs.w.512(<32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_w_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsw{{.*}}{%k1} 
define <32 x i16>@test_int_x86_avx512_mask_pabs_w_512(<32 x i16> %x0, <32 x i16> %x1, i32 %x2) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pabs.w.512(<32 x i16> %x0, <32 x i16> %x1, i32 %x2)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pabs.w.512(<32 x i16> %x0, <32 x i16> %x1, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.pabs.b.512(<64 x i8>, <64 x i8>, i64)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_b_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsb{{.*}}{%k1} 
define <64 x i8>@test_int_x86_avx512_mask_pabs_b_512(<64 x i8> %x0, <64 x i8> %x1, i64 %x2) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.pabs.b.512(<64 x i8> %x0, <64 x i8> %x1, i64 %x2)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.pabs.b.512(<64 x i8> %x0, <64 x i8> %x1, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pmulhu.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmulhu_w_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhuw {{.*}}encoding: [0x62
define <32 x i16>@test_int_x86_avx512_mask_pmulhu_w_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmulhu.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pmulhu.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pmulh.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmulh_w_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhw {{.*}}encoding: [0x62
define <32 x i16>@test_int_x86_avx512_mask_pmulh_w_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmulh.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pmulh.w.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmulhr_sw_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: {%k1} 
; CHECK: vpmulhrsw {{.*}}encoding: [0x62
define <32 x i16>@test_int_x86_avx512_mask_pmulhr_sw_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pmul.hr.sw.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <32 x i8> @llvm.x86.avx512.mask.pmov.wb.512(<32 x i16>, <32 x i8>, i32)

define <32 x i8>@test_int_x86_avx512_mask_pmov_wb_512(<32 x i16> %x0, <32 x i8> %x1, i32 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_wb_512:
; CHECK:       vpmovwb %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovwb %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovwb %zmm0, %ymm0
    %res0 = call <32 x i8> @llvm.x86.avx512.mask.pmov.wb.512(<32 x i16> %x0, <32 x i8> %x1, i32 -1)
    %res1 = call <32 x i8> @llvm.x86.avx512.mask.pmov.wb.512(<32 x i16> %x0, <32 x i8> %x1, i32 %x2)
    %res2 = call <32 x i8> @llvm.x86.avx512.mask.pmov.wb.512(<32 x i16> %x0, <32 x i8> zeroinitializer, i32 %x2)
    %res3 = add <32 x i8> %res0, %res1
    %res4 = add <32 x i8> %res3, %res2
    ret <32 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.wb.mem.512(i8* %ptr, <32 x i16>, i32)

define void @test_int_x86_avx512_mask_pmov_wb_mem_512(i8* %ptr, <32 x i16> %x1, i32 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_wb_mem_512:
; CHECK:  vpmovwb %zmm0, (%rdi)
; CHECK:  vpmovwb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.wb.mem.512(i8* %ptr, <32 x i16> %x1, i32 -1)
    call void @llvm.x86.avx512.mask.pmov.wb.mem.512(i8* %ptr, <32 x i16> %x1, i32 %x2)
    ret void
}

declare <32 x i8> @llvm.x86.avx512.mask.pmovs.wb.512(<32 x i16>, <32 x i8>, i32)

define <32 x i8>@test_int_x86_avx512_mask_pmovs_wb_512(<32 x i16> %x0, <32 x i8> %x1, i32 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_wb_512:
; CHECK:       vpmovswb %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovswb %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovswb %zmm0, %ymm0
    %res0 = call <32 x i8> @llvm.x86.avx512.mask.pmovs.wb.512(<32 x i16> %x0, <32 x i8> %x1, i32 -1)
    %res1 = call <32 x i8> @llvm.x86.avx512.mask.pmovs.wb.512(<32 x i16> %x0, <32 x i8> %x1, i32 %x2)
    %res2 = call <32 x i8> @llvm.x86.avx512.mask.pmovs.wb.512(<32 x i16> %x0, <32 x i8> zeroinitializer, i32 %x2)
    %res3 = add <32 x i8> %res0, %res1
    %res4 = add <32 x i8> %res3, %res2
    ret <32 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.wb.mem.512(i8* %ptr, <32 x i16>, i32)

define void @test_int_x86_avx512_mask_pmovs_wb_mem_512(i8* %ptr, <32 x i16> %x1, i32 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_wb_mem_512:
; CHECK:  vpmovswb %zmm0, (%rdi)
; CHECK:  vpmovswb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.wb.mem.512(i8* %ptr, <32 x i16> %x1, i32 -1)
    call void @llvm.x86.avx512.mask.pmovs.wb.mem.512(i8* %ptr, <32 x i16> %x1, i32 %x2)
    ret void
}

declare <32 x i8> @llvm.x86.avx512.mask.pmovus.wb.512(<32 x i16>, <32 x i8>, i32)

define <32 x i8>@test_int_x86_avx512_mask_pmovus_wb_512(<32 x i16> %x0, <32 x i8> %x1, i32 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_wb_512:
; CHECK:       vpmovuswb %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovuswb %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovuswb %zmm0, %ymm0
    %res0 = call <32 x i8> @llvm.x86.avx512.mask.pmovus.wb.512(<32 x i16> %x0, <32 x i8> %x1, i32 -1)
    %res1 = call <32 x i8> @llvm.x86.avx512.mask.pmovus.wb.512(<32 x i16> %x0, <32 x i8> %x1, i32 %x2)
    %res2 = call <32 x i8> @llvm.x86.avx512.mask.pmovus.wb.512(<32 x i16> %x0, <32 x i8> zeroinitializer, i32 %x2)
    %res3 = add <32 x i8> %res0, %res1
    %res4 = add <32 x i8> %res3, %res2
    ret <32 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.wb.mem.512(i8* %ptr, <32 x i16>, i32)

define void @test_int_x86_avx512_mask_pmovus_wb_mem_512(i8* %ptr, <32 x i16> %x1, i32 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_wb_mem_512:
; CHECK:  vpmovuswb %zmm0, (%rdi)
; CHECK:  vpmovuswb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.wb.mem.512(i8* %ptr, <32 x i16> %x1, i32 -1)
    call void @llvm.x86.avx512.mask.pmovus.wb.mem.512(i8* %ptr, <32 x i16> %x1, i32 %x2)
    ret void
}

declare <32 x i16> @llvm.x86.avx512.mask.pmaddubs.w.512(<64 x i8>, <64 x i8>, <32 x i16>, i32)

define <32 x i16>@test_int_x86_avx512_mask_pmaddubs_w_512(<64 x i8> %x0, <64 x i8> %x1, <32 x i16> %x2, i32 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmaddubs_w_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k1 ## encoding: [0xc5,0xfb,0x92,0xcf]
; CHECK-NEXT:    vpmaddubsw %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vpmaddubsw %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddw %zmm0, %zmm2, %zmm0 
; CHECK-NEXT:    retq
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmaddubs.w.512(<64 x i8> %x0, <64 x i8> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.pmaddubs.w.512(<64 x i8> %x0, <64 x i8> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.pmaddw.d.512(<32 x i16>, <32 x i16>, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_pmaddw_d_512(<32 x i16> %x0, <32 x i16> %x1, <16 x i32> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmaddw_d_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmaddwd %zmm1, %zmm0, %zmm2 {%k1} 
; CHECK-NEXT:    vpmaddwd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm2, %zmm0 
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmaddw.d.512(<32 x i16> %x0, <32 x i16> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.pmaddw.d.512(<32 x i16> %x0, <32 x i16> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.punpckhb.w.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

define <64 x i8>@test_int_x86_avx512_mask_punpckhb_w_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhb_w_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovq %rdi, %k1
; CHECK-NEXT:    vpunpckhbw %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[8],k1[8],zmm2[9],k1[9],zmm2[10],k1[10],zmm2[11],k1[11],zmm2[12],k1[12],zmm2[13],k1[13],zmm2[14],k1[14],zmm2[15],k1[15],zmm2[24],k1[24],zmm2[25],k1[25],zmm2[26],k1[26],zmm2[27],k1[27],zmm2[28],k1[28],zmm2[29],k1[29],zmm2[30],k1[30],zmm2[31],k1[31],zmm2[40],k1[40],zmm2[41],k1[41],zmm2[42],k1[42],zmm2[43],k1[43],zmm2[44],k1[44],zmm2[45],k1[45],zmm2[46],k1[46],zmm2[47],k1[47],zmm2[56],k1[56],zmm2[57],k1[57],zmm2[58],k1[58],zmm2[59],k1[59],zmm2[60],k1[60],zmm2[61],k1[61],zmm2[62],k1[62],zmm2[63],k1[63]
; CHECK-NEXT:    vpunpckhbw %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[12],zmm1[12],zmm0[13],zmm1[13],zmm0[14],zmm1[14],zmm0[15],zmm1[15],zmm0[24],zmm1[24],zmm0[25],zmm1[25],zmm0[26],zmm1[26],zmm0[27],zmm1[27],zmm0[28],zmm1[28],zmm0[29],zmm1[29],zmm0[30],zmm1[30],zmm0[31],zmm1[31],zmm0[40],zmm1[40],zmm0[41],zmm1[41],zmm0[42],zmm1[42],zmm0[43],zmm1[43],zmm0[44],zmm1[44],zmm0[45],zmm1[45],zmm0[46],zmm1[46],zmm0[47],zmm1[47],zmm0[56],zmm1[56],zmm0[57],zmm1[57],zmm0[58],zmm1[58],zmm0[59],zmm1[59],zmm0[60],zmm1[60],zmm0[61],zmm1[61],zmm0[62],zmm1[62],zmm0[63],zmm1[63]
; CHECK-NEXT:    vpaddb %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <64 x i8> @llvm.x86.avx512.mask.punpckhb.w.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.punpckhb.w.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.punpcklb.w.512(<64 x i8>, <64 x i8>, <64 x i8>, i64)

define <64 x i8>@test_int_x86_avx512_mask_punpcklb_w_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklb_w_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovq %rdi, %k1
; CHECK-NEXT:    vpunpcklbw %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[0],k1[0],zmm2[1],k1[1],zmm2[2],k1[2],zmm2[3],k1[3],zmm2[4],k1[4],zmm2[5],k1[5],zmm2[6],k1[6],zmm2[7],k1[7],zmm2[16],k1[16],zmm2[17],k1[17],zmm2[18],k1[18],zmm2[19],k1[19],zmm2[20],k1[20],zmm2[21],k1[21],zmm2[22],k1[22],zmm2[23],k1[23],zmm2[32],k1[32],zmm2[33],k1[33],zmm2[34],k1[34],zmm2[35],k1[35],zmm2[36],k1[36],zmm2[37],k1[37],zmm2[38],k1[38],zmm2[39],k1[39],zmm2[48],k1[48],zmm2[49],k1[49],zmm2[50],k1[50],zmm2[51],k1[51],zmm2[52],k1[52],zmm2[53],k1[53],zmm2[54],k1[54],zmm2[55],k1[55]
; CHECK-NEXT:    vpunpcklbw %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[16],zmm1[16],zmm0[17],zmm1[17],zmm0[18],zmm1[18],zmm0[19],zmm1[19],zmm0[20],zmm1[20],zmm0[21],zmm1[21],zmm0[22],zmm1[22],zmm0[23],zmm1[23],zmm0[32],zmm1[32],zmm0[33],zmm1[33],zmm0[34],zmm1[34],zmm0[35],zmm1[35],zmm0[36],zmm1[36],zmm0[37],zmm1[37],zmm0[38],zmm1[38],zmm0[39],zmm1[39],zmm0[48],zmm1[48],zmm0[49],zmm1[49],zmm0[50],zmm1[50],zmm0[51],zmm1[51],zmm0[52],zmm1[52],zmm0[53],zmm1[53],zmm0[54],zmm1[54],zmm0[55],zmm1[55]
; CHECK-NEXT:    vpaddb %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <64 x i8> @llvm.x86.avx512.mask.punpcklb.w.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 %x3)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.punpcklb.w.512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2, i64 -1)
  %res2 = add <64 x i8> %res, %res1
  ret <64 x i8> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.punpckhw.d.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16>@test_int_x86_avx512_mask_punpckhw_d_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhw_d_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpunpckhwd %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[4],k1[4],zmm2[5],k1[5],zmm2[6],k1[6],zmm2[7],k1[7],zmm2[12],k1[12],zmm2[13],k1[13],zmm2[14],k1[14],zmm2[15],k1[15],zmm2[20],k1[20],zmm2[21],k1[21],zmm2[22],k1[22],zmm2[23],k1[23],zmm2[28],k1[28],zmm2[29],k1[29],zmm2[30],k1[30],zmm2[31],k1[31]
; CHECK-NEXT:    vpunpckhwd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[12],zmm1[12],zmm0[13],zmm1[13],zmm0[14],zmm1[14],zmm0[15],zmm1[15],zmm0[20],zmm1[20],zmm0[21],zmm1[21],zmm0[22],zmm1[22],zmm0[23],zmm1[23],zmm0[28],zmm1[28],zmm0[29],zmm1[29],zmm0[30],zmm1[30],zmm0[31],zmm1[31]
; CHECK-NEXT:    vpaddw %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <32 x i16> @llvm.x86.avx512.mask.punpckhw.d.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.punpckhw.d.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <32 x i16> @llvm.x86.avx512.mask.punpcklw.d.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16>@test_int_x86_avx512_mask_punpcklw_d_512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklw_d_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpunpcklwd %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[0],k1[0],zmm2[1],k1[1],zmm2[2],k1[2],zmm2[3],k1[3],zmm2[8],k1[8],zmm2[9],k1[9],zmm2[10],k1[10],zmm2[11],k1[11],zmm2[16],k1[16],zmm2[17],k1[17],zmm2[18],k1[18],zmm2[19],k1[19],zmm2[24],k1[24],zmm2[25],k1[25],zmm2[26],k1[26],zmm2[27],k1[27]
; CHECK-NEXT:    vpunpcklwd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[16],zmm1[16],zmm0[17],zmm1[17],zmm0[18],zmm1[18],zmm0[19],zmm1[19],zmm0[24],zmm1[24],zmm0[25],zmm1[25],zmm0[26],zmm1[26],zmm0[27],zmm1[27]
; CHECK-NEXT:    vpaddw %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <32 x i16> @llvm.x86.avx512.mask.punpcklw.d.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 %x3)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.punpcklw.d.512(<32 x i16> %x0, <32 x i16> %x1, <32 x i16> %x2, i32 -1)
  %res2 = add <32 x i16> %res, %res1
  ret <32 x i16> %res2
}

declare <64 x i8> @llvm.x86.avx512.mask.palignr.512(<64 x i8>, <64 x i8>, i32, <64 x i8>, i64)

define <64 x i8>@test_int_x86_avx512_mask_palignr_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x3, i64 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_palignr_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovq %rdi, %k1
; CHECK-NEXT:    vpalignr $2, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vpalignr $2, %zmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vpalignr $2, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddb %zmm3, %zmm2, %zmm1
; CHECK-NEXT:    vpaddb %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <64 x i8> @llvm.x86.avx512.mask.palignr.512(<64 x i8> %x0, <64 x i8> %x1, i32 2, <64 x i8> %x3, i64 %x4)
  %res1 = call <64 x i8> @llvm.x86.avx512.mask.palignr.512(<64 x i8> %x0, <64 x i8> %x1, i32 2, <64 x i8> zeroinitializer, i64 %x4)
  %res2 = call <64 x i8> @llvm.x86.avx512.mask.palignr.512(<64 x i8> %x0, <64 x i8> %x1, i32 2, <64 x i8> %x3, i64 -1)
  %res3 = add <64 x i8> %res, %res1
  %res4 = add <64 x i8> %res3, %res2
  ret <64 x i8> %res4
}

declare <32 x i16> @llvm.x86.avx512.mask.dbpsadbw.512(<64 x i8>, <64 x i8>, i32, <32 x i16>, i32)

define <32 x i16>@test_int_x86_avx512_mask_dbpsadbw_512(<64 x i8> %x0, <64 x i8> %x1, <32 x i16> %x3, i32 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_dbpsadbw_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vdbpsadbw $2, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vdbpsadbw $2, %zmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vdbpsadbw $2, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddw %zmm3, %zmm2, %zmm1
; CHECK-NEXT:    vpaddw %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <32 x i16> @llvm.x86.avx512.mask.dbpsadbw.512(<64 x i8> %x0, <64 x i8> %x1, i32 2, <32 x i16> %x3, i32 %x4)
  %res1 = call <32 x i16> @llvm.x86.avx512.mask.dbpsadbw.512(<64 x i8> %x0, <64 x i8> %x1, i32 2, <32 x i16> zeroinitializer, i32 %x4)
  %res2 = call <32 x i16> @llvm.x86.avx512.mask.dbpsadbw.512(<64 x i8> %x0, <64 x i8> %x1, i32 2, <32 x i16> %x3, i32 -1)
  %res3 = add <32 x i16> %res, %res1
  %res4 = add <32 x i16> %res3, %res2
  ret <32 x i16> %res4
}

declare <8 x i64> @llvm.x86.avx512.psll.dq.512(<8 x i64>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_psll_dq_512
; CHECK-NOT: call 
; CHECK: vpslldq 
; CHECK: vpslldq 
define <8 x i64>@test_int_x86_avx512_mask_psll_dq_512(<8 x i64> %x0) {
  %res = call <8 x i64> @llvm.x86.avx512.psll.dq.512(<8 x i64> %x0, i32 8)
  %res1 = call <8 x i64> @llvm.x86.avx512.psll.dq.512(<8 x i64> %x0, i32 4)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <8 x i64> @llvm.x86.avx512.psrl.dq.512(<8 x i64>, i32)

; CHECK-LABEL: @test_int_x86_avx512_mask_psrl_dq_512
; CHECK-NOT: call 
; CHECK: vpsrldq 
; CHECK: vpsrldq
define <8 x i64>@test_int_x86_avx512_mask_psrl_dq_512(<8 x i64> %x0) {
  %res = call <8 x i64> @llvm.x86.avx512.psrl.dq.512(<8 x i64> %x0, i32 8)
  %res1 = call <8 x i64> @llvm.x86.avx512.psrl.dq.512(<8 x i64> %x0, i32 4)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}
declare  <8 x i64> @llvm.x86.avx512.psad.bw.512(<64 x i8>, <64 x i8>)

; CHECK-LABEL: @test_int_x86_avx512_mask_psadb_w_512
; CHECK-NOT: call 
; CHECK: vpsadbw %zmm1
; CHECK: vpsadbw %zmm2
define  <8 x i64>@test_int_x86_avx512_mask_psadb_w_512(<64 x i8> %x0, <64 x i8> %x1, <64 x i8> %x2){
  %res = call  <8 x i64> @llvm.x86.avx512.psad.bw.512(<64 x i8> %x0, <64 x i8> %x1)
  %res1 = call  <8 x i64> @llvm.x86.avx512.psad.bw.512(<64 x i8> %x0, <64 x i8> %x2)
  %res2 = add  <8 x i64> %res, %res1
  ret  <8 x i64> %res2
}
