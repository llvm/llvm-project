; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vl --show-mc-encoding| FileCheck %s

; 256-bit

define i8 @test_pcmpeq_d_256(<8 x i32> %a, <8 x i32> %b) {
; CHECK-LABEL: test_pcmpeq_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.256(<8 x i32> %a, <8 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_d_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.256(<8 x i32> %a, <8 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.d.256(<8 x i32>, <8 x i32>, i8)

define i8 @test_pcmpeq_q_256(<4 x i64> %a, <4 x i64> %b) {
; CHECK-LABEL: test_pcmpeq_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.256(<4 x i64> %a, <4 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_q_256(<4 x i64> %a, <4 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.256(<4 x i64> %a, <4 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.q.256(<4 x i64>, <4 x i64>, i8)

define i8 @test_pcmpgt_d_256(<8 x i32> %a, <8 x i32> %b) {
; CHECK-LABEL: test_pcmpgt_d_256
; CHECK: vpcmpgtd %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.256(<8 x i32> %a, <8 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_d_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_d_256
; CHECK: vpcmpgtd %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.256(<8 x i32> %a, <8 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.d.256(<8 x i32>, <8 x i32>, i8)

define i8 @test_pcmpgt_q_256(<4 x i64> %a, <4 x i64> %b) {
; CHECK-LABEL: test_pcmpgt_q_256
; CHECK: vpcmpgtq %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.256(<4 x i64> %a, <4 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_q_256(<4 x i64> %a, <4 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_q_256
; CHECK: vpcmpgtq %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.256(<4 x i64> %a, <4 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.q.256(<4 x i64>, <4 x i64>, i8)

define <8 x i8> @test_cmp_d_256(<8 x i32> %a0, <8 x i32> %a1) {
; CHECK-LABEL: test_cmp_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_d_256(<8 x i32> %a0, <8 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32>, <8 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_d_256(<8 x i32> %a0, <8 x i32> %a1) {
; CHECK-LABEL: test_ucmp_d_256
; CHECK: vpcmpequd %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_d_256(<8 x i32> %a0, <8 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_d_256
; CHECK: vpcmpequd %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32>, <8 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_cmp_q_256(<4 x i64> %a0, <4 x i64> %a1) {
; CHECK-LABEL: test_cmp_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_q_256(<4 x i64> %a0, <4 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64>, <4 x i64>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_q_256(<4 x i64> %a0, <4 x i64> %a1) {
; CHECK-LABEL: test_ucmp_q_256
; CHECK: vpcmpequq %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_q_256(<4 x i64> %a0, <4 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_q_256
; CHECK: vpcmpequq %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64>, <4 x i64>, i32, i8) nounwind readnone

; 128-bit

define i8 @test_pcmpeq_d_128(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_pcmpeq_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.128(<4 x i32> %a, <4 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_d_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.128(<4 x i32> %a, <4 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.d.128(<4 x i32>, <4 x i32>, i8)

define i8 @test_pcmpeq_q_128(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_pcmpeq_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.128(<2 x i64> %a, <2 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_q_128(<2 x i64> %a, <2 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.128(<2 x i64> %a, <2 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.q.128(<2 x i64>, <2 x i64>, i8)

define i8 @test_pcmpgt_d_128(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_pcmpgt_d_128
; CHECK: vpcmpgtd %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.128(<4 x i32> %a, <4 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_d_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_d_128
; CHECK: vpcmpgtd %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.128(<4 x i32> %a, <4 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.d.128(<4 x i32>, <4 x i32>, i8)

define i8 @test_pcmpgt_q_128(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_pcmpgt_q_128
; CHECK: vpcmpgtq %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.128(<2 x i64> %a, <2 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_q_128(<2 x i64> %a, <2 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_q_128
; CHECK: vpcmpgtq %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.128(<2 x i64> %a, <2 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.q.128(<2 x i64>, <2 x i64>, i8)

define <8 x i8> @test_cmp_d_128(<4 x i32> %a0, <4 x i32> %a1) {
; CHECK-LABEL: test_cmp_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_d_128(<4 x i32> %a0, <4 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32>, <4 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_d_128(<4 x i32> %a0, <4 x i32> %a1) {
; CHECK-LABEL: test_ucmp_d_128
; CHECK: vpcmpequd %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_d_128(<4 x i32> %a0, <4 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_d_128
; CHECK: vpcmpequd %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32>, <4 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_cmp_q_128(<2 x i64> %a0, <2 x i64> %a1) {
; CHECK-LABEL: test_cmp_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_q_128(<2 x i64> %a0, <2 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64>, <2 x i64>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_q_128(<2 x i64> %a0, <2 x i64> %a1) {
; CHECK-LABEL: test_ucmp_q_128
; CHECK: vpcmpequq %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_q_128(<2 x i64> %a0, <2 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_q_128
; CHECK: vpcmpequq %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64>, <2 x i64>, i32, i8) nounwind readnone

; CHECK-LABEL: compr1
; CHECK: vcompresspd %zmm0
define void @compr1(i8* %addr, <8 x double> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.pd.512(i8* %addr, <8 x double> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.pd.512(i8* %addr, <8 x double> %data, i8 %mask)

; CHECK-LABEL: compr2
; CHECK: vcompresspd %ymm0
define void @compr2(i8* %addr, <4 x double> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.pd.256(i8* %addr, <4 x double> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.pd.256(i8* %addr, <4 x double> %data, i8 %mask)

; CHECK-LABEL: compr3
; CHECK: vcompressps %xmm0
define void @compr3(i8* %addr, <4 x float> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.ps.128(i8* %addr, <4 x float> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.ps.128(i8* %addr, <4 x float> %data, i8 %mask)

; CHECK-LABEL: compr4
; CHECK: vcompresspd %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xc9,0x8a,0xc0]
define <8 x double> @compr4(i8* %addr, <8 x double> %data, i8 %mask) {
  %res = call <8 x double> @llvm.x86.avx512.mask.compress.pd.512(<8 x double> %data, <8 x double> zeroinitializer, i8 %mask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.compress.pd.512(<8 x double> %data, <8 x double> %src0, i8 %mask)

; CHECK-LABEL: compr5
; CHECK: vcompresspd %ymm0, %ymm1 {%k1}  ## encoding: [0x62,0xf2,0xfd,0x29,0x8a,0xc1]
define <4 x double> @compr5(<4 x double> %data, <4 x double> %src0, i8 %mask) {
  %res = call <4 x double> @llvm.x86.avx512.mask.compress.pd.256( <4 x double> %data, <4 x double> %src0, i8 %mask)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.compress.pd.256(<4 x double> %data, <4 x double> %src0, i8 %mask)

; CHECK-LABEL: compr6
; CHECK: vcompressps %xmm0
define <4 x float> @compr6(<4 x float> %data, i8 %mask) {
  %res = call <4 x float> @llvm.x86.avx512.mask.compress.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.compress.ps.128(<4 x float> %data, <4 x float> %src0, i8 %mask)

; CHECK-LABEL: compr7
; CHECK-NOT: vcompress
; CHECK: vmovupd
define void @compr7(i8* %addr, <8 x double> %data) {
  call void @llvm.x86.avx512.mask.compress.store.pd.512(i8* %addr, <8 x double> %data, i8 -1)
  ret void
}

; CHECK-LABEL: compr8
; CHECK-NOT: vcompressps %xmm0
define <4 x float> @compr8(<4 x float> %data) {
  %res = call <4 x float> @llvm.x86.avx512.mask.compress.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}

; CHECK-LABEL: compr9
; CHECK: vpcompressq %zmm0, (%rdi) {%k1}  ## encoding: [0x62,0xf2,0xfd,0x49,0x8b,0x07]
define void @compr9(i8* %addr, <8 x i64> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.q.512(i8* %addr, <8 x i64> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.q.512(i8* %addr, <8 x i64> %data, i8 %mask)

; CHECK-LABEL: compr10
; CHECK: vpcompressd %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0x89,0x8b,0xc0]
define <4 x i32> @compr10(<4 x i32> %data, i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.compress.d.128(<4 x i32> %data, <4 x i32>zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.compress.d.128(<4 x i32> %data, <4 x i32> %src0, i8 %mask)

; Expand

; CHECK-LABEL: expand1
; CHECK: vexpandpd (%rdi), %zmm0 {%k1}  ## encoding: [0x62,0xf2,0xfd,0x49,0x88,0x07]
define <8 x double> @expand1(i8* %addr, <8 x double> %data, i8 %mask) {
  %res = call <8 x double> @llvm.x86.avx512.mask.expand.load.pd.512(i8* %addr, <8 x double> %data, i8 %mask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.expand.load.pd.512(i8* %addr, <8 x double> %data, i8 %mask)

; CHECK-LABEL: expand2
; CHECK: vexpandpd (%rdi), %ymm0 {%k1} ## encoding: [0x62,0xf2,0xfd,0x29,0x88,0x07]
define <4 x double> @expand2(i8* %addr, <4 x double> %data, i8 %mask) {
  %res = call <4 x double> @llvm.x86.avx512.mask.expand.load.pd.256(i8* %addr, <4 x double> %data, i8 %mask)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.expand.load.pd.256(i8* %addr, <4 x double> %data, i8 %mask)

; CHECK-LABEL: expand3
; CHECK: vexpandps (%rdi), %xmm0 {%k1} ## encoding: [0x62,0xf2,0x7d,0x09,0x88,0x07]
define <4 x float> @expand3(i8* %addr, <4 x float> %data, i8 %mask) {
  %res = call <4 x float> @llvm.x86.avx512.mask.expand.load.ps.128(i8* %addr, <4 x float> %data, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.expand.load.ps.128(i8* %addr, <4 x float> %data, i8 %mask)

; CHECK-LABEL: expand4
; CHECK: vexpandpd %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xc9,0x88,0xc0]
define <8 x double> @expand4(i8* %addr, <8 x double> %data, i8 %mask) {
  %res = call <8 x double> @llvm.x86.avx512.mask.expand.pd.512(<8 x double> %data, <8 x double> zeroinitializer, i8 %mask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.expand.pd.512(<8 x double> %data, <8 x double> %src0, i8 %mask)

; CHECK-LABEL: expand5
; CHECK: vexpandpd %ymm0, %ymm1 {%k1}  ## encoding: [0x62,0xf2,0xfd,0x29,0x88,0xc8]
define <4 x double> @expand5(<4 x double> %data, <4 x double> %src0, i8 %mask) {
  %res = call <4 x double> @llvm.x86.avx512.mask.expand.pd.256( <4 x double> %data, <4 x double> %src0, i8 %mask)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.expand.pd.256(<4 x double> %data, <4 x double> %src0, i8 %mask)

; CHECK-LABEL: expand6
; CHECK: vexpandps %xmm0
define <4 x float> @expand6(<4 x float> %data, i8 %mask) {
  %res = call <4 x float> @llvm.x86.avx512.mask.expand.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.expand.ps.128(<4 x float> %data, <4 x float> %src0, i8 %mask)

; CHECK-LABEL: expand7
; CHECK-NOT: vexpand
; CHECK: vmovupd
define <8 x double> @expand7(i8* %addr, <8 x double> %data) {
  %res = call <8 x double> @llvm.x86.avx512.mask.expand.load.pd.512(i8* %addr, <8 x double> %data, i8 -1)
  ret <8 x double> %res
}

; CHECK-LABEL: expand8
; CHECK-NOT: vexpandps %xmm0
define <4 x float> @expand8(<4 x float> %data) {
  %res = call <4 x float> @llvm.x86.avx512.mask.expand.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}

; CHECK-LABEL: expand9
; CHECK: vpexpandq (%rdi), %zmm0 {%k1} ## encoding: [0x62,0xf2,0xfd,0x49,0x89,0x07]
define <8 x i64> @expand9(i8* %addr, <8 x i64> %data, i8 %mask) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.expand.load.q.512(i8* %addr, <8 x i64> %data, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.expand.load.q.512(i8* %addr, <8 x i64> %data, i8 %mask)

; CHECK-LABEL: expand10
; CHECK: vpexpandd %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0x89,0x89,0xc0]
define <4 x i32> @expand10(<4 x i32> %data, i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.expand.d.128(<4 x i32> %data, <4 x i32>zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.expand.d.128(<4 x i32> %data, <4 x i32> %src0, i8 %mask)

define <8 x float> @test_x86_mask_blend_ps_256(i8 %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: vblendmps %ymm1, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.blend.ps.256(<8 x float> %a1, <8 x float> %a2, i8 %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}

declare <8 x float> @llvm.x86.avx512.mask.blend.ps.256(<8 x float>, <8 x float>, i8) nounwind readonly

define <4 x double> @test_x86_mask_blend_pd_256(i8 %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK: vblendmpd %ymm1, %ymm0
  %res = call <4 x double> @llvm.x86.avx512.mask.blend.pd.256(<4 x double> %a1, <4 x double> %a2, i8 %a0) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}

define <4 x double> @test_x86_mask_blend_pd_256_memop(<4 x double> %a, <4 x double>* %ptr, i8 %mask) {
  ; CHECK-LABEL: test_x86_mask_blend_pd_256_memop
  ; CHECK: vblendmpd (%
  %b = load <4 x double>, <4 x double>* %ptr
  %res = call <4 x double> @llvm.x86.avx512.mask.blend.pd.256(<4 x double> %a, <4 x double> %b, i8 %mask) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx512.mask.blend.pd.256(<4 x double>, <4 x double>, i8) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_d_256
; CHECK: vpblendmd
define <8 x i32> @test_x86_mask_blend_d_256(i8 %a0, <8 x i32> %a1, <8 x i32> %a2) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.blend.d.256(<8 x i32> %a1, <8 x i32> %a2, i8 %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx512.mask.blend.d.256(<8 x i32>, <8 x i32>, i8) nounwind readonly

define <4 x i64> @test_x86_mask_blend_q_256(i8 %a0, <4 x i64> %a1, <4 x i64> %a2) {
  ; CHECK: vpblendmq
  %res = call <4 x i64> @llvm.x86.avx512.mask.blend.q.256(<4 x i64> %a1, <4 x i64> %a2, i8 %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx512.mask.blend.q.256(<4 x i64>, <4 x i64>, i8) nounwind readonly

define <4 x float> @test_x86_mask_blend_ps_128(i8 %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: vblendmps %xmm1, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.blend.ps.128(<4 x float> %a1, <4 x float> %a2, i8 %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.blend.ps.128(<4 x float>, <4 x float>, i8) nounwind readonly

define <2 x double> @test_x86_mask_blend_pd_128(i8 %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: vblendmpd %xmm1, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.mask.blend.pd.128(<2 x double> %a1, <2 x double> %a2, i8 %a0) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}

define <2 x double> @test_x86_mask_blend_pd_128_memop(<2 x double> %a, <2 x double>* %ptr, i8 %mask) {
  ; CHECK-LABEL: test_x86_mask_blend_pd_128_memop
  ; CHECK: vblendmpd (%
  %b = load <2 x double>, <2 x double>* %ptr
  %res = call <2 x double> @llvm.x86.avx512.mask.blend.pd.128(<2 x double> %a, <2 x double> %b, i8 %mask) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.mask.blend.pd.128(<2 x double>, <2 x double>, i8) nounwind readonly

define <4 x i32> @test_x86_mask_blend_d_128(i8 %a0, <4 x i32> %a1, <4 x i32> %a2) {
  ; CHECK: vpblendmd
  %res = call <4 x i32> @llvm.x86.avx512.mask.blend.d.128(<4 x i32> %a1, <4 x i32> %a2, i8 %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx512.mask.blend.d.128(<4 x i32>, <4 x i32>, i8) nounwind readonly

define <2 x i64> @test_x86_mask_blend_q_128(i8 %a0, <2 x i64> %a1, <2 x i64> %a2) {
  ; CHECK: vpblendmq
  %res = call <2 x i64> @llvm.x86.avx512.mask.blend.q.128(<2 x i64> %a1, <2 x i64> %a2, i8 %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx512.mask.blend.q.128(<2 x i64>, <2 x i64>, i8) nounwind readonly


define < 2 x i64> @test_mask_mul_epi32_rr_128(< 4 x i32> %a, < 4 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rr_128
  ;CHECK: vpmuldq %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf2,0xfd,0x08,0x28,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rrk_128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrk_128
  ;CHECK: vpmuldq %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf2,0xfd,0x09,0x28,0xd1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rrkz_128(< 4 x i32> %a, < 4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrkz_128
  ;CHECK: vpmuldq %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0x89,0x28,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rm_128(< 4 x i32> %a, < 4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rm_128
  ;CHECK: vpmuldq (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf2,0xfd,0x08,0x28,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmk_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmk_128
  ;CHECK: vpmuldq (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x09,0x28,0x0f]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmkz_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmkz_128
  ;CHECK: vpmuldq (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0x89,0x28,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmb_128(< 4 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmb_128
  ;CHECK: vpmuldq (%rdi){1to2}, %xmm0, %xmm0  ## encoding: [0x62,0xf2,0xfd,0x18,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmbk_128(< 4 x i32> %a, i64* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbk_128
  ;CHECK: vpmuldq (%rdi){1to2}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x19,0x28,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmbkz_128(< 4 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbkz_128
  ;CHECK: vpmuldq (%rdi){1to2}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0x99,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, < 2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

declare < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32>, < 4 x i32>, < 2 x i64>, i8)

define < 4 x i64> @test_mask_mul_epi32_rr_256(< 8 x i32> %a, < 8 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rr_256
  ;CHECK: vpmuldq %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf2,0xfd,0x28,0x28,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rrk_256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrk_256
  ;CHECK: vpmuldq %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf2,0xfd,0x29,0x28,0xd1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rrkz_256(< 8 x i32> %a, < 8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrkz_256
  ;CHECK: vpmuldq %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xa9,0x28,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rm_256(< 8 x i32> %a, < 8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rm_256
  ;CHECK: vpmuldq (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf2,0xfd,0x28,0x28,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmk_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmk_256
  ;CHECK: vpmuldq (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x29,0x28,0x0f]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmkz_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmkz_256
  ;CHECK: vpmuldq (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xa9,0x28,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmb_256(< 8 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmb_256
  ;CHECK: vpmuldq (%rdi){1to4}, %ymm0, %ymm0  ## encoding: [0x62,0xf2,0xfd,0x38,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmbk_256(< 8 x i32> %a, i64* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbk_256
  ;CHECK: vpmuldq (%rdi){1to4}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x39,0x28,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmbkz_256(< 8 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbkz_256
  ;CHECK: vpmuldq (%rdi){1to4}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xb9,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

declare < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32>, < 8 x i32>, < 4 x i64>, i8)

define < 2 x i64> @test_mask_mul_epu32_rr_128(< 4 x i32> %a, < 4 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rr_128
  ;CHECK: vpmuludq %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xf4,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rrk_128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrk_128
  ;CHECK: vpmuludq %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xf4,0xd1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rrkz_128(< 4 x i32> %a, < 4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrkz_128
  ;CHECK: vpmuludq %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xf4,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rm_128(< 4 x i32> %a, < 4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rm_128
  ;CHECK: vpmuludq (%rdi), %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xf4,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmk_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmk_128
  ;CHECK: vpmuludq (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xf4,0x0f]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmkz_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmkz_128
  ;CHECK: vpmuludq (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xf4,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmb_128(< 4 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmb_128
  ;CHECK: vpmuludq (%rdi){1to2}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0xfd,0x18,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmbk_128(< 4 x i32> %a, i64* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbk_128
  ;CHECK: vpmuludq (%rdi){1to2}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x19,0xf4,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmbkz_128(< 4 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbkz_128
  ;CHECK: vpmuludq (%rdi){1to2}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x99,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, < 2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

declare < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32>, < 4 x i32>, < 2 x i64>, i8)

define < 4 x i64> @test_mask_mul_epu32_rr_256(< 8 x i32> %a, < 8 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rr_256
  ;CHECK: vpmuludq %ymm1, %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0xf4,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rrk_256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrk_256
  ;CHECK: vpmuludq %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xf4,0xd1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rrkz_256(< 8 x i32> %a, < 8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrkz_256
  ;CHECK: vpmuludq %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xf4,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rm_256(< 8 x i32> %a, < 8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rm_256
  ;CHECK: vpmuludq (%rdi), %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0xf4,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmk_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmk_256
  ;CHECK: vpmuludq (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xf4,0x0f]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmkz_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmkz_256
  ;CHECK: vpmuludq (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xf4,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmb_256(< 8 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmb_256
  ;CHECK: vpmuludq (%rdi){1to4}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0xfd,0x38,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmbk_256(< 8 x i32> %a, i64* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbk_256
  ;CHECK: vpmuludq (%rdi){1to4}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x39,0xf4,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmbkz_256(< 8 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbkz_256
  ;CHECK: vpmuludq (%rdi){1to4}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xb9,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

declare < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32>, < 8 x i32>, < 4 x i64>, i8)

define <4 x i32> @test_mask_add_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_add_epi32_rr_128
  ;CHECK: vpaddd %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xfe,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrk_128
  ;CHECK: vpaddd %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfe,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrkz_128
  ;CHECK: vpaddd %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfe,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rm_128
  ;CHECK: vpaddd (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xfe,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmk_128
  ;CHECK: vpaddd (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfe,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmkz_128
  ;CHECK: vpaddd (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfe,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rmb_128
  ;CHECK: vpaddd (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbk_128
  ;CHECK: vpaddd (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xfe,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbkz_128
  ;CHECK: vpaddd (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32> @test_mask_sub_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rr_128
  ;CHECK: vpsubd %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xfa,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrk_128
  ;CHECK: vpsubd %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfa,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrkz_128
  ;CHECK: vpsubd %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfa,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rm_128
  ;CHECK: (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xfa,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmk_128
  ;CHECK: vpsubd (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfa,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmkz_128
  ;CHECK: vpsubd (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfa,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmb_128
  ;CHECK: vpsubd (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbk_128
  ;CHECK: vpsubd (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xfa,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbkz_128
  ;CHECK: vpsubd (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_sub_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rr_256
  ;CHECK: vpsubd %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xfa,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrk_256
  ;CHECK: vpsubd %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfa,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrkz_256
  ;CHECK: vpsubd %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfa,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rm_256
  ;CHECK: vpsubd (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xfa,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmk_256
  ;CHECK: vpsubd (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfa,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmkz_256
  ;CHECK: vpsubd (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfa,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmb_256
  ;CHECK: vpsubd (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbk_256
  ;CHECK: vpsubd (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xfa,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbkz_256
  ;CHECK: vpsubd (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32> @test_mask_add_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_add_epi32_rr_256
  ;CHECK: vpaddd %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xfe,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrk_256
  ;CHECK: vpaddd %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfe,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrkz_256
  ;CHECK: vpaddd %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfe,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rm_256
  ;CHECK: vpaddd (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xfe,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmk_256
  ;CHECK: vpaddd (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfe,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmkz_256
  ;CHECK: vpaddd (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfe,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rmb_256
  ;CHECK: vpaddd (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbk_256
  ;CHECK: vpaddd (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xfe,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbkz_256
  ;CHECK: vpaddd (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_and_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_and_epi32_rr_128
  ;CHECK: vpandd  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xdb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrk_128
  ;CHECK: vpandd  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdb,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrkz_128
  ;CHECK: vpandd  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rm_128
  ;CHECK: vpandd  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xdb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmk_128
  ;CHECK: vpandd  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdb,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmkz_128
  ;CHECK: vpandd  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rmb_128
  ;CHECK: vpandd  (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbk_128
  ;CHECK: vpandd  (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xdb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbkz_128
  ;CHECK: vpandd  (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_and_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_and_epi32_rr_256
  ;CHECK: vpandd  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xdb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrk_256
  ;CHECK: vpandd  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdb,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrkz_256
  ;CHECK: vpandd  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rm_256
  ;CHECK: vpandd  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xdb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmk_256
  ;CHECK: vpandd  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdb,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmkz_256
  ;CHECK: vpandd  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rmb_256
  ;CHECK: vpandd  (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbk_256
  ;CHECK: vpandd  (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xdb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbkz_256
  ;CHECK: vpandd  (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_or_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_or_epi32_rr_128
  ;CHECK: vpord   %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xeb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrk_128
  ;CHECK: vpord   %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xeb,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrkz_128
  ;CHECK: vpord   %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xeb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rm_128
  ;CHECK: vpord   (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xeb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmk_128
  ;CHECK: vpord   (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xeb,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmkz_128
  ;CHECK: vpord   (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xeb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rmb_128
  ;CHECK: vpord   (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbk_128
  ;CHECK: vpord   (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xeb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbkz_128
  ;CHECK: vpord   (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_or_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_or_epi32_rr_256
  ;CHECK: vpord   %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xeb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrk_256
  ;CHECK: vpord   %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xeb,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrkz_256
  ;CHECK: vpord   %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xeb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rm_256
  ;CHECK: vpord   (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xeb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmk_256
  ;CHECK: vpord   (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xeb,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmkz_256
  ;CHECK: vpord   (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xeb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rmb_256
  ;CHECK: vpord   (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbk_256
  ;CHECK: vpord   (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xeb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbkz_256
  ;CHECK: vpord   (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_xor_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rr_128
  ;CHECK: vpxord  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xef,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrk_128
  ;CHECK: vpxord  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xef,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrkz_128
  ;CHECK: vpxord  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xef,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rm_128
  ;CHECK: vpxord  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xef,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmk_128
  ;CHECK: vpxord  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xef,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmkz_128
  ;CHECK: vpxord  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xef,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmb_128
  ;CHECK: vpxord  (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xef,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbk_128
  ;CHECK: vpxord  (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xef,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbkz_128
  ;CHECK: vpxord  (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xef,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_xor_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rr_256
  ;CHECK: vpxord  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xef,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrk_256
  ;CHECK: vpxord  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xef,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrkz_256
  ;CHECK: vpxord  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xef,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rm_256
  ;CHECK: vpxord  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xef,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmk_256
  ;CHECK: vpxord  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xef,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmkz_256
  ;CHECK: vpxord  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xef,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmb_256
  ;CHECK: vpxord  (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xef,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbk_256
  ;CHECK: vpxord  (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xef,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbkz_256
  ;CHECK: vpxord  (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xef,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_andnot_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rr_128
  ;CHECK: vpandnd  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xdf,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrk_128
  ;CHECK: vpandnd  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdf,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrkz_128
  ;CHECK: vpandnd  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdf,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rm_128
  ;CHECK: vpandnd  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xdf,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmk_128
  ;CHECK: vpandnd  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdf,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmkz_128
  ;CHECK: vpandnd  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdf,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmb_128
  ;CHECK: vpandnd  (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbk_128
  ;CHECK: vpandnd  (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xdf,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbkz_128
  ;CHECK: vpandnd  (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_andnot_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rr_256
  ;CHECK: vpandnd  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xdf,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrk_256
  ;CHECK: vpandnd  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdf,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrkz_256
  ;CHECK: vpandnd  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdf,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rm_256
  ;CHECK: vpandnd  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xdf,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmk_256
  ;CHECK: vpandnd  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdf,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmkz_256
  ;CHECK: vpandnd  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdf,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmb_256
  ;CHECK: vpandnd  (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbk_256
  ;CHECK: vpandnd  (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xdf,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbkz_256
  ;CHECK: vpandnd  (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <2 x i64> @test_mask_andnot_epi64_rr_128(<2 x i64> %a, <2 x i64> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rr_128
  ;CHECK: vpandnq  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0xfd,0x08,0xdf,0xc1]
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 -1)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rrk_128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrk_128
  ;CHECK: vpandnq  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xdf,0xd1]
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rrkz_128(<2 x i64> %a, <2 x i64> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrkz_128
  ;CHECK: vpandnq  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xdf,0xc1]
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rm_128(<2 x i64> %a, <2 x i64>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rm_128
  ;CHECK: vpandnq  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0xfd,0x08,0xdf,0x07]
  %b = load <2 x i64>, <2 x i64>* %ptr_b
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 -1)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmk_128(<2 x i64> %a, <2 x i64>* %ptr_b, <2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmk_128
  ;CHECK: vpandnq  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xdf,0x0f]
  %b = load <2 x i64>, <2 x i64>* %ptr_b
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmkz_128(<2 x i64> %a, <2 x i64>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmkz_128
  ;CHECK: vpandnq  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xdf,0x07]
  %b = load <2 x i64>, <2 x i64>* %ptr_b
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmb_128(<2 x i64> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmb_128
  ;CHECK: vpandnq  (%rdi){1to2}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0xfd,0x18,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %b = shufflevector <2 x i64> %vecinit.i, <2 x i64> undef, <2 x i32> zeroinitializer
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 -1)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmbk_128(<2 x i64> %a, i64* %ptr_b, <2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbk_128
  ;CHECK: vpandnq  (%rdi){1to2}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x19,0xdf,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %b = shufflevector <2 x i64> %vecinit.i, <2 x i64> undef, <2 x i32> zeroinitializer
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmbkz_128(<2 x i64> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbkz_128
  ;CHECK: vpandnq  (%rdi){1to2}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x99,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %b = shufflevector <2 x i64> %vecinit.i, <2 x i64> undef, <2 x i32> zeroinitializer
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 %mask)
  ret <2 x i64> %res
}

declare <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <4 x i64> @test_mask_andnot_epi64_rr_256(<4 x i64> %a, <4 x i64> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rr_256
  ;CHECK: vpandnq  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0xfd,0x28,0xdf,0xc1]
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 -1)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rrk_256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrk_256
  ;CHECK: vpandnq  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xdf,0xd1]
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rrkz_256(<4 x i64> %a, <4 x i64> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrkz_256
  ;CHECK: vpandnq  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xdf,0xc1]
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rm_256(<4 x i64> %a, <4 x i64>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rm_256
  ;CHECK: vpandnq  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0xfd,0x28,0xdf,0x07]
  %b = load <4 x i64>, <4 x i64>* %ptr_b
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 -1)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmk_256(<4 x i64> %a, <4 x i64>* %ptr_b, <4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmk_256
  ;CHECK: vpandnq  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xdf,0x0f]
  %b = load <4 x i64>, <4 x i64>* %ptr_b
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmkz_256(<4 x i64> %a, <4 x i64>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmkz_256
  ;CHECK: vpandnq  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xdf,0x07]
  %b = load <4 x i64>, <4 x i64>* %ptr_b
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmb_256(<4 x i64> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmb_256
  ;CHECK: vpandnq  (%rdi){1to4}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0xfd,0x38,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %b = shufflevector <4 x i64> %vecinit.i, <4 x i64> undef, <4 x i32> zeroinitializer
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 -1)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmbk_256(<4 x i64> %a, i64* %ptr_b, <4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbk_256
  ;CHECK: vpandnq  (%rdi){1to4}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x39,0xdf,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %b = shufflevector <4 x i64> %vecinit.i, <4 x i64> undef, <4 x i32> zeroinitializer
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmbkz_256(<4 x i64> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbkz_256
  ;CHECK: vpandnq  (%rdi){1to4}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xb9,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %b = shufflevector <4 x i64> %vecinit.i, <4 x i64> undef, <4 x i32> zeroinitializer
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 %mask)
  ret <4 x i64> %res
}

declare <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define i8 @test_cmpps_256(<8 x float> %a, <8 x float> %b) {
 ;CHECK: vcmpleps  %ymm1, %ymm0, %k0  ## encoding: [0x62,0xf1,0x7c,0x28,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %a, <8 x float> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> , <8 x float> , i32, i8)

define i8 @test_cmpps_128(<4 x float> %a, <4 x float> %b) {
 ;CHECK: vcmpleps  %xmm1, %xmm0, %k0  ## encoding: [0x62,0xf1,0x7c,0x08,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %a, <4 x float> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> , <4 x float> , i32, i8)

define i8 @test_cmppd_256(<4 x double> %a, <4 x double> %b) {
 ;CHECK: vcmplepd  %ymm1, %ymm0, %k0  ## encoding: [0x62,0xf1,0xfd,0x28,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %a, <4 x double> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> , <4 x double> , i32, i8)

define i8 @test_cmppd_128(<2 x double> %a, <2 x double> %b) {
 ;CHECK: vcmplepd  %xmm1, %xmm0, %k0  ## encoding: [0x62,0xf1,0xfd,0x08,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %a, <2 x double> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> , <2 x double> , i32, i8)

define <8 x float> @test_mm512_maskz_add_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_ps_256
  ;CHECK: vaddps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_add_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_ps_256
  ;CHECK: vaddps %ymm1, %ymm0, %ymm2 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_add_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_add_ps_256
  ;CHECK: vaddps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_add_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_ps_128
  ;CHECK: vaddps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_add_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_ps_128
  ;CHECK: vaddps %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_add_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_add_ps_128
  ;CHECK: vaddps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_sub_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_sub_ps_256
  ;CHECK: vsubps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_sub_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_ps_256
  ;CHECK: vsubps %ymm1, %ymm0, %ymm2 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_sub_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_sub_ps_256
  ;CHECK: vsubps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_sub_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_sub_ps_128
  ;CHECK: vsubps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_sub_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_ps_128
  ;CHECK: vsubps %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_sub_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_sub_ps_128
  ;CHECK: vsubps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_mul_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_mul_ps_256
  ;CHECK: vmulps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_mul_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_mul_ps_256
  ;CHECK: vmulps %ymm1, %ymm0, %ymm2 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mul_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mul_ps_256
  ;CHECK: vmulps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_mul_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_mul_ps_128
  ;CHECK: vmulps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_mul_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_mul_ps_128
  ;CHECK: vmulps %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mul_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mul_ps_128
  ;CHECK: vmulps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_div_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_ps_256
  ;CHECK: vdivps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_div_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_ps_256
  ;CHECK: vdivps %ymm1, %ymm0, %ymm2 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_div_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_div_ps_256
  ;CHECK: vdivps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_div_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_ps_128
  ;CHECK: vdivps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_div_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_ps_128
  ;CHECK: vdivps %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_div_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_div_ps_128
  ;CHECK: vdivps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_max_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_max_ps_256
  ;CHECK: vmaxps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_max_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_max_ps_256
  ;CHECK: vmaxps %ymm1, %ymm0, %ymm2 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_max_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_max_ps_256
  ;CHECK: vmaxps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_max_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_max_ps_128
  ;CHECK: vmaxps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_max_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_max_ps_128
  ;CHECK: vmaxps %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_max_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_max_ps_128
  ;CHECK: vmaxps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_min_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_min_ps_256
  ;CHECK: vminps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_min_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_min_ps_256
  ;CHECK: vminps %ymm1, %ymm0, %ymm2 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_min_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_min_ps_256
  ;CHECK: vminps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_min_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_min_ps_128
  ;CHECK: vminps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_min_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_min_ps_128
  ;CHECK: vminps %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_min_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_min_ps_128
  ;CHECK: vminps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x double> @test_sqrt_pd_256(<4 x double> %a0, i8 %mask) {
  ; CHECK-LABEL: test_sqrt_pd_256
  ; CHECK: vsqrtpd
  %res = call <4 x double> @llvm.x86.avx512.mask.sqrt.pd.256(<4 x double> %a0,  <4 x double> zeroinitializer, i8 %mask)
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx512.mask.sqrt.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone

define <8 x float> @test_sqrt_ps_256(<8 x float> %a0, i8 %mask) {
  ; CHECK-LABEL: test_sqrt_ps_256
  ; CHECK: vsqrtps
  %res = call <8 x float> @llvm.x86.avx512.mask.sqrt.ps.256(<8 x float> %a0, <8 x float> zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

declare <8 x float> @llvm.x86.avx512.mask.sqrt.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone

define <4 x double> @test_getexp_pd_256(<4 x double> %a0) {
  ; CHECK-LABEL: test_getexp_pd_256
  ; CHECK: vgetexppd
  %res = call <4 x double> @llvm.x86.avx512.mask.getexp.pd.256(<4 x double> %a0,  <4 x double> zeroinitializer, i8 -1)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.getexp.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone

define <8 x float> @test_getexp_ps_256(<8 x float> %a0) {
  ; CHECK-LABEL: test_getexp_ps_256
  ; CHECK: vgetexpps
  %res = call <8 x float> @llvm.x86.avx512.mask.getexp.ps.256(<8 x float> %a0, <8 x float> zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.getexp.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone

declare <4 x i32> @llvm.x86.avx512.mask.pmaxs.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_d_128
; CHECK-NOT: call
; CHECK: vpmaxsd %xmm
; CHECK: {%k1}
define <4 x i32>@test_int_x86_avx512_mask_pmaxs_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmaxs.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2 ,i8 %mask)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmaxs.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %mask)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.pmaxs.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_d_256
; CHECK-NOT: call
; CHECK: vpmaxsd %ymm
; CHECK: {%k1}
define <8 x i32>@test_int_x86_avx512_mask_pmaxs_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmaxs.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmaxs.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.pmaxs.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_q_128
; CHECK-NOT: call
; CHECK: vpmaxsq %xmm
; CHECK: {%k1}
define <2 x i64>@test_int_x86_avx512_mask_pmaxs_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmaxs.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmaxs.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.pmaxs.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_q_256
; CHECK-NOT: call
; CHECK: vpmaxsq %ymm
; CHECK: {%k1}
define <4 x i64>@test_int_x86_avx512_mask_pmaxs_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask) {
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmaxs.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmaxs.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %mask)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.pmaxu.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_d_128
; CHECK-NOT: call
; CHECK: vpmaxud %xmm
; CHECK: {%k1}
define <4 x i32>@test_int_x86_avx512_mask_pmaxu_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2,i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmaxu.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %mask)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmaxu.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %mask)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.pmaxu.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_d_256
; CHECK-NOT: call
; CHECK: vpmaxud %ymm
; CHECK: {%k1}
define <8 x i32>@test_int_x86_avx512_mask_pmaxu_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmaxu.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmaxu.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.pmaxu.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_q_128
; CHECK-NOT: call
; CHECK: vpmaxuq %xmm
; CHECK: {%k1}
define <2 x i64>@test_int_x86_avx512_mask_pmaxu_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmaxu.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmaxu.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.pmaxu.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_q_256
; CHECK-NOT: call
; CHECK: vpmaxuq %ymm
; CHECK: {%k1}
define <4 x i64>@test_int_x86_avx512_mask_pmaxu_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask) {
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmaxu.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmaxu.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %mask)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.pmins.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_d_128
; CHECK-NOT: call
; CHECK: vpminsd %xmm
; CHECK: {%k1}
define <4 x i32>@test_int_x86_avx512_mask_pmins_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmins.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %mask)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmins.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %mask)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.pmins.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_d_256
; CHECK-NOT: call
; CHECK: vpminsd %ymm
; CHECK: {%k1}
define <8 x i32>@test_int_x86_avx512_mask_pmins_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmins.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmins.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.pmins.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_q_128
; CHECK-NOT: call
; CHECK: vpminsq %xmm
; CHECK: {%k1}
define <2 x i64>@test_int_x86_avx512_mask_pmins_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmins.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmins.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.pmins.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_q_256
; CHECK-NOT: call
; CHECK: vpminsq %ymm
; CHECK: {%k1}
define <4 x i64>@test_int_x86_avx512_mask_pmins_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask) {
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmins.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmins.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %mask)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.pminu.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_d_128
; CHECK-NOT: call
; CHECK: vpminud %xmm
; CHECK: {%k1}
define <4 x i32>@test_int_x86_avx512_mask_pminu_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.pminu.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %mask)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pminu.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %mask)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.pminu.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_d_256
; CHECK-NOT: call
; CHECK: vpminud %ymm
; CHECK: {%k1}
define <8 x i32>@test_int_x86_avx512_mask_pminu_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.pminu.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pminu.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.pminu.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_q_128
; CHECK-NOT: call
; CHECK: vpminuq %xmm
; CHECK: {%k1}
define <2 x i64>@test_int_x86_avx512_mask_pminu_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
  %res = call <2 x i64> @llvm.x86.avx512.mask.pminu.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pminu.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.pminu.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_q_256
; CHECK-NOT: call
; CHECK: vpminuq %ymm
; CHECK: {%k1}
define <4 x i64>@test_int_x86_avx512_mask_pminu_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask) {
  %res = call <4 x i64> @llvm.x86.avx512.mask.pminu.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %mask)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pminu.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %mask)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.vpermt2var.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermt2var_d_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermt2d %xmm{{.*}}{%k1}
; CHECK-NOT: {z}
define <4 x i32>@test_int_x86_avx512_mask_vpermt2var_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpermt2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpermt2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_d_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermt2d %xmm{{.*}}{%k1} {z}
define <4 x i32>@test_int_x86_avx512_maskz_vpermt2var_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
  %res = call <4 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.vpermt2var.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermt2var_d_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermt2d %ymm{{.*}}{%k1}
; CHECK-NOT: {z}
define <8 x i32>@test_int_x86_avx512_mask_vpermt2var_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpermt2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpermt2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_d_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermt2d {{.*}}{%k1} {z}
define <8 x i32>@test_int_x86_avx512_maskz_vpermt2var_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
  %res = call <8 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.vpermi2var.pd.128(<2 x double>, <2 x i64>, <2 x double>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_pd_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermi2pd %xmm{{.*}}{%k1}
define <2 x double>@test_int_x86_avx512_mask_vpermi2var_pd_128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2, i8 %x3) {
  %res = call <2 x double> @llvm.x86.avx512.mask.vpermi2var.pd.128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.vpermi2var.pd.128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.vpermi2var.pd.256(<4 x double>, <4 x i64>, <4 x double>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_pd_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermi2pd %ymm{{.*}}{%k1}
define <4 x double>@test_int_x86_avx512_mask_vpermi2var_pd_256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 %x3) {
  %res = call <4 x double> @llvm.x86.avx512.mask.vpermi2var.pd.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.vpermi2var.pd.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.vpermi2var.ps.128(<4 x float>, <4 x i32>, <4 x float>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_ps_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermi2ps %xmm{{.*}}{%k1}
define <4 x float>@test_int_x86_avx512_mask_vpermi2var_ps_128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2, i8 %x3) {
  %res = call <4 x float> @llvm.x86.avx512.mask.vpermi2var.ps.128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vpermi2var.ps.128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.vpermi2var.ps.256(<8 x float>, <8 x i32>, <8 x float>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_ps_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpermi2ps %ymm{{.*}}{%k1}
define <8 x float>@test_int_x86_avx512_mask_vpermi2var_ps_256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 %x3) {
  %res = call <8 x float> @llvm.x86.avx512.mask.vpermi2var.ps.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.vpermi2var.ps.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.pabs.q.128(<2 x i64>, <2 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_q_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpabsq{{.*}}{%k1}
define <2 x i64>@test_int_x86_avx512_mask_pabs_q_128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2) {
  %res = call <2 x i64> @llvm.x86.avx512.mask.pabs.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pabs.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.pabs.q.256(<4 x i64>, <4 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_q_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpabsq{{.*}}{%k1}
define <4 x i64>@test_int_x86_avx512_mask_pabs_q_256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2) {
  %res = call <4 x i64> @llvm.x86.avx512.mask.pabs.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pabs.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.pabs.d.128(<4 x i32>, <4 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_d_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpabsd{{.*}}{%k1}
define <4 x i32>@test_int_x86_avx512_mask_pabs_d_128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.pabs.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pabs.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.pabs.d.256(<8 x i32>, <8 x i32>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_d_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vpabsd{{.*}}{%k1}
define <8 x i32>@test_int_x86_avx512_mask_pabs_d_256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.pabs.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pabs.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}


declare <2 x double> @llvm.x86.avx512.mask.scalef.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_pd_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vscalefpd{{.*}}{%k1}
define <2 x double>@test_int_x86_avx512_mask_scalef_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
  %res = call <2 x double> @llvm.x86.avx512.mask.scalef.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.scalef.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.scalef.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_pd_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vscalefpd{{.*}}{%k1}
define <4 x double>@test_int_x86_avx512_mask_scalef_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
  %res = call <4 x double> @llvm.x86.avx512.mask.scalef.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.scalef.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.scalef.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)
; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_ps_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vscalefps{{.*}}{%k1}
define <4 x float>@test_int_x86_avx512_mask_scalef_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
  %res = call <4 x float> @llvm.x86.avx512.mask.scalef.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.scalef.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.scalef.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)
; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_ps_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vscalefps{{.*}}{%k1}
define <8 x float>@test_int_x86_avx512_mask_scalef_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
  %res = call <8 x float> @llvm.x86.avx512.mask.scalef.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.scalef.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.unpckh.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_unpckh_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckh_pd_128:
; CHECK:         vunpckhpd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[1],k1[1]
; CHECK-NEXT:    vunpckhpd %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0xfd,0x08,0x15,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[1],xmm1[1]
  %res = call <2 x double> @llvm.x86.avx512.mask.unpckh.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.unpckh.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.unpckh.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_unpckh_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckh_pd_256:
; CHECK:         vunpckhpd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[1],k1[1],ymm2[3],k1[3]
; CHECK-NEXT:    vunpckhpd %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0xfd,0x28,0x15,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[1],ymm1[1],ymm0[3],ymm1[3]
  %res = call <4 x double> @llvm.x86.avx512.mask.unpckh.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.unpckh.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.unpckh.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_unpckh_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckh_ps_128:
; CHECK:         vunpckhps %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[2],k1[2],xmm2[3],k1[3]
; CHECK-NEXT:    vunpckhps %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7c,0x08,0x15,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[2],xmm1[2],xmm0[3],xmm1[3]
  %res = call <4 x float> @llvm.x86.avx512.mask.unpckh.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.unpckh.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.unpckh.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_unpckh_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckh_ps_256:
; CHECK:       ## BB#0:
; CHECK:         vunpckhps %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[2],k1[2],ymm2[3],k1[3],ymm2[6],k1[6],ymm2[7],k1[7]
; CHECK-NEXT:    vunpckhps %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7c,0x28,0x15,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[6],ymm1[6],ymm0[7],ymm1[7]
  %res = call <8 x float> @llvm.x86.avx512.mask.unpckh.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.unpckh.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.unpckl.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_unpckl_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckl_pd_128:
; CHECK:         vunpcklpd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[0],k1[0]
; CHECK-NEXT:    vunpcklpd %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0xfd,0x08,0x14,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[0],xmm1[0]
  %res = call <2 x double> @llvm.x86.avx512.mask.unpckl.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.unpckl.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.unpckl.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_unpckl_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckl_pd_256:
; CHECK:         vunpcklpd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[0],k1[0],ymm2[2],k1[2]
; CHECK-NEXT:    vunpcklpd %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0xfd,0x28,0x14,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[0],ymm1[0],ymm0[2],ymm1[2]
  %res = call <4 x double> @llvm.x86.avx512.mask.unpckl.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.unpckl.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.unpckl.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_unpckl_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckl_ps_128:
; CHECK:         vunpcklps %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[0],k1[0],xmm2[1],k1[1]
; CHECK-NEXT:    vunpcklps %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7c,0x08,0x14,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
  %res = call <4 x float> @llvm.x86.avx512.mask.unpckl.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.unpckl.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.unpckl.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_unpckl_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckl_ps_256:
; CHECK:         vunpcklps %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[0],k1[0],ymm2[1],k1[1],ymm2[4],k1[4],ymm2[5],k1[5]
; CHECK-NEXT:    vunpcklps %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7c,0x28,0x14,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[4],ymm1[4],ymm0[5],ymm1[5]
  %res = call <8 x float> @llvm.x86.avx512.mask.unpckl.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.unpckl.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.punpckhd.q.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_punpckhd_q_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhd_q_128:
; CHECK:         vpunpckhdq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[2],k1[2],xmm2[3],k1[3]
; CHECK-NEXT:    vpunpckhdq %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7d,0x08,0x6a,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[2],xmm1[2],xmm0[3],xmm1[3]
  %res = call <4 x i32> @llvm.x86.avx512.mask.punpckhd.q.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.punpckhd.q.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.punpckld.q.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_punpckld_q_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckld_q_128:
; CHECK:         vpunpckldq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[0],k1[0],xmm2[1],k1[1]
; CHECK-NEXT:    vpunpckldq %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0x7d,0x08,0x62,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.punpckld.q.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.punpckld.q.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.punpckhd.q.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_punpckhd_q_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhd_q_256:
; CHECK:       ## BB#0:
; CHECK:         vpunpckhdq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[2],k1[2],ymm2[3],k1[3],ymm2[6],k1[6],ymm2[7],k1[7]
; CHECK-NEXT:    vpunpckhdq %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7d,0x28,0x6a,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[6],ymm1[6],ymm0[7],ymm1[7]
  %res = call <8 x i32> @llvm.x86.avx512.mask.punpckhd.q.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.punpckhd.q.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.punpckld.q.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_punpckld_q_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckld_q_256:
; CHECK:         vpunpckldq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[0],k1[0],ymm2[1],k1[1],ymm2[4],k1[4],ymm2[5],k1[5]
; CHECK-NEXT:    vpunpckldq %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0x7d,0x28,0x62,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[4],ymm1[4],ymm0[5],ymm1[5]
  %res = call <8 x i32> @llvm.x86.avx512.mask.punpckld.q.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.punpckld.q.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.punpckhqd.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_punpckhqd_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhqd_q_128:
; CHECK:         vpunpckhqdq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[1],k1[1]
; CHECK-NEXT:    vpunpckhqdq %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0xfd,0x08,0x6d,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[1],xmm1[1]
  %res = call <2 x i64> @llvm.x86.avx512.mask.punpckhqd.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.punpckhqd.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.punpcklqd.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_punpcklqd_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklqd_q_128:
; CHECK:         vpunpcklqdq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[0],k1[0]
; CHECK-NEXT:    vpunpcklqdq %xmm1, %xmm0, %xmm0 ## encoding: [0x62,0xf1,0xfd,0x08,0x6c,0xc1]
; CHECK-NEXT:    ## xmm0 = xmm0[0],xmm1[0]
  %res = call <2 x i64> @llvm.x86.avx512.mask.punpcklqd.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.punpcklqd.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.punpcklqd.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_punpcklqd_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklqd_q_256:
; CHECK:         vpunpcklqdq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[0],k1[0],ymm2[2],k1[2]
; CHECK-NEXT:    vpunpcklqdq %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0xfd,0x28,0x6c,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[0],ymm1[0],ymm0[2],ymm1[2]
  %res = call <4 x i64> @llvm.x86.avx512.mask.punpcklqd.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.punpcklqd.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.punpckhqd.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_punpckhqd_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhqd_q_256:
; CHECK:         vpunpckhqdq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[1],k1[1],ymm2[3],k1[3]
; CHECK-NEXT:    vpunpckhqdq %ymm1, %ymm0, %ymm0 ## encoding: [0x62,0xf1,0xfd,0x28,0x6d,0xc1]
; CHECK-NEXT:    ## ymm0 = ymm0[1],ymm1[1],ymm0[3],ymm1[3]
  %res = call <4 x i64> @llvm.x86.avx512.mask.punpckhqd.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.punpckhqd.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.qb.128(<2 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmov_qb_128(<2 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qb_128:
; CHECK:       vpmovqb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.128(<2 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.128(<2 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.128(<2 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qb.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qb_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qb_mem_128:
; CHECK:  vpmovqb %xmm0, (%rdi)
; CHECK:  vpmovqb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qb.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qb.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.128(<2 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_qb_128(<2 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qb_128:
; CHECK:       vpmovsqb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.128(<2 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.128(<2 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.128(<2 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qb.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qb_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qb_mem_128:
; CHECK:  vpmovsqb %xmm0, (%rdi)
; CHECK:  vpmovsqb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qb.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qb.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.128(<2 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_qb_128(<2 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qb_128:
; CHECK:       vpmovusqb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.128(<2 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.128(<2 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.128(<2 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qb.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qb_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qb_mem_128:
; CHECK:  vpmovusqb %xmm0, (%rdi)
; CHECK:  vpmovusqb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qb.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qb.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.qb.256(<4 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmov_qb_256(<4 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qb_256:
; CHECK:       vpmovqb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.256(<4 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.256(<4 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.256(<4 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qb.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qb_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qb_mem_256:
; CHECK:  vpmovqb %ymm0, (%rdi)
; CHECK:  vpmovqb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qb.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qb.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.256(<4 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_qb_256(<4 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qb_256:
; CHECK:       vpmovsqb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.256(<4 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.256(<4 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.256(<4 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qb.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qb_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qb_mem_256:
; CHECK:  vpmovsqb %ymm0, (%rdi)
; CHECK:  vpmovsqb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qb.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qb.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.256(<4 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_qb_256(<4 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qb_256:
; CHECK:       vpmovusqb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.256(<4 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.256(<4 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.256(<4 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qb.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qb_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qb_mem_256:
; CHECK:  vpmovusqb %ymm0, (%rdi)
; CHECK:  vpmovusqb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qb.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qb.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmov.qw.128(<2 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmov_qw_128(<2 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qw_128:
; CHECK:       vpmovqw %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqw %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqw %xmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.128(<2 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.128(<2 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.128(<2 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qw.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qw_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qw_mem_128:
; CHECK:  vpmovqw %xmm0, (%rdi)
; CHECK:  vpmovqw %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qw.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qw.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.128(<2 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovs_qw_128(<2 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qw_128:
; CHECK:       vpmovsqw %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqw %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqw %xmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.128(<2 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.128(<2 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.128(<2 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qw.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qw_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qw_mem_128:
; CHECK:  vpmovsqw %xmm0, (%rdi)
; CHECK:  vpmovsqw %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qw.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qw.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.128(<2 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovus_qw_128(<2 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qw_128:
; CHECK:       vpmovusqw %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqw %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqw %xmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.128(<2 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.128(<2 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.128(<2 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qw.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qw_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qw_mem_128:
; CHECK:  vpmovusqw %xmm0, (%rdi)
; CHECK:  vpmovusqw %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qw.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qw.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmov.qw.256(<4 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmov_qw_256(<4 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qw_256:
; CHECK:       vpmovqw %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqw %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqw %ymm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.256(<4 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.256(<4 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.256(<4 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qw.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qw_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qw_mem_256:
; CHECK:  vpmovqw %ymm0, (%rdi)
; CHECK:  vpmovqw %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qw.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qw.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.256(<4 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovs_qw_256(<4 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qw_256:
; CHECK:       vpmovsqw %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqw %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqw %ymm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.256(<4 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.256(<4 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.256(<4 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qw.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qw_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qw_mem_256:
; CHECK:  vpmovsqw %ymm0, (%rdi)
; CHECK:  vpmovsqw %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qw.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qw.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.256(<4 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovus_qw_256(<4 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qw_256:
; CHECK:       vpmovusqw %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqw %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqw %ymm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.256(<4 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.256(<4 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.256(<4 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qw.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qw_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qw_mem_256:
; CHECK:  vpmovusqw %ymm0, (%rdi)
; CHECK:  vpmovusqw %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qw.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qw.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <4 x i32> @llvm.x86.avx512.mask.pmov.qd.128(<2 x i64>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmov_qd_128(<2 x i64> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qd_128:
; CHECK:       vpmovqd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqd %xmm0, %xmm0
    %res0 = call <4 x i32> @llvm.x86.avx512.mask.pmov.qd.128(<2 x i64> %x0, <4 x i32> %x1, i8 -1)
    %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmov.qd.128(<2 x i64> %x0, <4 x i32> %x1, i8 %x2)
    %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmov.qd.128(<2 x i64> %x0, <4 x i32> zeroinitializer, i8 %x2)
    %res3 = add <4 x i32> %res0, %res1
    %res4 = add <4 x i32> %res3, %res2
    ret <4 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qd.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qd_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qd_mem_128:
; CHECK:  vpmovqd %xmm0, (%rdi)
; CHECK:  vpmovqd %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qd.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qd.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.128(<2 x i64>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovs_qd_128(<2 x i64> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qd_128:
; CHECK:       vpmovsqd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqd %xmm0, %xmm0
    %res0 = call <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.128(<2 x i64> %x0, <4 x i32> %x1, i8 -1)
    %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.128(<2 x i64> %x0, <4 x i32> %x1, i8 %x2)
    %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.128(<2 x i64> %x0, <4 x i32> zeroinitializer, i8 %x2)
    %res3 = add <4 x i32> %res0, %res1
    %res4 = add <4 x i32> %res3, %res2
    ret <4 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qd.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qd_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qd_mem_128:
; CHECK:  vpmovsqd %xmm0, (%rdi)
; CHECK:  vpmovsqd %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qd.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qd.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.128(<2 x i64>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovus_qd_128(<2 x i64> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qd_128:
; CHECK:       vpmovusqd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqd %xmm0, %xmm0
    %res0 = call <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.128(<2 x i64> %x0, <4 x i32> %x1, i8 -1)
    %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.128(<2 x i64> %x0, <4 x i32> %x1, i8 %x2)
    %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.128(<2 x i64> %x0, <4 x i32> zeroinitializer, i8 %x2)
    %res3 = add <4 x i32> %res0, %res1
    %res4 = add <4 x i32> %res3, %res2
    ret <4 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qd.mem.128(i8* %ptr, <2 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qd_mem_128(i8* %ptr, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qd_mem_128:
; CHECK:  vpmovusqd %xmm0, (%rdi)
; CHECK:  vpmovusqd %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qd.mem.128(i8* %ptr, <2 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qd.mem.128(i8* %ptr, <2 x i64> %x1, i8 %x2)
    ret void
}

declare <4 x i32> @llvm.x86.avx512.mask.pmov.qd.256(<4 x i64>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmov_qd_256(<4 x i64> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qd_256:
; CHECK:       vpmovqd %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqd %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqd %ymm0, %xmm0
    %res0 = call <4 x i32> @llvm.x86.avx512.mask.pmov.qd.256(<4 x i64> %x0, <4 x i32> %x1, i8 -1)
    %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmov.qd.256(<4 x i64> %x0, <4 x i32> %x1, i8 %x2)
    %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmov.qd.256(<4 x i64> %x0, <4 x i32> zeroinitializer, i8 %x2)
    %res3 = add <4 x i32> %res0, %res1
    %res4 = add <4 x i32> %res3, %res2
    ret <4 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qd.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qd_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qd_mem_256:
; CHECK:  vpmovqd %ymm0, (%rdi)
; CHECK:  vpmovqd %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qd.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qd.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.256(<4 x i64>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovs_qd_256(<4 x i64> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qd_256:
; CHECK:       vpmovsqd %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqd %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqd %ymm0, %xmm0
    %res0 = call <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.256(<4 x i64> %x0, <4 x i32> %x1, i8 -1)
    %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.256(<4 x i64> %x0, <4 x i32> %x1, i8 %x2)
    %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.256(<4 x i64> %x0, <4 x i32> zeroinitializer, i8 %x2)
    %res3 = add <4 x i32> %res0, %res1
    %res4 = add <4 x i32> %res3, %res2
    ret <4 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qd.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qd_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qd_mem_256:
; CHECK:  vpmovsqd %ymm0, (%rdi)
; CHECK:  vpmovsqd %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qd.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qd.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.256(<4 x i64>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovus_qd_256(<4 x i64> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qd_256:
; CHECK:       vpmovusqd %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqd %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqd %ymm0, %xmm0
    %res0 = call <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.256(<4 x i64> %x0, <4 x i32> %x1, i8 -1)
    %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.256(<4 x i64> %x0, <4 x i32> %x1, i8 %x2)
    %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.256(<4 x i64> %x0, <4 x i32> zeroinitializer, i8 %x2)
    %res3 = add <4 x i32> %res0, %res1
    %res4 = add <4 x i32> %res3, %res2
    ret <4 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qd.mem.256(i8* %ptr, <4 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qd_mem_256(i8* %ptr, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qd_mem_256:
; CHECK:  vpmovusqd %ymm0, (%rdi)
; CHECK:  vpmovusqd %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qd.mem.256(i8* %ptr, <4 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qd.mem.256(i8* %ptr, <4 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.db.128(<4 x i32>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmov_db_128(<4 x i32> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_db_128:
; CHECK:       vpmovdb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovdb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovdb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.128(<4 x i32> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.128(<4 x i32> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.128(<4 x i32> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.db.mem.128(i8* %ptr, <4 x i32>, i8)

define void @test_int_x86_avx512_mask_pmov_db_mem_128(i8* %ptr, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_db_mem_128:
; CHECK:  vpmovdb %xmm0, (%rdi)
; CHECK:  vpmovdb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.db.mem.128(i8* %ptr, <4 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.db.mem.128(i8* %ptr, <4 x i32> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.db.128(<4 x i32>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_db_128(<4 x i32> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_db_128:
; CHECK:       vpmovsdb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsdb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsdb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.128(<4 x i32> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.128(<4 x i32> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.128(<4 x i32> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.db.mem.128(i8* %ptr, <4 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovs_db_mem_128(i8* %ptr, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_db_mem_128:
; CHECK:  vpmovsdb %xmm0, (%rdi)
; CHECK:  vpmovsdb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.db.mem.128(i8* %ptr, <4 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.db.mem.128(i8* %ptr, <4 x i32> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.db.128(<4 x i32>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_db_128(<4 x i32> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_db_128:
; CHECK:       vpmovusdb %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusdb %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusdb %xmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.128(<4 x i32> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.128(<4 x i32> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.128(<4 x i32> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.db.mem.128(i8* %ptr, <4 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovus_db_mem_128(i8* %ptr, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_db_mem_128:
; CHECK:  vpmovusdb %xmm0, (%rdi)
; CHECK:  vpmovusdb %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.db.mem.128(i8* %ptr, <4 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.db.mem.128(i8* %ptr, <4 x i32> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.db.256(<8 x i32>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmov_db_256(<8 x i32> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_db_256:
; CHECK:       vpmovdb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovdb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovdb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.256(<8 x i32> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.256(<8 x i32> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.256(<8 x i32> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.db.mem.256(i8* %ptr, <8 x i32>, i8)

define void @test_int_x86_avx512_mask_pmov_db_mem_256(i8* %ptr, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_db_mem_256:
; CHECK:  vpmovdb %ymm0, (%rdi)
; CHECK:  vpmovdb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.db.mem.256(i8* %ptr, <8 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.db.mem.256(i8* %ptr, <8 x i32> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.db.256(<8 x i32>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_db_256(<8 x i32> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_db_256:
; CHECK:       vpmovsdb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsdb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsdb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.256(<8 x i32> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.256(<8 x i32> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.256(<8 x i32> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.db.mem.256(i8* %ptr, <8 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovs_db_mem_256(i8* %ptr, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_db_mem_256:
; CHECK:  vpmovsdb %ymm0, (%rdi)
; CHECK:  vpmovsdb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.db.mem.256(i8* %ptr, <8 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.db.mem.256(i8* %ptr, <8 x i32> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.db.256(<8 x i32>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_db_256(<8 x i32> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_db_256:
; CHECK:       vpmovusdb %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusdb %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusdb %ymm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.256(<8 x i32> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.256(<8 x i32> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.256(<8 x i32> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.db.mem.256(i8* %ptr, <8 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovus_db_mem_256(i8* %ptr, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_db_mem_256:
; CHECK:  vpmovusdb %ymm0, (%rdi)
; CHECK:  vpmovusdb %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.db.mem.256(i8* %ptr, <8 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.db.mem.256(i8* %ptr, <8 x i32> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmov.dw.128(<4 x i32>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmov_dw_128(<4 x i32> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_dw_128:
; CHECK:       vpmovdw %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovdw %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovdw %xmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmov.dw.128(<4 x i32> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmov.dw.128(<4 x i32> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmov.dw.128(<4 x i32> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmov.dw.mem.128(i8* %ptr, <4 x i32>, i8)

define void @test_int_x86_avx512_mask_pmov_dw_mem_128(i8* %ptr, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_dw_mem_128:
; CHECK:  vpmovdw %xmm0, (%rdi)
; CHECK:  vpmovdw %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.dw.mem.128(i8* %ptr, <4 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.dw.mem.128(i8* %ptr, <4 x i32> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.128(<4 x i32>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovs_dw_128(<4 x i32> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_dw_128:
; CHECK:       vpmovsdw %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsdw %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsdw %xmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.128(<4 x i32> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.128(<4 x i32> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.128(<4 x i32> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.dw.mem.128(i8* %ptr, <4 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovs_dw_mem_128(i8* %ptr, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_dw_mem_128:
; CHECK:  vpmovsdw %xmm0, (%rdi)
; CHECK:  vpmovsdw %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.dw.mem.128(i8* %ptr, <4 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.dw.mem.128(i8* %ptr, <4 x i32> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.128(<4 x i32>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovus_dw_128(<4 x i32> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_dw_128:
; CHECK:       vpmovusdw %xmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusdw %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusdw %xmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.128(<4 x i32> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.128(<4 x i32> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.128(<4 x i32> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.dw.mem.128(i8* %ptr, <4 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovus_dw_mem_128(i8* %ptr, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_dw_mem_128:
; CHECK:  vpmovusdw %xmm0, (%rdi)
; CHECK:  vpmovusdw %xmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.dw.mem.128(i8* %ptr, <4 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.dw.mem.128(i8* %ptr, <4 x i32> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmov.dw.256(<8 x i32>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmov_dw_256(<8 x i32> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_dw_256:
; CHECK:       vpmovdw %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovdw %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovdw %ymm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmov.dw.256(<8 x i32> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmov.dw.256(<8 x i32> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmov.dw.256(<8 x i32> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmov.dw.mem.256(i8* %ptr, <8 x i32>, i8)

define void @test_int_x86_avx512_mask_pmov_dw_mem_256(i8* %ptr, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_dw_mem_256:
; CHECK:  vpmovdw %ymm0, (%rdi)
; CHECK:  vpmovdw %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.dw.mem.256(i8* %ptr, <8 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.dw.mem.256(i8* %ptr, <8 x i32> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.256(<8 x i32>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovs_dw_256(<8 x i32> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_dw_256:
; CHECK:       vpmovsdw %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsdw %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsdw %ymm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.256(<8 x i32> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.256(<8 x i32> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.256(<8 x i32> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.dw.mem.256(i8* %ptr, <8 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovs_dw_mem_256(i8* %ptr, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_dw_mem_256:
; CHECK:  vpmovsdw %ymm0, (%rdi)
; CHECK:  vpmovsdw %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.dw.mem.256(i8* %ptr, <8 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.dw.mem.256(i8* %ptr, <8 x i32> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.256(<8 x i32>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovus_dw_256(<8 x i32> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_dw_256:
; CHECK:       vpmovusdw %ymm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusdw %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusdw %ymm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.256(<8 x i32> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.256(<8 x i32> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.256(<8 x i32> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.dw.mem.256(i8* %ptr, <8 x i32>, i8)

define void @test_int_x86_avx512_mask_pmovus_dw_mem_256(i8* %ptr, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_dw_mem_256:
; CHECK:  vpmovusdw %ymm0, (%rdi)
; CHECK:  vpmovusdw %ymm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.dw.mem.256(i8* %ptr, <8 x i32> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.dw.mem.256(i8* %ptr, <8 x i32> %x1, i8 %x2)
    ret void
}

declare <2 x double> @llvm.x86.avx512.mask.cvtdq2pd.128(<4 x i32>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_cvt_dq2pd_128(<4 x i32> %x0, <2 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_dq2pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtdq2pd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtdq2pd %xmm0, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.cvtdq2pd.128(<4 x i32> %x0, <2 x double> %x1, i8 %x2)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.cvtdq2pd.128(<4 x i32> %x0, <2 x double> %x1, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.cvtdq2pd.256(<4 x i32>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_cvt_dq2pd_256(<4 x i32> %x0, <4 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_dq2pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtdq2pd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtdq2pd %xmm0, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.cvtdq2pd.256(<4 x i32> %x0, <4 x double> %x1, i8 %x2)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.cvtdq2pd.256(<4 x i32> %x0, <4 x double> %x1, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.cvtdq2ps.128(<4 x i32>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_cvt_dq2ps_128(<4 x i32> %x0, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_dq2ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtdq2ps %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtdq2ps %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.cvtdq2ps.128(<4 x i32> %x0, <4 x float> %x1, i8 %x2)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.cvtdq2ps.128(<4 x i32> %x0, <4 x float> %x1, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.cvtdq2ps.256(<8 x i32>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_cvt_dq2ps_256(<8 x i32> %x0, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_dq2ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtdq2ps %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtdq2ps %ymm0, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.cvtdq2ps.256(<8 x i32> %x0, <8 x float> %x1, i8 %x2)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.cvtdq2ps.256(<8 x i32> %x0, <8 x float> %x1, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvtpd2dq.128(<2 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvt_pd2dq_128(<2 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2dq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtpd2dq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtpd2dq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2dq.128(<2 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2dq.128(<2 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvtpd2dq.256(<4 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvt_pd2dq_256(<4 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2dq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtpd2dq %ymm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtpd2dq %ymm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2dq.256(<4 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2dq.256(<4 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.cvtpd2ps.256(<4 x double>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_cvt_pd2ps_256(<4 x double> %x0, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtpd2ps %ymm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtpd2ps %ymm0, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.cvtpd2ps.256(<4 x double> %x0, <4 x float> %x1, i8 %x2)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.cvtpd2ps.256(<4 x double> %x0, <4 x float> %x1, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.cvtpd2ps(<2 x double>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_cvt_pd2ps(<2 x double> %x0, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2ps:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtpd2ps %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtpd2ps %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.cvtpd2ps(<2 x double> %x0, <4 x float> %x1, i8 %x2)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.cvtpd2ps(<2 x double> %x0, <4 x float> %x1, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvtpd2udq.128(<2 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvt_pd2udq_128(<2 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2udq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtpd2udq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtpd2udq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2udq.128(<2 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2udq.128(<2 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvtpd2udq.256(<4 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvt_pd2udq_256(<4 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2udq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtpd2udq %ymm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtpd2udq %ymm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2udq.256(<4 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvtpd2udq.256(<4 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvtps2dq.128(<4 x float>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvt_ps2dq_128(<4 x float> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2dq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2dq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtps2dq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvtps2dq.128(<4 x float> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvtps2dq.128(<4 x float> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvtps2dq.256(<8 x float>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_cvt_ps2dq_256(<8 x float> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2dq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2dq %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtps2dq %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvtps2dq.256(<8 x float> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvtps2dq.256(<8 x float> %x0, <8 x i32> %x1, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.cvtps2pd.128(<4 x float>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_cvt_ps2pd_128(<4 x float> %x0, <2 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2pd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtps2pd %xmm0, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.cvtps2pd.128(<4 x float> %x0, <2 x double> %x1, i8 %x2)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.cvtps2pd.128(<4 x float> %x0, <2 x double> %x1, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.cvtps2pd.256(<4 x float>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_cvt_ps2pd_256(<4 x float> %x0, <4 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2pd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtps2pd %xmm0, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.cvtps2pd.256(<4 x float> %x0, <4 x double> %x1, i8 %x2)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.cvtps2pd.256(<4 x float> %x0, <4 x double> %x1, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvtps2udq.128(<4 x float>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvt_ps2udq_128(<4 x float> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2udq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2udq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtps2udq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvtps2udq.128(<4 x float> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvtps2udq.128(<4 x float> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvtps2udq.256(<8 x float>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_cvt_ps2udq_256(<8 x float> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2udq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2udq %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtps2udq %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvtps2udq.256(<8 x float> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvtps2udq.256(<8 x float> %x0, <8 x i32> %x1, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvttpd2dq.128(<2 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvtt_pd2dq_128(<2 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_pd2dq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttpd2dq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvttpd2dq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2dq.128(<2 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2dq.128(<2 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvttpd2dq.256(<4 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvtt_pd2dq_256(<4 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_pd2dq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttpd2dq %ymm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvttpd2dq %ymm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2dq.256(<4 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2dq.256(<4 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvttpd2udq.128(<2 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvtt_pd2udq_128(<2 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_pd2udq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttpd2udq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvttpd2udq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2udq.128(<2 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2udq.128(<2 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvttpd2udq.256(<4 x double>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvtt_pd2udq_256(<4 x double> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_pd2udq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttpd2udq %ymm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvttpd2udq %ymm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2udq.256(<4 x double> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvttpd2udq.256(<4 x double> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvttps2dq.128(<4 x float>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvtt_ps2dq_128(<4 x float> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_ps2dq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttps2dq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvttps2dq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvttps2dq.128(<4 x float> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvttps2dq.128(<4 x float> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvttps2dq.256(<8 x float>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_cvtt_ps2dq_256(<8 x float> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_ps2dq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttps2dq %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvttps2dq %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvttps2dq.256(<8 x float> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvttps2dq.256(<8 x float> %x0, <8 x i32> %x1, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.cvttps2udq.128(<4 x float>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_cvtt_ps2udq_128(<4 x float> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_ps2udq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttps2udq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvttps2udq %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.cvttps2udq.128(<4 x float> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.cvttps2udq.128(<4 x float> %x0, <4 x i32> %x1, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvttps2udq.256(<8 x float>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_cvtt_ps2udq_256(<8 x float> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_ps2udq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttps2udq %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvttps2udq %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvttps2udq.256(<8 x float> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvttps2udq.256(<8 x float> %x0, <8 x i32> %x1, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.cvtudq2pd.128(<4 x i32>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_cvt_udq2pd_128(<4 x i32> %x0, <2 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_udq2pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtudq2pd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtudq2pd %xmm0, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.cvtudq2pd.128(<4 x i32> %x0, <2 x double> %x1, i8 %x2)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.cvtudq2pd.128(<4 x i32> %x0, <2 x double> %x1, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.cvtudq2pd.256(<4 x i32>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_cvt_udq2pd_256(<4 x i32> %x0, <4 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_udq2pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtudq2pd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtudq2pd %xmm0, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.cvtudq2pd.256(<4 x i32> %x0, <4 x double> %x1, i8 %x2)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.cvtudq2pd.256(<4 x i32> %x0, <4 x double> %x1, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.cvtudq2ps.128(<4 x i32>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_cvt_udq2ps_128(<4 x i32> %x0, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_udq2ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtudq2ps %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vcvtudq2ps %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.cvtudq2ps.128(<4 x i32> %x0, <4 x float> %x1, i8 %x2)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.cvtudq2ps.128(<4 x i32> %x0, <4 x float> %x1, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.cvtudq2ps.256(<8 x i32>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_cvt_udq2ps_256(<8 x i32> %x0, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_udq2ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtudq2ps %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtudq2ps %ymm0, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.cvtudq2ps.256(<8 x i32> %x0, <8 x float> %x1, i8 %x2)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.cvtudq2ps.256(<8 x i32> %x0, <8 x float> %x1, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.rndscale.pd.128(<2 x double>, i32, <2 x double>, i8)
; CHECK-LABEL: @test_int_x86_avx512_mask_rndscale_pd_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vrndscalepd {{.*}}{%k1}
; CHECK: vrndscalepd
define <2 x double>@test_int_x86_avx512_mask_rndscale_pd_128(<2 x double> %x0, <2 x double> %x2, i8 %x3) {
  %res = call <2 x double> @llvm.x86.avx512.mask.rndscale.pd.128(<2 x double> %x0, i32 4, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.rndscale.pd.128(<2 x double> %x0, i32 88, <2 x double> %x2, i8 -1)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.rndscale.pd.256(<4 x double>, i32, <4 x double>, i8)
; CHECK-LABEL: @test_int_x86_avx512_mask_rndscale_pd_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vrndscalepd {{.*}}{%k1}
; CHECK: vrndscalepd
define <4 x double>@test_int_x86_avx512_mask_rndscale_pd_256(<4 x double> %x0, <4 x double> %x2, i8 %x3) {
  %res = call <4 x double> @llvm.x86.avx512.mask.rndscale.pd.256(<4 x double> %x0, i32 4, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.rndscale.pd.256(<4 x double> %x0, i32 88, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.rndscale.ps.128(<4 x float>, i32, <4 x float>, i8)
; CHECK-LABEL: @test_int_x86_avx512_mask_rndscale_ps_128
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vrndscaleps {{.*}}{%k1}
; CHECK: vrndscaleps
define <4 x float>@test_int_x86_avx512_mask_rndscale_ps_128(<4 x float> %x0, <4 x float> %x2, i8 %x3) {
  %res = call <4 x float> @llvm.x86.avx512.mask.rndscale.ps.128(<4 x float> %x0, i32 88, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.rndscale.ps.128(<4 x float> %x0, i32 4, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.rndscale.ps.256(<8 x float>, i32, <8 x float>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_rndscale_ps_256
; CHECK-NOT: call
; CHECK: kmov
; CHECK: vrndscaleps {{.*}}{%k1}
; CHECK: vrndscaleps
define <8 x float>@test_int_x86_avx512_mask_rndscale_ps_256(<8 x float> %x0, <8 x float> %x2, i8 %x3) {
  %res = call <8 x float> @llvm.x86.avx512.mask.rndscale.ps.256(<8 x float> %x0, i32 5, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.rndscale.ps.256(<8 x float> %x0, i32 66, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.shuf.f32x4.256(<8 x float>, <8 x float>, i32, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_shuf_f32x4_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_f32x4_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshuff32x4 $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; CHECK-NEXT:    vshuff32x4 $22, %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    ## ymm3 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; CHECK-NEXT:    vshuff32x4 $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; CHECK-NEXT:    vaddps %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.shuf.f32x4.256(<8 x float> %x0, <8 x float> %x1, i32 22, <8 x float> %x3, i8 %x4)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.shuf.f32x4.256(<8 x float> %x0, <8 x float> %x1, i32 22, <8 x float> %x3, i8 -1)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.shuf.f32x4.256(<8 x float> %x0, <8 x float> %x1, i32 22, <8 x float> zeroinitializer, i8 %x4)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res2, %res3
  ret <8 x float> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.shuf.f64x2.256(<4 x double>, <4 x double>, i32, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_shuf_f64x2_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_f64x2_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshuff64x2 $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm0[0,1],ymm1[2,3]
; CHECK-NEXT:    vshuff64x2 $22, %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    ## ymm3 = ymm0[0,1],ymm1[2,3]
; CHECK-NEXT:    vshuff64x2 $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0,1],ymm1[2,3]
; CHECK-NEXT:    vaddpd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.shuf.f64x2.256(<4 x double> %x0, <4 x double> %x1, i32 22, <4 x double> %x3, i8 %x4)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.shuf.f64x2.256(<4 x double> %x0, <4 x double> %x1, i32 22, <4 x double> %x3, i8 -1)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.shuf.f64x2.256(<4 x double> %x0, <4 x double> %x1, i32 22, <4 x double> zeroinitializer, i8 %x4)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res2, %res3
  ret <4 x double> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.shuf.i32x4.256(<8 x i32>, <8 x i32>, i32, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_shuf_i32x4_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_i32x4_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufi32x4 $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; CHECK-NEXT:    vshufi32x4 $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; CHECK-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.shuf.i32x4.256(<8 x i32> %x0, <8 x i32> %x1, i32 22, <8 x i32> %x3, i8 %x4)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.shuf.i32x4.256(<8 x i32> %x0, <8 x i32> %x1, i32 22, <8 x i32> %x3, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.shuf.i64x2.256(<4 x i64>, <4 x i64>, i32, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_shuf_i64x2_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_i64x2_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufi64x2 $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm0[0,1],ymm1[2,3]
; CHECK-NEXT:    vshufi64x2 $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0,1],ymm1[2,3]
; CHECK-NEXT:    vpaddq %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.shuf.i64x2.256(<4 x i64> %x0, <4 x i64> %x1, i32 22, <4 x i64> %x3, i8 %x4)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.shuf.i64x2.256(<4 x i64> %x0, <4 x i64> %x1, i32 22, <4 x i64> %x3, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.vextractf32x4.256(<8 x float>, i32, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_vextractf32x4_256(<8 x float> %x0, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vextractf32x4_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vextractf32x4 $1, %ymm0, %xmm1 {%k1}
; CHECK-NEXT:    vextractf32x4 $1, %ymm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vextractf32x4 $1, %ymm0, %xmm0
; CHECK-NEXT:    vaddps %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res  = call <4 x float> @llvm.x86.avx512.mask.vextractf32x4.256(<8 x float> %x0, i32 1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vextractf32x4.256(<8 x float> %x0, i32 1, <4 x float> zeroinitializer, i8 %x3)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.vextractf32x4.256(<8 x float> %x0, i32 1, <4 x float> zeroinitializer, i8 -1)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res2, %res3
  ret <4 x float> %res4
}

declare <2 x double> @llvm.x86.avx512.mask.getmant.pd.128(<2 x double>, i32, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_getmant_pd_128(<2 x double> %x0, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vgetmantpd $11, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vgetmantpd $11, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vgetmantpd $11, %xmm0, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.getmant.pd.128(<2 x double> %x0, i32 11, <2 x double> %x2, i8 %x3)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.getmant.pd.128(<2 x double> %x0, i32 11, <2 x double> zeroinitializer, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.getmant.pd.128(<2 x double> %x0, i32 11, <2 x double> %x2, i8 -1)
  %res3 = fadd <2 x double> %res, %res1
  %res4 = fadd <2 x double> %res2, %res3
  ret <2 x double> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.getmant.pd.256(<4 x double>, i32, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_getmant_pd_256(<4 x double> %x0, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vgetmantpd $11, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vgetmantpd $11, %ymm0, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.getmant.pd.256(<4 x double> %x0, i32 11, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.getmant.pd.256(<4 x double> %x0, i32 11, <4 x double> %x2, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.getmant.ps.128(<4 x float>, i32, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_getmant_ps_128(<4 x float> %x0, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vgetmantps $11, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vgetmantps $11, %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.getmant.ps.128(<4 x float> %x0, i32 11, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.getmant.ps.128(<4 x float> %x0, i32 11, <4 x float> %x2, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.getmant.ps.256(<8 x float>, i32, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_getmant_ps_256(<8 x float> %x0, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vgetmantps $11, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vgetmantps $11, %ymm0, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.getmant.ps.256(<8 x float> %x0, i32 11, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.getmant.ps.256(<8 x float> %x0, i32 11, <8 x float> %x2, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.shuf.pd.128(<2 x double>, <2 x double>, i32, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_shuf_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x double> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufpd $22, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[0],k1[1]
; CHECK-NEXT:    vshufpd $22, %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    ## xmm3 = k1[0],xmm0[1]
; CHECK-NEXT:    vshufpd $22, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[0],xmm1[1]
; CHECK-NEXT:    vaddpd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.shuf.pd.128(<2 x double> %x0, <2 x double> %x1, i32 22, <2 x double> %x3, i8 %x4)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.shuf.pd.128(<2 x double> %x0, <2 x double> %x1, i32 22, <2 x double> %x3, i8 -1)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.shuf.pd.128(<2 x double> %x0, <2 x double> %x1, i32 22, <2 x double> zeroinitializer, i8 %x4)
  %res3 = fadd <2 x double> %res, %res1
  %res4 = fadd <2 x double> %res2, %res3
  ret <2 x double> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.shuf.pd.256(<4 x double>, <4 x double>, i32, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_shuf_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x double> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufpd $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[0],k1[1],ymm2[3],k1[2]
; CHECK-NEXT:    vshufpd $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0],ymm1[1],ymm0[3],ymm1[2]
; CHECK-NEXT:    vaddpd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.shuf.pd.256(<4 x double> %x0, <4 x double> %x1, i32 22, <4 x double> %x3, i8 %x4)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.shuf.pd.256(<4 x double> %x0, <4 x double> %x1, i32 22, <4 x double> %x3, i8 -1)
  %res2 = fadd <4 x double> %res, %res1
  ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.shuf.ps.128(<4 x float>, <4 x float>, i32, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_shuf_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x float> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufps $22, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    ## xmm2 = xmm2[2,1],k1[1,0]
; CHECK-NEXT:    vshufps $22, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[2,1],xmm1[1,0]
; CHECK-NEXT:    vaddps %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.shuf.ps.128(<4 x float> %x0, <4 x float> %x1, i32 22, <4 x float> %x3, i8 %x4)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.shuf.ps.128(<4 x float> %x0, <4 x float> %x1, i32 22, <4 x float> %x3, i8 -1)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.shuf.ps.256(<8 x float>, <8 x float>, i32, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_shuf_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x float> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufps $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    ## ymm2 = ymm2[2,1],k1[1,0],ymm2[6,5],k1[5,4]
; CHECK-NEXT:    vshufps $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[2,1],ymm1[1,0],ymm0[6,5],ymm1[5,4]
; CHECK-NEXT:    vaddps %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.shuf.ps.256(<8 x float> %x0, <8 x float> %x1, i32 22, <8 x float> %x3, i8 %x4)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.shuf.ps.256(<8 x float> %x0, <8 x float> %x1, i32 22, <8 x float> %x3, i8 -1)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.valign.d.128(<4 x i32>, <4 x i32>, i32, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_valign_d_128(<4 x i32> %x0, <4 x i32> %x1,<4 x i32> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_valign_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    valignd $22, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    valignd $22, %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    valignd $22, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.valign.d.128(<4 x i32> %x0, <4 x i32> %x1, i32 22, <4 x i32> %x3, i8 %x4)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.valign.d.128(<4 x i32> %x0, <4 x i32> %x1, i32 22, <4 x i32> %x3, i8 -1)
    %res2 = call <4 x i32> @llvm.x86.avx512.mask.valign.d.128(<4 x i32> %x0, <4 x i32> %x1, i32 22, <4 x i32> zeroinitializer,i8 %x4)
  %res3 = add <4 x i32> %res, %res1
    %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.valign.d.256(<8 x i32>, <8 x i32>, i32, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_valign_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_valign_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    valignd $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    valignd $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.valign.d.256(<8 x i32> %x0, <8 x i32> %x1, i32 22, <8 x i32> %x3, i8 %x4)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.valign.d.256(<8 x i32> %x0, <8 x i32> %x1, i32 22, <8 x i32> %x3, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.valign.q.128(<2 x i64>, <2 x i64>, i32, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_valign_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_valign_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    valignq $22, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    valignq $22, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.valign.q.128(<2 x i64> %x0, <2 x i64> %x1, i32 22, <2 x i64> %x3, i8 %x4)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.valign.q.128(<2 x i64> %x0, <2 x i64> %x1, i32 22, <2 x i64> %x3, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.valign.q.256(<4 x i64>, <4 x i64>, i32, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_valign_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_valign_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    valignq $22, %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    valignq $22, %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.valign.q.256(<4 x i64> %x0, <4 x i64> %x1, i32 22, <4 x i64> %x3, i8 %x4)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.valign.q.256(<4 x i64> %x0, <4 x i64> %x1, i32 22, <4 x i64> %x3, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.vpermil.pd.256(<4 x double>, i32, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_vpermil_pd_256(<4 x double> %x0, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermil_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilpd $22, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    ## ymm1 = ymm1[0,1,3,2]
; CHECK-NEXT:    vpermilpd $22, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    ## ymm2 = k1[0,1,3,2]
; CHECK-NEXT:    vpermilpd $22, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0,1,3,2]
; CHECK-NEXT:    vaddpd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.vpermil.pd.256(<4 x double> %x0, i32 22, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.vpermil.pd.256(<4 x double> %x0, i32 22, <4 x double> zeroinitializer, i8 %x3)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.vpermil.pd.256(<4 x double> %x0, i32 22, <4 x double> %x2, i8 -1)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res2, %res3
  ret <4 x double> %res4
}

declare <2 x double> @llvm.x86.avx512.mask.vpermil.pd.128(<2 x double>, i32, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_vpermil_pd_128(<2 x double> %x0, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermil_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilpd $1, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    ## xmm1 = xmm1[1,0]
; CHECK-NEXT:    vpermilpd $1, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    ## xmm2 = k1[1,0]
; CHECK-NEXT:    vpermilpd $1, %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[1,0]
; CHECK-NEXT:    vaddpd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.vpermil.pd.128(<2 x double> %x0, i32 1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.vpermil.pd.128(<2 x double> %x0, i32 1, <2 x double> zeroinitializer, i8 %x3)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.vpermil.pd.128(<2 x double> %x0, i32 1, <2 x double> %x2, i8 -1)
  %res3 = fadd <2 x double> %res, %res1
  %res4 = fadd <2 x double> %res3, %res2
  ret <2 x double> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.vpermil.ps.256(<8 x float>, i32, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_vpermil_ps_256(<8 x float> %x0, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermil_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilps $22, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    ## ymm1 = ymm1[2,1,1,0,6,5,5,4]
; CHECK-NEXT:    vpermilps $22, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    ## ymm2 = k1[2,1,1,0,6,5,5,4]
; CHECK-NEXT:    vpermilps $22, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[2,1,1,0,6,5,5,4]
; CHECK-NEXT:    vaddps %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.vpermil.ps.256(<8 x float> %x0, i32 22, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.vpermil.ps.256(<8 x float> %x0, i32 22, <8 x float> zeroinitializer, i8 %x3)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.vpermil.ps.256(<8 x float> %x0, i32 22, <8 x float> %x2, i8 -1)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res3, %res2
  ret <8 x float> %res4
}

declare <4 x float> @llvm.x86.avx512.mask.vpermil.ps.128(<4 x float>, i32, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_vpermil_ps_128(<4 x float> %x0, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermil_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilps $22, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    ## xmm1 = xmm1[2,1,1,0]
; CHECK-NEXT:    vpermilps $22, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    ## xmm2 = k1[2,1,1,0]
; CHECK-NEXT:    vpermilps $22, %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[2,1,1,0]
; CHECK-NEXT:    vaddps %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.vpermil.ps.128(<4 x float> %x0, i32 22, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vpermil.ps.128(<4 x float> %x0, i32 22, <4 x float> zeroinitializer, i8 %x3)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.vpermil.ps.128(<4 x float> %x0, i32 22, <4 x float> %x2, i8 -1)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res2, %res3
  ret <4 x float> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.vpermilvar.pd.256(<4 x double>, <4 x i64>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_vpermilvar_pd_256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermilvar_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilpd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpermilpd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpermilpd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vaddpd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.vpermilvar.pd.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.vpermilvar.pd.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> zeroinitializer, i8 %x3)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.vpermilvar.pd.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 -1)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res2, %res3
  ret <4 x double> %res4
}

declare <2 x double> @llvm.x86.avx512.mask.vpermilvar.pd.128(<2 x double>, <2 x i64>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_vpermilvar_pd_128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermilvar_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilpd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpermilpd %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpermilpd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddpd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.vpermilvar.pd.128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2, i8 %x3)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.vpermilvar.pd.128(<2 x double> %x0, <2 x i64> %x1, <2 x double> zeroinitializer, i8 %x3)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.vpermilvar.pd.128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2, i8 -1)
  %res3 = fadd <2 x double> %res, %res1
  %res4 = fadd <2 x double> %res3, %res2
  ret <2 x double> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.vpermilvar.ps.256(<8 x float>, <8 x i32>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_vpermilvar_ps_256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermilvar_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilps %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpermilps %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpermilps %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vaddps %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.vpermilvar.ps.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.vpermilvar.ps.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> zeroinitializer, i8 %x3)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.vpermilvar.ps.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 -1)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res3, %res2
  ret <8 x float> %res4
}

declare <4 x float> @llvm.x86.avx512.mask.vpermilvar.ps.128(<4 x float>, <4 x i32>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_vpermilvar_ps_128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermilvar_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilps %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpermilps %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpermilps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.vpermilvar.ps.128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2, i8 %x3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.vpermilvar.ps.128(<4 x float> %x0, <4 x i32> %x1, <4 x float> zeroinitializer, i8 %x3)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.vpermilvar.ps.128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2, i8 -1)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res2, %res3
  ret <4 x float> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.insertf32x4.256(<8 x float>, <4 x float>, i32, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_insertf32x4_256(<8 x float> %x0, <4 x float> %x1, <8 x float> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_insertf32x4_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vinsertf32x4 $1, %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vinsertf32x4 $1, %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vinsertf32x4 $1, %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.insertf32x4.256(<8 x float> %x0, <4 x float> %x1, i32 1, <8 x float> %x3, i8 %x4)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.insertf32x4.256(<8 x float> %x0, <4 x float> %x1, i32 1, <8 x float> %x3, i8 -1)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.insertf32x4.256(<8 x float> %x0, <4 x float> %x1, i32 1, <8 x float> zeroinitializer, i8 %x4)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res2, %res3
  ret <8 x float> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.inserti32x4.256(<8 x i32>, <4 x i32>, i32, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_inserti32x4_256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_inserti32x4_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq

  %res = call <8 x i32> @llvm.x86.avx512.mask.inserti32x4.256(<8 x i32> %x0, <4 x i32> %x1, i32 1, <8 x i32> %x3, i8 %x4)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.inserti32x4.256(<8 x i32> %x0, <4 x i32> %x1, i32 1, <8 x i32> %x3, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.inserti32x4.256(<8 x i32> %x0, <4 x i32> %x1, i32 1, <8 x i32> zeroinitializer, i8 %x4)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.pternlog.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i32, i8)

define <4 x i32>@test_int_x86_avx512_mask_pternlog_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_pternlog_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogd $33, %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vpternlogd $33, %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.pternlog.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i32 33, i8 %x4)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pternlog.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i32 33, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <4 x i32> @llvm.x86.avx512.maskz.pternlog.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i32, i8)

define <4 x i32>@test_int_x86_avx512_maskz_pternlog_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_pternlog_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogd $33, %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpternlogd $33, %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.maskz.pternlog.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i32 33, i8 %x4)
  %res1 = call <4 x i32> @llvm.x86.avx512.maskz.pternlog.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i32 33, i8 -1)
  %res2 = add <4 x i32> %res, %res1
  ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.pternlog.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i32, i8)

define <8 x i32>@test_int_x86_avx512_mask_pternlog_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_pternlog_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogd $33, %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vpternlogd $33, %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.pternlog.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i32 33, i8 %x4)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pternlog.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i32 33, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.maskz.pternlog.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i32, i8)

define <8 x i32>@test_int_x86_avx512_maskz_pternlog_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_pternlog_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogd $33, %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpternlogd $33, %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.maskz.pternlog.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i32 33, i8 %x4)
  %res1 = call <8 x i32> @llvm.x86.avx512.maskz.pternlog.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i32 33, i8 -1)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.pternlog.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i32, i8)

define <2 x i64>@test_int_x86_avx512_mask_pternlog_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_pternlog_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogq $33, %xmm2, %xmm1, %xmm3 {%k1}
; CHECK-NEXT:    vpternlogq $33, %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.pternlog.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i32 33, i8 %x4)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pternlog.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i32 33, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <2 x i64> @llvm.x86.avx512.maskz.pternlog.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i32, i8)

define <2 x i64>@test_int_x86_avx512_maskz_pternlog_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_pternlog_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogq $33, %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpternlogq $33, %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm3, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.maskz.pternlog.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i32 33, i8 %x4)
  %res1 = call <2 x i64> @llvm.x86.avx512.maskz.pternlog.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i32 33, i8 -1)
  %res2 = add <2 x i64> %res, %res1
  ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.pternlog.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i32, i8)

define <4 x i64>@test_int_x86_avx512_mask_pternlog_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_pternlog_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogq $33, %ymm2, %ymm1, %ymm3 {%k1}
; CHECK-NEXT:    vpternlogq $33, %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.pternlog.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i32 33, i8 %x4)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pternlog.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i32 33, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.maskz.pternlog.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i32, i8)

define <4 x i64>@test_int_x86_avx512_maskz_pternlog_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_pternlog_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogq $33, %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpternlogq $33, %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.maskz.pternlog.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i32 33, i8 %x4)
  %res1 = call <4 x i64> @llvm.x86.avx512.maskz.pternlog.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i32 33, i8 -1)
  %res2 = add <4 x i64> %res, %res1
  ret <4 x i64> %res2
}

declare <8 x i32> @llvm.x86.avx512.pbroadcastd.256(<4 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_pbroadcastd_256(<4 x i32> %x0, <8 x i32> %x1, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpbroadcastd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpbroadcastd %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastd %xmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.pbroadcastd.256(<4 x i32> %x0, <8 x i32> %x1, i8 -1)
  %res1 = call <8 x i32> @llvm.x86.avx512.pbroadcastd.256(<4 x i32> %x0, <8 x i32> %x1, i8 %mask)
  %res2 = call <8 x i32> @llvm.x86.avx512.pbroadcastd.256(<4 x i32> %x0, <8 x i32> zeroinitializer, i8 %mask)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.pbroadcastd.128(<4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_pbroadcastd_128(<4 x i32> %x0, <4 x i32> %x1, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpbroadcastd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpbroadcastd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastd %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.pbroadcastd.128(<4 x i32> %x0, <4 x i32> %x1, i8 -1)
  %res1 = call <4 x i32> @llvm.x86.avx512.pbroadcastd.128(<4 x i32> %x0, <4 x i32> %x1, i8 %mask)
  %res2 = call <4 x i32> @llvm.x86.avx512.pbroadcastd.128(<4 x i32> %x0, <4 x i32> zeroinitializer, i8 %mask)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <4 x i64> @llvm.x86.avx512.pbroadcastq.256(<2 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_pbroadcastq_256(<2 x i64> %x0, <4 x i64> %x1, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpbroadcastq %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpbroadcastq %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastq %xmm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.pbroadcastq.256(<2 x i64> %x0, <4 x i64> %x1,i8 -1)
  %res1 = call <4 x i64> @llvm.x86.avx512.pbroadcastq.256(<2 x i64> %x0, <4 x i64> %x1,i8 %mask)
  %res2 = call <4 x i64> @llvm.x86.avx512.pbroadcastq.256(<2 x i64> %x0, <4 x i64> zeroinitializer,i8 %mask)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res2, %res3
  ret <4 x i64> %res4
}

declare <2 x i64> @llvm.x86.avx512.pbroadcastq.128(<2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_pbroadcastq_128(<2 x i64> %x0, <2 x i64> %x1, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpbroadcastq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpbroadcastq %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.pbroadcastq.128(<2 x i64> %x0, <2 x i64> %x1,i8 -1)
  %res1 = call <2 x i64> @llvm.x86.avx512.pbroadcastq.128(<2 x i64> %x0, <2 x i64> %x1,i8 %mask)
  %res2 = call <2 x i64> @llvm.x86.avx512.pbroadcastq.128(<2 x i64> %x0, <2 x i64> zeroinitializer,i8 %mask)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res2, %res3
  ret <2 x i64> %res4
}

define <4 x float> @test_x86_vcvtph2ps_128(<8 x i16> %a0) {
  ; CHECK: test_x86_vcvtph2ps_128
  ; CHECK: vcvtph2ps  %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.vcvtph2ps.128(<8 x i16> %a0, <4 x float> zeroinitializer, i8 -1)
  ret <4 x float> %res
}

define <4 x float> @test_x86_vcvtph2ps_128_rrk(<8 x i16> %a0,<4 x float> %a1, i8 %mask) {
  ; CHECK: test_x86_vcvtph2ps_128_rrk
  ; CHECK: vcvtph2ps  %xmm0, %xmm1 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.vcvtph2ps.128(<8 x i16> %a0, <4 x float> %a1, i8 %mask)
  ret <4 x float> %res
}


define <4 x float> @test_x86_vcvtph2ps_128_rrkz(<8 x i16> %a0, i8 %mask) {
  ; CHECK: test_x86_vcvtph2ps_128_rrkz
  ; CHECK: vcvtph2ps  %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.vcvtph2ps.128(<8 x i16> %a0, <4 x float> zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.vcvtph2ps.128(<8 x i16>, <4 x float>, i8) nounwind readonly

define <8 x float> @test_x86_vcvtph2ps_256(<8 x i16> %a0) {
  ; CHECK: test_x86_vcvtph2ps_256
  ; CHECK: vcvtph2ps  %xmm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.vcvtph2ps.256(<8 x i16> %a0, <8 x float> zeroinitializer, i8 -1)
  ret <8 x float> %res
}

define <8 x float> @test_x86_vcvtph2ps_256_rrk(<8 x i16> %a0,<8 x float> %a1, i8 %mask) {
  ; CHECK: test_x86_vcvtph2ps_256_rrk
  ; CHECK: vcvtph2ps  %xmm0, %ymm1 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.mask.vcvtph2ps.256(<8 x i16> %a0, <8 x float> %a1, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_x86_vcvtph2ps_256_rrkz(<8 x i16> %a0, i8 %mask) {
  ; CHECK: test_x86_vcvtph2ps_256_rrkz
  ; CHECK: vcvtph2ps  %xmm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.vcvtph2ps.256(<8 x i16> %a0, <8 x float> zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

declare <8 x float> @llvm.x86.avx512.mask.vcvtph2ps.256(<8 x i16>, <8 x float>, i8) nounwind readonly

define <8 x i16> @test_x86_vcvtps2ph_128(<4 x float> %a0) {
  ; CHECK: test_x86_vcvtps2ph_128
  ; CHECK: vcvtps2ph $2, %xmm0, %xmm0
  %res = call <8 x i16> @llvm.x86.avx512.mask.vcvtps2ph.128(<4 x float> %a0, i32 2, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}


declare <8 x i16> @llvm.x86.avx512.mask.vcvtps2ph.128(<4 x float>, i32, <8 x i16>, i8) nounwind readonly

define <8 x i16> @test_x86_vcvtps2ph_256(<8 x float> %a0) {
  ; CHECK: test_x86_vcvtps2ph_256
  ; CHECK: vcvtps2ph $2, %ymm0, %xmm0
  %res = call <8 x i16> @llvm.x86.avx512.mask.vcvtps2ph.256(<8 x float> %a0, i32 2, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.vcvtps2ph.256(<8 x float>, i32, <8 x i16>, i8) nounwind readonly

declare <4 x float> @llvm.x86.avx512.mask.movsldup.128(<4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_movsldup_128(<4 x float> %x0, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movsldup_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovsldup %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    ## xmm1 = xmm0[0,0,2,2]
; CHECK-NEXT:    vmovsldup %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    ## xmm2 = xmm0[0,0,2,2]
; CHECK-NEXT:    vmovsldup %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[0,0,2,2]
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.movsldup.128(<4 x float> %x0, <4 x float> %x1, i8 %x2)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.movsldup.128(<4 x float> %x0, <4 x float> %x1, i8 -1)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.movsldup.128(<4 x float> %x0, <4 x float> zeroinitializer, i8 %x2)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res2, %res3
  ret <4 x float> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.movsldup.256(<8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_movsldup_256(<8 x float> %x0, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movsldup_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovsldup %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    ## ymm1 = ymm0[0,0,2,2,4,4,6,6]
; CHECK-NEXT:    vmovsldup %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    ## ymm2 = ymm0[0,0,2,2,4,4,6,6]
; CHECK-NEXT:    vmovsldup %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0,0,2,2,4,4,6,6]
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.movsldup.256(<8 x float> %x0, <8 x float> %x1, i8 %x2)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.movsldup.256(<8 x float> %x0, <8 x float> %x1, i8 -1)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.movsldup.256(<8 x float> %x0, <8 x float> zeroinitializer, i8 %x2)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res2, %res3
  ret <8 x float> %res4
}

declare <4 x float> @llvm.x86.avx512.mask.movshdup.128(<4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_movshdup_128(<4 x float> %x0, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movshdup_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovshdup %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    ## xmm1 = xmm0[1,1,3,3]
; CHECK-NEXT:    vmovshdup %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    ## xmm2 = xmm0[1,1,3,3]
; CHECK-NEXT:    vmovshdup %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[1,1,3,3]
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.movshdup.128(<4 x float> %x0, <4 x float> %x1, i8 %x2)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.movshdup.128(<4 x float> %x0, <4 x float> %x1, i8 -1)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.movshdup.128(<4 x float> %x0, <4 x float> zeroinitializer, i8 %x2)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res2, %res3
  ret <4 x float> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.movshdup.256(<8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_movshdup_256(<8 x float> %x0, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movshdup_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovshdup %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    ## ymm1 = ymm0[1,1,3,3,5,5,7,7]
; CHECK-NEXT:    vmovshdup %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    ## ymm2 = ymm0[1,1,3,3,5,5,7,7]
; CHECK-NEXT:    vmovshdup %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[1,1,3,3,5,5,7,7]
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.movshdup.256(<8 x float> %x0, <8 x float> %x1, i8 %x2)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.movshdup.256(<8 x float> %x0, <8 x float> %x1, i8 -1)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.movshdup.256(<8 x float> %x0, <8 x float> zeroinitializer, i8 %x2)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res2, %res3
  ret <8 x float> %res4
}
declare <2 x double> @llvm.x86.avx512.mask.movddup.128(<2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_movddup_128(<2 x double> %x0, <2 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movddup_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovddup %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    ## xmm1 = xmm0[0,0]
; CHECK-NEXT:    vmovddup %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    ## xmm2 = xmm0[0,0]
; CHECK-NEXT:    vmovddup %xmm0, %xmm0
; CHECK-NEXT:    ## xmm0 = xmm0[0,0]
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.movddup.128(<2 x double> %x0, <2 x double> %x1, i8 %x2)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.movddup.128(<2 x double> %x0, <2 x double> %x1, i8 -1)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.movddup.128(<2 x double> %x0, <2 x double> zeroinitializer, i8 %x2)
  %res3 = fadd <2 x double> %res, %res1
  %res4 = fadd <2 x double> %res2, %res3
  ret <2 x double> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.movddup.256(<4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_movddup_256(<4 x double> %x0, <4 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movddup_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovddup %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    ## ymm1 = ymm0[0,0,2,2]
; CHECK-NEXT:    vmovddup %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    ## ymm2 = ymm0[0,0,2,2]
; CHECK-NEXT:    vmovddup %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[0,0,2,2]
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    vaddpd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.movddup.256(<4 x double> %x0, <4 x double> %x1, i8 %x2)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.movddup.256(<4 x double> %x0, <4 x double> %x1, i8 -1)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.movddup.256(<4 x double> %x0, <4 x double> zeroinitializer, i8 %x2)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res2, %res3
  ret <4 x double> %res4
}

define <8 x float> @test_rsqrt_ps_256_rr(<8 x float> %a0) {
; CHECK-LABEL: test_rsqrt_ps_256_rr:
; CHECK: vrsqrt14ps %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float> %a0, <8 x float> zeroinitializer, i8 -1)
  ret <8 x float> %res
}

define <8 x float> @test_rsqrt_ps_256_rrkz(<8 x float> %a0, i8 %mask) {
; CHECK-LABEL: test_rsqrt_ps_256_rrkz:
; CHECK: vrsqrt14ps %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float> %a0, <8 x float> zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_rsqrt_ps_256_rrk(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
; CHECK-LABEL: test_rsqrt_ps_256_rrk:
; CHECK: vrsqrt14ps %ymm0, %ymm1 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float> %a0, <8 x float> %a1, i8 %mask)
  ret <8 x float> %res
}

define <4 x float> @test_rsqrt_ps_128_rr(<4 x float> %a0) {
; CHECK-LABEL: test_rsqrt_ps_128_rr:
; CHECK: vrsqrt14ps %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.rsqrt14.ps.128(<4 x float> %a0, <4 x float> zeroinitializer, i8 -1)
  ret <4 x float> %res
}

define <4 x float> @test_rsqrt_ps_128_rrkz(<4 x float> %a0, i8 %mask) {
; CHECK-LABEL: test_rsqrt_ps_128_rrkz:
; CHECK: vrsqrt14ps %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.rsqrt14.ps.128(<4 x float> %a0, <4 x float> zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_rsqrt_ps_128_rrk(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
; CHECK-LABEL: test_rsqrt_ps_128_rrk:
; CHECK: vrsqrt14ps %xmm0, %xmm1 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.rsqrt14.ps.128(<4 x float> %a0, <4 x float> %a1, i8 %mask)
  ret <4 x float> %res
}

declare <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone
declare <4 x float> @llvm.x86.avx512.rsqrt14.ps.128(<4 x float>, <4 x float>, i8) nounwind readnone

define <8 x float> @test_rcp_ps_256_rr(<8 x float> %a0) {
; CHECK-LABEL: test_rcp_ps_256_rr:
; CHECK: vrcp14ps %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float> %a0, <8 x float> zeroinitializer, i8 -1)
  ret <8 x float> %res
}

define <8 x float> @test_rcp_ps_256_rrkz(<8 x float> %a0, i8 %mask) {
; CHECK-LABEL: test_rcp_ps_256_rrkz:
; CHECK: vrcp14ps %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float> %a0, <8 x float> zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_rcp_ps_256_rrk(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
; CHECK-LABEL: test_rcp_ps_256_rrk:
; CHECK: vrcp14ps %ymm0, %ymm1 {%k1}
  %res = call <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float> %a0, <8 x float> %a1, i8 %mask)
  ret <8 x float> %res
}

define <4 x float> @test_rcp_ps_128_rr(<4 x float> %a0) {
; CHECK-LABEL: test_rcp_ps_128_rr:
; CHECK: vrcp14ps %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.rcp14.ps.128(<4 x float> %a0, <4 x float> zeroinitializer, i8 -1)
  ret <4 x float> %res
}

define <4 x float> @test_rcp_ps_128_rrkz(<4 x float> %a0, i8 %mask) {
; CHECK-LABEL: test_rcp_ps_128_rrkz:
; CHECK: vrcp14ps %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.rcp14.ps.128(<4 x float> %a0, <4 x float> zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_rcp_ps_128_rrk(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
; CHECK-LABEL: test_rcp_ps_128_rrk:
; CHECK: vrcp14ps %xmm0, %xmm1 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.rcp14.ps.128(<4 x float> %a0, <4 x float> %a1, i8 %mask)
  ret <4 x float> %res
}

declare <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone
declare <4 x float> @llvm.x86.avx512.rcp14.ps.128(<4 x float>, <4 x float>, i8) nounwind readnone


define <4 x double> @test_rsqrt_pd_256_rr(<4 x double> %a0) {
; CHECK-LABEL: test_rsqrt_pd_256_rr:
; CHECK: vrsqrt14pd %ymm0, %ymm0
  %res = call <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double> %a0, <4 x double> zeroinitializer, i8 -1)
  ret <4 x double> %res
}

define <4 x double> @test_rsqrt_pd_256_rrkz(<4 x double> %a0, i8 %mask) {
; CHECK-LABEL: test_rsqrt_pd_256_rrkz:
; CHECK: vrsqrt14pd %ymm0, %ymm0 {%k1} {z}
  %res = call <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double> %a0, <4 x double> zeroinitializer, i8 %mask)
  ret <4 x double> %res
}

define <4 x double> @test_rsqrt_pd_256_rrk(<4 x double> %a0, <4 x double> %a1, i8 %mask) {
; CHECK-LABEL: test_rsqrt_pd_256_rrk:
; CHECK: vrsqrt14pd %ymm0, %ymm1 {%k1}
  %res = call <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double> %a0, <4 x double> %a1, i8 %mask)
  ret <4 x double> %res
}

define <2 x double> @test_rsqrt_pd_128_rr(<2 x double> %a0) {
; CHECK-LABEL: test_rsqrt_pd_128_rr:
; CHECK: vrsqrt14pd %xmm0, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.rsqrt14.pd.128(<2 x double> %a0, <2 x double> zeroinitializer, i8 -1)
  ret <2 x double> %res
}

define <2 x double> @test_rsqrt_pd_128_rrkz(<2 x double> %a0, i8 %mask) {
; CHECK-LABEL: test_rsqrt_pd_128_rrkz:
; CHECK: vrsqrt14pd %xmm0, %xmm0 {%k1} {z}
  %res = call <2 x double> @llvm.x86.avx512.rsqrt14.pd.128(<2 x double> %a0, <2 x double> zeroinitializer, i8 %mask)
  ret <2 x double> %res
}

define <2 x double> @test_rsqrt_pd_128_rrk(<2 x double> %a0, <2 x double> %a1, i8 %mask) {
; CHECK-LABEL: test_rsqrt_pd_128_rrk:
; CHECK: vrsqrt14pd %xmm0, %xmm1 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.rsqrt14.pd.128(<2 x double> %a0, <2 x double> %a1, i8 %mask)
  ret <2 x double> %res
}

declare <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone
declare <2 x double> @llvm.x86.avx512.rsqrt14.pd.128(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x double> @test_rcp_pd_256_rr(<4 x double> %a0) {
; CHECK-LABEL: test_rcp_pd_256_rr:
; CHECK: vrcp14pd %ymm0, %ymm0
  %res = call <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double> %a0, <4 x double> zeroinitializer, i8 -1)
  ret <4 x double> %res
}

define <4 x double> @test_rcp_pd_256_rrkz(<4 x double> %a0, i8 %mask) {
; CHECK-LABEL: test_rcp_pd_256_rrkz:
; CHECK: vrcp14pd %ymm0, %ymm0 {%k1} {z}
  %res = call <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double> %a0, <4 x double> zeroinitializer, i8 %mask)
  ret <4 x double> %res
}

define <4 x double> @test_rcp_pd_256_rrk(<4 x double> %a0, <4 x double> %a1, i8 %mask) {
; CHECK-LABEL: test_rcp_pd_256_rrk:
; CHECK: vrcp14pd %ymm0, %ymm1 {%k1}
  %res = call <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double> %a0, <4 x double> %a1, i8 %mask)
  ret <4 x double> %res
}

define <2 x double> @test_rcp_pd_128_rr(<2 x double> %a0) {
; CHECK-LABEL: test_rcp_pd_128_rr:
; CHECK: vrcp14pd %xmm0, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.rcp14.pd.128(<2 x double> %a0, <2 x double> zeroinitializer, i8 -1)
  ret <2 x double> %res
}

define <2 x double> @test_rcp_pd_128_rrkz(<2 x double> %a0, i8 %mask) {
; CHECK-LABEL: test_rcp_pd_128_rrkz:
; CHECK: vrcp14pd %xmm0, %xmm0 {%k1} {z}
  %res = call <2 x double> @llvm.x86.avx512.rcp14.pd.128(<2 x double> %a0, <2 x double> zeroinitializer, i8 %mask)
  ret <2 x double> %res
}

define <2 x double> @test_rcp_pd_128_rrk(<2 x double> %a0, <2 x double> %a1, i8 %mask) {
; CHECK-LABEL: test_rcp_pd_128_rrk:
; CHECK: vrcp14pd %xmm0, %xmm1 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.rcp14.pd.128(<2 x double> %a0, <2 x double> %a1, i8 %mask)
  ret <2 x double> %res
}

declare <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone
declare <2 x double> @llvm.x86.avx512.rcp14.pd.128(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x double> @test_x86_vbroadcast_sd_pd_256(<2 x double> %a0, <4 x double> %a1, i8 %mask ) {
; CHECK-LABEL: test_x86_vbroadcast_sd_pd_256:
; CHECK: kmovw   %edi, %k1
; CHECK-NEXT:    vbroadcastsd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vbroadcastsd %xmm0, %ymm2 {%k1} {z} 
; CHECK-NEXT:    vbroadcastsd %xmm0, %ymm0
; CHECK-NEXT:    vaddpd %ymm1, %ymm0, %ymm0

  %res = call <4 x double> @llvm.x86.avx512.mask.broadcast.sd.pd.256(<2 x double> %a0, <4 x double> zeroinitializer, i8 -1)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.broadcast.sd.pd.256(<2 x double> %a0, <4 x double> %a1, i8 %mask)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.broadcast.sd.pd.256(<2 x double> %a0, <4 x double> zeroinitializer, i8 %mask)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res2, %res3
  ret <4 x double> %res4
}
declare <4 x double> @llvm.x86.avx512.mask.broadcast.sd.pd.256(<2 x double>, <4 x double>, i8) nounwind readonly

define <8 x float> @test_x86_vbroadcast_ss_ps_256(<4 x float> %a0, <8 x float> %a1, i8 %mask ) {
; CHECK-LABEL: test_x86_vbroadcast_ss_ps_256:
; CHECK: kmovw   %edi, %k1
; CHECK-NEXT: vbroadcastss %xmm0, %ymm1 {%k1}
; CHECK-NEXT: vbroadcastss %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT: vbroadcastss %xmm0, %ymm0
; CHECK-NEXT: vaddps %ymm1, %ymm0, %ymm0

  %res = call <8 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.256(<4 x float> %a0, <8 x float> zeroinitializer, i8 -1)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.256(<4 x float> %a0, <8 x float> %a1, i8 %mask)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.256(<4 x float> %a0, <8 x float> zeroinitializer, i8 %mask)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res2, %res3
  ret <8 x float> %res4
}
declare <8 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.256(<4 x float>, <8 x float>, i8) nounwind readonly

define <4 x float> @test_x86_vbroadcast_ss_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask ) {
; CHECK-LABEL: test_x86_vbroadcast_ss_ps_128:
; CHECK: kmovw   %edi, %k1
; CHECK-NEXT: vbroadcastss %xmm0, %xmm1 {%k1}
; CHECK-NEXT: vbroadcastss %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT: vbroadcastss %xmm0, %xmm0
; CHECK-NEXT: vaddps %xmm1, %xmm0, %xmm0

  %res = call <4 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.128(<4 x float> %a0, <4 x float> zeroinitializer, i8 -1)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.128(<4 x float> %a0, <4 x float> %a1, i8 %mask)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.128(<4 x float> %a0, <4 x float> zeroinitializer, i8 %mask)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res2, %res3
  ret <4 x float> %res4
}
declare <4 x float> @llvm.x86.avx512.mask.broadcast.ss.ps.128(<4 x float>, <4 x float>, i8) nounwind readonly


declare <8 x float> @llvm.x86.avx512.mask.broadcastf32x4.256(<4 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_broadcastf32x4_256(<4 x float> %x0, <8 x float> %x2, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_mask_broadcastf32x4_256:
; CHECK: kmovw %edi, %k1
; CHECK: vshuff32x4 $0, %ymm0, %ymm0, %ymm2 {%k1} {z}
; CHECK: vshuff32x4 $0, %ymm0, %ymm0, %ymm1 {%k1}
; CHECK: vshuff32x4 $0, %ymm0, %ymm0, %ymm0
; CHECK: vaddps %ymm1, %ymm0, %ymm0
; CHECK: vaddps %ymm0, %ymm2, %ymm0

  %res1 = call <8 x float> @llvm.x86.avx512.mask.broadcastf32x4.256(<4 x float> %x0, <8 x float> %x2, i8 -1)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.broadcastf32x4.256(<4 x float> %x0, <8 x float> %x2, i8 %mask)
  %res3 = call <8 x float> @llvm.x86.avx512.mask.broadcastf32x4.256(<4 x float> %x0, <8 x float> zeroinitializer, i8 %mask)
  %res4 = fadd <8 x float> %res1, %res2
  %res5 = fadd <8 x float> %res3, %res4
  ret <8 x float> %res5
}

declare <8 x i32> @llvm.x86.avx512.mask.broadcasti32x4.256(<4 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_broadcasti32x4_256(<4 x i32> %x0, <8 x i32> %x2, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_mask_broadcasti32x4_256:
; CHECK: kmovw %edi, %k1
; CHECK: vshufi32x4 $0, %ymm0, %ymm0, %ymm2 {%k1} {z}
; CHECK: vshufi32x4 $0, %ymm0, %ymm0, %ymm1 {%k1}
; CHECK: vshufi32x4 $0, %ymm0, %ymm0, %ymm0
; CHECK: vpaddd %ymm1, %ymm0, %ymm0
; CHECK: vpaddd %ymm0, %ymm2, %ymm0

  %res1 = call <8 x i32> @llvm.x86.avx512.mask.broadcasti32x4.256(<4 x i32> %x0, <8 x i32> %x2, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.broadcasti32x4.256(<4 x i32> %x0, <8 x i32> %x2, i8 %mask)
  %res3 = call <8 x i32> @llvm.x86.avx512.mask.broadcasti32x4.256(<4 x i32> %x0, <8 x i32> zeroinitializer, i8 %mask)
  %res4 = add <8 x i32> %res1, %res2
  %res5 = add <8 x i32> %res3, %res4
  ret <8 x i32> %res5
}

declare <2 x i64> @llvm.x86.avx512.mask.psrl.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psrl_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsrlq %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsrlq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    vpaddq %xmm3, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psrl.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psrl.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psrl.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psrl.q.256(<4 x i64>, <2 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psrl_q_256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlq %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsrlq %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsrlq %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm0, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psrl.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psrl.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psrl.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.psrl.qi.128(<2 x i64>, i8, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psrl_qi_128(<2 x i64> %x0, i8 %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_qi_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrlq $255, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsrlq $255, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsrlq $255, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vpaddq %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psrl.qi.128(<2 x i64> %x0, i8 255, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psrl.qi.128(<2 x i64> %x0, i8 255, <2 x i64> %x2, i8 -1)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psrl.qi.128(<2 x i64> %x0, i8 255, <2 x i64> zeroinitializer, i8 %x3)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res2, %res3
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psrl.qi.256(<4 x i64>, i8, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psrl_qi_256(<4 x i64> %x0, i8 %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_qi_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrlq $255, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsrlq $255, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsrlq $255, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    vpaddq %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psrl.qi.256(<4 x i64> %x0, i8 255, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psrl.qi.256(<4 x i64> %x0, i8 255, <4 x i64> %x2, i8 -1)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psrl.qi.256(<4 x i64> %x0, i8 255, <4 x i64> zeroinitializer, i8 %x3)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res2, %res3
  ret <4 x i64> %res4
}
declare <4 x i32> @llvm.x86.avx512.mask.psrl.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
define <4 x i32>@test_int_x86_avx512_mask_psrl_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrld %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsrld %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsrld %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psrl.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psrl.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psrl.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psrl.d.256(<8 x i32>, <4 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psrl_d_256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrld %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsrld %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsrld %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm3, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psrl.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psrl.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psrl.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.psrl.di.128(<4 x i32>, i8, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psrl_di_128(<4 x i32> %x0, i8 %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_di_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrld $255, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsrld $255, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsrld $255, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psrl.di.128(<4 x i32> %x0, i8 255, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psrl.di.128(<4 x i32> %x0, i8 255, <4 x i32> %x2, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psrl.di.128(<4 x i32> %x0, i8 255, <4 x i32> zeroinitializer, i8 %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psrl.di.256(<8 x i32>, i8, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psrl_di_256(<8 x i32> %x0, i8 %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_di_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrld $255, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsrld $255, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsrld $255, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psrl.di.256(<8 x i32> %x0, i8 255, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psrl.di.256(<8 x i32> %x0, i8 255, <8 x i32> %x2, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psrl.di.256(<8 x i32> %x0, i8 255, <8 x i32> zeroinitializer, i8 %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <16 x i32> @llvm.x86.avx512.mask.psrl.di.512(<16 x i32>, i8, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_psrl_di_512(<16 x i32> %x0, i8 %x1, <16 x i32> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrl_di_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrld $255, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vpsrld $255, %zmm0, %zmm2 {%k1} {z}
; CHECK-NEXT:    vpsrld $255, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrl.di.512(<16 x i32> %x0, i8 255, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.psrl.di.512(<16 x i32> %x0, i8 255, <16 x i32> %x2, i16 -1)
  %res2 = call <16 x i32> @llvm.x86.avx512.mask.psrl.di.512(<16 x i32> %x0, i8 255, <16 x i32> zeroinitializer, i16 %x3)
  %res3 = add <16 x i32> %res, %res1
  %res4 = add <16 x i32> %res2, %res3
  ret <16 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.psrlv2.di(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psrlv2_di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrlv2_di:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlvq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsrlvq %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsrlvq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psrlv2.di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psrlv2.di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psrlv2.di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psrlv4.di(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psrlv4_di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrlv4_di:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlvq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsrlvq %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsrlvq %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psrlv4.di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psrlv4.di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psrlv4.di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.psrlv4.si(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psrlv4_si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrlv4_si:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlvd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsrlvd %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsrlvd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psrlv4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psrlv4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psrlv4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psrlv8.si(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psrlv8_si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrlv8_si:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrlvd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsrlvd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsrlvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psrlv8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psrlv8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psrlv8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.psra.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psra_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrad %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsrad %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsrad %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psra.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psra.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psra.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psra.d.256(<8 x i32>, <4 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psra_d_256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsrad %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsrad %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsrad %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psra.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psra.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psra.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.psra.di.128(<4 x i32>, i8, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psra_di_128(<4 x i32> %x0, i8 %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_di_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrad $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsrad $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsrad $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psra.di.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psra.di.128(<4 x i32> %x0, i8 3, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psra.di.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psra.di.256(<8 x i32>, i8, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psra_di_256(<8 x i32> %x0, i8 %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_di_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsrad $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsrad $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsrad $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psra.di.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psra.di.256(<8 x i32> %x0, i8 3, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psra.di.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.psra.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psra_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsraq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsraq %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsraq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psra.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psra.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psra.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psra.q.256(<4 x i64>, <2 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psra_q_256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsraq %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsraq %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsraq %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psra.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psra.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psra.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.psra.qi.128(<2 x i64>, i8, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psra_qi_128(<2 x i64> %x0, i8 %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_qi_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsraq $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsraq $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsraq $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psra.qi.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psra.qi.128(<2 x i64> %x0, i8 3, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psra.qi.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psra.qi.256(<4 x i64>, i8, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psra_qi_256(<4 x i64> %x0, i8 %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psra_qi_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsraq $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsraq $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsraq $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psra.qi.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psra.qi.256(<4 x i64> %x0, i8 3, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psra.qi.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}


declare <4 x i32> @llvm.x86.avx512.mask.psll.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psll_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpslld %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpslld %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpslld %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psll.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psll.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psll.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psll.d.256(<8 x i32>, <4 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psll_d_256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpslld %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpslld %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpslld %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psll.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psll.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psll.d.256(<8 x i32> %x0, <4 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.psll.di.128(<4 x i32>, i8, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psll_di_128(<4 x i32> %x0, i8 %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_di_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpslld $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpslld $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpslld $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psll.di.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psll.di.128(<4 x i32> %x0, i8 3, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psll.di.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psll.di.256(<8 x i32>, i8, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psll_di_256(<8 x i32> %x0, i8 %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_di_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpslld $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpslld $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpslld $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psll.di.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psll.di.256(<8 x i32> %x0, i8 3, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psll.di.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psll.q.256(<4 x i64>, <2 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psll_q_256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsllq %xmm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsllq %xmm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsllq %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psll.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psll.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psll.q.256(<4 x i64> %x0, <2 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.psll.qi.128(<2 x i64>, i8, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psll_qi_128(<2 x i64> %x0, i8 %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_qi_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsllq $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpsllq $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpsllq $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psll.qi.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psll.qi.128(<2 x i64> %x0, i8 3, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psll.qi.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psll.qi.256(<4 x i64>, i8, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psll_qi_256(<4 x i64> %x0, i8 %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psll_qi_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpsllq $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpsllq $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpsllq $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psll.qi.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psll.qi.256(<4 x i64> %x0, i8 3, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psll.qi.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
 ret <4 x i64> %res4
}

define <8 x float> @test_mask_load_aligned_ps_256(<8 x float> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %ymm0
; CHECK-NEXT:    vmovaps (%rdi), %ymm0 {%k1}
; CHECK-NEXT:    vmovaps (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.load.ps.256(i8* %ptr, <8 x float> zeroinitializer, i8 -1)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.load.ps.256(i8* %ptr, <8 x float> %res, i8 %mask)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.load.ps.256(i8* %ptr, <8 x float> zeroinitializer, i8 %mask)
  %res4 = fadd <8 x float> %res2, %res1
  ret <8 x float> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.load.ps.256(i8*, <8 x float>, i8)

define <8 x float> @test_mask_load_unaligned_ps_256(<8 x float> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovups (%rdi), %ymm0
; CHECK-NEXT:    vmovups (%rdi), %ymm0 {%k1}
; CHECK-NEXT:    vmovups (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.loadu.ps.256(i8* %ptr, <8 x float> zeroinitializer, i8 -1)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.loadu.ps.256(i8* %ptr, <8 x float> %res, i8 %mask)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.loadu.ps.256(i8* %ptr, <8 x float> zeroinitializer, i8 %mask)
  %res4 = fadd <8 x float> %res2, %res1
  ret <8 x float> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.loadu.ps.256(i8*, <8 x float>, i8)

define <4 x double> @test_mask_load_aligned_pd_256(<4 x double> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovapd (%rdi), %ymm0
; CHECK-NEXT:    vmovapd (%rdi), %ymm0 {%k1}
; CHECK-NEXT:    vmovapd (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.load.pd.256(i8* %ptr, <4 x double> zeroinitializer, i8 -1)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.load.pd.256(i8* %ptr, <4 x double> %res, i8 %mask)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.load.pd.256(i8* %ptr, <4 x double> zeroinitializer, i8 %mask)
  %res4 = fadd <4 x double> %res2, %res1
  ret <4 x double> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.load.pd.256(i8*, <4 x double>, i8)

define <4 x double> @test_mask_load_unaligned_pd_256(<4 x double> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovupd (%rdi), %ymm0
; CHECK-NEXT:    vmovupd (%rdi), %ymm0 {%k1}
; CHECK-NEXT:    vmovupd (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.loadu.pd.256(i8* %ptr, <4 x double> zeroinitializer, i8 -1)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.loadu.pd.256(i8* %ptr, <4 x double> %res, i8 %mask)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.loadu.pd.256(i8* %ptr, <4 x double> zeroinitializer, i8 %mask)
  %res4 = fadd <4 x double> %res2, %res1
  ret <4 x double> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.loadu.pd.256(i8*, <4 x double>, i8)

define <4 x float> @test_mask_load_aligned_ps_128(<4 x float> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %xmm0
; CHECK-NEXT:    vmovaps (%rdi), %xmm0 {%k1}
; CHECK-NEXT:    vmovaps (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.load.ps.128(i8* %ptr, <4 x float> zeroinitializer, i8 -1)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.load.ps.128(i8* %ptr, <4 x float> %res, i8 %mask)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.load.ps.128(i8* %ptr, <4 x float> zeroinitializer, i8 %mask)
  %res4 = fadd <4 x float> %res2, %res1
  ret <4 x float> %res4
}

declare <4 x float> @llvm.x86.avx512.mask.load.ps.128(i8*, <4 x float>, i8)

define <4 x float> @test_mask_load_unaligned_ps_128(<4 x float> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovups (%rdi), %xmm0
; CHECK-NEXT:    vmovups (%rdi), %xmm0 {%k1}
; CHECK-NEXT:    vmovups (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.loadu.ps.128(i8* %ptr, <4 x float> zeroinitializer, i8 -1)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.loadu.ps.128(i8* %ptr, <4 x float> %res, i8 %mask)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.loadu.ps.128(i8* %ptr, <4 x float> zeroinitializer, i8 %mask)
  %res4 = fadd <4 x float> %res2, %res1
  ret <4 x float> %res4
}

declare <4 x float> @llvm.x86.avx512.mask.loadu.ps.128(i8*, <4 x float>, i8)

define <2 x double> @test_mask_load_aligned_pd_128(<2 x double> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovapd (%rdi), %xmm0
; CHECK-NEXT:    vmovapd (%rdi), %xmm0 {%k1}
; CHECK-NEXT:    vmovapd (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.load.pd.128(i8* %ptr, <2 x double> zeroinitializer, i8 -1)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.load.pd.128(i8* %ptr, <2 x double> %res, i8 %mask)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.load.pd.128(i8* %ptr, <2 x double> zeroinitializer, i8 %mask)
  %res4 = fadd <2 x double> %res2, %res1
  ret <2 x double> %res4
}

declare <2 x double> @llvm.x86.avx512.mask.load.pd.128(i8*, <2 x double>, i8)

define <2 x double> @test_mask_load_unaligned_pd_128(<2 x double> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovupd (%rdi), %xmm0
; CHECK-NEXT:    vmovupd (%rdi), %xmm0 {%k1}
; CHECK-NEXT:    vmovupd (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.loadu.pd.128(i8* %ptr, <2 x double> zeroinitializer, i8 -1)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.loadu.pd.128(i8* %ptr, <2 x double> %res, i8 %mask)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.loadu.pd.128(i8* %ptr, <2 x double> zeroinitializer, i8 %mask)
  %res4 = fadd <2 x double> %res2, %res1
  ret <2 x double> %res4
}

declare <2 x double> @llvm.x86.avx512.mask.loadu.pd.128(i8*, <2 x double>, i8)

declare <4 x i32> @llvm.x86.avx512.mask.psrav4.si(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psrav4_si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrav4_si:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsravd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsravd %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsravd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psrav4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psrav4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psrav4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psrav8.si(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psrav8_si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrav8_si:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsravd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsravd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsravd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psrav8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psrav8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psrav8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.psrav.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psrav_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrav_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsravq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsravq %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsravq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psrav.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psrav.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psrav.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psrav.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psrav_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psrav_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsravq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsravq %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsravq %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psrav.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psrav.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psrav.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.psllv2.di(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_psllv2_di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psllv2_di:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsllvq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsllvq %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsllvq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.psllv2.di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.psllv2.di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.psllv2.di(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.psllv4.di(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_psllv4_di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psllv4_di:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsllvq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsllvq %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsllvq %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.psllv4.di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.psllv4.di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.psllv4.di(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.psllv4.si(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_psllv4_si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psllv4_si:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsllvd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vpsllvd %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vpsllvd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.psllv4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.psllv4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.psllv4.si(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.psllv8.si(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_psllv8_si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_psllv8_si:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpsllvd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpsllvd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpsllvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.psllv8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.psllv8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.psllv8.si(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.prorv.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_prorv_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prorv_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprorvd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vprorvd %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vprorvd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.prorv.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.prorv.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.prorv.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.prorv.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_prorv_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prorv_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprorvd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vprorvd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vprorvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.prorv.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.prorv.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.prorv.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.prorv.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_prorv_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prorv_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprorvq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vprorvq %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vprorvq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.prorv.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.prorv.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.prorv.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.prorv.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_prorv_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prorv_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprorvq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vprorvq %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vprorvq %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.prorv.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.prorv.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.prorv.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.loadu.d.128(i8*, <4 x i32>, i8)

define <4 x i32> @test_mask_load_unaligned_d_128(i8* %ptr, i8* %ptr2, <4 x i32> %data, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu32 (%rdi), %xmm0
; CHECK-NEXT:    vmovdqu32 (%rsi), %xmm0 {%k1}
; CHECK-NEXT:    vmovdqu32 (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.loadu.d.128(i8* %ptr, <4 x i32> zeroinitializer, i8 -1)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.loadu.d.128(i8* %ptr2, <4 x i32> %res, i8 %mask)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.loadu.d.128(i8* %ptr, <4 x i32> zeroinitializer, i8 %mask)
  %res4 = add <4 x i32> %res2, %res1
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.loadu.d.256(i8*, <8 x i32>, i8)

define <8 x i32> @test_mask_load_unaligned_d_256(i8* %ptr, i8* %ptr2, <8 x i32> %data, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu32 (%rdi), %ymm0
; CHECK-NEXT:    vmovdqu32 (%rsi), %ymm0 {%k1}
; CHECK-NEXT:    vmovdqu32 (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.loadu.d.256(i8* %ptr, <8 x i32> zeroinitializer, i8 -1)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.loadu.d.256(i8* %ptr2, <8 x i32> %res, i8 %mask)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.loadu.d.256(i8* %ptr, <8 x i32> zeroinitializer, i8 %mask)
  %res4 = add <8 x i32> %res2, %res1
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.loadu.q.128(i8*, <2 x i64>, i8)

define <2 x i64> @test_mask_load_unaligned_q_128(i8* %ptr, i8* %ptr2, <2 x i64> %data, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu64 (%rdi), %xmm0
; CHECK-NEXT:    vmovdqu64 (%rsi), %xmm0 {%k1}
; CHECK-NEXT:    vmovdqu64 (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.loadu.q.128(i8* %ptr, <2 x i64> zeroinitializer, i8 -1)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.loadu.q.128(i8* %ptr2, <2 x i64> %res, i8 %mask)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.loadu.q.128(i8* %ptr, <2 x i64> zeroinitializer, i8 %mask)
  %res4 = add <2 x i64> %res2, %res1
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.loadu.q.256(i8*, <4 x i64>, i8)

define <4 x i64> @test_mask_load_unaligned_q_256(i8* %ptr, i8* %ptr2, <4 x i64> %data, i8 %mask) {
; CHECK-LABEL: test_mask_load_unaligned_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu64 (%rdi), %ymm0
; CHECK-NEXT:    vmovdqu64 (%rsi), %ymm0 {%k1}
; CHECK-NEXT:    vmovdqu64 (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.loadu.q.256(i8* %ptr, <4 x i64> zeroinitializer, i8 -1)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.loadu.q.256(i8* %ptr2, <4 x i64> %res, i8 %mask)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.loadu.q.256(i8* %ptr, <4 x i64> zeroinitializer, i8 %mask)
  %res4 = add <4 x i64> %res2, %res1
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.prol.d.128(<4 x i32>, i8, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_prol_d_128(<4 x i32> %x0, i8 %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prol_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprold $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vprold $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vprold $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.prol.d.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.prol.d.128(<4 x i32> %x0, i8 3, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.prol.d.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.prol.d.256(<8 x i32>, i8, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_prol_d_256(<8 x i32> %x0, i8 %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prol_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprold $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vprold $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vprold $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.prol.d.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.prol.d.256(<8 x i32> %x0, i8 3, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.prol.d.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.prol.q.128(<2 x i64>, i8, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_prol_q_128(<2 x i64> %x0, i8 %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prol_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprolq $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vprolq $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vprolq $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.prol.q.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.prol.q.128(<2 x i64> %x0, i8 3, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.prol.q.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.prol.q.256(<4 x i64>, i8, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_prol_q_256(<4 x i64> %x0, i8 %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prol_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprolq $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vprolq $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vprolq $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.prol.q.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.prol.q.256(<4 x i64> %x0, i8 3, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.prol.q.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.load.d.128(i8*, <4 x i32>, i8)

define <4 x i32> @test_mask_load_aligned_d_128(<4 x i32> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovdqa32 (%rdi), %xmm0
; CHECK-NEXT:    vmovdqa32 (%rdi), %xmm0 {%k1}
; CHECK-NEXT:    vmovdqa32 (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.load.d.128(i8* %ptr, <4 x i32> zeroinitializer, i8 -1)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.load.d.128(i8* %ptr, <4 x i32> %res, i8 %mask)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.load.d.128(i8* %ptr, <4 x i32> zeroinitializer, i8 %mask)
  %res4 = add <4 x i32> %res2, %res1
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.load.d.256(i8*, <8 x i32>, i8)

define <8 x i32> @test_mask_load_aligned_d_256(<8 x i32> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovdqa32 (%rdi), %ymm0
; CHECK-NEXT:    vmovdqa32 (%rdi), %ymm0 {%k1}
; CHECK-NEXT:    vmovdqa32 (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.load.d.256(i8* %ptr, <8 x i32> zeroinitializer, i8 -1)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.load.d.256(i8* %ptr, <8 x i32> %res, i8 %mask)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.load.d.256(i8* %ptr, <8 x i32> zeroinitializer, i8 %mask)
  %res4 = add <8 x i32> %res2, %res1
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.load.q.128(i8*, <2 x i64>, i8)

define <2 x i64> @test_mask_load_aligned_q_128(<2 x i64> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovdqa64 (%rdi), %xmm0
; CHECK-NEXT:    vmovdqa64 (%rdi), %xmm0 {%k1}
; CHECK-NEXT:    vmovdqa64 (%rdi), %xmm1 {%k1} {z}
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.load.q.128(i8* %ptr, <2 x i64> zeroinitializer, i8 -1)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.load.q.128(i8* %ptr, <2 x i64> %res, i8 %mask)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.load.q.128(i8* %ptr, <2 x i64> zeroinitializer, i8 %mask)
  %res4 = add <2 x i64> %res2, %res1
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.load.q.256(i8*, <4 x i64>, i8)

define <4 x i64> @test_mask_load_aligned_q_256(<4 x i64> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_load_aligned_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovdqa64 (%rdi), %ymm0
; CHECK-NEXT:    vmovdqa64 (%rdi), %ymm0 {%k1}
; CHECK-NEXT:    vmovdqa64 (%rdi), %ymm1 {%k1} {z}
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.load.q.256(i8* %ptr, <4 x i64> zeroinitializer, i8 -1)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.load.q.256(i8* %ptr, <4 x i64> %res, i8 %mask)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.load.q.256(i8* %ptr, <4 x i64> zeroinitializer, i8 %mask)
  %res4 = add <4 x i64> %res2, %res1
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.prolv.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_prolv_d_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prolv_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprolvd %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vprolvd %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vprolvd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.prolv.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.prolv.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.prolv.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.prolv.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_prolv_d_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prolv_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprolvd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vprolvd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vprolvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.prolv.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.prolv.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.prolv.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.prolv.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_prolv_q_128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prolv_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprolvq %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vprolvq %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vprolvq %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.prolv.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.prolv.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.prolv.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.prolv.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_prolv_q_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_prolv_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vprolvq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vprolvq %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vprolvq %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.prolv.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.prolv.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.prolv.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.pror.d.128(<4 x i32>, i8, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pror_d_128(<4 x i32> %x0, i8 %x1, <4 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pror_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprord $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vprord $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vprord $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.pror.d.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pror.d.128(<4 x i32> %x0, i8 3, <4 x i32> zeroinitializer, i8 %x3)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.pror.d.128(<4 x i32> %x0, i8 3, <4 x i32> %x2, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.pror.d.256(<8 x i32>, i8, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pror_d_256(<8 x i32> %x0, i32 %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pror_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprord $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vprord $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vprord $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.pror.d.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pror.d.256(<8 x i32> %x0, i8 3, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.pror.d.256(<8 x i32> %x0, i8 3, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.pror.q.128(<2 x i64>, i8, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_pror_q_128(<2 x i64> %x0, i8 %x1, <2 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pror_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprorq $3, %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vprorq $3, %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vprorq $3, %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.pror.q.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 %x3)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pror.q.128(<2 x i64> %x0, i8 3, <2 x i64> zeroinitializer, i8 %x3)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.pror.q.128(<2 x i64> %x0, i8 3, <2 x i64> %x2, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.pror.q.256(<4 x i64>, i8, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_pror_q_256(<4 x i64> %x0, i8 %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_pror_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vprorq $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vprorq $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vprorq $3, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.pror.q.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pror.q.256(<4 x i64> %x0, i8 3, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.pror.q.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovzxb.d.128(<16 x i8>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovzxb_d_128(<16 x i8> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxb_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxbd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovzxbd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxbd %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmovzxb.d.128(<16 x i8> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovzxb.d.128(<16 x i8> %x0, <4 x i32> zeroinitializer, i8 %x2)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovzxb.d.128(<16 x i8> %x0, <4 x i32> %x1, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.pmovzxb.d.256(<16 x i8>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmovzxb_d_256(<16 x i8> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxb_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxbd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovzxbd %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxbd %xmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmovzxb.d.256(<16 x i8> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmovzxb.d.256(<16 x i8> %x0, <8 x i32> zeroinitializer, i8 %x2)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.pmovzxb.d.256(<16 x i8> %x0, <8 x i32> %x1, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.pmovzxb.q.128(<16 x i8>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_pmovzxb_q_128(<16 x i8> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxb_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxbq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovzxbq %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxbq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmovzxb.q.128(<16 x i8> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmovzxb.q.128(<16 x i8> %x0, <2 x i64> zeroinitializer, i8 %x2)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.pmovzxb.q.128(<16 x i8> %x0, <2 x i64> %x1, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.pmovzxb.q.256(<16 x i8>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_pmovzxb_q_256(<16 x i8> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxb_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxbq %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovzxbq %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxbq %xmm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmovzxb.q.256(<16 x i8> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmovzxb.q.256(<16 x i8> %x0, <4 x i64> zeroinitializer, i8 %x2)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.pmovzxb.q.256(<16 x i8> %x0, <4 x i64> %x1, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.pmovzxd.q.128(<4 x i32>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_pmovzxd_q_128(<4 x i32> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxd_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxdq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovzxdq %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxdq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmovzxd.q.128(<4 x i32> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmovzxd.q.128(<4 x i32> %x0, <2 x i64> zeroinitializer, i8 %x2)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.pmovzxd.q.128(<4 x i32> %x0, <2 x i64> %x1, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.pmovzxd.q.256(<4 x i32>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_pmovzxd_q_256(<4 x i32> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxd_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxdq %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovzxdq %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxdq %xmm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmovzxd.q.256(<4 x i32> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmovzxd.q.256(<4 x i32> %x0, <4 x i64> zeroinitializer, i8 %x2)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.pmovzxd.q.256(<4 x i32> %x0, <4 x i64> %x1, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovzxw.d.128(<8 x i16>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovzxw_d_128(<8 x i16> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxw_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxwd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovzxwd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxwd %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmovzxw.d.128(<8 x i16> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovzxw.d.128(<8 x i16> %x0, <4 x i32> zeroinitializer, i8 %x2)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovzxw.d.128(<8 x i16> %x0, <4 x i32> %x1, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.pmovzxw.d.256(<8 x i16>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmovzxw_d_256(<8 x i16> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxw_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxwd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovzxwd %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxwd %xmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmovzxw.d.256(<8 x i16> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmovzxw.d.256(<8 x i16> %x0, <8 x i32> zeroinitializer, i8 %x2)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.pmovzxw.d.256(<8 x i16> %x0, <8 x i32> %x1, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.pmovzxw.q.128(<8 x i16>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_pmovzxw_q_128(<8 x i16> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxw_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxwq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovzxwq %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxwq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmovzxw.q.128(<8 x i16> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmovzxw.q.128(<8 x i16> %x0, <2 x i64> zeroinitializer, i8 %x2)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.pmovzxw.q.128(<8 x i16> %x0, <2 x i64> %x1, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.pmovzxw.q.256(<8 x i16>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_pmovzxw_q_256(<8 x i16> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovzxw_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovzxwq %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovzxwq %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovzxwq %xmm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmovzxw.q.256(<8 x i16> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmovzxw.q.256(<8 x i16> %x0, <4 x i64> zeroinitializer, i8 %x2)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.pmovzxw.q.256(<8 x i16> %x0, <4 x i64> %x1, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovsxb.d.128(<16 x i8>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovsxb_d_128(<16 x i8> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxb_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxbd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovsxbd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxbd %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmovsxb.d.128(<16 x i8> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovsxb.d.128(<16 x i8> %x0, <4 x i32> zeroinitializer, i8 %x2)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovsxb.d.128(<16 x i8> %x0, <4 x i32> %x1, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.pmovsxb.d.256(<16 x i8>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmovsxb_d_256(<16 x i8> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxb_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxbd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovsxbd %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxbd %xmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmovsxb.d.256(<16 x i8> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmovsxb.d.256(<16 x i8> %x0, <8 x i32> zeroinitializer, i8 %x2)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.pmovsxb.d.256(<16 x i8> %x0, <8 x i32> %x1, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.pmovsxb.q.128(<16 x i8>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_pmovsxb_q_128(<16 x i8> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxb_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxbq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovsxbq %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxbq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmovsxb.q.128(<16 x i8> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmovsxb.q.128(<16 x i8> %x0, <2 x i64> zeroinitializer, i8 %x2)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.pmovsxb.q.128(<16 x i8> %x0, <2 x i64> %x1, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.pmovsxb.q.256(<16 x i8>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_pmovsxb_q_256(<16 x i8> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxb_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxbq %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovsxbq %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxbq %xmm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmovsxb.q.256(<16 x i8> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmovsxb.q.256(<16 x i8> %x0, <4 x i64> zeroinitializer, i8 %x2)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.pmovsxb.q.256(<16 x i8> %x0, <4 x i64> %x1, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.pmovsxw.d.128(<8 x i16>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_pmovsxw_d_128(<8 x i16> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxw_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxwd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovsxwd %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxwd %xmm0, %xmm0
; CHECK-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x i32> @llvm.x86.avx512.mask.pmovsxw.d.128(<8 x i16> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.pmovsxw.d.128(<8 x i16> %x0, <4 x i32> zeroinitializer, i8 %x2)
  %res2 = call <4 x i32> @llvm.x86.avx512.mask.pmovsxw.d.128(<8 x i16> %x0, <4 x i32> %x1, i8 -1)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res3, %res2
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.pmovsxw.d.256(<8 x i16>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmovsxw_d_256(<8 x i16> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxw_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxwd %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovsxwd %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxwd %xmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.pmovsxw.d.256(<8 x i16> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmovsxw.d.256(<8 x i16> %x0, <8 x i32> zeroinitializer, i8 %x2)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.pmovsxw.d.256(<8 x i16> %x0, <8 x i32> %x1, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x i64> @llvm.x86.avx512.mask.pmovsxw.q.128(<8 x i16>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_pmovsxw_q_128(<8 x i16> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxw_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxwq %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vpmovsxwq %xmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxwq %xmm0, %xmm0
; CHECK-NEXT:    vpaddq %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x i64> @llvm.x86.avx512.mask.pmovsxw.q.128(<8 x i16> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call <2 x i64> @llvm.x86.avx512.mask.pmovsxw.q.128(<8 x i16> %x0, <2 x i64> zeroinitializer, i8 %x2)
  %res2 = call <2 x i64> @llvm.x86.avx512.mask.pmovsxw.q.128(<8 x i16> %x0, <2 x i64> %x1, i8 -1)
  %res3 = add <2 x i64> %res, %res1
  %res4 = add <2 x i64> %res3, %res2
  ret <2 x i64> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.pmovsxw.q.256(<8 x i16>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_pmovsxw_q_256(<8 x i16> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovsxw_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpmovsxwq %xmm0, %ymm1 {%k1}
; CHECK-NEXT:    vpmovsxwq %xmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpmovsxwq %xmm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.pmovsxw.q.256(<8 x i16> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.pmovsxw.q.256(<8 x i16> %x0, <4 x i64> zeroinitializer, i8 %x2)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.pmovsxw.q.256(<8 x i16> %x0, <4 x i64> %x1, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}

declare <4 x double> @llvm.x86.avx512.mask.perm.df.256(<4 x double>, i8, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_perm_df_256(<4 x double> %x0, i8 %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_perm_df_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpermpd $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpermpd $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpermpd $3, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[3,0,0,0]
; CHECK-NEXT:    vaddpd %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.perm.df.256(<4 x double> %x0, i8 3, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.perm.df.256(<4 x double> %x0, i8 3, <4 x double> zeroinitializer, i8 %x3)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.perm.df.256(<4 x double> %x0, i8 3, <4 x double> %x2, i8 -1)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res3, %res2
  ret <4 x double> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.perm.di.256(<4 x i64>, i8, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_perm_di_256(<4 x i64> %x0, i8 %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_perm_di_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vpermq $3, %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vpermq $3, %ymm0, %ymm2 {%k1} {z}
; CHECK-NEXT:    vpermq $3, %ymm0, %ymm0
; CHECK-NEXT:    ## ymm0 = ymm0[3,0,0,0]
; CHECK-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.perm.di.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.perm.di.256(<4 x i64> %x0, i8 3, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.perm.di.256(<4 x i64> %x0, i8 3, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}
declare <4 x double> @llvm.x86.avx512.mask.permvar.df.256(<4 x double>, <4 x i64>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_permvar_df_256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_permvar_df_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermpd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpermpd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpermpd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vaddpd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x double> @llvm.x86.avx512.mask.permvar.df.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 %x3)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.permvar.df.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> zeroinitializer, i8 %x3)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.permvar.df.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 -1)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res3, %res2
  ret <4 x double> %res4
}

declare <4 x i64> @llvm.x86.avx512.mask.permvar.di.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_permvar_di_256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_permvar_di_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermq %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpermq %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpermq %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddq %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <4 x i64> @llvm.x86.avx512.mask.permvar.di.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 %x3)
  %res1 = call <4 x i64> @llvm.x86.avx512.mask.permvar.di.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> zeroinitializer, i8 %x3)
  %res2 = call <4 x i64> @llvm.x86.avx512.mask.permvar.di.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  %res3 = add <4 x i64> %res, %res1
  %res4 = add <4 x i64> %res3, %res2
  ret <4 x i64> %res4
}



declare <8 x float> @llvm.x86.avx512.mask.permvar.sf.256(<8 x float>, <8 x i32>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_permvar_sf_256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_permvar_sf_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermps %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpermps %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpermps %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vaddps %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.permvar.sf.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 %x3)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.permvar.sf.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> zeroinitializer, i8 %x3)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.permvar.sf.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 -1)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res3, %res2
  ret <8 x float> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.permvar.si.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_permvar_si_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_permvar_si_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermd %ymm1, %ymm0, %ymm2 {%k1}
; CHECK-NEXT:    vpermd %ymm1, %ymm0, %ymm3 {%k1} {z}
; CHECK-NEXT:    vpermd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm3, %ymm2, %ymm1
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.permvar.si.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.permvar.si.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> zeroinitializer, i8 %x3)
  %res2 = call <8 x i32> @llvm.x86.avx512.mask.permvar.si.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res3, %res2
  ret <8 x i32> %res4
}

declare <2 x double> @llvm.x86.avx512.mask.mova.pd.128(<2 x double>, <2 x double>, i8)

define <2 x double>@test_int_x86_avx512_mask_mova_pd_128(<2 x double> %x0, <2 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovapd %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vmovapd %xmm0, %xmm0 {%k1} {z}
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
    %res = call <2 x double> @llvm.x86.avx512.mask.mova.pd.128(<2 x double> %x0, <2 x double> %x1, i8 %x2)
    %res1 = call <2 x double> @llvm.x86.avx512.mask.mova.pd.128(<2 x double> %x0, <2 x double> zeroinitializer, i8 %x2)
    %res2 = fadd <2 x double> %res, %res1
    ret <2 x double> %res2
}

declare <4 x double> @llvm.x86.avx512.mask.mova.pd.256(<4 x double>, <4 x double>, i8)

define <4 x double>@test_int_x86_avx512_mask_mova_pd_256(<4 x double> %x0, <4 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovapd %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vmovapd %ymm0, %ymm0 {%k1} {z}
; CHECK-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
    %res = call <4 x double> @llvm.x86.avx512.mask.mova.pd.256(<4 x double> %x0, <4 x double> %x1, i8 %x2)
    %res1 = call <4 x double> @llvm.x86.avx512.mask.mova.pd.256(<4 x double> %x0, <4 x double> zeroinitializer, i8 %x2)
    %res2 = fadd <4 x double> %res, %res1
    ret <4 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.mova.ps.128(<4 x float>, <4 x float>, i8)

define <4 x float>@test_int_x86_avx512_mask_mova_ps_128(<4 x float> %x0, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vmovaps %xmm0, %xmm0 {%k1} {z}
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
    %res = call <4 x float> @llvm.x86.avx512.mask.mova.ps.128(<4 x float> %x0, <4 x float> %x1, i8 %x2)
    %res1 = call <4 x float> @llvm.x86.avx512.mask.mova.ps.128(<4 x float> %x0, <4 x float> zeroinitializer, i8 %x2)
    %res2 = fadd <4 x float> %res, %res1
    ret <4 x float> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.mova.ps.256(<8 x float>, <8 x float>, i8)

define <8 x float>@test_int_x86_avx512_mask_mova_ps_256(<8 x float> %x0, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vmovaps %ymm0, %ymm0 {%k1} {z}
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
    %res = call <8 x float> @llvm.x86.avx512.mask.mova.ps.256(<8 x float> %x0, <8 x float> %x1, i8 %x2)
    %res1 = call <8 x float> @llvm.x86.avx512.mask.mova.ps.256(<8 x float> %x0, <8 x float> zeroinitializer, i8 %x2)
    %res2 = fadd <8 x float> %res, %res1
    ret <8 x float> %res2
}

declare <2 x i64> @llvm.x86.avx512.mask.mova.q.128(<2 x i64>, <2 x i64>, i8)

define <2 x i64>@test_int_x86_avx512_mask_mova_q_128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovdqa64 %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vmovdqa64 %xmm0, %xmm0 {%k1} {z}
; CHECK-NEXT:    vpaddq %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
    %res = call <2 x i64> @llvm.x86.avx512.mask.mova.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2)
    %res1 = call <2 x i64> @llvm.x86.avx512.mask.mova.q.128(<2 x i64> %x0, <2 x i64> zeroinitializer, i8 %x2)
    %res2 = add <2 x i64> %res, %res1
    ret <2 x i64> %res2
}

declare <4 x i64> @llvm.x86.avx512.mask.mova.q.256(<4 x i64>, <4 x i64>, i8)

define <4 x i64>@test_int_x86_avx512_mask_mova_q_256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovdqa64 %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vmovdqa64 %ymm0, %ymm0 {%k1} {z}
; CHECK-NEXT:    vpaddq %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
    %res = call <4 x i64> @llvm.x86.avx512.mask.mova.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2)
    %res1 = call <4 x i64> @llvm.x86.avx512.mask.mova.q.256(<4 x i64> %x0, <4 x i64> zeroinitializer, i8 %x2)
    %res2 = add <4 x i64> %res, %res1
    ret <4 x i64> %res2
}

declare <4 x i32> @llvm.x86.avx512.mask.mova.d.128(<4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_mova_d_128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovdqa32 %xmm0, %xmm1 {%k1}
; CHECK-NEXT:    vmovdqa32 %xmm0, %xmm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
    %res = call <4 x i32> @llvm.x86.avx512.mask.mova.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2)
    %res1 = call <4 x i32> @llvm.x86.avx512.mask.mova.d.128(<4 x i32> %x0, <4 x i32> zeroinitializer, i8 %x2)
    %res2 = add <4 x i32> %res, %res1
    ret <4 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.mova.d.256(<8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_mova_d_256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_mova_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovdqa32 %ymm0, %ymm1 {%k1}
; CHECK-NEXT:    vmovdqa32 %ymm0, %ymm0 {%k1} {z}
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
    %res = call <8 x i32> @llvm.x86.avx512.mask.mova.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2)
    %res1 = call <8 x i32> @llvm.x86.avx512.mask.mova.d.256(<8 x i32> %x0, <8 x i32> zeroinitializer, i8 %x2)
    %res2 = add <8 x i32> %res, %res1
    ret <8 x i32> %res2
}

declare void @llvm.x86.avx512.mask.store.pd.128(i8*, <2 x double>, i8)

define void@test_int_x86_avx512_mask_store_pd_128(i8* %ptr1, i8* %ptr2, <2 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovapd %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovapd %xmm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.pd.128(i8* %ptr1, <2 x double> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.store.pd.128(i8* %ptr2, <2 x double> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.store.pd.256(i8*, <4 x double>, i8)

define void@test_int_x86_avx512_mask_store_pd_256(i8* %ptr1, i8* %ptr2, <4 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovapd %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovapd %ymm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.pd.256(i8* %ptr1, <4 x double> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.store.pd.256(i8* %ptr2, <4 x double> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.pd.128(i8*, <2 x double>, i8)

define void@test_int_x86_avx512_mask_storeu_pd_128(i8* %ptr1, i8* %ptr2, <2 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_pd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovupd %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovupd %xmm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.storeu.pd.128(i8* %ptr1, <2 x double> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.storeu.pd.128(i8* %ptr2, <2 x double> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.pd.256(i8*, <4 x double>, i8)

define void@test_int_x86_avx512_mask_storeu_pd_256(i8* %ptr1, i8* %ptr2, <4 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_pd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovupd %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovupd %ymm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.storeu.pd.256(i8* %ptr1, <4 x double> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.storeu.pd.256(i8* %ptr2, <4 x double> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.store.ps.128(i8*, <4 x float>, i8)

define void@test_int_x86_avx512_mask_store_ps_128(i8* %ptr1, i8* %ptr2, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovaps %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovaps %xmm0, (%rsi)
; CHECK-NEXT:    retq
    call void @llvm.x86.avx512.mask.store.ps.128(i8* %ptr1, <4 x float> %x1, i8 %x2)
    call void @llvm.x86.avx512.mask.store.ps.128(i8* %ptr2, <4 x float> %x1, i8 -1)
    ret void
}

declare void @llvm.x86.avx512.mask.store.ps.256(i8*, <8 x float>, i8)

define void@test_int_x86_avx512_mask_store_ps_256(i8* %ptr1, i8* %ptr2, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovaps %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovaps %ymm0, (%rsi)
; CHECK-NEXT:    retq
    call void @llvm.x86.avx512.mask.store.ps.256(i8* %ptr1, <8 x float> %x1, i8 %x2)
    call void @llvm.x86.avx512.mask.store.ps.256(i8* %ptr2, <8 x float> %x1, i8 -1)
    ret void
}

declare void @llvm.x86.avx512.mask.storeu.ps.128(i8*, <4 x float>, i8)

define void@test_int_x86_avx512_mask_storeu_ps_128(i8* %ptr1, i8* %ptr2, <4 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_ps_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovups %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovups %xmm0, (%rsi)
; CHECK-NEXT:    retq
    call void @llvm.x86.avx512.mask.storeu.ps.128(i8* %ptr1, <4 x float> %x1, i8 %x2)
    call void @llvm.x86.avx512.mask.storeu.ps.128(i8* %ptr2, <4 x float> %x1, i8 -1)
    ret void
}

declare void @llvm.x86.avx512.mask.storeu.ps.256(i8*, <8 x float>, i8)

define void@test_int_x86_avx512_mask_storeu_ps_256(i8* %ptr1, i8* %ptr2, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_ps_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovups %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovups %ymm0, (%rsi)
; CHECK-NEXT:    retq
    call void @llvm.x86.avx512.mask.storeu.ps.256(i8* %ptr1, <8 x float> %x1, i8 %x2)
    call void @llvm.x86.avx512.mask.storeu.ps.256(i8* %ptr2, <8 x float> %x1, i8 -1)
    ret void
}

declare void @llvm.x86.avx512.mask.storeu.q.128(i8*, <2 x i64>, i8)

define void@test_int_x86_avx512_mask_storeu_q_128(i8* %ptr1, i8* %ptr2, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu64 %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqu64 %xmm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.storeu.q.128(i8* %ptr1, <2 x i64> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.storeu.q.128(i8* %ptr2, <2 x i64> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.q.256(i8*, <4 x i64>, i8)

define void@test_int_x86_avx512_mask_storeu_q_256(i8* %ptr1, i8* %ptr2, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu64 %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqu64 %ymm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.storeu.q.256(i8* %ptr1, <4 x i64> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.storeu.q.256(i8* %ptr2, <4 x i64> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.d.128(i8*, <4 x i32>, i8)

define void@test_int_x86_avx512_mask_storeu_d_128(i8* %ptr1, i8* %ptr2, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu32 %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqu32 %xmm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.storeu.d.128(i8* %ptr1, <4 x i32> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.storeu.d.128(i8* %ptr2, <4 x i32> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.d.256(i8*, <8 x i32>, i8)

define void@test_int_x86_avx512_mask_storeu_d_256(i8* %ptr1, i8* %ptr2, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_storeu_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqu32 %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqu32 %ymm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.storeu.d.256(i8* %ptr1, <8 x i32> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.storeu.d.256(i8* %ptr2, <8 x i32> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.store.q.128(i8*, <2 x i64>, i8)

define void@test_int_x86_avx512_mask_store_q_128(i8* %ptr1, i8* %ptr2, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqa64 %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqa64 %xmm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.q.128(i8* %ptr1, <2 x i64> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.store.q.128(i8* %ptr2, <2 x i64> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.store.q.256(i8*, <4 x i64>, i8)

define void@test_int_x86_avx512_mask_store_q_256(i8* %ptr1, i8* %ptr2, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqa64 %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqa64 %ymm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.q.256(i8* %ptr1, <4 x i64> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.store.q.256(i8* %ptr2, <4 x i64> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.store.d.128(i8*, <4 x i32>, i8)

define void@test_int_x86_avx512_mask_store_d_128(i8* %ptr1, i8* %ptr2, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqa32 %xmm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqa32 %xmm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.d.128(i8* %ptr1, <4 x i32> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.store.d.128(i8* %ptr2, <4 x i32> %x1, i8 -1)
  ret void
}

declare void @llvm.x86.avx512.mask.store.d.256(i8*, <8 x i32>, i8)

define void@test_int_x86_avx512_mask_store_d_256(i8* %ptr1, i8* %ptr2, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_store_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edx, %k1
; CHECK-NEXT:    vmovdqa32 %ymm0, (%rdi) {%k1}
; CHECK-NEXT:    vmovdqa32 %ymm0, (%rsi)
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.d.256(i8* %ptr1, <8 x i32> %x1, i8 %x2)
  call void @llvm.x86.avx512.mask.store.d.256(i8* %ptr2, <8 x i32> %x1, i8 -1)
  ret void
}

declare <2 x double> @llvm.x86.avx512.mask.fixupimm.pd.128(<2 x double>, <2 x double>, <2 x i64>, i32, i8)

define <2 x double>@test_int_x86_avx512_mask_fixupimm_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x i64> %x2, i8 %x4) {
; CHECK-LABEL:test_int_x86_avx512_mask_fixupimm_pd_128
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmpd  $5, %xmm2, %xmm1, %xmm3 {%k1} 
; CHECK: vpxor  %xmm4, %xmm4, %xmm4
; CHECK: vfixupimmpd  $4, %xmm2, %xmm1, %xmm4 {%k1} {z} 
; CHECK: vfixupimmpd  $3, %xmm2, %xmm1, %xmm0 
; CHECK: vaddpd  %xmm4, %xmm3, %xmm1
; CHECK: vaddpd  %xmm0, %xmm1, %xmm0

  %res = call <2 x double> @llvm.x86.avx512.mask.fixupimm.pd.128(<2 x double> %x0, <2 x double> %x1,<2 x i64> %x2, i32 5, i8 %x4)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.fixupimm.pd.128(<2 x double> zeroinitializer, <2 x double> %x1, <2 x i64> %x2, i32 4, i8 %x4)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.fixupimm.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x i64> %x2, i32 3, i8 -1)
  %res3 = fadd <2 x double> %res, %res1
  %res4 = fadd <2 x double> %res3, %res2
  ret <2 x double> %res4
}

declare <2 x double> @llvm.x86.avx512.maskz.fixupimm.pd.128(<2 x double>, <2 x double>, <2 x i64>, i32, i8)

define <2 x double>@test_int_x86_avx512_maskz_fixupimm_pd_128(<2 x double> %x0, <2 x double> %x1, <2 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_fixupimm_pd_128
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmpd  $5, %xmm2, %xmm1, %xmm3 {%k1} {z}
; CHECK: vpxor  %xmm2, %xmm2, %xmm2
; CHECK: vfixupimmpd  $3, %xmm2, %xmm1, %xmm0 {%k1} {z}
; CHECK: vaddpd  %xmm0, %xmm3, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.maskz.fixupimm.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x i64> %x2, i32 5, i8 %x4)
  %res1 = call <2 x double> @llvm.x86.avx512.maskz.fixupimm.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x i64> zeroinitializer, i32 3, i8 %x4)
  ;%res2 = call <2 x double> @llvm.x86.avx512.maskz.fixupimm.pd.128(<2 x double> %x0, <2 x double> %x1, <2 x i64> %x2, i32 4, i8 -1)
  %res3 = fadd <2 x double> %res, %res1
  ;%res4 = fadd <2 x double> %res3, %res2
  ret <2 x double> %res3
}

declare <4 x double> @llvm.x86.avx512.mask.fixupimm.pd.256(<4 x double>, <4 x double>, <4 x i64>, i32, i8)

define <4 x double>@test_int_x86_avx512_mask_fixupimm_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_fixupimm_pd_256
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmpd  $4, %ymm2, %ymm1, %ymm3 {%k1} 
; CHECK: vpxor  %ymm4, %ymm4, %ymm4
; CHECK: vfixupimmpd  $5, %ymm2, %ymm1, %ymm4 {%k1} {z} 
; CHECK: vfixupimmpd  $3, %ymm2, %ymm1, %ymm0 
; CHECK: vaddpd  %ymm4, %ymm3, %ymm1
; CHECK: vaddpd  %ymm0, %ymm1, %ymm0

  %res = call <4 x double> @llvm.x86.avx512.mask.fixupimm.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x i64> %x2, i32 4, i8 %x4)
  %res1 = call <4 x double> @llvm.x86.avx512.mask.fixupimm.pd.256(<4 x double> zeroinitializer, <4 x double> %x1, <4 x i64> %x2 , i32 5, i8 %x4)
  %res2 = call <4 x double> @llvm.x86.avx512.mask.fixupimm.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x i64> %x2, i32 3, i8 -1)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res3, %res2
  ret <4 x double> %res4
}

declare <4 x double> @llvm.x86.avx512.maskz.fixupimm.pd.256(<4 x double>, <4 x double>, <4 x i64>, i32, i8)

define <4 x double>@test_int_x86_avx512_maskz_fixupimm_pd_256(<4 x double> %x0, <4 x double> %x1, <4 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_fixupimm_pd_256
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmpd  $5, %ymm2, %ymm1, %ymm3 {%k1} {z}
; CHECK: vpxor  %ymm4, %ymm4, %ymm4
; CHECK: vmovaps  %zmm0, %zmm5
; CHECK: vfixupimmpd  $4, %ymm4, %ymm1, %ymm5 {%k1} {z}
; CHECK: vfixupimmpd  $3, %ymm2, %ymm1, %ymm0 
; CHECK: vaddpd  %ymm5, %ymm3, %ymm1
; CHECK: vaddpd  %ymm0, %ymm1, %ymm0

  %res = call <4 x double> @llvm.x86.avx512.maskz.fixupimm.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x i64> %x2, i32 5, i8 %x4)
  %res1 = call <4 x double> @llvm.x86.avx512.maskz.fixupimm.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x i64> zeroinitializer, i32 4, i8 %x4)
  %res2 = call <4 x double> @llvm.x86.avx512.maskz.fixupimm.pd.256(<4 x double> %x0, <4 x double> %x1, <4 x i64> %x2, i32 3, i8 -1)
  %res3 = fadd <4 x double> %res, %res1
  %res4 = fadd <4 x double> %res3, %res2
  ret <4 x double> %res4
}

declare <4 x float> @llvm.x86.avx512.mask.fixupimm.ps.128(<4 x float>, <4 x float>, <4 x i32>, i32, i8)

define <4 x float>@test_int_x86_avx512_mask_fixupimm_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_fixupimm_ps_128
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmps  $5, %xmm2, %xmm1, %xmm3 {%k1} 
; CHECK: vmovaps  %zmm0, %zmm4
; CHECK: vfixupimmps  $5, %xmm2, %xmm1, %xmm4 
; CHECK: vpxor  %xmm2, %xmm2, %xmm2
; CHECK: vfixupimmps  $5, %xmm2, %xmm1, %xmm0 {%k1} 
; CHECK: vaddps  %xmm0, %xmm3, %xmm0
; CHECK: vaddps  %xmm4, %xmm0, %xmm0

  %res = call <4 x float> @llvm.x86.avx512.mask.fixupimm.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x i32> %x2, i32 5, i8 %x4)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.fixupimm.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x i32> zeroinitializer, i32 5, i8 %x4)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.fixupimm.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x i32> %x2, i32 5, i8 -1)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res3, %res2
  ret <4 x float> %res4
}

declare <4 x float> @llvm.x86.avx512.maskz.fixupimm.ps.128(<4 x float>, <4 x float>, <4 x i32>, i32, i8)

define <4 x float>@test_int_x86_avx512_maskz_fixupimm_ps_128(<4 x float> %x0, <4 x float> %x1, <4 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_fixupimm_ps_128
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmps  $5, %xmm2, %xmm1, %xmm3 {%k1} {z} 
; CHECK: vmovaps  %zmm0, %zmm4
; CHECK: vfixupimmps  $5, %xmm2, %xmm1, %xmm4 
; CHECK: vpxor  %xmm2, %xmm2, %xmm2
; CHECK: vfixupimmps  $5, %xmm2, %xmm1, %xmm0 {%k1} {z} 
; CHECK: vaddps  %xmm0, %xmm3, %xmm0
; CHECK: vaddps  %xmm4, %xmm0, %xmm0

  %res = call <4 x float> @llvm.x86.avx512.maskz.fixupimm.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x i32> %x2, i32 5, i8 %x4)
  %res1 = call <4 x float> @llvm.x86.avx512.maskz.fixupimm.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x i32> zeroinitializer, i32 5, i8 %x4)
  %res2 = call <4 x float> @llvm.x86.avx512.maskz.fixupimm.ps.128(<4 x float> %x0, <4 x float> %x1, <4 x i32> %x2, i32 5, i8 -1)
  %res3 = fadd <4 x float> %res, %res1
  %res4 = fadd <4 x float> %res3, %res2
  ret <4 x float> %res4
}

declare <8 x float> @llvm.x86.avx512.mask.fixupimm.ps.256(<8 x float>, <8 x float>, <8 x i32>, i32, i8)

define <8 x float>@test_int_x86_avx512_mask_fixupimm_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_fixupimm_ps_256
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmps  $5, %ymm2, %ymm1, %ymm3 {%k1} 
; CHECK: vmovaps  %zmm0, %zmm4
; CHECK: vfixupimmps  $5, %ymm2, %ymm1, %ymm4 
; CHECK: vpxor  %ymm2, %ymm2, %ymm2
; CHECK: vfixupimmps  $5, %ymm2, %ymm1, %ymm0 {%k1} 
; CHECK: vaddps  %ymm0, %ymm3, %ymm0
; CHECK: vaddps  %ymm4, %ymm0, %ymm0

  %res = call <8 x float> @llvm.x86.avx512.mask.fixupimm.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x i32> %x2, i32 5, i8 %x4)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.fixupimm.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x i32> zeroinitializer, i32 5, i8 %x4)
  %res2 = call <8 x float> @llvm.x86.avx512.mask.fixupimm.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x i32> %x2, i32 5, i8 -1)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res3, %res2
  ret <8 x float> %res4
}

declare <8 x float> @llvm.x86.avx512.maskz.fixupimm.ps.256(<8 x float>, <8 x float>, <8 x i32>, i32, i8)

define <8 x float>@test_int_x86_avx512_maskz_fixupimm_ps_256(<8 x float> %x0, <8 x float> %x1, <8 x i32> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_fixupimm_ps_256
; CHECK: kmovw  %edi, %k1
; CHECK: vmovaps  %zmm0, %zmm3
; CHECK: vfixupimmps  $5, %ymm2, %ymm1, %ymm3 {%k1} {z} 
; CHECK: vmovaps  %zmm0, %zmm4
; CHECK: vfixupimmps  $5, %ymm2, %ymm1, %ymm4 
; CHECK: vpxor  %ymm2, %ymm2, %ymm2
; CHECK: vfixupimmps  $5, %ymm2, %ymm1, %ymm0 {%k1} {z} 
; CHECK: vaddps  %ymm0, %ymm3, %ymm0
; CHECK: vaddps  %ymm4, %ymm0, %ymm0

  %res = call <8 x float> @llvm.x86.avx512.maskz.fixupimm.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x i32> %x2, i32 5, i8 %x4)
  %res1 = call <8 x float> @llvm.x86.avx512.maskz.fixupimm.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x i32> zeroinitializer, i32 5, i8 %x4)
  %res2 = call <8 x float> @llvm.x86.avx512.maskz.fixupimm.ps.256(<8 x float> %x0, <8 x float> %x1, <8 x i32> %x2, i32 5, i8 -1)
  %res3 = fadd <8 x float> %res, %res1
  %res4 = fadd <8 x float> %res3, %res2
  ret <8 x float> %res4
}

declare i8 @llvm.x86.avx512.ptestm.d.128(<4 x i32>, <4 x i32>,i8)

define i8@test_int_x86_avx512_ptestm_d_128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestm_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 ## encoding: [0xc5,0xf8,0x92,0xcf]
; CHECK-NEXT:    vptestmd %xmm1, %xmm0, %k0 {%k1} ## encoding: [0x62,0xf2,0x7d,0x09,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %ecx ## encoding: [0xc5,0xf8,0x93,0xc8]
; CHECK-NEXT:    vptestmd %xmm1, %xmm0, %k0 ## encoding: [0x62,0xf2,0x7d,0x08,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %eax ## encoding: [0xc5,0xf8,0x93,0xc0]
; CHECK-NEXT:    addb %cl, %al ## encoding: [0x00,0xc8]
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call i8 @llvm.x86.avx512.ptestm.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestm.d.128(<4 x i32> %x0, <4 x i32> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

declare i8 @llvm.x86.avx512.ptestm.d.256(<8 x i32>, <8 x i32>, i8)

define i8@test_int_x86_avx512_ptestm_d_256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestm_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 ## encoding: [0xc5,0xf8,0x92,0xcf]
; CHECK-NEXT:    vptestmd %ymm1, %ymm0, %k0 {%k1} ## encoding: [0x62,0xf2,0x7d,0x29,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %ecx ## encoding: [0xc5,0xf8,0x93,0xc8]
; CHECK-NEXT:    vptestmd %ymm1, %ymm0, %k0 ## encoding: [0x62,0xf2,0x7d,0x28,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %eax ## encoding: [0xc5,0xf8,0x93,0xc0]
; CHECK-NEXT:    addb %cl, %al ## encoding: [0x00,0xc8]
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call i8 @llvm.x86.avx512.ptestm.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestm.d.256(<8 x i32> %x0, <8 x i32> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

declare i8 @llvm.x86.avx512.ptestm.q.128(<2 x i64>, <2 x i64>, i8)

define i8@test_int_x86_avx512_ptestm_q_128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestm_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 ## encoding: [0xc5,0xf8,0x92,0xcf]
; CHECK-NEXT:    vptestmq %xmm1, %xmm0, %k0 {%k1} ## encoding: [0x62,0xf2,0xfd,0x09,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %ecx ## encoding: [0xc5,0xf8,0x93,0xc8]
; CHECK-NEXT:    vptestmq %xmm1, %xmm0, %k0 ## encoding: [0x62,0xf2,0xfd,0x08,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %eax ## encoding: [0xc5,0xf8,0x93,0xc0]
; CHECK-NEXT:    addb %cl, %al ## encoding: [0x00,0xc8]
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call i8 @llvm.x86.avx512.ptestm.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestm.q.128(<2 x i64> %x0, <2 x i64> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

declare i8 @llvm.x86.avx512.ptestm.q.256(<4 x i64>, <4 x i64>, i8)

define i8@test_int_x86_avx512_ptestm_q_256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestm_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 ## encoding: [0xc5,0xf8,0x92,0xcf]
; CHECK-NEXT:    vptestmq %ymm1, %ymm0, %k0 {%k1} ## encoding: [0x62,0xf2,0xfd,0x29,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %ecx ## encoding: [0xc5,0xf8,0x93,0xc8]
; CHECK-NEXT:    vptestmq %ymm1, %ymm0, %k0 ## encoding: [0x62,0xf2,0xfd,0x28,0x27,0xc1]
; CHECK-NEXT:    kmovw %k0, %eax ## encoding: [0xc5,0xf8,0x93,0xc0]
; CHECK-NEXT:    addb %cl, %al ## encoding: [0x00,0xc8]
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call i8 @llvm.x86.avx512.ptestm.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestm.q.256(<4 x i64> %x0, <4 x i64> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

declare i8 @llvm.x86.avx512.ptestnm.d.128(<4 x i32>, <4 x i32>, i8 %x2)

define i8@test_int_x86_avx512_ptestnm_d_128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestnm_d_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vptestnmd %xmm1, %xmm0, %k0 {%k1} 
; CHECK-NEXT:    kmovw %k0, %ecx 
; CHECK-NEXT:    vptestnmd %xmm1, %xmm0, %k0 
; CHECK-NEXT:    kmovw %k0, %eax 
; CHECK-NEXT:    addb %cl, %al 
; CHECK-NEXT:    retq 
  %res = call i8 @llvm.x86.avx512.ptestnm.d.128(<4 x i32> %x0, <4 x i32> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestnm.d.128(<4 x i32> %x0, <4 x i32> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

declare i8 @llvm.x86.avx512.ptestnm.d.256(<8 x i32>, <8 x i32>, i8 %x2)

define i8@test_int_x86_avx512_ptestnm_d_256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestnm_d_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vptestnmd %ymm1, %ymm0, %k0 {%k1} 
; CHECK-NEXT:    kmovw %k0, %ecx 
; CHECK-NEXT:    vptestnmd %ymm1, %ymm0, %k0 
; CHECK-NEXT:    kmovw %k0, %eax 
; CHECK-NEXT:    addb %cl, %al 
; CHECK-NEXT:    retq 
  %res = call i8 @llvm.x86.avx512.ptestnm.d.256(<8 x i32> %x0, <8 x i32> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestnm.d.256(<8 x i32> %x0, <8 x i32> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

declare i8 @llvm.x86.avx512.ptestnm.q.128(<2 x i64>, <2 x i64>, i8 %x2)

define i8@test_int_x86_avx512_ptestnm_q_128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestnm_q_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vptestnmq %xmm1, %xmm0, %k0 {%k1} 
; CHECK-NEXT:    kmovw %k0, %ecx 
; CHECK-NEXT:    vptestnmq %xmm1, %xmm0, %k0 
; CHECK-NEXT:    kmovw %k0, %eax 
; CHECK-NEXT:    addb %cl, %al 
; CHECK-NEXT:    retq 
  %res = call i8 @llvm.x86.avx512.ptestnm.q.128(<2 x i64> %x0, <2 x i64> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestnm.q.128(<2 x i64> %x0, <2 x i64> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

declare i8 @llvm.x86.avx512.ptestnm.q.256(<4 x i64>, <4 x i64>, i8 %x2)

define i8@test_int_x86_avx512_ptestnm_q_256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_ptestnm_q_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vptestnmq %ymm1, %ymm0, %k0 {%k1} 
; CHECK-NEXT:    kmovw %k0, %ecx 
; CHECK-NEXT:    vptestnmq %ymm1, %ymm0, %k0 
; CHECK-NEXT:    kmovw %k0, %eax 
; CHECK-NEXT:    addb %cl, %al 
; CHECK-NEXT:    retq 
  %res = call i8 @llvm.x86.avx512.ptestnm.q.256(<4 x i64> %x0, <4 x i64> %x1, i8 %x2)
  %res1 = call i8 @llvm.x86.avx512.ptestnm.q.256(<4 x i64> %x0, <4 x i64> %x1, i8-1)
  %res2 = add i8 %res, %res1
  ret i8 %res2
}

