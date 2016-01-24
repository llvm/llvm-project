; RUN: llc -mtriple=x86_64-apple-darwin -mattr=avx < %s | FileCheck %s --check-prefix=AVX1
; RUN: llc -mtriple=x86_64-apple-darwin -mattr=avx2 < %s | FileCheck %s --check-prefix=AVX2
; RUN: llc -mtriple=x86_64-apple-darwin -mattr=avx512f < %s | FileCheck %s --check-prefix=AVX512
; RUN: llc -mtriple=x86_64-apple-darwin -mattr=avx512f,avx512bw,avx512vl < %s | FileCheck %s --check-prefix=SKX

; FIXME: AVX1 supports vmaskmovp[s/d], so its codegen should be identical to AVX2 for FP cases.
; For integer cases, AVX1 could use the FP instructions in place of vpmaskmov?

; To test for the case where masked load/store is not legal, we should add a run with a target 
; that does not have AVX, but that case should probably be a separate test file using less tests
; because it takes over 1.2 seconds to codegen these tests on Haswell 4GHz if there's no maskmov. 

define <16 x i32> @test1(<16 x i32> %trigger, <16 x i32>* %addr) {
; Bypassing exact checking here because it's over 300 lines.
; AVX1-LABEL: test1:
; AVX1-NOT:   maskmov

; AVX2-LABEL: test1:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpmaskmovd 32(%rdi), %ymm1, %ymm1
; AVX2-NEXT:    vpmaskmovd (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test1:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512-NEXT:    vpcmpeqd %zmm1, %zmm0, %k1
; AVX512-NEXT:    vmovdqu32 (%rdi), %zmm0 {%k1} {z}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test1:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; SKX-NEXT:    vpcmpeqd %zmm1, %zmm0, %k1
; SKX-NEXT:    vmovdqu32 (%rdi), %zmm0 {%k1} {z}
; SKX-NEXT:    retq
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  %res = call <16 x i32> @llvm.masked.load.v16i32(<16 x i32>* %addr, i32 4, <16 x i1>%mask, <16 x i32>undef)
  ret <16 x i32> %res
}

define <16 x i32> @test2(<16 x i32> %trigger, <16 x i32>* %addr) {
; Bypassing exact checking here because it's over 300 lines.
; AVX1-LABEL: test2:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test2:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpmaskmovd 32(%rdi), %ymm1, %ymm1
; AVX2-NEXT:    vpmaskmovd (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test2:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512-NEXT:    vpcmpeqd %zmm1, %zmm0, %k1
; AVX512-NEXT:    vmovdqu32 (%rdi), %zmm0 {%k1} {z}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test2:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; SKX-NEXT:    vpcmpeqd %zmm1, %zmm0, %k1
; SKX-NEXT:    vmovdqu32 (%rdi), %zmm0 {%k1} {z}
; SKX-NEXT:    retq
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  %res = call <16 x i32> @llvm.masked.load.v16i32(<16 x i32>* %addr, i32 4, <16 x i1>%mask, <16 x i32>zeroinitializer)
  ret <16 x i32> %res
}

define void @test3(<16 x i32> %trigger, <16 x i32>* %addr, <16 x i32> %val) {
; Bypassing exact checking here because it's over 300 lines.
; AVX1-LABEL: test3:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test3:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm4, %ymm4, %ymm4
; AVX2-NEXT:    vpcmpeqd %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpeqd %ymm4, %ymm1, %ymm1
; AVX2-NEXT:    vpmaskmovd %ymm3, %ymm1, 32(%rdi)
; AVX2-NEXT:    vpmaskmovd %ymm2, %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test3:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; AVX512-NEXT:    vpcmpeqd %zmm2, %zmm0, %k1
; AVX512-NEXT:    vmovdqu32 %zmm1, (%rdi) {%k1}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test3:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; SKX-NEXT:    vpcmpeqd %zmm2, %zmm0, %k1
; SKX-NEXT:    vmovdqu32 %zmm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v16i32(<16 x i32>%val, <16 x i32>* %addr, i32 4, <16 x i1>%mask)
  ret void
}

define <16 x float> @test4(<16 x i32> %trigger, <16 x float>* %addr, <16 x float> %dst) {
; Bypassing exact checking here because it's over 300 lines.
; AVX1-LABEL: test4:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test4:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm4, %ymm4, %ymm4
; AVX2-NEXT:    vpcmpeqd %ymm4, %ymm1, %ymm1
; AVX2-NEXT:    vpcmpeqd %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vmaskmovps (%rdi), %ymm0, %ymm4
; AVX2-NEXT:    vblendvps %ymm0, %ymm4, %ymm2, %ymm0
; AVX2-NEXT:    vmaskmovps 32(%rdi), %ymm1, %ymm2
; AVX2-NEXT:    vblendvps %ymm1, %ymm2, %ymm3, %ymm1
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test4:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; AVX512-NEXT:    vpcmpeqd %zmm2, %zmm0, %k1
; AVX512-NEXT:    vmovups (%rdi), %zmm1 {%k1}
; AVX512-NEXT:    vmovaps %zmm1, %zmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test4:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; SKX-NEXT:    vpcmpeqd %zmm2, %zmm0, %k1
; SKX-NEXT:    vmovups (%rdi), %zmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  %res = call <16 x float> @llvm.masked.load.v16f32(<16 x float>* %addr, i32 4, <16 x i1>%mask, <16 x float> %dst)
  ret <16 x float> %res
}

define <8 x double> @test5(<8 x i32> %trigger, <8 x double>* %addr, <8 x double> %dst) {
; Bypassing exact checking here because it's over 100 lines.
; AVX1-LABEL: test5:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test5:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm3
; AVX2-NEXT:    vpxor %xmm4, %xmm4, %xmm4
; AVX2-NEXT:    vpcmpeqd %xmm4, %xmm3, %xmm3
; AVX2-NEXT:    vpmovsxdq %xmm3, %ymm3
; AVX2-NEXT:    vpcmpeqd %xmm4, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vmaskmovpd (%rdi), %ymm0, %ymm4
; AVX2-NEXT:    vblendvpd %ymm0, %ymm4, %ymm1, %ymm0
; AVX2-NEXT:    vmaskmovpd 32(%rdi), %ymm3, %ymm1
; AVX2-NEXT:    vblendvpd %ymm3, %ymm1, %ymm2, %ymm1
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test5:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX512-NEXT:    vpcmpeqd %zmm2, %zmm0, %k1
; AVX512-NEXT:    vmovupd (%rdi), %zmm1 {%k1}
; AVX512-NEXT:    vmovaps %zmm1, %zmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test5:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; SKX-NEXT:    vpcmpeqd %ymm2, %ymm0, %k1
; SKX-NEXT:    vmovupd (%rdi), %zmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  %res = call <8 x double> @llvm.masked.load.v8f64(<8 x double>* %addr, i32 4, <8 x i1>%mask, <8 x double>%dst)
  ret <8 x double> %res
}

define <2 x double> @test6(<2 x i64> %trigger, <2 x double>* %addr, <2 x double> %dst) {
; AVX1-LABEL: test6:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm3
; AVX1-NEXT:    vpextrb $0, %xmm3, %eax
; AVX1-NEXT:    ## implicit-def: %XMM2
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB5_2
; AVX1-NEXT:  ## BB#1: ## %cond.load
; AVX1-NEXT:    vmovsd {{.*#+}} xmm2 = mem[0],zero
; AVX1-NEXT:  LBB5_2: ## %else
; AVX1-NEXT:    vpextrb $8, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB5_4
; AVX1-NEXT:  ## BB#3: ## %cond.load1
; AVX1-NEXT:    vmovhpd 8(%rdi), %xmm2, %xmm2
; AVX1-NEXT:  LBB5_4: ## %else2
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpcmpeqq %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vblendvpd %xmm0, %xmm2, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test6:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vmaskmovpd (%rdi), %xmm0, %xmm2
; AVX2-NEXT:    vblendvpd %xmm0, %xmm2, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test6:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vmaskmovpd (%rdi), %xmm0, %xmm2
; AVX512-NEXT:    vblendvpd %xmm0, %xmm2, %xmm1, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test6:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpcmpeqq %xmm2, %xmm0, %k1
; SKX-NEXT:    vmovupd (%rdi), %xmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <2 x i64> %trigger, zeroinitializer
  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %addr, i32 4, <2 x i1>%mask, <2 x double>%dst)
  ret <2 x double> %res
}

define <4 x float> @test7(<4 x i32> %trigger, <4 x float>* %addr, <4 x float> %dst) {
; AVX1-LABEL: test7:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm3
; AVX1-NEXT:    vpextrb $0, %xmm3, %eax
; AVX1-NEXT:    ## implicit-def: %XMM2
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB6_2
; AVX1-NEXT:  ## BB#1: ## %cond.load
; AVX1-NEXT:    vmovss {{.*#+}} xmm2 = mem[0],zero,zero,zero
; AVX1-NEXT:  LBB6_2: ## %else
; AVX1-NEXT:    vpextrb $4, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB6_4
; AVX1-NEXT:  ## BB#3: ## %cond.load1
; AVX1-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0],mem[0],xmm2[2,3]
; AVX1-NEXT:  LBB6_4: ## %else2
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpcmpeqd %xmm3, %xmm0, %xmm3
; AVX1-NEXT:    vpextrb $8, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB6_6
; AVX1-NEXT:  ## BB#5: ## %cond.load4
; AVX1-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0,1],mem[0],xmm2[3]
; AVX1-NEXT:  LBB6_6: ## %else5
; AVX1-NEXT:    vpextrb $12, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB6_8
; AVX1-NEXT:  ## BB#7: ## %cond.load7
; AVX1-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0,1,2],mem[0]
; AVX1-NEXT:  LBB6_8: ## %else8
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpcmpeqd %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test7:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm2
; AVX2-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test7:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm2
; AVX512-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test7:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpcmpeqd %xmm2, %xmm0, %k1
; SKX-NEXT:    vmovups (%rdi), %xmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x float> @llvm.masked.load.v4f32(<4 x float>* %addr, i32 4, <4 x i1>%mask, <4 x float>%dst)
  ret <4 x float> %res
}

define <4 x i32> @test8(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %dst) {
; AVX1-LABEL: test8:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm3
; AVX1-NEXT:    vpextrb $0, %xmm3, %eax
; AVX1-NEXT:    ## implicit-def: %XMM2
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB7_2
; AVX1-NEXT:  ## BB#1: ## %cond.load
; AVX1-NEXT:    vmovd {{.*#+}} xmm2 = mem[0],zero,zero,zero
; AVX1-NEXT:  LBB7_2: ## %else
; AVX1-NEXT:    vpextrb $4, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB7_4
; AVX1-NEXT:  ## BB#3: ## %cond.load1
; AVX1-NEXT:    vpinsrd $1, 4(%rdi), %xmm2, %xmm2
; AVX1-NEXT:  LBB7_4: ## %else2
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpcmpeqd %xmm3, %xmm0, %xmm3
; AVX1-NEXT:    vpextrb $8, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB7_6
; AVX1-NEXT:  ## BB#5: ## %cond.load4
; AVX1-NEXT:    vpinsrd $2, 8(%rdi), %xmm2, %xmm2
; AVX1-NEXT:  LBB7_6: ## %else5
; AVX1-NEXT:    vpextrb $12, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB7_8
; AVX1-NEXT:  ## BB#7: ## %cond.load7
; AVX1-NEXT:    vpinsrd $3, 12(%rdi), %xmm2, %xmm2
; AVX1-NEXT:  LBB7_8: ## %else8
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpcmpeqd %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test8:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpmaskmovd (%rdi), %xmm0, %xmm2
; AVX2-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test8:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vpmaskmovd (%rdi), %xmm0, %xmm2
; AVX512-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test8:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpcmpeqd %xmm2, %xmm0, %k1
; SKX-NEXT:    vmovdqu32 (%rdi), %xmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x i32> @llvm.masked.load.v4i32(<4 x i32>* %addr, i32 4, <4 x i1>%mask, <4 x i32>%dst)
  ret <4 x i32> %res
}

define void @test9(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %val) {
; AVX1-LABEL: test9:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm2
; AVX1-NEXT:    vpextrb $0, %xmm2, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB8_2
; AVX1-NEXT:  ## BB#1: ## %cond.store
; AVX1-NEXT:    vmovd %xmm1, (%rdi)
; AVX1-NEXT:  LBB8_2: ## %else
; AVX1-NEXT:    vpextrb $4, %xmm2, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB8_4
; AVX1-NEXT:  ## BB#3: ## %cond.store1
; AVX1-NEXT:    vpextrd $1, %xmm1, 4(%rdi)
; AVX1-NEXT:  LBB8_4: ## %else2
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpextrb $8, %xmm0, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB8_6
; AVX1-NEXT:  ## BB#5: ## %cond.store3
; AVX1-NEXT:    vpextrd $2, %xmm1, 8(%rdi)
; AVX1-NEXT:  LBB8_6: ## %else4
; AVX1-NEXT:    vpextrb $12, %xmm0, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB8_8
; AVX1-NEXT:  ## BB#7: ## %cond.store5
; AVX1-NEXT:    vpextrd $3, %xmm1, 12(%rdi)
; AVX1-NEXT:  LBB8_8: ## %else6
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test9:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test9:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX512-NEXT:    retq
;
; SKX-LABEL: test9:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpcmpeqd %xmm2, %xmm0, %k1
; SKX-NEXT:    vmovdqu32 %xmm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v4i32(<4 x i32>%val, <4 x i32>* %addr, i32 4, <4 x i1>%mask)
  ret void
}

define <4 x double> @test10(<4 x i32> %trigger, <4 x double>* %addr, <4 x double> %dst) {
; AVX1-LABEL: test10:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm3
; AVX1-NEXT:    vpextrb $0, %xmm3, %eax
; AVX1-NEXT:    ## implicit-def: %YMM2
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB9_2
; AVX1-NEXT:  ## BB#1: ## %cond.load
; AVX1-NEXT:    vmovsd {{.*#+}} xmm2 = mem[0],zero
; AVX1-NEXT:  LBB9_2: ## %else
; AVX1-NEXT:    vpextrb $4, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB9_4
; AVX1-NEXT:  ## BB#3: ## %cond.load1
; AVX1-NEXT:    vmovhpd 8(%rdi), %xmm2, %xmm3
; AVX1-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3]
; AVX1-NEXT:  LBB9_4: ## %else2
; AVX1-NEXT:    vxorpd %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpcmpeqd %xmm3, %xmm0, %xmm3
; AVX1-NEXT:    vpextrb $8, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB9_6
; AVX1-NEXT:  ## BB#5: ## %cond.load4
; AVX1-NEXT:    vextractf128 $1, %ymm2, %xmm4
; AVX1-NEXT:    vmovlpd 16(%rdi), %xmm4, %xmm4
; AVX1-NEXT:    vinsertf128 $1, %xmm4, %ymm2, %ymm2
; AVX1-NEXT:  LBB9_6: ## %else5
; AVX1-NEXT:    vpextrb $12, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB9_8
; AVX1-NEXT:  ## BB#7: ## %cond.load7
; AVX1-NEXT:    vextractf128 $1, %ymm2, %xmm3
; AVX1-NEXT:    vmovhpd 24(%rdi), %xmm3, %xmm3
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm2, %ymm2
; AVX1-NEXT:  LBB9_8: ## %else8
; AVX1-NEXT:    vxorpd %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpcmpeqd %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm3, %ymm0
; AVX1-NEXT:    vblendvpd %ymm0, %ymm2, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test10:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vmaskmovpd (%rdi), %ymm0, %ymm2
; AVX2-NEXT:    vblendvpd %ymm0, %ymm2, %ymm1, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test10:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX512-NEXT:    vmaskmovpd (%rdi), %ymm0, %ymm2
; AVX512-NEXT:    vblendvpd %ymm0, %ymm2, %ymm1, %ymm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test10:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpcmpeqd %xmm2, %xmm0, %k1
; SKX-NEXT:    vmovapd (%rdi), %ymm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x double> @llvm.masked.load.v4f64(<4 x double>* %addr, i32 32, <4 x i1>%mask, <4 x double>%dst)
  ret <4 x double> %res
}

define <8 x float> @test11a(<8 x i32> %trigger, <8 x float>* %addr, <8 x float> %dst) {
; Bypassing exact checking here because it's over 100 lines.
; AVX1-LABEL: test11a:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test11a:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vmaskmovps (%rdi), %ymm0, %ymm2
; AVX2-NEXT:    vblendvps %ymm0, %ymm2, %ymm1, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test11a:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX512-NEXT:    vpcmpeqd %zmm2, %zmm0, %k0
; AVX512-NEXT:    kshiftlw $8, %k0, %k0
; AVX512-NEXT:    kshiftrw $8, %k0, %k1
; AVX512-NEXT:    vmovups (%rdi), %zmm1 {%k1}
; AVX512-NEXT:    vmovaps %zmm1, %zmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test11a:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; SKX-NEXT:    vpcmpeqd %ymm2, %ymm0, %k1
; SKX-NEXT:    vmovaps (%rdi), %ymm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  %res = call <8 x float> @llvm.masked.load.v8f32(<8 x float>* %addr, i32 32, <8 x i1>%mask, <8 x float>%dst)
  ret <8 x float> %res
}

define <8 x i32> @test11b(<8 x i1> %mask, <8 x i32>* %addr, <8 x i32> %dst) {
; Bypassing exact checking here because it's over 70 lines.
; AVX1-LABEL: test11b:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test11b:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vpslld $31, %ymm0, %ymm0
; AVX2-NEXT:    vpsrad $31, %ymm0, %ymm0
; AVX2-NEXT:    vpmaskmovd (%rdi), %ymm0, %ymm2
; AVX2-NEXT:    vblendvps %ymm0, %ymm2, %ymm1, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test11b:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxwq %xmm0, %zmm0
; AVX512-NEXT:    vpsllq $63, %zmm0, %zmm0
; AVX512-NEXT:    vptestmq %zmm0, %zmm0, %k0
; AVX512-NEXT:    kshiftlw $8, %k0, %k0
; AVX512-NEXT:    kshiftrw $8, %k0, %k1
; AVX512-NEXT:    vmovdqu32 (%rdi), %zmm1 {%k1}
; AVX512-NEXT:    vmovaps %zmm1, %zmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test11b:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $15, %xmm0, %xmm0
; SKX-NEXT:    vpmovw2m %xmm0, %k1
; SKX-NEXT:    vmovdqu32 (%rdi), %ymm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %res = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %addr, i32 4, <8 x i1>%mask, <8 x i32>%dst)
  ret <8 x i32> %res
}

define <8 x float> @test11c(<8 x i1> %mask, <8 x float>* %addr) {
; Bypassing exact checking here because it's over 70 lines.
; AVX1-LABEL: test11c:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test11c:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vpslld $31, %ymm0, %ymm0
; AVX2-NEXT:    vpsrad $31, %ymm0, %ymm0
; AVX2-NEXT:    vmaskmovps (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test11c:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxwq %xmm0, %zmm0
; AVX512-NEXT:    vpsllq $63, %zmm0, %zmm0
; AVX512-NEXT:    vptestmq %zmm0, %zmm0, %k0
; AVX512-NEXT:    kshiftlw $8, %k0, %k0
; AVX512-NEXT:    kshiftrw $8, %k0, %k1
; AVX512-NEXT:    vmovups (%rdi), %zmm0 {%k1} {z}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test11c:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $15, %xmm0, %xmm0
; SKX-NEXT:    vpmovw2m %xmm0, %k1
; SKX-NEXT:    vmovaps (%rdi), %ymm0 {%k1} {z}
; SKX-NEXT:    retq
  %res = call <8 x float> @llvm.masked.load.v8f32(<8 x float>* %addr, i32 32, <8 x i1> %mask, <8 x float> zeroinitializer)
  ret <8 x float> %res
}

define <8 x i32> @test11d(<8 x i1> %mask, <8 x i32>* %addr) {
; Bypassing exact checking here because it's over 70 lines.
; AVX1-LABEL: test11d:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test11d:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vpslld $31, %ymm0, %ymm0
; AVX2-NEXT:    vpsrad $31, %ymm0, %ymm0
; AVX2-NEXT:    vpmaskmovd (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test11d:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxwq %xmm0, %zmm0
; AVX512-NEXT:    vpsllq $63, %zmm0, %zmm0
; AVX512-NEXT:    vptestmq %zmm0, %zmm0, %k0
; AVX512-NEXT:    kshiftlw $8, %k0, %k0
; AVX512-NEXT:    kshiftrw $8, %k0, %k1
; AVX512-NEXT:    vmovdqu32 (%rdi), %zmm0 {%k1} {z}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test11d:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $15, %xmm0, %xmm0
; SKX-NEXT:    vpmovw2m %xmm0, %k1
; SKX-NEXT:    vmovdqu32 (%rdi), %ymm0 {%k1} {z}
; SKX-NEXT:    retq
  %res = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %addr, i32 4, <8 x i1> %mask, <8 x i32> zeroinitializer)
  ret <8 x i32> %res
}

define void @test12(<8 x i32> %trigger, <8 x i32>* %addr, <8 x i32> %val) {
; Bypassing exact checking here because it's over 90 lines.
; AVX1-LABEL: test12:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test12:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpmaskmovd %ymm1, %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test12:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX512-NEXT:    vpcmpeqd %zmm2, %zmm0, %k0
; AVX512-NEXT:    kshiftlw $8, %k0, %k0
; AVX512-NEXT:    kshiftrw $8, %k0, %k1
; AVX512-NEXT:    vmovdqu32 %zmm1, (%rdi) {%k1}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test12:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; SKX-NEXT:    vpcmpeqd %ymm2, %ymm0, %k1
; SKX-NEXT:    vmovdqu32 %ymm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v8i32(<8 x i32>%val, <8 x i32>* %addr, i32 4, <8 x i1>%mask)
  ret void
}

define void @test13(<16 x i32> %trigger, <16 x float>* %addr, <16 x float> %val) {
; Bypassing exact checking here because it's over 300 lines.
; AVX1-LABEL: test13:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test13:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm4, %ymm4, %ymm4
; AVX2-NEXT:    vpcmpeqd %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpeqd %ymm4, %ymm1, %ymm1
; AVX2-NEXT:    vmaskmovps %ymm3, %ymm1, 32(%rdi)
; AVX2-NEXT:    vmaskmovps %ymm2, %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test13:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; AVX512-NEXT:    vpcmpeqd %zmm2, %zmm0, %k1
; AVX512-NEXT:    vmovups %zmm1, (%rdi) {%k1}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test13:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; SKX-NEXT:    vpcmpeqd %zmm2, %zmm0, %k1
; SKX-NEXT:    vmovups %zmm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v16f32(<16 x float>%val, <16 x float>* %addr, i32 4, <16 x i1>%mask)
  ret void
}

define void @test14(<2 x i32> %trigger, <2 x float>* %addr, <2 x float> %val) {
; AVX1-LABEL: test14:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpblendw {{.*#+}} xmm3 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm3, %xmm3
; AVX1-NEXT:    vpextrb $0, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB16_2
; AVX1-NEXT:  ## BB#1: ## %cond.store
; AVX1-NEXT:    vmovss %xmm1, (%rdi)
; AVX1-NEXT:  LBB16_2: ## %else
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpextrb $8, %xmm0, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB16_4
; AVX1-NEXT:  ## BB#3: ## %cond.store1
; AVX1-NEXT:    vextractps $1, %xmm1, 4(%rdi)
; AVX1-NEXT:  LBB16_4: ## %else2
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test14:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX2-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX2-NEXT:    vmaskmovps %xmm1, %xmm0, (%rdi)
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test14:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX512-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX512-NEXT:    vmaskmovps %xmm1, %xmm0, (%rdi)
; AVX512-NEXT:    retq
;
; SKX-LABEL: test14:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; SKX-NEXT:    vpcmpeqq %xmm2, %xmm0, %k0
; SKX-NEXT:    kshiftlw $2, %k0, %k0
; SKX-NEXT:    kshiftrw $2, %k0, %k1
; SKX-NEXT:    vmovups %xmm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v2f32(<2 x float>%val, <2 x float>* %addr, i32 4, <2 x i1>%mask)
  ret void
}

define void @test15(<2 x i32> %trigger, <2 x i32>* %addr, <2 x i32> %val) {
; AVX1-LABEL: test15:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpblendw {{.*#+}} xmm3 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm3, %xmm3
; AVX1-NEXT:    vpextrb $0, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB17_2
; AVX1-NEXT:  ## BB#1: ## %cond.store
; AVX1-NEXT:    vmovd %xmm1, (%rdi)
; AVX1-NEXT:  LBB17_2: ## %else
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpextrb $8, %xmm0, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB17_4
; AVX1-NEXT:  ## BB#3: ## %cond.store1
; AVX1-NEXT:    vpextrd $2, %xmm1, 4(%rdi)
; AVX1-NEXT:  LBB17_4: ## %else2
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test15:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX2-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; AVX2-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test15:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX512-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX512-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; AVX512-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX512-NEXT:    retq
;
; SKX-LABEL: test15:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; SKX-NEXT:    vpcmpeqq %xmm2, %xmm0, %k1
; SKX-NEXT:    vpmovqd %xmm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v2i32(<2 x i32>%val, <2 x i32>* %addr, i32 4, <2 x i1>%mask)
  ret void
}

define <2 x float> @test16(<2 x i32> %trigger, <2 x float>* %addr, <2 x float> %dst) {
; AVX1-LABEL: test16:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $0, %xmm2, %eax
; AVX1-NEXT:    ## implicit-def: %XMM2
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB18_2
; AVX1-NEXT:  ## BB#1: ## %cond.load
; AVX1-NEXT:    vmovss {{.*#+}} xmm2 = mem[0],zero,zero,zero
; AVX1-NEXT:  LBB18_2: ## %else
; AVX1-NEXT:    vpblendw {{.*#+}} xmm4 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm3, %xmm4, %xmm3
; AVX1-NEXT:    vpextrb $8, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB18_4
; AVX1-NEXT:  ## BB#3: ## %cond.load1
; AVX1-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0],mem[0],xmm2[2,3]
; AVX1-NEXT:  LBB18_4: ## %else2
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX1-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX1-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test16:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX2-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX2-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm2
; AVX2-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test16:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX512-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX512-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm2
; AVX512-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test16:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; SKX-NEXT:    vpcmpeqq %xmm2, %xmm0, %k0
; SKX-NEXT:    kshiftlw $2, %k0, %k0
; SKX-NEXT:    kshiftrw $2, %k0, %k1
; SKX-NEXT:    vmovups (%rdi), %xmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  %res = call <2 x float> @llvm.masked.load.v2f32(<2 x float>* %addr, i32 4, <2 x i1>%mask, <2 x float>%dst)
  ret <2 x float> %res
}

define <2 x i32> @test17(<2 x i32> %trigger, <2 x i32>* %addr, <2 x i32> %dst) {
; AVX1-LABEL: test17:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $0, %xmm2, %eax
; AVX1-NEXT:    ## implicit-def: %XMM2
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB19_2
; AVX1-NEXT:  ## BB#1: ## %cond.load
; AVX1-NEXT:    vmovd {{.*#+}} xmm2 = mem[0],zero,zero,zero
; AVX1-NEXT:  LBB19_2: ## %else
; AVX1-NEXT:    vpblendw {{.*#+}} xmm4 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm3, %xmm4, %xmm3
; AVX1-NEXT:    vpextrb $8, %xmm3, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB19_4
; AVX1-NEXT:  ## BB#3: ## %cond.load1
; AVX1-NEXT:    movl 4(%rdi), %eax
; AVX1-NEXT:    vpinsrq $1, %rax, %xmm2, %xmm2
; AVX1-NEXT:  LBB19_4: ## %else2
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vblendvpd %xmm0, %xmm2, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test17:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX2-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX2-NEXT:    vpmaskmovd (%rdi), %xmm0, %xmm2
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; AVX2-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test17:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX512-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX512-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX512-NEXT:    vpmaskmovd (%rdi), %xmm0, %xmm2
; AVX512-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; AVX512-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX512-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test17:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; SKX-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; SKX-NEXT:    vpcmpeqq %xmm2, %xmm0, %k0
; SKX-NEXT:    kshiftlw $2, %k0, %k0
; SKX-NEXT:    kshiftrw $2, %k0, %k1
; SKX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm1[0,2,2,3]
; SKX-NEXT:    vmovdqu32 (%rdi), %xmm0 {%k1}
; SKX-NEXT:    vpmovsxdq %xmm0, %xmm0
; SKX-NEXT:    retq
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  %res = call <2 x i32> @llvm.masked.load.v2i32(<2 x i32>* %addr, i32 4, <2 x i1>%mask, <2 x i32>%dst)
  ret <2 x i32> %res
}

define <2 x float> @test18(<2 x i32> %trigger, <2 x float>* %addr) {
; AVX1-LABEL: test18:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpextrb $0, %xmm1, %eax
; AVX1-NEXT:    ## implicit-def: %XMM1
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB20_2
; AVX1-NEXT:  ## BB#1: ## %cond.load
; AVX1-NEXT:    vmovss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; AVX1-NEXT:  LBB20_2: ## %else
; AVX1-NEXT:    vpblendw {{.*#+}} xmm3 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm2, %eax
; AVX1-NEXT:    testb $1, %al
; AVX1-NEXT:    je LBB20_4
; AVX1-NEXT:  ## BB#3: ## %cond.load1
; AVX1-NEXT:    vinsertps {{.*#+}} xmm1 = xmm1[0],mem[0],xmm1[2,3]
; AVX1-NEXT:  LBB20_4: ## %else2
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpcmpeqq %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX1-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX1-NEXT:    vblendvps %xmm0, %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test18:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; AVX2-NEXT:    vpcmpeqq %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX2-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test18:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX512-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; AVX512-NEXT:    vpcmpeqq %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = xmm0[0],zero
; AVX512-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test18:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; SKX-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; SKX-NEXT:    vpcmpeqq %xmm1, %xmm0, %k0
; SKX-NEXT:    kshiftlw $2, %k0, %k0
; SKX-NEXT:    kshiftrw $2, %k0, %k1
; SKX-NEXT:    vmovups (%rdi), %xmm0 {%k1} {z}
; SKX-NEXT:    retq
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  %res = call <2 x float> @llvm.masked.load.v2f32(<2 x float>* %addr, i32 4, <2 x i1>%mask, <2 x float>undef)
  ret <2 x float> %res
}

define <4 x float> @test19(<4 x i32> %trigger, <4 x float>* %addr) {
; AVX1-LABEL: test19:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vmovups (%rdi), %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test19:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpcmpeqd %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test19:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpcmpeqd %xmm0, %xmm0, %xmm0
; AVX512-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test19:
; SKX:       ## BB#0:
; SKX-NEXT:    kxnorw %k0, %k0, %k1
; SKX-NEXT:    vmovups (%rdi), %xmm0 {%k1} {z}
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x float> @llvm.masked.load.v4f32(<4 x float>* %addr, i32 4, <4 x i1><i1 true, i1 true, i1 true, i1 true>, <4 x float>undef)
  ret <4 x float> %res
}

define <4 x float> @test20(<4 x i32> %trigger, <4 x float>* %addr, <4 x float> %src0) {
; AVX1-LABEL: test20:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vblendps {{.*#+}} xmm0 = mem[0],xmm1[1],mem[2,3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test20:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vmovaps {{.*#+}} xmm0 = [4294967295,0,4294967295,4294967295]
; AVX2-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm2
; AVX2-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test20:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vmovaps {{.*#+}} xmm0 = [4294967295,0,4294967295,4294967295]
; AVX512-NEXT:    vmaskmovps (%rdi), %xmm0, %xmm2
; AVX512-NEXT:    vblendvps %xmm0, %xmm2, %xmm1, %xmm0
; AVX512-NEXT:    retq
;
; SKX-LABEL: test20:
; SKX:       ## BB#0:
; SKX-NEXT:    movb $13, %al
; SKX-NEXT:    kmovw %eax, %k1
; SKX-NEXT:    vmovaps (%rdi), %xmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x float> @llvm.masked.load.v4f32(<4 x float>* %addr, i32 16, <4 x i1><i1 true, i1 false, i1 true, i1 true>, <4 x float> %src0)
  ret <4 x float> %res
}

define void @test21(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %val) {
; AVX1-LABEL: test21:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vmovups %xmm1, (%rdi)
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test21:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpcmpeqd %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test21:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpcmpeqd %xmm0, %xmm0, %xmm0
; AVX512-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX512-NEXT:    retq
;
; SKX-LABEL: test21:
; SKX:       ## BB#0:
; SKX-NEXT:    kxnorw %k0, %k0, %k1
; SKX-NEXT:    vmovdqu32 %xmm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v4i32(<4 x i32>%val, <4 x i32>* %addr, i32 4, <4 x i1><i1 true, i1 true, i1 true, i1 true>)
  ret void
}

define void @test22(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %val) {
; AVX1-LABEL: test22:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vmovd %xmm1, (%rdi)
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test22:
; AVX2:       ## BB#0:
; AVX2-NEXT:    movl $-1, %eax
; AVX2-NEXT:    vmovd %eax, %xmm0
; AVX2-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test22:
; AVX512:       ## BB#0:
; AVX512-NEXT:    movl $-1, %eax
; AVX512-NEXT:    vmovd %eax, %xmm0
; AVX512-NEXT:    vpmaskmovd %xmm1, %xmm0, (%rdi)
; AVX512-NEXT:    retq
;
; SKX-LABEL: test22:
; SKX:       ## BB#0:
; SKX-NEXT:    movb $1, %al
; SKX-NEXT:    kmovw %eax, %k1
; SKX-NEXT:    vmovdqu32 %xmm1, (%rdi) {%k1}
; SKX-NEXT:    retq
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v4i32(<4 x i32>%val, <4 x i32>* %addr, i32 4, <4 x i1><i1 true, i1 false, i1 false, i1 false>)
  ret void
}

declare <16 x i32> @llvm.masked.load.v16i32(<16 x i32>*, i32, <16 x i1>, <16 x i32>)
declare <4 x i32> @llvm.masked.load.v4i32(<4 x i32>*, i32, <4 x i1>, <4 x i32>)
declare <2 x i32> @llvm.masked.load.v2i32(<2 x i32>*, i32, <2 x i1>, <2 x i32>)
declare void @llvm.masked.store.v16i32(<16 x i32>, <16 x i32>*, i32, <16 x i1>)
declare void @llvm.masked.store.v8i32(<8 x i32>, <8 x i32>*, i32, <8 x i1>)
declare void @llvm.masked.store.v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)
declare void @llvm.masked.store.v2f32(<2 x float>, <2 x float>*, i32, <2 x i1>)
declare void @llvm.masked.store.v2i32(<2 x i32>, <2 x i32>*, i32, <2 x i1>)
declare void @llvm.masked.store.v16f32(<16 x float>, <16 x float>*, i32, <16 x i1>)
declare void @llvm.masked.store.v16f32p(<16 x float>*, <16 x float>**, i32, <16 x i1>)
declare <16 x float> @llvm.masked.load.v16f32(<16 x float>*, i32, <16 x i1>, <16 x float>)
declare <8 x float> @llvm.masked.load.v8f32(<8 x float>*, i32, <8 x i1>, <8 x float>)
declare <8 x i32> @llvm.masked.load.v8i32(<8 x i32>*, i32, <8 x i1>, <8 x i32>)
declare <4 x float> @llvm.masked.load.v4f32(<4 x float>*, i32, <4 x i1>, <4 x float>)
declare <2 x float> @llvm.masked.load.v2f32(<2 x float>*, i32, <2 x i1>, <2 x float>)
declare <8 x double> @llvm.masked.load.v8f64(<8 x double>*, i32, <8 x i1>, <8 x double>)
declare <4 x double> @llvm.masked.load.v4f64(<4 x double>*, i32, <4 x i1>, <4 x double>)
declare <2 x double> @llvm.masked.load.v2f64(<2 x double>*, i32, <2 x i1>, <2 x double>)
declare void @llvm.masked.store.v8f64(<8 x double>, <8 x double>*, i32, <8 x i1>)
declare void @llvm.masked.store.v2f64(<2 x double>, <2 x double>*, i32, <2 x i1>)
declare void @llvm.masked.store.v2i64(<2 x i64>, <2 x i64>*, i32, <2 x i1>)

declare <16 x i32*> @llvm.masked.load.v16p0i32(<16 x i32*>*, i32, <16 x i1>, <16 x i32*>)

define <16 x i32*> @test23(<16 x i32*> %trigger, <16 x i32*>* %addr) {
; Bypassing exact checking here because it's over 700 lines.
; AVX1-LABEL: test23:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test23:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpxor %ymm4, %ymm4, %ymm4
; AVX2-NEXT:    vpcmpeqq %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpeqq %ymm4, %ymm1, %ymm1
; AVX2-NEXT:    vpcmpeqq %ymm4, %ymm2, %ymm2
; AVX2-NEXT:    vpcmpeqq %ymm4, %ymm3, %ymm3
; AVX2-NEXT:    vpmaskmovq 96(%rdi), %ymm3, %ymm3
; AVX2-NEXT:    vpmaskmovq 64(%rdi), %ymm2, %ymm2
; AVX2-NEXT:    vpmaskmovq 32(%rdi), %ymm1, %ymm1
; AVX2-NEXT:    vpmaskmovq (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test23:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; AVX512-NEXT:    vpcmpeqq %zmm2, %zmm0, %k1
; AVX512-NEXT:    vpcmpeqq %zmm2, %zmm1, %k2
; AVX512-NEXT:    vmovdqu64 64(%rdi), %zmm1 {%k2} {z}
; AVX512-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test23:
; SKX:       ## BB#0:
; SKX-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; SKX-NEXT:    vpcmpeqq %zmm2, %zmm0, %k1
; SKX-NEXT:    vpcmpeqq %zmm2, %zmm1, %k2
; SKX-NEXT:    vmovdqu64 64(%rdi), %zmm1 {%k2} {z}
; SKX-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; SKX-NEXT:    retq
  %mask = icmp eq <16 x i32*> %trigger, zeroinitializer
  %res = call <16 x i32*> @llvm.masked.load.v16p0i32(<16 x i32*>* %addr, i32 4, <16 x i1>%mask, <16 x i32*>zeroinitializer)
  ret <16 x i32*> %res
}

%mystruct = type { i16, i16, [1 x i8*] }

declare <16 x %mystruct*> @llvm.masked.load.v16p0mystruct(<16 x %mystruct*>*, i32, <16 x i1>, <16 x %mystruct*>)

define <16 x %mystruct*> @test24(<16 x i1> %mask, <16 x %mystruct*>* %addr) {
; Bypassing exact checking here because it's over 100 lines.
; AVX1-LABEL: test24:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test24:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vpmaskmovq (%rdi), %ymm1, %ymm4
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vpmaskmovq 96(%rdi), %ymm1, %ymm3
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vpmaskmovq 64(%rdi), %ymm1, %ymm2
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vpmaskmovq 32(%rdi), %ymm0, %ymm1
; AVX2-NEXT:    vmovdqa %ymm4, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test24:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512-NEXT:    vpslld $31, %zmm0, %zmm0
; AVX512-NEXT:    vptestmd %zmm0, %zmm0, %k1
; AVX512-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; AVX512-NEXT:    kshiftrw $8, %k1, %k1
; AVX512-NEXT:    vmovdqu64 64(%rdi), %zmm1 {%k1} {z}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test24:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $7, %xmm0, %xmm0
; SKX-NEXT:    vpmovb2m %xmm0, %k1
; SKX-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; SKX-NEXT:    kshiftrw $8, %k1, %k1
; SKX-NEXT:    vmovdqu64 64(%rdi), %zmm1 {%k1} {z}
; SKX-NEXT:    retq
  %res = call <16 x %mystruct*> @llvm.masked.load.v16p0mystruct(<16 x %mystruct*>* %addr, i32 4, <16 x i1>%mask, <16 x %mystruct*>zeroinitializer)
  ret <16 x %mystruct*> %res
}

define void @test_store_16i64(<16 x i64>* %ptrs, <16 x i1> %mask, <16 x i64> %src0)  {
; Bypassing exact checking here because it's over 100 lines.
; AVX1-LABEL: test_store_16i64:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test_store_16i64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm5 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm5, %xmm5
; AVX2-NEXT:    vpsrad $31, %xmm5, %xmm5
; AVX2-NEXT:    vpmovsxdq %xmm5, %ymm5
; AVX2-NEXT:    vpmaskmovq %ymm1, %ymm5, (%rdi)
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vpmaskmovq %ymm4, %ymm1, 96(%rdi)
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vpmaskmovq %ymm3, %ymm1, 64(%rdi)
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vpmaskmovq %ymm2, %ymm0, 32(%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test_store_16i64:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512-NEXT:    vpslld $31, %zmm0, %zmm0
; AVX512-NEXT:    vptestmd %zmm0, %zmm0, %k1
; AVX512-NEXT:    vmovdqu64 %zmm1, (%rdi) {%k1}
; AVX512-NEXT:    kshiftrw $8, %k1, %k1
; AVX512-NEXT:    vmovdqu64 %zmm2, 64(%rdi) {%k1}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test_store_16i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $7, %xmm0, %xmm0
; SKX-NEXT:    vpmovb2m %xmm0, %k1
; SKX-NEXT:    vmovdqu64 %zmm1, (%rdi) {%k1}
; SKX-NEXT:    kshiftrw $8, %k1, %k1
; SKX-NEXT:    vmovdqu64 %zmm2, 64(%rdi) {%k1}
; SKX-NEXT:    retq
  call void @llvm.masked.store.v16i64(<16 x i64> %src0, <16 x i64>* %ptrs, i32 4, <16 x i1> %mask)
  ret void
}
declare void @llvm.masked.store.v16i64(<16 x i64> %src0, <16 x i64>* %ptrs, i32, <16 x i1> %mask)

define void @test_store_16f64(<16 x double>* %ptrs, <16 x i1> %mask, <16 x double> %src0)  {
; Bypassing exact checking here because it's over 100 lines.
; AVX1-LABEL: test_store_16f64:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test_store_16f64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm5 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm5, %xmm5
; AVX2-NEXT:    vpsrad $31, %xmm5, %xmm5
; AVX2-NEXT:    vpmovsxdq %xmm5, %ymm5
; AVX2-NEXT:    vmaskmovpd %ymm1, %ymm5, (%rdi)
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vmaskmovpd %ymm4, %ymm1, 96(%rdi)
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vmaskmovpd %ymm3, %ymm1, 64(%rdi)
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vmaskmovpd %ymm2, %ymm0, 32(%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test_store_16f64:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512-NEXT:    vpslld $31, %zmm0, %zmm0
; AVX512-NEXT:    vptestmd %zmm0, %zmm0, %k1
; AVX512-NEXT:    vmovupd %zmm1, (%rdi) {%k1}
; AVX512-NEXT:    kshiftrw $8, %k1, %k1
; AVX512-NEXT:    vmovupd %zmm2, 64(%rdi) {%k1}
; AVX512-NEXT:    retq
;
; SKX-LABEL: test_store_16f64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $7, %xmm0, %xmm0
; SKX-NEXT:    vpmovb2m %xmm0, %k1
; SKX-NEXT:    vmovupd %zmm1, (%rdi) {%k1}
; SKX-NEXT:    kshiftrw $8, %k1, %k1
; SKX-NEXT:    vmovupd %zmm2, 64(%rdi) {%k1}
; SKX-NEXT:    retq
  call void @llvm.masked.store.v16f64(<16 x double> %src0, <16 x double>* %ptrs, i32 4, <16 x i1> %mask)
  ret void
}
declare void @llvm.masked.store.v16f64(<16 x double> %src0, <16 x double>* %ptrs, i32, <16 x i1> %mask)

define <16 x i64> @test_load_16i64(<16 x i64>* %ptrs, <16 x i1> %mask, <16 x i64> %src0)  {
; Bypassing exact checking here because it's over 100 lines.
; AVX1-LABEL: test_load_16i64:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test_load_16i64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm5 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm5, %xmm5
; AVX2-NEXT:    vpsrad $31, %xmm5, %xmm5
; AVX2-NEXT:    vpmovsxdq %xmm5, %ymm5
; AVX2-NEXT:    vpmaskmovq (%rdi), %ymm5, %ymm6
; AVX2-NEXT:    vblendvpd %ymm5, %ymm6, %ymm1, %ymm5
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[1,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vpmaskmovq 32(%rdi), %ymm1, %ymm6
; AVX2-NEXT:    vblendvpd %ymm1, %ymm6, %ymm2, %ymm1
; AVX2-NEXT:    vpshufd {{.*#+}} xmm2 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm2 = xmm2[0],zero,zero,zero,xmm2[1],zero,zero,zero,xmm2[2],zero,zero,zero,xmm2[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm2, %xmm2
; AVX2-NEXT:    vpsrad $31, %xmm2, %xmm2
; AVX2-NEXT:    vpmovsxdq %xmm2, %ymm2
; AVX2-NEXT:    vpmaskmovq 64(%rdi), %ymm2, %ymm6
; AVX2-NEXT:    vblendvpd %ymm2, %ymm6, %ymm3, %ymm2
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[3,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vpmaskmovq 96(%rdi), %ymm0, %ymm3
; AVX2-NEXT:    vblendvpd %ymm0, %ymm3, %ymm4, %ymm3
; AVX2-NEXT:    vmovapd %ymm5, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test_load_16i64:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512-NEXT:    vpslld $31, %zmm0, %zmm0
; AVX512-NEXT:    vptestmd %zmm0, %zmm0, %k1
; AVX512-NEXT:    vmovdqu64 (%rdi), %zmm1 {%k1}
; AVX512-NEXT:    kshiftrw $8, %k1, %k1
; AVX512-NEXT:    vmovdqu64 64(%rdi), %zmm2 {%k1}
; AVX512-NEXT:    vmovaps %zmm1, %zmm0
; AVX512-NEXT:    vmovaps %zmm2, %zmm1
; AVX512-NEXT:    retq
;
; SKX-LABEL: test_load_16i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $7, %xmm0, %xmm0
; SKX-NEXT:    vpmovb2m %xmm0, %k1
; SKX-NEXT:    vmovdqu64 (%rdi), %zmm1 {%k1}
; SKX-NEXT:    kshiftrw $8, %k1, %k1
; SKX-NEXT:    vmovdqu64 64(%rdi), %zmm2 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    vmovaps %zmm2, %zmm1
; SKX-NEXT:    retq
  %res = call <16 x i64> @llvm.masked.load.v16i64(<16 x i64>* %ptrs, i32 4, <16 x i1> %mask, <16 x i64> %src0)
  ret <16 x i64> %res
}
declare <16 x i64> @llvm.masked.load.v16i64(<16 x i64>* %ptrs, i32, <16 x i1> %mask, <16 x i64> %src0)

define <16 x double> @test_load_16f64(<16 x double>* %ptrs, <16 x i1> %mask, <16 x double> %src0)  {
; Bypassing exact checking here because it's over 100 lines.
; AVX1-LABEL: test_load_16f64:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test_load_16f64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm5 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm5, %xmm5
; AVX2-NEXT:    vpsrad $31, %xmm5, %xmm5
; AVX2-NEXT:    vpmovsxdq %xmm5, %ymm5
; AVX2-NEXT:    vmaskmovpd (%rdi), %ymm5, %ymm6
; AVX2-NEXT:    vblendvpd %ymm5, %ymm6, %ymm1, %ymm5
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[1,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vmaskmovpd 32(%rdi), %ymm1, %ymm6
; AVX2-NEXT:    vblendvpd %ymm1, %ymm6, %ymm2, %ymm1
; AVX2-NEXT:    vpshufd {{.*#+}} xmm2 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm2 = xmm2[0],zero,zero,zero,xmm2[1],zero,zero,zero,xmm2[2],zero,zero,zero,xmm2[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm2, %xmm2
; AVX2-NEXT:    vpsrad $31, %xmm2, %xmm2
; AVX2-NEXT:    vpmovsxdq %xmm2, %ymm2
; AVX2-NEXT:    vmaskmovpd 64(%rdi), %ymm2, %ymm6
; AVX2-NEXT:    vblendvpd %ymm2, %ymm6, %ymm3, %ymm2
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[3,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vmaskmovpd 96(%rdi), %ymm0, %ymm3
; AVX2-NEXT:    vblendvpd %ymm0, %ymm3, %ymm4, %ymm3
; AVX2-NEXT:    vmovapd %ymm5, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test_load_16f64:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512-NEXT:    vpslld $31, %zmm0, %zmm0
; AVX512-NEXT:    vptestmd %zmm0, %zmm0, %k1
; AVX512-NEXT:    vmovupd (%rdi), %zmm1 {%k1}
; AVX512-NEXT:    kshiftrw $8, %k1, %k1
; AVX512-NEXT:    vmovupd 64(%rdi), %zmm2 {%k1}
; AVX512-NEXT:    vmovaps %zmm1, %zmm0
; AVX512-NEXT:    vmovaps %zmm2, %zmm1
; AVX512-NEXT:    retq
;
; SKX-LABEL: test_load_16f64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $7, %xmm0, %xmm0
; SKX-NEXT:    vpmovb2m %xmm0, %k1
; SKX-NEXT:    vmovupd (%rdi), %zmm1 {%k1}
; SKX-NEXT:    kshiftrw $8, %k1, %k1
; SKX-NEXT:    vmovupd 64(%rdi), %zmm2 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    vmovaps %zmm2, %zmm1
; SKX-NEXT:    retq
  %res = call <16 x double> @llvm.masked.load.v16f64(<16 x double>* %ptrs, i32 4, <16 x i1> %mask, <16 x double> %src0)
  ret <16 x double> %res
}
declare <16 x double> @llvm.masked.load.v16f64(<16 x double>* %ptrs, i32, <16 x i1> %mask, <16 x double> %src0)

define <32 x double> @test_load_32f64(<32 x double>* %ptrs, <32 x i1> %mask, <32 x double> %src0)  {
; Bypassing exact checking here because it's over 300 lines.
; AVX1-LABEL: test_load_32f64:
; AVX1-NOT:   maskmov
;
; AVX2-LABEL: test_load_32f64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    pushq %rbp
; AVX2-NEXT:  Ltmp0:
; AVX2-NEXT:    .cfi_def_cfa_offset 16
; AVX2-NEXT:  Ltmp1:
; AVX2-NEXT:    .cfi_offset %rbp, -16
; AVX2-NEXT:    movq %rsp, %rbp
; AVX2-NEXT:  Ltmp2:
; AVX2-NEXT:    .cfi_def_cfa_register %rbp
; AVX2-NEXT:    andq $-32, %rsp
; AVX2-NEXT:    subq $32, %rsp
; AVX2-NEXT:    vmovapd 16(%rbp), %ymm8
; AVX2-NEXT:    vpshufd {{.*#+}} xmm9 = xmm0[1,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm9 = xmm9[0],zero,zero,zero,xmm9[1],zero,zero,zero,xmm9[2],zero,zero,zero,xmm9[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm9, %xmm9
; AVX2-NEXT:    vpsrad $31, %xmm9, %xmm9
; AVX2-NEXT:    vpmovsxdq %xmm9, %ymm9
; AVX2-NEXT:    vmaskmovpd 32(%rsi), %ymm9, %ymm10
; AVX2-NEXT:    vblendvpd %ymm9, %ymm10, %ymm2, %ymm9
; AVX2-NEXT:    vpshufd {{.*#+}} xmm2 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm2 = xmm2[0],zero,zero,zero,xmm2[1],zero,zero,zero,xmm2[2],zero,zero,zero,xmm2[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm2, %xmm2
; AVX2-NEXT:    vpsrad $31, %xmm2, %xmm2
; AVX2-NEXT:    vpmovsxdq %xmm2, %ymm2
; AVX2-NEXT:    vmaskmovpd 64(%rsi), %ymm2, %ymm10
; AVX2-NEXT:    vblendvpd %ymm2, %ymm10, %ymm3, %ymm11
; AVX2-NEXT:    vpshufd {{.*#+}} xmm2 = xmm0[3,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm2 = xmm2[0],zero,zero,zero,xmm2[1],zero,zero,zero,xmm2[2],zero,zero,zero,xmm2[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm2, %xmm2
; AVX2-NEXT:    vpsrad $31, %xmm2, %xmm2
; AVX2-NEXT:    vpmovsxdq %xmm2, %ymm2
; AVX2-NEXT:    vmaskmovpd 96(%rsi), %ymm2, %ymm10
; AVX2-NEXT:    vblendvpd %ymm2, %ymm10, %ymm4, %ymm4
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX2-NEXT:    vpshufd {{.*#+}} xmm3 = xmm2[1,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm3 = xmm3[0],zero,zero,zero,xmm3[1],zero,zero,zero,xmm3[2],zero,zero,zero,xmm3[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm3, %xmm3
; AVX2-NEXT:    vpsrad $31, %xmm3, %xmm3
; AVX2-NEXT:    vpmovsxdq %xmm3, %ymm3
; AVX2-NEXT:    vmaskmovpd 160(%rsi), %ymm3, %ymm10
; AVX2-NEXT:    vblendvpd %ymm3, %ymm10, %ymm6, %ymm3
; AVX2-NEXT:    vpshufd {{.*#+}} xmm6 = xmm2[2,3,0,1]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm6 = xmm6[0],zero,zero,zero,xmm6[1],zero,zero,zero,xmm6[2],zero,zero,zero,xmm6[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm6, %xmm6
; AVX2-NEXT:    vpsrad $31, %xmm6, %xmm6
; AVX2-NEXT:    vpmovsxdq %xmm6, %ymm6
; AVX2-NEXT:    vmaskmovpd 192(%rsi), %ymm6, %ymm10
; AVX2-NEXT:    vblendvpd %ymm6, %ymm10, %ymm7, %ymm6
; AVX2-NEXT:    vpshufd {{.*#+}} xmm7 = xmm2[3,1,2,3]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm7 = xmm7[0],zero,zero,zero,xmm7[1],zero,zero,zero,xmm7[2],zero,zero,zero,xmm7[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm7, %xmm7
; AVX2-NEXT:    vpsrad $31, %xmm7, %xmm7
; AVX2-NEXT:    vpmovsxdq %xmm7, %ymm7
; AVX2-NEXT:    vmaskmovpd 224(%rsi), %ymm7, %ymm10
; AVX2-NEXT:    vblendvpd %ymm7, %ymm10, %ymm8, %ymm7
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    vmaskmovpd (%rsi), %ymm0, %ymm8
; AVX2-NEXT:    vblendvpd %ymm0, %ymm8, %ymm1, %ymm0
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm2[0],zero,zero,zero,xmm2[1],zero,zero,zero,xmm2[2],zero,zero,zero,xmm2[3],zero,zero,zero
; AVX2-NEXT:    vpslld $31, %xmm1, %xmm1
; AVX2-NEXT:    vpsrad $31, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxdq %xmm1, %ymm1
; AVX2-NEXT:    vmaskmovpd 128(%rsi), %ymm1, %ymm2
; AVX2-NEXT:    vblendvpd %ymm1, %ymm2, %ymm5, %ymm1
; AVX2-NEXT:    vmovapd %ymm1, 128(%rdi)
; AVX2-NEXT:    vmovapd %ymm0, (%rdi)
; AVX2-NEXT:    vmovapd %ymm7, 224(%rdi)
; AVX2-NEXT:    vmovapd %ymm6, 192(%rdi)
; AVX2-NEXT:    vmovapd %ymm3, 160(%rdi)
; AVX2-NEXT:    vmovapd %ymm4, 96(%rdi)
; AVX2-NEXT:    vmovapd %ymm11, 64(%rdi)
; AVX2-NEXT:    vmovapd %ymm9, 32(%rdi)
; AVX2-NEXT:    movq %rdi, %rax
; AVX2-NEXT:    movq %rbp, %rsp
; AVX2-NEXT:    popq %rbp
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512-LABEL: test_load_32f64:
; AVX512:       ## BB#0:
; AVX512-NEXT:    vextractf128 $1, %ymm0, %xmm5
; AVX512-NEXT:    vpmovsxbd %xmm5, %zmm5
; AVX512-NEXT:    vpslld $31, %zmm5, %zmm5
; AVX512-NEXT:    vptestmd %zmm5, %zmm5, %k1
; AVX512-NEXT:    vmovupd 128(%rdi), %zmm3 {%k1}
; AVX512-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512-NEXT:    vpslld $31, %zmm0, %zmm0
; AVX512-NEXT:    vptestmd %zmm0, %zmm0, %k2
; AVX512-NEXT:    vmovupd (%rdi), %zmm1 {%k2}
; AVX512-NEXT:    kshiftrw $8, %k1, %k1
; AVX512-NEXT:    vmovupd 192(%rdi), %zmm4 {%k1}
; AVX512-NEXT:    kshiftrw $8, %k2, %k1
; AVX512-NEXT:    vmovupd 64(%rdi), %zmm2 {%k1}
; AVX512-NEXT:    vmovaps %zmm1, %zmm0
; AVX512-NEXT:    vmovaps %zmm2, %zmm1
; AVX512-NEXT:    vmovaps %zmm3, %zmm2
; AVX512-NEXT:    vmovaps %zmm4, %zmm3
; AVX512-NEXT:    retq
;
; SKX-LABEL: test_load_32f64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpsllw $7, %ymm0, %ymm0
; SKX-NEXT:    vpmovb2m %ymm0, %k1
; SKX-NEXT:    vmovupd (%rdi), %zmm1 {%k1}
; SKX-NEXT:    kshiftrd $16, %k1, %k2
; SKX-NEXT:    vmovupd 128(%rdi), %zmm3 {%k2}
; SKX-NEXT:    kshiftrw $8, %k1, %k1
; SKX-NEXT:    vmovupd 64(%rdi), %zmm2 {%k1}
; SKX-NEXT:    kshiftrw $8, %k2, %k1
; SKX-NEXT:    vmovupd 192(%rdi), %zmm4 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    vmovaps %zmm2, %zmm1
; SKX-NEXT:    vmovaps %zmm3, %zmm2
; SKX-NEXT:    vmovaps %zmm4, %zmm3
; SKX-NEXT:    retq
  %res = call <32 x double> @llvm.masked.load.v32f64(<32 x double>* %ptrs, i32 4, <32 x i1> %mask, <32 x double> %src0)
  ret <32 x double> %res
}
declare <32 x double> @llvm.masked.load.v32f64(<32 x double>* %ptrs, i32, <32 x i1> %mask, <32 x double> %src0)
