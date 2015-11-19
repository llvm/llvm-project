; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding| FileCheck %s

declare i32 @llvm.x86.avx512.kortestz.w(i16, i16) nounwind readnone
; CHECK-LABEL: test_kortestz
; CHECK: kortestw
; CHECK: sete
define i32 @test_kortestz(i16 %a0, i16 %a1) {
  %res = call i32 @llvm.x86.avx512.kortestz.w(i16 %a0, i16 %a1)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.kortestc.w(i16, i16) nounwind readnone
; CHECK-LABEL: test_kortestc
; CHECK: kortestw
; CHECK: sbbl
define i32 @test_kortestc(i16 %a0, i16 %a1) {
  %res = call i32 @llvm.x86.avx512.kortestc.w(i16 %a0, i16 %a1)
  ret i32 %res
}

declare i16 @llvm.x86.avx512.kand.w(i16, i16) nounwind readnone
; CHECK-LABEL: test_kand
; CHECK: kandw
; CHECK: kandw
define i16 @test_kand(i16 %a0, i16 %a1) {
  %t1 = call i16 @llvm.x86.avx512.kand.w(i16 %a0, i16 8)
  %t2 = call i16 @llvm.x86.avx512.kand.w(i16 %t1, i16 %a1)
  ret i16 %t2
}

declare i16 @llvm.x86.avx512.knot.w(i16) nounwind readnone
; CHECK-LABEL: test_knot
; CHECK: knotw
define i16 @test_knot(i16 %a0) {
  %res = call i16 @llvm.x86.avx512.knot.w(i16 %a0)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.kunpck.bw(i16, i16) nounwind readnone

; CHECK-LABEL: unpckbw_test
; CHECK: kunpckbw
; CHECK:ret
define i16 @unpckbw_test(i16 %a0, i16 %a1) {
  %res = call i16 @llvm.x86.avx512.kunpck.bw(i16 %a0, i16 %a1)
  ret i16 %res
}

define <16 x float> @test_rcp_ps_512(<16 x float> %a0) {
  ; CHECK: vrcp14ps {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x4c,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone

define <8 x double> @test_rcp_pd_512(<8 x double> %a0) {
  ; CHECK: vrcp14pd {{.*}}encoding: [0x62,0xf2,0xfd,0x48,0x4c,0xc0]
  %res = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %a0, <8 x double> zeroinitializer, i8 -1) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double>, <8 x double>, i8) nounwind readnone

declare <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>, i32, <8 x double>, i8, i32)

define <8 x double> @test7(<8 x double> %a) {
; CHECK: vrndscalepd {{.*}}encoding: [0x62,0xf3,0xfd,0x48,0x09,0xc0,0x0b]
  %res = call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double> %a, i32 11, <8 x double> %a, i8 -1, i32 4)
  ret <8 x double>%res
}

declare <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>, i32, <16 x float>, i16, i32)

define <16 x float> @test8(<16 x float> %a) {
; CHECK: vrndscaleps {{.*}}encoding: [0x62,0xf3,0x7d,0x48,0x08,0xc0,0x0b]
  %res = call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %a, i32 11, <16 x float> %a, i16 -1, i32 4)
  ret <16 x float>%res
}

define <16 x float> @test_rsqrt_ps_512(<16 x float> %a0) {
  ; CHECK: vrsqrt14ps {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x4e,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone

define <4 x float> @test_rsqrt14_ss(<4 x float> %a0) {
  ; CHECK: vrsqrt14ss {{.*}}encoding: [0x62,0xf2,0x7d,0x08,0x4f,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_rcp14_ss(<4 x float> %a0) {
  ; CHECK: vrcp14ss {{.*}}encoding: [0x62,0xf2,0x7d,0x08,0x4d,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <8 x double> @test_sqrt_pd_512(<8 x double> %a0) {
  ; CHECK-LABEL: test_sqrt_pd_512
  ; CHECK: vsqrtpd
  %res = call <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double> %a0,  <8 x double> zeroinitializer, i8 -1, i32 4) 
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_sqrt_ps_512(<16 x float> %a0) {
  ; CHECK-LABEL: test_sqrt_ps_512
  ; CHECK: vsqrtps
  %res = call <16 x float> @llvm.x86.avx512.mask.sqrt.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 4) 
  ret <16 x float> %res
}
define <16 x float> @test_sqrt_round_ps_512(<16 x float> %a0) {
  ; CHECK-LABEL: test_sqrt_round_ps_512
  ; CHECK: vsqrtps {rz-sae}
  %res = call <16 x float> @llvm.x86.avx512.mask.sqrt.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 3) 
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.sqrt.ps.512(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_getexp_pd_512(<8 x double> %a0) {
  ; CHECK-LABEL: test_getexp_pd_512
  ; CHECK: vgetexppd
  %res = call <8 x double> @llvm.x86.avx512.mask.getexp.pd.512(<8 x double> %a0,  <8 x double> zeroinitializer, i8 -1, i32 4) 
  ret <8 x double> %res
}
define <8 x double> @test_getexp_round_pd_512(<8 x double> %a0) {
  ; CHECK-LABEL: test_getexp_round_pd_512
  ; CHECK: vgetexppd {sae}
  %res = call <8 x double> @llvm.x86.avx512.mask.getexp.pd.512(<8 x double> %a0,  <8 x double> zeroinitializer, i8 -1, i32 8) 
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.getexp.pd.512(<8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_getexp_ps_512(<16 x float> %a0) {
  ; CHECK-LABEL: test_getexp_ps_512
  ; CHECK: vgetexpps
  %res = call <16 x float> @llvm.x86.avx512.mask.getexp.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 4) 
  ret <16 x float> %res
}

define <16 x float> @test_getexp_round_ps_512(<16 x float> %a0) {
  ; CHECK-LABEL: test_getexp_round_ps_512
  ; CHECK: vgetexpps {sae}
  %res = call <16 x float> @llvm.x86.avx512.mask.getexp.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 8) 
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.getexp.ps.512(<16 x float>, <16 x float>, i16, i32) nounwind readnone

declare <4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_sqrt_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_sqrt_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vsqrtss %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vsqrtss {rd-sae}, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vsqrtss {ru-sae}, %xmm1, %xmm0, %xmm4 {%k1} {z}
; CHECK-NEXT:    vsqrtss {rz-sae}, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm2, %xmm3, %xmm1
; CHECK-NEXT:    vaddps %xmm0, %xmm4, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res0 = call <4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 4)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 1)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %mask, i32 2)
  %res3 = call <4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 -1, i32 3)

  %res.1 = fadd <4 x float> %res0, %res1
  %res.2 = fadd <4 x float> %res2, %res3
  %res   = fadd <4 x float> %res.1, %res.2
  ret <4 x float> %res
}

declare <2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double>, <2 x double>, <2 x double>, i8, i32) nounwind readnone

define <2 x double> @test_sqrt_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_sqrt_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vsqrtsd %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vsqrtsd {rd-sae}, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vsqrtsd {ru-sae}, %xmm1, %xmm0, %xmm4 {%k1} {z}
; CHECK-NEXT:    vsqrtsd {rz-sae}, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddpd %xmm2, %xmm3, %xmm1
; CHECK-NEXT:    vaddpd %xmm0, %xmm4, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res0 = call <2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 4)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 1)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 %mask, i32 2)
  %res3 = call <2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 -1, i32 3)

  %res.1 = fadd <2 x double> %res0, %res1
  %res.2 = fadd <2 x double> %res2, %res3
  %res   = fadd <2 x double> %res.1, %res.2
  ret <2 x double> %res
}

define i64 @test_x86_sse2_cvtsd2si64(<2 x double> %a0) {
  ; CHECK: vcvtsd2si {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone

define <2 x double> @test_x86_sse2_cvtsi642sd(<2 x double> %a0, i64 %a1) {
  ; CHECK: vcvtsi2sdq {{.*}}encoding: [0x62
  %res = call <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double> %a0, i64 %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double>, i64) nounwind readnone

define i64 @test_x86_avx512_cvttsd2si64(<2 x double> %a0) {
  ; CHECK: vcvttsd2si {{.*}}encoding: [0x62
  ; CHECK: vcvttsd2si {sae}{{.*}}encoding: [0x62
  %res0 = call i64 @llvm.x86.avx512.cvttsd2si64(<2 x double> %a0, i32 4) ;
  %res1 = call i64 @llvm.x86.avx512.cvttsd2si64(<2 x double> %a0, i32 8) ;
  %res2 = add i64 %res0, %res1
  ret i64 %res2
}
declare i64 @llvm.x86.avx512.cvttsd2si64(<2 x double>, i32) nounwind readnone

define i32 @test_x86_avx512_cvttsd2usi(<2 x double> %a0) {
  ; CHECK: vcvttsd2usi {{.*}}encoding: [0x62
  ; CHECK: vcvttsd2usi {sae}{{.*}}encoding: [0x62
  %res0 = call i32 @llvm.x86.avx512.cvttsd2usi(<2 x double> %a0, i32 4) ;
  %res1 = call i32 @llvm.x86.avx512.cvttsd2usi(<2 x double> %a0, i32 8) ;
  %res2 = add i32 %res0, %res1
  ret i32 %res2
}
declare i32 @llvm.x86.avx512.cvttsd2usi(<2 x double>, i32) nounwind readnone

define i32 @test_x86_avx512_cvttsd2si(<2 x double> %a0) {
  ; CHECK: vcvttsd2si {{.*}}encoding: [0x62
  ; CHECK: vcvttsd2si {sae}{{.*}}encoding: [0x62
  %res0 = call i32 @llvm.x86.avx512.cvttsd2si(<2 x double> %a0, i32 4) ;
  %res1 = call i32 @llvm.x86.avx512.cvttsd2si(<2 x double> %a0, i32 8) ;
  %res2 = add i32 %res0, %res1
  ret i32 %res2
}
declare i32 @llvm.x86.avx512.cvttsd2si(<2 x double>, i32) nounwind readnone



define i64 @test_x86_avx512_cvttsd2usi64(<2 x double> %a0) {
  ; CHECK: vcvttsd2usi {{.*}}encoding: [0x62
  ; CHECK: vcvttsd2usi {sae}{{.*}}encoding: [0x62
  %res0 = call i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double> %a0, i32 4) ;
  %res1 = call i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double> %a0, i32 8) ;
  %res2 = add i64 %res0, %res1
  ret i64 %res2
}
declare i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double>, i32) nounwind readnone

define i64 @test_x86_sse_cvtss2si64(<4 x float> %a0) {
  ; CHECK: vcvtss2si {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cvtsi642ss(<4 x float> %a0, i64 %a1) {
  ; CHECK: vcvtsi2ssq {{.*}}encoding: [0x62
  %res = call <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float> %a0, i64 %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float>, i64) nounwind readnone


define i32 @test_x86_avx512_cvttss2si(<4 x float> %a0) {
  ; CHECK: vcvttss2si {sae}{{.*}}encoding: [0x62
  ; CHECK: vcvttss2si {{.*}}encoding: [0x62
  %res0 = call i32 @llvm.x86.avx512.cvttss2si(<4 x float> %a0, i32 8) ;
  %res1 = call i32 @llvm.x86.avx512.cvttss2si(<4 x float> %a0, i32 4) ;
  %res2 = add i32 %res0, %res1
  ret i32 %res2
}
declare i32 @llvm.x86.avx512.cvttss2si(<4 x float>, i32) nounwind readnone

define i64 @test_x86_avx512_cvttss2si64(<4 x float> %a0) {
  ; CHECK: vcvttss2si {{.*}}encoding: [0x62
  ; CHECK: vcvttss2si {sae}{{.*}}encoding: [0x62
  %res0 = call i64 @llvm.x86.avx512.cvttss2si64(<4 x float> %a0, i32 4) ;
  %res1 = call i64 @llvm.x86.avx512.cvttss2si64(<4 x float> %a0, i32 8) ;
  %res2 = add i64 %res0, %res1
  ret i64 %res2
}
declare i64 @llvm.x86.avx512.cvttss2si64(<4 x float>, i32) nounwind readnone

define i32 @test_x86_avx512_cvttss2usi(<4 x float> %a0) {
  ; CHECK: vcvttss2usi {sae}{{.*}}encoding: [0x62
  ; CHECK: vcvttss2usi {{.*}}encoding: [0x62
  %res0 = call i32 @llvm.x86.avx512.cvttss2usi(<4 x float> %a0, i32 8) ;
  %res1 = call i32 @llvm.x86.avx512.cvttss2usi(<4 x float> %a0, i32 4) ;
  %res2 = add i32 %res0, %res1
  ret i32 %res2
}
declare i32 @llvm.x86.avx512.cvttss2usi(<4 x float>, i32) nounwind readnone

define i64 @test_x86_avx512_cvttss2usi64(<4 x float> %a0) {
  ; CHECK: vcvttss2usi {{.*}}encoding: [0x62
  ; CHECK: vcvttss2usi {sae}{{.*}}encoding: [0x62
  %res0 = call i64 @llvm.x86.avx512.cvttss2usi64(<4 x float> %a0, i32 4) ;
  %res1 = call i64 @llvm.x86.avx512.cvttss2usi64(<4 x float> %a0, i32 8) ;
  %res2 = add i64 %res0, %res1
  ret i64 %res2
}
declare i64 @llvm.x86.avx512.cvttss2usi64(<4 x float>, i32) nounwind readnone

define i64 @test_x86_avx512_cvtsd2usi64(<2 x double> %a0) {
  ; CHECK: vcvtsd2usi {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.avx512.cvtsd2usi64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.avx512.cvtsd2usi64(<2 x double>) nounwind readnone

define <16 x float> @test_x86_vcvtph2ps_512(<16 x i16> %a0) {
  ; CHECK: test_x86_vcvtph2ps_512
  ; CHECK: vcvtph2ps  %ymm0, %zmm0    ## encoding: [0x62,0xf2,0x7d,0x48,0x13,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %a0, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_x86_vcvtph2ps_512_sae(<16 x i16> %a0) {
; CHECK: test_x86_vcvtph2ps_512_sae
  ; CHECK: vcvtph2ps  {sae}, %ymm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %a0, <16 x float> zeroinitializer, i16 -1, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_x86_vcvtph2ps_512_rrk(<16 x i16> %a0,<16 x float> %a1, i16 %mask) {
  ; CHECK: test_x86_vcvtph2ps_512_rrk
  ; CHECK: vcvtph2ps  %ymm0, %zmm1 {%k1}
  %res = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %a0, <16 x float> %a1, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_x86_vcvtph2ps_512_sae_rrkz(<16 x i16> %a0, i16 %mask) {
  ; CHECK: test_x86_vcvtph2ps_512_sae_rrkz
  ; CHECK: vcvtph2ps  {sae}, %ymm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %a0, <16 x float> zeroinitializer, i16 %mask, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_x86_vcvtph2ps_512_rrkz(<16 x i16> %a0, i16 %mask) {
  ; CHECK: test_x86_vcvtph2ps_512_rrkz
  ; CHECK: vcvtph2ps  %ymm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %a0, <16 x float> zeroinitializer, i16 %mask, i32 4)
  ret <16 x float> %res
}

declare <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16>, <16 x float>, i16, i32) nounwind readonly


define <16 x i16> @test_x86_vcvtps2ph_256(<16 x float> %a0) {
  ; CHECK: vcvtps2ph $2, %zmm0, %ymm0  ## encoding: [0x62,0xf3,0x7d,0x48,0x1d,0xc0,0x02]
  %res = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %a0, i32 2, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float>, i32, <16 x i16>, i16) nounwind readonly

define <16 x float> @test_x86_vbroadcast_ss_512(i8* %a0) {
  ; CHECK: vbroadcastss
  %res = call <16 x float> @llvm.x86.avx512.vbroadcast.ss.512(i8* %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.vbroadcast.ss.512(i8*) nounwind readonly

define <8 x double> @test_x86_vbroadcast_sd_512(i8* %a0) {
  ; CHECK: vbroadcastsd
  %res = call <8 x double> @llvm.x86.avx512.vbroadcast.sd.512(i8* %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.vbroadcast.sd.512(i8*) nounwind readonly

define <16 x float> @test_x86_vbroadcast_ss_ps_512(<4 x float> %a0) {
  ; CHECK: vbroadcastss
  %res = call <16 x float> @llvm.x86.avx512.vbroadcast.ss.ps.512(<4 x float> %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.vbroadcast.ss.ps.512(<4 x float>) nounwind readonly

define <8 x double> @test_x86_vbroadcast_sd_pd_512(<2 x double> %a0) {
  ; CHECK: vbroadcastsd
  %res = call <8 x double> @llvm.x86.avx512.vbroadcast.sd.pd.512(<2 x double> %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.vbroadcast.sd.pd.512(<2 x double>) nounwind readonly

define <16 x i32>@test_int_x86_avx512_pbroadcastd_512(<4 x i32> %x0, <16 x i32> %x1, i16 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpbroadcastd %xmm0, %zmm1 {%k1}
; CHECK-NEXT:    vpbroadcastd %xmm0, %zmm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastd %xmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.pbroadcastd.512(<4 x i32> %x0, <16 x i32> %x1, i16 -1)
  %res1 = call <16 x i32> @llvm.x86.avx512.pbroadcastd.512(<4 x i32> %x0, <16 x i32> %x1, i16 %mask)
  %res2 = call <16 x i32> @llvm.x86.avx512.pbroadcastd.512(<4 x i32> %x0, <16 x i32> zeroinitializer, i16 %mask)
  %res3 = add <16 x i32> %res, %res1
  %res4 = add <16 x i32> %res2, %res3
  ret <16 x i32> %res4
}
declare <16 x i32> @llvm.x86.avx512.pbroadcastd.512(<4 x i32>, <16 x i32>, i16)

define <16 x i32> @test_x86_pbroadcastd_i32_512(i32  %a0) {
  ; CHECK: vpbroadcastd
  %res = call <16 x i32> @llvm.x86.avx512.pbroadcastd.i32.512(i32 %a0) ; <<16 x i32>> [#uses=1]
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.pbroadcastd.i32.512(i32) nounwind readonly

define <8 x i64>@test_int_x86_avx512_pbroadcastq_512(<2 x i64> %x0, <8 x i64> %x1, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_pbroadcastq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpbroadcastq %xmm0, %zmm1 {%k1}
; CHECK-NEXT:    vpbroadcastq %xmm0, %zmm2 {%k1} {z}
; CHECK-NEXT:    vpbroadcastq %xmm0, %zmm0
; CHECK-NEXT:    vpaddq %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddq %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.pbroadcastq.512(<2 x i64> %x0, <8 x i64> %x1,i8 -1)
  %res1 = call <8 x i64> @llvm.x86.avx512.pbroadcastq.512(<2 x i64> %x0, <8 x i64> %x1,i8 %mask)
  %res2 = call <8 x i64> @llvm.x86.avx512.pbroadcastq.512(<2 x i64> %x0, <8 x i64> zeroinitializer,i8 %mask)
  %res3 = add <8 x i64> %res, %res1
  %res4 = add <8 x i64> %res2, %res3
  ret <8 x i64> %res4
}
declare <8 x i64> @llvm.x86.avx512.pbroadcastq.512(<2 x i64>, <8 x i64>, i8)

define <8 x i64> @test_x86_pbroadcastq_i64_512(i64 %a0) {
  ; CHECK: vpbroadcastq
  %res = call <8 x i64> @llvm.x86.avx512.pbroadcastq.i64.512(i64 %a0) ; <<8 x i64>> [#uses=1]
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.pbroadcastq.i64.512(i64) nounwind readonly

define <16 x i32> @test_conflict_d(<16 x i32> %a) {
; CHECK-LABEL: test_conflict_d:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpconflictd %zmm0, %zmm0
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %a, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32>, <16 x i32>, i16) nounwind readonly

define <8 x i64> @test_conflict_q(<8 x i64> %a) {
; CHECK-LABEL: test_conflict_q:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpconflictq %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %a, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64>, <8 x i64>, i8) nounwind readonly

define <16 x i32> @test_maskz_conflict_d(<16 x i32> %a, i16 %mask) {
; CHECK-LABEL: test_maskz_conflict_d:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpconflictd %zmm0, %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %a, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

define <8 x i64> @test_mask_conflict_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_conflict_q:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpconflictq %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret <8 x i64> %res
}

define <16 x i32> @test_lzcnt_d(<16 x i32> %a) {
; CHECK-LABEL: test_lzcnt_d:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vplzcntd %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.lzcnt.d.512(<16 x i32> %a, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.lzcnt.d.512(<16 x i32>, <16 x i32>, i16) nounwind readonly

define <8 x i64> @test_lzcnt_q(<8 x i64> %a) {
; CHECK-LABEL: test_lzcnt_q:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vplzcntq %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.lzcnt.q.512(<8 x i64> %a, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.lzcnt.q.512(<8 x i64>, <8 x i64>, i8) nounwind readonly


define <16 x i32> @test_mask_lzcnt_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
; CHECK-LABEL: test_mask_lzcnt_d:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vplzcntd %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.lzcnt.d.512(<16 x i32> %a, <16 x i32> %b, i16 %mask)
  ret <16 x i32> %res
}

define <8 x i64> @test_mask_lzcnt_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_lzcnt_q:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vplzcntq %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq ## encoding: [0xc3]
  %res = call <8 x i64> @llvm.x86.avx512.mask.lzcnt.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret <8 x i64> %res
}

define <16 x float> @test_x86_mask_blend_ps_512(i16 %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK: vblendmps %zmm1, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.blend.ps.512(<16 x float> %a1, <16 x float> %a2, i16 %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}

declare <16 x float> @llvm.x86.avx512.mask.blend.ps.512(<16 x float>, <16 x float>, i16) nounwind readonly

define <8 x double> @test_x86_mask_blend_pd_512(i8 %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK: vblendmpd %zmm1, %zmm0
  %res = call <8 x double> @llvm.x86.avx512.mask.blend.pd.512(<8 x double> %a1, <8 x double> %a2, i8 %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}

define <8 x double> @test_x86_mask_blend_pd_512_memop(<8 x double> %a, <8 x double>* %ptr, i8 %mask) {
  ; CHECK-LABEL: test_x86_mask_blend_pd_512_memop
  ; CHECK: vblendmpd (%
  %b = load <8 x double>, <8 x double>* %ptr
  %res = call <8 x double> @llvm.x86.avx512.mask.blend.pd.512(<8 x double> %a, <8 x double> %b, i8 %mask) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.blend.pd.512(<8 x double>, <8 x double>, i8) nounwind readonly

define <16 x i32> @test_x86_mask_blend_d_512(i16 %a0, <16 x i32> %a1, <16 x i32> %a2) {
  ; CHECK: vpblendmd
  %res = call <16 x i32> @llvm.x86.avx512.mask.blend.d.512(<16 x i32> %a1, <16 x i32> %a2, i16 %a0) ; <<16 x i32>> [#uses=1]
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.mask.blend.d.512(<16 x i32>, <16 x i32>, i16) nounwind readonly

define <8 x i64> @test_x86_mask_blend_q_512(i8 %a0, <8 x i64> %a1, <8 x i64> %a2) {
  ; CHECK: vpblendmq
  %res = call <8 x i64> @llvm.x86.avx512.mask.blend.q.512(<8 x i64> %a1, <8 x i64> %a2, i8 %a0) ; <<8 x i64>> [#uses=1]
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.mask.blend.q.512(<8 x i64>, <8 x i64>, i8) nounwind readonly

 define i16 @test_cmpps(<16 x float> %a, <16 x float> %b) {
 ;CHECK: vcmpleps {sae}{{.*}}encoding: [0x62,0xf1,0x7c,0x18,0xc2,0xc1,0x02]
   %res = call i16 @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %a, <16 x float> %b, i32 2, i16 -1, i32 8)
   ret i16 %res
 }
 declare i16 @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> , <16 x float> , i32, i16, i32)

 define i8 @test_cmppd(<8 x double> %a, <8 x double> %b) {
 ;CHECK: vcmpneqpd %zmm{{.*}}encoding: [0x62,0xf1,0xfd,0x48,0xc2,0xc1,0x04]
   %res = call i8 @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %a, <8 x double> %b, i32 4, i8 -1, i32 4)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> , <8 x double> , i32, i8, i32)

 ; fp min - max
define <8 x double> @test_vmaxpd(<8 x double> %a0, <8 x double> %a1) {
  ; CHECK: vmaxpd
  %res = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double>zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)

define <8 x double> @test_vminpd(<8 x double> %a0, <8 x double> %a1) {
  ; CHECK: vminpd
  %res = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double>zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)

 declare <16 x i32> @llvm.x86.avx512.mask.pabs.d.512(<16 x i32>, <16 x i32>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_d_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsd{{.*}}{%k1} 
define <16 x i32>@test_int_x86_avx512_mask_pabs_d_512(<16 x i32> %x0, <16 x i32> %x1, i16 %x2) {
  %res = call <16 x i32> @llvm.x86.avx512.mask.pabs.d.512(<16 x i32> %x0, <16 x i32> %x1, i16 %x2)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.pabs.d.512(<16 x i32> %x0, <16 x i32> %x1, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.pabs.q.512(<8 x i64>, <8 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pabs_q_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpabsq{{.*}}{%k1} 
define <8 x i64>@test_int_x86_avx512_mask_pabs_q_512(<8 x i64> %x0, <8 x i64> %x1, i8 %x2) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pabs.q.512(<8 x i64> %x0, <8 x i64> %x1, i8 %x2)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.pabs.q.512(<8 x i64> %x0, <8 x i64> %x1, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

define i8 @test_vptestmq(<8 x i64> %a0, <8 x i64> %a1) {
  ; CHECK: vptestmq {{.*}}encoding: [0x62,0xf2,0xfd,0x48,0x27,0xc1]
  %res = call i8 @llvm.x86.avx512.mask.ptestm.q.512(<8 x i64> %a0, <8 x i64> %a1, i8 -1)
  ret i8 %res
}
declare i8 @llvm.x86.avx512.mask.ptestm.q.512(<8 x i64>, <8 x i64>, i8)

define i16 @test_vptestmd(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK: vptestmd {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x27,0xc1]
  %res = call i16 @llvm.x86.avx512.mask.ptestm.d.512(<16 x i32> %a0, <16 x i32> %a1, i16 -1)
  ret i16 %res
}
declare i16 @llvm.x86.avx512.mask.ptestm.d.512(<16 x i32>, <16 x i32>, i16)

define void @test_store1(<16 x float> %data, i8* %ptr, i16 %mask) {
; CHECK: vmovups {{.*}}encoding: [0x62,0xf1,0x7c,0x49,0x11,0x07]
  call void @llvm.x86.avx512.mask.storeu.ps.512(i8* %ptr, <16 x float> %data, i16 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.ps.512(i8*, <16 x float>, i16 )

define void @test_store2(<8 x double> %data, i8* %ptr, i8 %mask) {
; CHECK: vmovupd {{.*}}encoding: [0x62,0xf1,0xfd,0x49,0x11,0x07]
  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %ptr, <8 x double> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.pd.512(i8*, <8 x double>, i8)

define void @test_mask_store_aligned_ps(<16 x float> %data, i8* %ptr, i16 %mask) {
; CHECK-LABEL: test_mask_store_aligned_ps:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps %zmm0, (%rdi) {%k1}
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.ps.512(i8* %ptr, <16 x float> %data, i16 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.store.ps.512(i8*, <16 x float>, i16 )

define void @test_mask_store_aligned_pd(<8 x double> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_mask_store_aligned_pd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovapd %zmm0, (%rdi) {%k1}
; CHECK-NEXT:    retq
  call void @llvm.x86.avx512.mask.store.pd.512(i8* %ptr, <8 x double> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.store.pd.512(i8*, <8 x double>, i8)

define <16 x float> @test_maskz_load_aligned_ps(<16 x float> %data, i8* %ptr, i16 %mask) {
; CHECK-LABEL: test_maskz_load_aligned_ps:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.load.ps.512(i8* %ptr, <16 x float> zeroinitializer, i16 %mask)
  ret <16 x float> %res
}

declare <16 x float> @llvm.x86.avx512.mask.load.ps.512(i8*, <16 x float>, i16)

define <8 x double> @test_maskz_load_aligned_pd(<8 x double> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_maskz_load_aligned_pd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmovapd (%rdi), %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.load.pd.512(i8* %ptr, <8 x double> zeroinitializer, i8 %mask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.load.pd.512(i8*, <8 x double>, i8)

define <16 x float> @test_load_aligned_ps(<16 x float> %data, i8* %ptr, i16 %mask) {
; CHECK-LABEL: test_load_aligned_ps:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vmovaps (%rdi), %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.load.ps.512(i8* %ptr, <16 x float> zeroinitializer, i16 -1)
  ret <16 x float> %res
}

define <8 x double> @test_load_aligned_pd(<8 x double> %data, i8* %ptr, i8 %mask) {
; CHECK-LABEL: test_load_aligned_pd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vmovapd (%rdi), %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.load.pd.512(i8* %ptr, <8 x double> zeroinitializer, i8 -1)
  ret <8 x double> %res
}

declare <8 x i64> @llvm.x86.avx512.movntdqa(i8*)

define <8 x i64> @test_valign_q(<8 x i64> %a, <8 x i64> %b) {
; CHECK-LABEL: test_valign_q:
; CHECK: valignq $2, %zmm1, %zmm0, %zmm0
  %res = call <8 x i64> @llvm.x86.avx512.mask.valign.q.512(<8 x i64> %a, <8 x i64> %b, i32 2, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_mask_valign_q(<8 x i64> %a, <8 x i64> %b, <8 x i64> %src, i8 %mask) {
; CHECK-LABEL: test_mask_valign_q:
; CHECK: valignq $2, %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.valign.q.512(<8 x i64> %a, <8 x i64> %b, i32 2, <8 x i64> %src, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.valign.q.512(<8 x i64>, <8 x i64>, i32, <8 x i64>, i8)

define <16 x i32> @test_maskz_valign_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
; CHECK-LABEL: test_maskz_valign_d:
; CHECK: valignd $5, %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf3,0x7d,0xc9,0x03,0xc1,0x05]
  %res = call <16 x i32> @llvm.x86.avx512.mask.valign.d.512(<16 x i32> %a, <16 x i32> %b, i32 5, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.valign.d.512(<16 x i32>, <16 x i32>, i32, <16 x i32>, i16)

define void @test_mask_store_ss(i8* %ptr, <4 x float> %data, i8 %mask) {
 ; CHECK-LABEL: test_mask_store_ss
 ; CHECK: vmovss %xmm0, (%rdi) {%k1}     ## encoding: [0x62,0xf1,0x7e,0x09,0x11,0x07]
 call void @llvm.x86.avx512.mask.store.ss(i8* %ptr, <4 x float> %data, i8 %mask)
 ret void
}

declare void @llvm.x86.avx512.mask.store.ss(i8*, <4 x float>, i8 )

define i16 @test_pcmpeq_d(<16 x i32> %a, <16 x i32> %b) {
; CHECK-LABEL: test_pcmpeq_d
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.d.512(<16 x i32> %a, <16 x i32> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpeq_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_d
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.d.512(<16 x i32> %a, <16 x i32> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpeq.d.512(<16 x i32>, <16 x i32>, i16)

define i8 @test_pcmpeq_q(<8 x i64> %a, <8 x i64> %b) {
; CHECK-LABEL: test_pcmpeq_q
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.512(<8 x i64> %a, <8 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_q
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.q.512(<8 x i64>, <8 x i64>, i8)

define i16 @test_pcmpgt_d(<16 x i32> %a, <16 x i32> %b) {
; CHECK-LABEL: test_pcmpgt_d
; CHECK: vpcmpgtd %zmm1, %zmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.d.512(<16 x i32> %a, <16 x i32> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpgt_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_d
; CHECK: vpcmpgtd %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.d.512(<16 x i32> %a, <16 x i32> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpgt.d.512(<16 x i32>, <16 x i32>, i16)

define i8 @test_pcmpgt_q(<8 x i64> %a, <8 x i64> %b) {
; CHECK-LABEL: test_pcmpgt_q
; CHECK: vpcmpgtq %zmm1, %zmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.512(<8 x i64> %a, <8 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_q
; CHECK: vpcmpgtq %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.q.512(<8 x i64>, <8 x i64>, i8)

define <8 x i16> @test_cmp_d_512(<16 x i32> %a0, <16 x i32> %a1) {
; CHECK_LABEL: test_cmp_d_512
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltd %zmm1, %zmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpled %zmm1, %zmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordd %zmm1, %zmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqd %zmm1, %zmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltd %zmm1, %zmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnled %zmm1, %zmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordd %zmm1, %zmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_d_512(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_d_512
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltd %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpled %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordd %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqd %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltd %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnled %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordd %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32>, <16 x i32>, i32, i16) nounwind readnone

define <8 x i16> @test_ucmp_d_512(<16 x i32> %a0, <16 x i32> %a1) {
; CHECK_LABEL: test_ucmp_d_512
; CHECK: vpcmpequd %zmm1, %zmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltud %zmm1, %zmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleud %zmm1, %zmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordud %zmm1, %zmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequd %zmm1, %zmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltud %zmm1, %zmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleud %zmm1, %zmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordud %zmm1, %zmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_d_512(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_d_512
; CHECK: vpcmpequd %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltud %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleud %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordud %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequd %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltud %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleud %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordud %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32>, <16 x i32>, i32, i16) nounwind readnone

define <8 x i8> @test_cmp_q_512(<8 x i64> %a0, <8 x i64> %a1) {
; CHECK_LABEL: test_cmp_q_512
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %zmm1, %zmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %zmm1, %zmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %zmm1, %zmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %zmm1, %zmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %zmm1, %zmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %zmm1, %zmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %zmm1, %zmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_q_512(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_cmp_q_512
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64>, <8 x i64>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_q_512(<8 x i64> %a0, <8 x i64> %a1) {
; CHECK_LABEL: test_ucmp_q_512
; CHECK: vpcmpequq %zmm1, %zmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %zmm1, %zmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %zmm1, %zmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %zmm1, %zmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %zmm1, %zmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %zmm1, %zmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %zmm1, %zmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %zmm1, %zmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_q_512(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_ucmp_q_512
; CHECK: vpcmpequq %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64>, <8 x i64>, i32, i8) nounwind readnone

define <4 x float> @test_mask_vextractf32x4(<4 x float> %b, <16 x float> %a, i8 %mask) {
; CHECK-LABEL: test_mask_vextractf32x4:
; CHECK: vextractf32x4 $2, %zmm1, %xmm0 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.vextractf32x4.512(<16 x float> %a, i32 2, <4 x float> %b, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.vextractf32x4.512(<16 x float>, i32, <4 x float>, i8)

define <4 x i64> @test_mask_vextracti64x4(<4 x i64> %b, <8 x i64> %a, i8 %mask) {
; CHECK-LABEL: test_mask_vextracti64x4:
; CHECK: vextracti64x4 $2, %zmm1, %ymm0 {%k1}
  %res = call <4 x i64> @llvm.x86.avx512.mask.vextracti64x4.512(<8 x i64> %a, i32 2, <4 x i64> %b, i8 %mask)
  ret <4 x i64> %res
}

declare <4 x i64> @llvm.x86.avx512.mask.vextracti64x4.512(<8 x i64>, i32, <4 x i64>, i8)

define <4 x i32> @test_maskz_vextracti32x4(<16 x i32> %a, i8 %mask) {
; CHECK-LABEL: test_maskz_vextracti32x4:
; CHECK: vextracti32x4 $2, %zmm0, %xmm0 {%k1} {z}
  %res = call <4 x i32> @llvm.x86.avx512.mask.vextracti32x4.512(<16 x i32> %a, i32 2, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.vextracti32x4.512(<16 x i32>, i32, <4 x i32>, i8)

define <4 x double> @test_vextractf64x4(<8 x double> %a) {
; CHECK-LABEL: test_vextractf64x4:
; CHECK: vextractf64x4 $2, %zmm0, %ymm0 ##
  %res = call <4 x double> @llvm.x86.avx512.mask.vextractf64x4.512(<8 x double> %a, i32 2, <4 x double> zeroinitializer, i8 -1)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.vextractf64x4.512(<8 x double>, i32, <4 x double>, i8)

define <16 x i32> @test_x86_avx512_pslli_d(<16 x i32> %a0) {
  ; CHECK-LABEL: test_x86_avx512_pslli_d
  ; CHECK: vpslld
  %res = call <16 x i32> @llvm.x86.avx512.mask.pslli.d(<16 x i32> %a0, i32 7, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_pslli_d(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_pslli_d
  ; CHECK: vpslld $7, %zmm0, %zmm1 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.pslli.d(<16 x i32> %a0, i32 7, <16 x i32> %a1, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_pslli_d(<16 x i32> %a0, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_pslli_d
  ; CHECK: vpslld $7, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.pslli.d(<16 x i32> %a0, i32 7, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.pslli.d(<16 x i32>, i32, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_pslli_q(<8 x i64> %a0) {
  ; CHECK-LABEL: test_x86_avx512_pslli_q
  ; CHECK: vpsllq
  %res = call <8 x i64> @llvm.x86.avx512.mask.pslli.q(<8 x i64> %a0, i32 7, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_pslli_q(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_pslli_q
  ; CHECK: vpsllq $7, %zmm0, %zmm1 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.pslli.q(<8 x i64> %a0, i32 7, <8 x i64> %a1, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_pslli_q(<8 x i64> %a0, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_pslli_q
  ; CHECK: vpsllq $7, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.pslli.q(<8 x i64> %a0, i32 7, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.pslli.q(<8 x i64>, i32, <8 x i64>, i8) nounwind readnone

define <16 x i32> @test_x86_avx512_psrli_d(<16 x i32> %a0) {
  ; CHECK-LABEL: test_x86_avx512_psrli_d
  ; CHECK: vpsrld
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrli.d(<16 x i32> %a0, i32 7, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psrli_d(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrli_d
  ; CHECK: vpsrld $7, %zmm0, %zmm1 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrli.d(<16 x i32> %a0, i32 7, <16 x i32> %a1, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psrli_d(<16 x i32> %a0, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrli_d
  ; CHECK: vpsrld $7, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrli.d(<16 x i32> %a0, i32 7, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psrli.d(<16 x i32>, i32, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psrli_q(<8 x i64> %a0) {
  ; CHECK-LABEL: test_x86_avx512_psrli_q
  ; CHECK: vpsrlq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrli.q(<8 x i64> %a0, i32 7, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psrli_q(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrli_q
  ; CHECK: vpsrlq $7, %zmm0, %zmm1 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrli.q(<8 x i64> %a0, i32 7, <8 x i64> %a1, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psrli_q(<8 x i64> %a0, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrli_q
  ; CHECK: vpsrlq $7, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrli.q(<8 x i64> %a0, i32 7, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psrli.q(<8 x i64>, i32, <8 x i64>, i8) nounwind readnone

define <16 x i32> @test_x86_avx512_psrai_d(<16 x i32> %a0) {
  ; CHECK-LABEL: test_x86_avx512_psrai_d
  ; CHECK: vpsrad
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrai.d(<16 x i32> %a0, i32 7, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psrai_d(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrai_d
  ; CHECK: vpsrad $7, %zmm0, %zmm1 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrai.d(<16 x i32> %a0, i32 7, <16 x i32> %a1, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psrai_d(<16 x i32> %a0, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrai_d
  ; CHECK: vpsrad $7, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrai.d(<16 x i32> %a0, i32 7, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psrai.d(<16 x i32>, i32, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psrai_q(<8 x i64> %a0) {
  ; CHECK-LABEL: test_x86_avx512_psrai_q
  ; CHECK: vpsraq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrai.q(<8 x i64> %a0, i32 7, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psrai_q(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrai_q
  ; CHECK: vpsraq $7, %zmm0, %zmm1 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrai.q(<8 x i64> %a0, i32 7, <8 x i64> %a1, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psrai_q(<8 x i64> %a0, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrai_q
  ; CHECK: vpsraq $7, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrai.q(<8 x i64> %a0, i32 7, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psrai.q(<8 x i64>, i32, <8 x i64>, i8) nounwind readnone

define <16 x i32> @test_x86_avx512_psll_d(<16 x i32> %a0, <4 x i32> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psll_d
  ; CHECK: vpslld
  %res = call <16 x i32> @llvm.x86.avx512.mask.psll.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psll_d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psll_d
  ; CHECK: vpslld %xmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psll.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> %a2, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psll_d(<16 x i32> %a0, <4 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psll_d
  ; CHECK: vpslld %xmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psll.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psll.d(<16 x i32>, <4 x i32>, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psll_q(<8 x i64> %a0, <2 x i64> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psll_q
  ; CHECK: vpsllq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psll.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psll_q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> %a2, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psll_q
  ; CHECK: vpsllq %xmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psll.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> %a2, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psll_q(<8 x i64> %a0, <2 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psll_q
  ; CHECK: vpsllq %xmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psll.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psll.q(<8 x i64>, <2 x i64>, <8 x i64>, i8) nounwind readnone

define <16 x i32> @test_x86_avx512_psrl_d(<16 x i32> %a0, <4 x i32> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psrl_d
  ; CHECK: vpsrld
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrl.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psrl_d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrl_d
  ; CHECK: vpsrld %xmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrl.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> %a2, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psrl_d(<16 x i32> %a0, <4 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrl_d
  ; CHECK: vpsrld %xmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrl.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psrl.d(<16 x i32>, <4 x i32>, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psrl_q(<8 x i64> %a0, <2 x i64> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psrl_q
  ; CHECK: vpsrlq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrl.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psrl_q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> %a2, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrl_q
  ; CHECK: vpsrlq %xmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrl.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> %a2, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psrl_q(<8 x i64> %a0, <2 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrl_q
  ; CHECK: vpsrlq %xmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrl.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psrl.q(<8 x i64>, <2 x i64>, <8 x i64>, i8) nounwind readnone

define <16 x i32> @test_x86_avx512_psra_d(<16 x i32> %a0, <4 x i32> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psra_d
  ; CHECK: vpsrad
  %res = call <16 x i32> @llvm.x86.avx512.mask.psra.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psra_d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psra_d
  ; CHECK: vpsrad %xmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psra.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> %a2, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psra_d(<16 x i32> %a0, <4 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psra_d
  ; CHECK: vpsrad %xmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psra.d(<16 x i32> %a0, <4 x i32> %a1, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psra.d(<16 x i32>, <4 x i32>, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psra_q(<8 x i64> %a0, <2 x i64> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psra_q
  ; CHECK: vpsraq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psra.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psra_q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> %a2, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psra_q
  ; CHECK: vpsraq %xmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psra.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> %a2, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psra_q(<8 x i64> %a0, <2 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psra_q
  ; CHECK: vpsraq %xmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psra.q(<8 x i64> %a0, <2 x i64> %a1, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psra.q(<8 x i64>, <2 x i64>, <8 x i64>, i8) nounwind readnone

define <16 x i32> @test_x86_avx512_psllv_d(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psllv_d
  ; CHECK: vpsllvd
  %res = call <16 x i32> @llvm.x86.avx512.mask.psllv.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psllv_d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psllv_d
  ; CHECK: vpsllvd %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psllv.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psllv_d(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psllv_d
  ; CHECK: vpsllvd %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psllv.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psllv.d(<16 x i32>, <16 x i32>, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psllv_q(<8 x i64> %a0, <8 x i64> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psllv_q
  ; CHECK: vpsllvq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psllv.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psllv_q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> %a2, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psllv_q
  ; CHECK: vpsllvq %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psllv.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> %a2, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psllv_q(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psllv_q
  ; CHECK: vpsllvq %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psllv.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psllv.q(<8 x i64>, <8 x i64>, <8 x i64>, i8) nounwind readnone


define <16 x i32> @test_x86_avx512_psrav_d(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psrav_d
  ; CHECK: vpsravd
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrav.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psrav_d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrav_d
  ; CHECK: vpsravd %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrav.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psrav_d(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrav_d
  ; CHECK: vpsravd %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrav.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psrav.d(<16 x i32>, <16 x i32>, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psrav_q(<8 x i64> %a0, <8 x i64> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psrav_q
  ; CHECK: vpsravq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrav.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psrav_q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> %a2, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrav_q
  ; CHECK: vpsravq %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrav.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> %a2, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psrav_q(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrav_q
  ; CHECK: vpsravq %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrav.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psrav.q(<8 x i64>, <8 x i64>, <8 x i64>, i8) nounwind readnone

define <16 x i32> @test_x86_avx512_psrlv_d(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psrlv_d
  ; CHECK: vpsrlvd
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrlv.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_mask_psrlv_d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrlv_d
  ; CHECK: vpsrlvd %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrlv.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask)
  ret <16 x i32> %res
}

define <16 x i32> @test_x86_avx512_maskz_psrlv_d(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrlv_d
  ; CHECK: vpsrlvd %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x i32> @llvm.x86.avx512.mask.psrlv.d(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psrlv.d(<16 x i32>, <16 x i32>, <16 x i32>, i16) nounwind readnone

define <8 x i64> @test_x86_avx512_psrlv_q(<8 x i64> %a0, <8 x i64> %a1) {
  ; CHECK-LABEL: test_x86_avx512_psrlv_q
  ; CHECK: vpsrlvq
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrlv.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_mask_psrlv_q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> %a2, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_mask_psrlv_q
  ; CHECK: vpsrlvq %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrlv.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> %a2, i8 %mask)
  ret <8 x i64> %res
}

define <8 x i64> @test_x86_avx512_maskz_psrlv_q(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
  ; CHECK-LABEL: test_x86_avx512_maskz_psrlv_q
  ; CHECK: vpsrlvq %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrlv.q(<8 x i64> %a0, <8 x i64> %a1, <8 x i64> zeroinitializer, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psrlv.q(<8 x i64>, <8 x i64>, <8 x i64>, i8) nounwind readnone

define <8 x i64> @test_x86_avx512_psrlv_q_memop(<8 x i64> %a0, <8 x i64>* %ptr) {
  ; CHECK-LABEL: test_x86_avx512_psrlv_q_memop
  ; CHECK: vpsrlvq (%
  %b = load <8 x i64>, <8 x i64>* %ptr
  %res = call <8 x i64> @llvm.x86.avx512.mask.psrlv.q(<8 x i64> %a0, <8 x i64> %b, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

declare <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
declare <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
declare <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)

define <16 x float> @test_vsubps_rn(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vsubps_rn
  ; CHECK: vsubps {rn-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x18,0x5c,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 0)
  ret <16 x float> %res
}

define <16 x float> @test_vsubps_rd(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vsubps_rd
  ; CHECK: vsubps {rd-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x38,0x5c,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 1)
  ret <16 x float> %res
}

define <16 x float> @test_vsubps_ru(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vsubps_ru
  ; CHECK: vsubps {ru-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x58,0x5c,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_vsubps_rz(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vsubps_rz
  ; CHECK: vsubps {rz-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x78,0x5c,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 3)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_rn(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vmulps_rn
  ; CHECK: vmulps {rn-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x18,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 0)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_rd(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vmulps_rd
  ; CHECK: vmulps {rd-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x38,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 1)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_ru(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vmulps_ru
  ; CHECK: vmulps {ru-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x58,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_rz(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK-LABEL: test_vmulps_rz
  ; CHECK: vmulps {rz-sae}{{.*}} ## encoding: [0x62,0xf1,0x7c,0x78,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 -1, i32 3)
  ret <16 x float> %res
}

;; mask float
define <16 x float> @test_vmulps_mask_rn(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_rn
  ; CHECK: vmulps {rn-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0x7c,0x99,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 %mask, i32 0)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_mask_rd(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_rd
  ; CHECK: vmulps {rd-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0x7c,0xb9,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 %mask, i32 1)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_mask_ru(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_ru
  ; CHECK: vmulps {ru-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0x7c,0xd9,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 %mask, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_mask_rz(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_rz
  ; CHECK: vmulps {rz-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0x7c,0xf9,0x59,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> zeroinitializer, i16 %mask, i32 3)
  ret <16 x float> %res
}

;; With Passthru value
define <16 x float> @test_vmulps_mask_passthru_rn(<16 x float> %a0, <16 x float> %a1, <16 x float> %passthru, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_passthru_rn
  ; CHECK: vmulps {rn-sae}{{.*}}{%k1} ## encoding: [0x62,0xf1,0x7c,0x19,0x59,0xd1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> %passthru, i16 %mask, i32 0)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_mask_passthru_rd(<16 x float> %a0, <16 x float> %a1, <16 x float> %passthru, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_passthru_rd
  ; CHECK: vmulps {rd-sae}{{.*}}{%k1} ## encoding: [0x62,0xf1,0x7c,0x39,0x59,0xd1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> %passthru, i16 %mask, i32 1)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_mask_passthru_ru(<16 x float> %a0, <16 x float> %a1, <16 x float> %passthru, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_passthru_ru
  ; CHECK: vmulps {ru-sae}{{.*}}{%k1} ## encoding: [0x62,0xf1,0x7c,0x59,0x59,0xd1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> %passthru, i16 %mask, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_vmulps_mask_passthru_rz(<16 x float> %a0, <16 x float> %a1, <16 x float> %passthru, i16 %mask) {
  ; CHECK-LABEL: test_vmulps_mask_passthru_rz
  ; CHECK: vmulps {rz-sae}{{.*}}{%k1} ## encoding: [0x62,0xf1,0x7c,0x79,0x59,0xd1]
  %res = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float> %passthru, i16 %mask, i32 3)
  ret <16 x float> %res
}

;; mask double
define <8 x double> @test_vmulpd_mask_rn(<8 x double> %a0, <8 x double> %a1, i8 %mask) {
  ; CHECK-LABEL: test_vmulpd_mask_rn
  ; CHECK: vmulpd {rn-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x99,0x59,0xc1]
  %res = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double> zeroinitializer, i8 %mask, i32 0)
  ret <8 x double> %res
}

define <8 x double> @test_vmulpd_mask_rd(<8 x double> %a0, <8 x double> %a1, i8 %mask) {
  ; CHECK-LABEL: test_vmulpd_mask_rd
  ; CHECK: vmulpd {rd-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xb9,0x59,0xc1]
  %res = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double> zeroinitializer, i8 %mask, i32 1)
  ret <8 x double> %res
}

define <8 x double> @test_vmulpd_mask_ru(<8 x double> %a0, <8 x double> %a1, i8 %mask) {
  ; CHECK-LABEL: test_vmulpd_mask_ru
  ; CHECK: vmulpd {ru-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xd9,0x59,0xc1]
  %res = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double> zeroinitializer, i8 %mask, i32 2)
  ret <8 x double> %res
}

define <8 x double> @test_vmulpd_mask_rz(<8 x double> %a0, <8 x double> %a1, i8 %mask) {
  ; CHECK-LABEL: test_vmulpd_mask_rz
  ; CHECK: vmulpd {rz-sae}{{.*}}{%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xf9,0x59,0xc1]
  %res = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double> zeroinitializer, i8 %mask, i32 3)
  ret <8 x double> %res
}

define <16 x i32> @test_xor_epi32(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_xor_epi32
  ;CHECK: vpxord {{.*}}encoding: [0x62,0xf1,0x7d,0x48,0xef,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pxor.d.512(<16 x i32> %a,<16 x i32> %b, <16 x i32>zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_xor_epi32(<16 x i32> %a,<16 x i32> %b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32
  ;CHECK: vpxord %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xef,0xd1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pxor.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.pxor.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @test_or_epi32(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_or_epi32
  ;CHECK: vpord {{.*}}encoding: [0x62,0xf1,0x7d,0x48,0xeb,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.por.d.512(<16 x i32> %a,<16 x i32> %b, <16 x i32>zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_or_epi32(<16 x i32> %a,<16 x i32> %b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32
  ;CHECK: vpord %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xeb,0xd1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.por.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.por.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @test_and_epi32(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_and_epi32
  ;CHECK: vpandd {{.*}}encoding: [0x62,0xf1,0x7d,0x48,0xdb,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pand.d.512(<16 x i32> %a,<16 x i32> %b, <16 x i32>zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_and_epi32(<16 x i32> %a,<16 x i32> %b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32
  ;CHECK: vpandd %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xdb,0xd1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pand.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.pand.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <8 x i64> @test_xor_epi64(<8 x i64> %a, <8 x i64> %b) {
  ;CHECK-LABEL: test_xor_epi64
  ;CHECK: vpxorq {{.*}}encoding: [0x62,0xf1,0xfd,0x48,0xef,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pxor.q.512(<8 x i64> %a,<8 x i64> %b, <8 x i64>zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_xor_epi64(<8 x i64> %a,<8 x i64> %b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi64
  ;CHECK: vpxorq %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xef,0xd1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pxor.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.pxor.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64> @test_or_epi64(<8 x i64> %a, <8 x i64> %b) {
  ;CHECK-LABEL: test_or_epi64
  ;CHECK: vporq {{.*}}encoding: [0x62,0xf1,0xfd,0x48,0xeb,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.por.q.512(<8 x i64> %a,<8 x i64> %b, <8 x i64>zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_or_epi64(<8 x i64> %a,<8 x i64> %b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi64
  ;CHECK: vporq %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xeb,0xd1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.por.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.por.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64> @test_and_epi64(<8 x i64> %a, <8 x i64> %b) {
  ;CHECK-LABEL: test_and_epi64
  ;CHECK: vpandq {{.*}}encoding: [0x62,0xf1,0xfd,0x48,0xdb,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pand.q.512(<8 x i64> %a,<8 x i64> %b, <8 x i64>zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_and_epi64(<8 x i64> %a,<8 x i64> %b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi64
  ;CHECK: vpandq %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xdb,0xd1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pand.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.pand.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)


define <16 x i32> @test_mask_add_epi32_rr(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_add_epi32_rr
  ;CHECK: vpaddd %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xfe,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rrk(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrk
  ;CHECK: vpaddd %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfe,0xd1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rrkz(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrkz
  ;CHECK: vpaddd %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfe,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rm(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rm
  ;CHECK: vpaddd (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xfe,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rmk(<16 x i32> %a, <16 x i32>* %ptr_b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmk
  ;CHECK: vpaddd (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfe,0x0f]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rmkz(<16 x i32> %a, <16 x i32>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmkz
  ;CHECK: vpaddd (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfe,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rmb(<16 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rmb
  ;CHECK: vpaddd (%rdi){1to16}, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x58,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rmbk(<16 x i32> %a, i32* %ptr_b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbk
  ;CHECK: vpaddd (%rdi){1to16}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x59,0xfe,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_add_epi32_rmbkz(<16 x i32> %a, i32* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbkz
  ;CHECK: vpaddd (%rdi){1to16}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xd9,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.padd.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @test_mask_sub_epi32_rr(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rr
  ;CHECK: vpsubd %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xfa,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rrk(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrk
  ;CHECK: vpsubd %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfa,0xd1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rrkz(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrkz
  ;CHECK: vpsubd %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfa,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rm(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rm
  ;CHECK: vpsubd (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xfa,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rmk(<16 x i32> %a, <16 x i32>* %ptr_b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmk
  ;CHECK: vpsubd (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfa,0x0f]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rmkz(<16 x i32> %a, <16 x i32>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmkz
  ;CHECK: vpsubd (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfa,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rmb(<16 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmb
  ;CHECK: vpsubd (%rdi){1to16}, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x58,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rmbk(<16 x i32> %a, i32* %ptr_b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbk
  ;CHECK: vpsubd (%rdi){1to16}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x59,0xfa,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_sub_epi32_rmbkz(<16 x i32> %a, i32* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbkz
  ;CHECK: vpsubd (%rdi){1to16}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xd9,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.psub.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <8 x i64> @test_mask_add_epi64_rr(<8 x i64> %a, <8 x i64> %b) {
  ;CHECK-LABEL: test_mask_add_epi64_rr
  ;CHECK: vpaddq %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0xfd,0x48,0xd4,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rrk(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi64_rrk
  ;CHECK: vpaddq %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xd4,0xd1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rrkz(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi64_rrkz
  ;CHECK: vpaddq %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0xd4,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rm(<8 x i64> %a, <8 x i64>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi64_rm
  ;CHECK: vpaddq (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0xfd,0x48,0xd4,0x07]
  %b = load <8 x i64>, <8 x i64>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rmk(<8 x i64> %a, <8 x i64>* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi64_rmk
  ;CHECK: vpaddq (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xd4,0x0f]
  %b = load <8 x i64>, <8 x i64>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rmkz(<8 x i64> %a, <8 x i64>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi64_rmkz
  ;CHECK: vpaddq (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0xd4,0x07]
  %b = load <8 x i64>, <8 x i64>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rmb(<8 x i64> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi64_rmb
  ;CHECK: vpaddq (%rdi){1to8}, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x58,0xd4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rmbk(<8 x i64> %a, i64* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi64_rmbk
  ;CHECK: vpaddq (%rdi){1to8}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x59,0xd4,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_add_epi64_rmbkz(<8 x i64> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi64_rmbkz
  ;CHECK: vpaddq (%rdi){1to8}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xd9,0xd4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %res = call <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.padd.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64> @test_mask_sub_epi64_rr(<8 x i64> %a, <8 x i64> %b) {
  ;CHECK-LABEL: test_mask_sub_epi64_rr
  ;CHECK: vpsubq %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0xfd,0x48,0xfb,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rrk(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi64_rrk
  ;CHECK: vpsubq %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xfb,0xd1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rrkz(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi64_rrkz
  ;CHECK: vpsubq %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0xfb,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rm(<8 x i64> %a, <8 x i64>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi64_rm
  ;CHECK: vpsubq (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0xfd,0x48,0xfb,0x07]
  %b = load <8 x i64>, <8 x i64>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rmk(<8 x i64> %a, <8 x i64>* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi64_rmk
  ;CHECK: vpsubq (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xfb,0x0f]
  %b = load <8 x i64>, <8 x i64>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rmkz(<8 x i64> %a, <8 x i64>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi64_rmkz
  ;CHECK: vpsubq (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0xfb,0x07]
  %b = load <8 x i64>, <8 x i64>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rmb(<8 x i64> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi64_rmb
  ;CHECK: vpsubq (%rdi){1to8}, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x58,0xfb,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rmbk(<8 x i64> %a, i64* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi64_rmbk
  ;CHECK: vpsubq (%rdi){1to8}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x59,0xfb,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_sub_epi64_rmbkz(<8 x i64> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi64_rmbkz
  ;CHECK: vpsubq (%rdi){1to8}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xd9,0xfb,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %res = call <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64> %a, <8 x i64> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.psub.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64> @test_mask_mul_epi32_rr(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rr
  ;CHECK: vpmuldq %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf2,0xfd,0x48,0x28,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rrk(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrk
  ;CHECK: vpmuldq %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf2,0xfd,0x49,0x28,0xd1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rrkz(<16 x i32> %a, <16 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrkz
  ;CHECK: vpmuldq %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xc9,0x28,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rm(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rm
  ;CHECK: vpmuldq (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf2,0xfd,0x48,0x28,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rmk(<16 x i32> %a, <16 x i32>* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmk
  ;CHECK: vpmuldq (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x49,0x28,0x0f]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rmkz(<16 x i32> %a, <16 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmkz
  ;CHECK: vpmuldq (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xc9,0x28,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rmb(<16 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmb
  ;CHECK: vpmuldq (%rdi){1to8}, %zmm0, %zmm0  ## encoding: [0x62,0xf2,0xfd,0x58,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %b = bitcast <8 x i64> %b64 to <16 x i32>
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rmbk(<16 x i32> %a, i64* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbk
  ;CHECK: vpmuldq (%rdi){1to8}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x59,0x28,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %b = bitcast <8 x i64> %b64 to <16 x i32>
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epi32_rmbkz(<16 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbkz
  ;CHECK: vpmuldq (%rdi){1to8}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xd9,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %b = bitcast <8 x i64> %b64 to <16 x i32>
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.pmul.dq.512(<16 x i32>, <16 x i32>, <8 x i64>, i8)

define <8 x i64> @test_mask_mul_epu32_rr(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rr
  ;CHECK: vpmuludq %zmm1, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x48,0xf4,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rrk(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrk
  ;CHECK: vpmuludq %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xf4,0xd1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rrkz(<16 x i32> %a, <16 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrkz
  ;CHECK: vpmuludq %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0xf4,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rm(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rm
  ;CHECK: vpmuludq (%rdi), %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x48,0xf4,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rmk(<16 x i32> %a, <16 x i32>* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmk
  ;CHECK: vpmuludq (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0xf4,0x0f]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rmkz(<16 x i32> %a, <16 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmkz
  ;CHECK: vpmuludq (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0xf4,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rmb(<16 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmb
  ;CHECK: vpmuludq (%rdi){1to8}, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x58,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %b = bitcast <8 x i64> %b64 to <16 x i32>
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 -1)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rmbk(<16 x i32> %a, i64* %ptr_b, <8 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbk
  ;CHECK: vpmuludq (%rdi){1to8}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x59,0xf4,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %b = bitcast <8 x i64> %b64 to <16 x i32>
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> %passThru, i8 %mask)
  ret < 8 x i64> %res
}

define <8 x i64> @test_mask_mul_epu32_rmbkz(<16 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbkz
  ;CHECK: vpmuludq (%rdi){1to8}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xd9,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <8 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector <8 x i64> %vecinit.i, <8 x i64> undef, <8 x i32> zeroinitializer
  %b = bitcast <8 x i64> %b64 to <16 x i32>
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a, <16 x i32> %b, <8 x i64> zeroinitializer, i8 %mask)
  ret < 8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32>, <16 x i32>, <8 x i64>, i8)

define <16 x i32> @test_mask_mullo_epi32_rr_512(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rr_512
  ;CHECK: vpmulld %zmm1, %zmm0, %zmm0 ## encoding: [0x62,0xf2,0x7d,0x48,0x40,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rrk_512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rrk_512
  ;CHECK: vpmulld %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf2,0x7d,0x49,0x40,0xd1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rrkz_512(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rrkz_512
  ;CHECK: vpmulld %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0xc9,0x40,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rm_512(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rm_512
  ;CHECK: vpmulld (%rdi), %zmm0, %zmm0 ## encoding: [0x62,0xf2,0x7d,0x48,0x40,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rmk_512(<16 x i32> %a, <16 x i32>* %ptr_b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rmk_512
  ;CHECK: vpmulld (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf2,0x7d,0x49,0x40,0x0f]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rmkz_512(<16 x i32> %a, <16 x i32>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rmkz_512
  ;CHECK: vpmulld (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0xc9,0x40,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rmb_512(<16 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rmb_512
  ;CHECK: vpmulld (%rdi){1to16}, %zmm0, %zmm0 ## encoding: [0x62,0xf2,0x7d,0x58,0x40,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 -1)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rmbk_512(<16 x i32> %a, i32* %ptr_b, <16 x i32> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rmbk_512
  ;CHECK: vpmulld (%rdi){1to16}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf2,0x7d,0x59,0x40,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %passThru, i16 %mask)
  ret < 16 x i32> %res
}

define <16 x i32> @test_mask_mullo_epi32_rmbkz_512(<16 x i32> %a, i32* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi32_rmbkz_512
  ;CHECK: vpmulld (%rdi){1to16}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0xd9,0x40,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32> %a, <16 x i32> %b, <16 x i32> zeroinitializer, i16 %mask)
  ret < 16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.pmull.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x float> @test_mm512_maskz_add_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_round_ps_rn_sae
  ;CHECK: vaddps {rn-sae}, %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_maskz_add_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_round_ps_rd_sae
  ;CHECK: vaddps {rd-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_maskz_add_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_round_ps_ru_sae
  ;CHECK: vaddps  {ru-sae}, %zmm1, %zmm0, %zmm0  {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_maskz_add_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_round_ps_rz_sae
  ;CHECK: vaddps  {rz-sae}, %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 3)
  ret <16 x float> %res
}


define <16 x float> @test_mm512_maskz_add_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_round_ps_current
  ;CHECK: vaddps %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_add_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_round_ps_rn_sae
  ;CHECK: vaddps {rn-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_mask_add_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_round_ps_rd_sae
  ;CHECK: vaddps {rd-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_mask_add_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_round_ps_ru_sae
  ;CHECK: vaddps  {ru-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_add_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_round_ps_rz_sae
  ;CHECK: vaddps  {rz-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 3)
  ret <16 x float> %res
}


define <16 x float> @test_mm512_mask_add_round_ps_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_round_ps_current
  ;CHECK: vaddps %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 4)
  ret <16 x float> %res
}


define <16 x float> @test_mm512_add_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_add_round_ps_rn_sae
  ;CHECK: vaddps {rn-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_add_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_add_round_ps_rd_sae
  ;CHECK: vaddps {rd-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_add_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_add_round_ps_ru_sae
  ;CHECK: vaddps  {ru-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_add_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_add_round_ps_rz_sae
  ;CHECK: vaddps  {rz-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 3)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_add_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_add_round_ps_current
  ;CHECK: vaddps %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.add.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @test_mm512_mask_sub_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_round_ps_rn_sae
  ;CHECK: vsubps {rn-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_mask_sub_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_round_ps_rd_sae
  ;CHECK: vsubps {rd-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_mask_sub_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_round_ps_ru_sae
  ;CHECK: vsubps  {ru-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_sub_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_round_ps_rz_sae
  ;CHECK: vsubps  {rz-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 3)
  ret <16 x float> %res
}


define <16 x float> @test_mm512_mask_sub_round_ps_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_round_ps_current
  ;CHECK: vsubps %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_sub_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_sub_round_ps_rn_sae
  ;CHECK: vsubps {rn-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_sub_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_sub_round_ps_rd_sae
  ;CHECK: vsubps {rd-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_sub_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_sub_round_ps_ru_sae
  ;CHECK: vsubps  {ru-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_sub_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_sub_round_ps_rz_sae
  ;CHECK: vsubps  {rz-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 3)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_sub_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_sub_round_ps_current
  ;CHECK: vsubps %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.sub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_maskz_div_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_round_ps_rn_sae
  ;CHECK: vdivps {rn-sae}, %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_maskz_div_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_round_ps_rd_sae
  ;CHECK: vdivps {rd-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_maskz_div_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_round_ps_ru_sae
  ;CHECK: vdivps  {ru-sae}, %zmm1, %zmm0, %zmm0  {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_maskz_div_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_round_ps_rz_sae
  ;CHECK: vdivps  {rz-sae}, %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 3)
  ret <16 x float> %res
}


define <16 x float> @test_mm512_maskz_div_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_round_ps_current
  ;CHECK: vdivps %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_div_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_round_ps_rn_sae
  ;CHECK: vdivps {rn-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_mask_div_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_round_ps_rd_sae
  ;CHECK: vdivps {rd-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_mask_div_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_round_ps_ru_sae
  ;CHECK: vdivps  {ru-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_div_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_round_ps_rz_sae
  ;CHECK: vdivps  {rz-sae}, %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 3)
  ret <16 x float> %res
}


define <16 x float> @test_mm512_mask_div_round_ps_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_round_ps_current
  ;CHECK: vdivps %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 4)
  ret <16 x float> %res
}


define <16 x float> @test_mm512_div_round_ps_rn_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_div_round_ps_rn_sae
  ;CHECK: vdivps {rn-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 0)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_div_round_ps_rd_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_div_round_ps_rd_sae
  ;CHECK: vdivps {rd-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 1)
  ret <16 x float> %res
}
define <16 x float> @test_mm512_div_round_ps_ru_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_div_round_ps_ru_sae
  ;CHECK: vdivps  {ru-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 2)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_div_round_ps_rz_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_div_round_ps_rz_sae
  ;CHECK: vdivps  {rz-sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 3)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_div_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_div_round_ps_current
  ;CHECK: vdivps %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.div.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @test_mm512_maskz_min_round_ps_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_min_round_ps_sae
  ;CHECK: vminps {sae}, %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_maskz_min_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_min_round_ps_current
  ;CHECK: vminps %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_min_round_ps_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_min_round_ps_sae
  ;CHECK: vminps {sae}, %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_min_round_ps_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_min_round_ps_current
  ;CHECK: vminps %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_min_round_ps_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_min_round_ps_sae
  ;CHECK: vminps {sae}, %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_min_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_min_round_ps_current
  ;CHECK: vminps %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @test_mm512_maskz_max_round_ps_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_max_round_ps_sae
  ;CHECK: vmaxps {sae}, %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_maskz_max_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_max_round_ps_current
  ;CHECK: vmaxps %zmm1, %zmm0, %zmm0 {%k1} {z}
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_max_round_ps_sae(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_max_round_ps_sae
  ;CHECK: vmaxps {sae}, %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_mask_max_round_ps_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask) {
  ;CHECK-LABEL: test_mm512_mask_max_round_ps_current
  ;CHECK: vmaxps %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %src, i16 %mask, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_max_round_ps_sae(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_max_round_ps_sae
  ;CHECK: vmaxps {sae}, %zmm1, %zmm0, %zmm0 
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test_mm512_max_round_ps_current(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: test_mm512_max_round_ps_current
  ;CHECK: vmaxps %zmm1, %zmm0, %zmm0
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

declare <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_mask_add_ss_rn(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_ss_rn
; CHECK: vaddss  {rn-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 0)
  ret <4 x float> %res
}

define <4 x float> @test_mask_add_ss_rd(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_ss_rd
; CHECK: vaddss  {rd-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 1)
  ret <4 x float> %res
}

define <4 x float> @test_mask_add_ss_ru(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_ss_ru
; CHECK: vaddss  {ru-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 2)
  ret <4 x float> %res
}

define <4 x float> @test_mask_add_ss_rz(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_ss_rz
; CHECK: vaddss  {rz-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 3)
  ret <4 x float> %res
}

define <4 x float> @test_mask_add_ss_current(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_ss_current
; CHECK: vaddss %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 4)
  ret <4 x float> %res
}

define <4 x float> @test_maskz_add_ss_rn(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
; CHECK-LABEL: test_maskz_add_ss_rn
; CHECK: vaddss  {rn-sae}, %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %mask, i32 0)
  ret <4 x float> %res
}

define <4 x float> @test_add_ss_rn(<4 x float> %a0, <4 x float> %a1) {
; CHECK-LABEL: test_add_ss_rn
; CHECK: vaddss  {rn-sae}, %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 -1, i32 0)
  ret <4 x float> %res
}

declare <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>, <2 x double>, <2 x double>, i8, i32) nounwind readnone

define <2 x double> @test_mask_add_sd_rn(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_sd_rn
; CHECK: vaddsd  {rn-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 0)
  ret <2 x double> %res
}

define <2 x double> @test_mask_add_sd_rd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_sd_rd
; CHECK: vaddsd  {rd-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 1)
  ret <2 x double> %res
}

define <2 x double> @test_mask_add_sd_ru(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_sd_ru
; CHECK: vaddsd  {ru-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 2)
  ret <2 x double> %res
}

define <2 x double> @test_mask_add_sd_rz(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_sd_rz
; CHECK: vaddsd  {rz-sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 3)
  ret <2 x double> %res
}

define <2 x double> @test_mask_add_sd_current(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_add_sd_current
; CHECK: vaddsd %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 4)
  ret <2 x double> %res
}

define <2 x double> @test_maskz_add_sd_rn(<2 x double> %a0, <2 x double> %a1, i8 %mask) {
; CHECK-LABEL: test_maskz_add_sd_rn
; CHECK: vaddsd  {rn-sae}, %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 %mask, i32 0)
  ret <2 x double> %res
}

define <2 x double> @test_add_sd_rn(<2 x double> %a0, <2 x double> %a1) {
; CHECK-LABEL: test_add_sd_rn
; CHECK: vaddsd  {rn-sae}, %xmm1, %xmm0, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.mask.add.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 -1, i32 0)
  ret <2 x double> %res
}

declare <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_mask_max_ss_sae(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_max_ss_sae
; CHECK: vmaxss  {sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 8)
  ret <4 x float> %res
}

define <4 x float> @test_maskz_max_ss_sae(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
; CHECK-LABEL: test_maskz_max_ss_sae
; CHECK: vmaxss  {sae}, %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %mask, i32 8)
  ret <4 x float> %res
}

define <4 x float> @test_max_ss_sae(<4 x float> %a0, <4 x float> %a1) {
; CHECK-LABEL: test_max_ss_sae
; CHECK: vmaxss  {sae}, %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 -1, i32 8)
  ret <4 x float> %res
}

define <4 x float> @test_mask_max_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_max_ss
; CHECK: vmaxss  %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 4)
  ret <4 x float> %res
}

define <4 x float> @test_maskz_max_ss(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
; CHECK-LABEL: test_maskz_max_ss
; CHECK: vmaxss  %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %mask, i32 4)
  ret <4 x float> %res
}

define <4 x float> @test_max_ss(<4 x float> %a0, <4 x float> %a1) {
; CHECK-LABEL: test_max_ss
; CHECK: vmaxss  %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 -1, i32 4)
  ret <4 x float> %res
}
declare <2 x double> @llvm.x86.avx512.mask.max.sd.round(<2 x double>, <2 x double>, <2 x double>, i8, i32) nounwind readnone

define <2 x double> @test_mask_max_sd_sae(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_max_sd_sae
; CHECK: vmaxsd  {sae}, %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.mask.max.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 8)
  ret <2 x double> %res
}

define <2 x double> @test_maskz_max_sd_sae(<2 x double> %a0, <2 x double> %a1, i8 %mask) {
; CHECK-LABEL: test_maskz_max_sd_sae
; CHECK: vmaxsd  {sae}, %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <2 x double> @llvm.x86.avx512.mask.max.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 %mask, i32 8)
  ret <2 x double> %res
}

define <2 x double> @test_max_sd_sae(<2 x double> %a0, <2 x double> %a1) {
; CHECK-LABEL: test_max_sd_sae
; CHECK: vmaxsd  {sae}, %xmm1, %xmm0, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.mask.max.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 -1, i32 8)
  ret <2 x double> %res
}

define <2 x double> @test_mask_max_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_mask_max_sd
; CHECK: vmaxsd  %xmm1, %xmm0, %xmm2 {%k1}
  %res = call <2 x double> @llvm.x86.avx512.mask.max.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 4)
  ret <2 x double> %res
}

define <2 x double> @test_maskz_max_sd(<2 x double> %a0, <2 x double> %a1, i8 %mask) {
; CHECK-LABEL: test_maskz_max_sd
; CHECK: vmaxsd  %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <2 x double> @llvm.x86.avx512.mask.max.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 %mask, i32 4)
  ret <2 x double> %res
}

define <2 x double> @test_max_sd(<2 x double> %a0, <2 x double> %a1) {
; CHECK-LABEL: test_max_sd
; CHECK: vmaxsd  %xmm1, %xmm0, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.mask.max.sd.round(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 -1, i32 4)
  ret <2 x double> %res
}

define <2 x double> @test_x86_avx512_cvtsi2sd32(<2 x double> %a, i32 %b) {
; CHECK-LABEL: test_x86_avx512_cvtsi2sd32:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtsi2sdl %edi, {rz-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <2 x double> @llvm.x86.avx512.cvtsi2sd32(<2 x double> %a, i32 %b, i32 3) ; <<<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.cvtsi2sd32(<2 x double>, i32, i32) nounwind readnone

define <2 x double> @test_x86_avx512_cvtsi2sd64(<2 x double> %a, i64 %b) {
; CHECK-LABEL: test_x86_avx512_cvtsi2sd64:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtsi2sdq %rdi, {rz-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <2 x double> @llvm.x86.avx512.cvtsi2sd64(<2 x double> %a, i64 %b, i32 3) ; <<<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.cvtsi2sd64(<2 x double>, i64, i32) nounwind readnone

define <4 x float> @test_x86_avx512_cvtsi2ss32(<4 x float> %a, i32 %b) {
; CHECK-LABEL: test_x86_avx512_cvtsi2ss32:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtsi2ssl %edi, {rz-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <4 x float> @llvm.x86.avx512.cvtsi2ss32(<4 x float> %a, i32 %b, i32 3) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.cvtsi2ss32(<4 x float>, i32, i32) nounwind readnone

define <4 x float> @test_x86_avx512_cvtsi2ss64(<4 x float> %a, i64 %b) {
; CHECK-LABEL: test_x86_avx512_cvtsi2ss64:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtsi2ssq %rdi, {rz-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
  %res = call <4 x float> @llvm.x86.avx512.cvtsi2ss64(<4 x float> %a, i64 %b, i32 3) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.cvtsi2ss64(<4 x float>, i64, i32) nounwind readnone

define <4 x float> @test_x86_avx512__mm_cvt_roundu32_ss (<4 x float> %a, i32 %b)
; CHECK-LABEL: test_x86_avx512__mm_cvt_roundu32_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2ssl %edi, {rd-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %res = call <4 x float> @llvm.x86.avx512.cvtusi2ss(<4 x float> %a, i32 %b, i32 1) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}

define <4 x float> @test_x86_avx512__mm_cvt_roundu32_ss_mem(<4 x float> %a, i32* %ptr)
; CHECK-LABEL: test_x86_avx512__mm_cvt_roundu32_ss_mem:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movl (%rdi), %eax 
; CHECK-NEXT:    vcvtusi2ssl %eax, {rd-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %b = load i32, i32* %ptr
  %res = call <4 x float> @llvm.x86.avx512.cvtusi2ss(<4 x float> %a, i32 %b, i32 1) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}

define <4 x float> @test_x86_avx512__mm_cvtu32_ss(<4 x float> %a, i32 %b)
; CHECK-LABEL: test_x86_avx512__mm_cvtu32_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2ssl %edi, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %res = call <4 x float> @llvm.x86.avx512.cvtusi2ss(<4 x float> %a, i32 %b, i32 4) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}

define <4 x float> @test_x86_avx512__mm_cvtu32_ss_mem(<4 x float> %a, i32* %ptr)
; CHECK-LABEL: test_x86_avx512__mm_cvtu32_ss_mem:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2ssl (%rdi), %xmm0, %xmm0
; CHECK-NEXT:    retq 
{
  %b = load i32, i32* %ptr
  %res = call <4 x float> @llvm.x86.avx512.cvtusi2ss(<4 x float> %a, i32 %b, i32 4) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.cvtusi2ss(<4 x float>, i32, i32) nounwind readnone

define <4 x float> @_mm_cvt_roundu64_ss (<4 x float> %a, i64 %b)
; CHECK-LABEL: _mm_cvt_roundu64_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2ssq %rdi, {rd-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %res = call <4 x float> @llvm.x86.avx512.cvtusi642ss(<4 x float> %a, i64 %b, i32 1) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}

define <4 x float> @_mm_cvtu64_ss(<4 x float> %a, i64 %b)
; CHECK-LABEL: _mm_cvtu64_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2ssq %rdi, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %res = call <4 x float> @llvm.x86.avx512.cvtusi642ss(<4 x float> %a, i64 %b, i32 4) ; <<<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.cvtusi642ss(<4 x float>, i64, i32) nounwind readnone

define <2 x double> @test_x86_avx512_mm_cvtu32_sd(<2 x double> %a, i32 %b)
; CHECK-LABEL: test_x86_avx512_mm_cvtu32_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2sdl %edi, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %res = call <2 x double> @llvm.x86.avx512.cvtusi2sd(<2 x double> %a, i32 %b) ; <<<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.cvtusi2sd(<2 x double>, i32) nounwind readnone

define <2 x double> @test_x86_avx512_mm_cvtu64_sd(<2 x double> %a, i64 %b)
; CHECK-LABEL: test_x86_avx512_mm_cvtu64_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2sdq %rdi, {rd-sae}, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %res = call <2 x double> @llvm.x86.avx512.cvtusi642sd(<2 x double> %a, i64 %b, i32 1) ; <<<2 x double>> [#uses=1]
  ret <2 x double> %res
}

define <2 x double> @test_x86_avx512__mm_cvt_roundu64_sd(<2 x double> %a, i64 %b)
; CHECK-LABEL: test_x86_avx512__mm_cvt_roundu64_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvtusi2sdq %rdi, %xmm0, %xmm0 
; CHECK-NEXT:    retq 
{
  %res = call <2 x double> @llvm.x86.avx512.cvtusi642sd(<2 x double> %a, i64 %b, i32 4) ; <<<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.cvtusi642sd(<2 x double>, i64, i32) nounwind readnone

define <8 x i64> @test_vpmaxq(<8 x i64> %a0, <8 x i64> %a1) {
  ; CHECK: vpmaxsq {{.*}}encoding: [0x62,0xf2,0xfd,0x48,0x3d,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %a0, <8 x i64> %a1,
                    <8 x i64>zeroinitializer, i8 -1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <16 x i32> @test_vpminud(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK: vpminud {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x3b,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32> %a0, <16 x i32> %a1,
                    <16 x i32>zeroinitializer, i16 -1)
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @test_vpmaxsd(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK: vpmaxsd {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x3d,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32> %a0, <16 x i32> %a1,
                    <16 x i32>zeroinitializer, i16 -1)
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_d_512
; CHECK-NOT: call 
; CHECK: vpmaxsd %zmm
; CHECK: {%k1} 
define <16 x i32>@test_int_x86_avx512_mask_pmaxs_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxs_q_512
; CHECK-NOT: call 
; CHECK: vpmaxsq %zmm
; CHECK: {%k1} 
define <8 x i64>@test_int_x86_avx512_mask_pmaxs_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_d_512
; CHECK-NOT: call 
; CHECK: vpmaxud %zmm
; CHECK: {%k1} 
define <16 x i32>@test_int_x86_avx512_mask_pmaxu_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmaxu_q_512
; CHECK-NOT: call 
; CHECK: vpmaxuq %zmm
; CHECK: {%k1} 
define <8 x i64>@test_int_x86_avx512_mask_pmaxu_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_d_512
; CHECK-NOT: call 
; CHECK: vpminsd %zmm
; CHECK: {%k1} 
define <16 x i32>@test_int_x86_avx512_mask_pmins_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pmins_q_512
; CHECK-NOT: call 
; CHECK: vpminsq %zmm
; CHECK: {%k1} 
define <8 x i64>@test_int_x86_avx512_mask_pmins_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_d_512
; CHECK-NOT: call 
; CHECK: vpminud %zmm
; CHECK: {%k1} 
define <16 x i32>@test_int_x86_avx512_mask_pminu_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
  %res = call <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_pminu_q_512
; CHECK-NOT: call 
; CHECK: vpminuq %zmm
; CHECK: {%k1} 
define <8 x i64>@test_int_x86_avx512_mask_pminu_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.vpermi2var.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_d_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermi2d {{.*}}{%k1} 
define <16 x i32>@test_int_x86_avx512_mask_vpermi2var_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
  %res = call <16 x i32> @llvm.x86.avx512.mask.vpermi2var.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.vpermi2var.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double>, <8 x i64>, <8 x double>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_pd_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK:  vpermi2pd {{.*}}{%k1} 
define <8 x double>@test_int_x86_avx512_mask_vpermi2var_pd_512(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2, i8 %x3) {
  %res = call <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2, i8 %x3)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2, i8 -1)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x float> @llvm.x86.avx512.mask.vpermi2var.ps.512(<16 x float>, <16 x i32>, <16 x float>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_ps_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermi2ps {{.*}}{%k1} 
define <16 x float>@test_int_x86_avx512_mask_vpermi2var_ps_512(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2, i16 %x3) {
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermi2var.ps.512(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2, i16 %x3)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.vpermi2var.ps.512(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2, i16 -1)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.vpermi2var.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermi2var_q_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermi2q {{.*}}{%k1} 
define <8 x i64>@test_int_x86_avx512_mask_vpermi2var_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.vpermi2var.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.vpermi2var.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <16 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_d_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2d {{.*}}{%k1} {z}
define <16 x i32>@test_int_x86_avx512_maskz_vpermt2var_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
  %res = call <16 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.maskz.vpermt2var.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x double> @llvm.x86.avx512.maskz.vpermt2var.pd.512(<8 x i64>, <8 x double>, <8 x double>, i8)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_pd_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2pd {{.*}}{%k1} {z}
define <8 x double>@test_int_x86_avx512_maskz_vpermt2var_pd_512(<8 x i64> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3) {
  %res = call <8 x double> @llvm.x86.avx512.maskz.vpermt2var.pd.512(<8 x i64> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3)
  %res1 = call <8 x double> @llvm.x86.avx512.maskz.vpermt2var.pd.512(<8 x i64> %x0, <8 x double> %x1, <8 x double> %x2, i8 -1)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x float> @llvm.x86.avx512.maskz.vpermt2var.ps.512(<16 x i32>, <16 x float>, <16 x float>, i16)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_ps_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2ps {{.*}}{%k1} {z}
define <16 x float>@test_int_x86_avx512_maskz_vpermt2var_ps_512(<16 x i32> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3) {
  %res = call <16 x float> @llvm.x86.avx512.maskz.vpermt2var.ps.512(<16 x i32> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3)
  %res1 = call <16 x float> @llvm.x86.avx512.maskz.vpermt2var.ps.512(<16 x i32> %x0, <16 x float> %x1, <16 x float> %x2, i16 -1)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}


declare <8 x i64> @llvm.x86.avx512.maskz.vpermt2var.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

; CHECK-LABEL: @test_int_x86_avx512_maskz_vpermt2var_q_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2q {{.*}}{%k1} {z}
define <8 x i64>@test_int_x86_avx512_maskz_vpermt2var_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
  %res = call <8 x i64> @llvm.x86.avx512.maskz.vpermt2var.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.maskz.vpermt2var.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.vpermt2var.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

; CHECK-LABEL: @test_int_x86_avx512_mask_vpermt2var_d_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vpermt2d {{.*}}{%k1}
; CHECK-NOT: {z}
define <16 x i32>@test_int_x86_avx512_mask_vpermt2var_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
  %res = call <16 x i32> @llvm.x86.avx512.mask.vpermt2var.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.vpermt2var.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)
; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_pd_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vscalefpd{{.*}}{%k1} 
define <8 x double>@test_int_x86_avx512_mask_scalef_pd_512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3) {
  %res = call <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3, i32 3)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 -1, i32 0)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x float> @llvm.x86.avx512.mask.scalef.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_ps_512
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vscalefps{{.*}}{%k1} 
define <16 x float>@test_int_x86_avx512_mask_scalef_ps_512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3) {
  %res = call <16 x float> @llvm.x86.avx512.mask.scalef.ps.512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3, i32 2)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.scalef.ps.512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 -1, i32 0)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.unpckh.pd.512(<8 x double>, <8 x double>, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_unpckh_pd_512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckh_pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vunpckhpd %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[1],k1[1],zmm2[3],k1[3],zmm2[5],k1[5],zmm2[7],k1[7]
; CHECK-NEXT:    vunpckhpd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
  %res = call <8 x double> @llvm.x86.avx512.mask.unpckh.pd.512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.unpckh.pd.512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 -1)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x float> @llvm.x86.avx512.mask.unpckh.ps.512(<16 x float>, <16 x float>, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_unpckh_ps_512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckh_ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vunpckhps %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[2],k1[2],zmm2[3],k1[3],zmm2[6],k1[6],zmm2[7],k1[7],zmm2[10],k1[10],zmm2[11],k1[11],zmm2[14],k1[14],zmm2[15],k1[15]
; CHECK-NEXT:    vunpckhps %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
  %res = call <16 x float> @llvm.x86.avx512.mask.unpckh.ps.512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.unpckh.ps.512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 -1)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.unpckl.pd.512(<8 x double>, <8 x double>, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_unpckl_pd_512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckl_pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vunpcklpd %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[0],k1[0],zmm2[2],k1[2],zmm2[4],k1[4],zmm2[6],k1[6]
; CHECK-NEXT:    vunpcklpd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
  %res = call <8 x double> @llvm.x86.avx512.mask.unpckl.pd.512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 %x3)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.unpckl.pd.512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x2, i8 -1)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x float> @llvm.x86.avx512.mask.unpckl.ps.512(<16 x float>, <16 x float>, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_unpckl_ps_512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_unpckl_ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vunpcklps %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[0],k1[0],zmm2[1],k1[1],zmm2[4],k1[4],zmm2[5],k1[5],zmm2[8],k1[8],zmm2[9],k1[9],zmm2[12],k1[12],zmm2[13],k1[13]
; CHECK-NEXT:    vunpcklps %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
  %res = call <16 x float> @llvm.x86.avx512.mask.unpckl.ps.512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 %x3)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.unpckl.ps.512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x2, i16 -1)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.punpcklqd.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_mask_punpcklqd_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpcklqd_q_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpunpcklqdq %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[0],k1[0],zmm2[2],k1[2],zmm2[4],k1[4],zmm2[6],k1[6]
; CHECK-NEXT:    vpunpcklqdq %zmm1, %zmm0, %zmm3 {%k1} {z
; CHECK-NEXT:    ## zmm3 = k1[0],zmm0[0],k1[2],zmm0[2],k1[4],zmm0[4],k1[6],zmm0[6]
; CHECK-NEXT:    vpunpcklqdq %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; CHECK-NEXT:    vpaddq %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    vpaddq %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.punpcklqd.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.punpcklqd.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = call <8 x i64> @llvm.x86.avx512.mask.punpcklqd.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> zeroinitializer,i8 %x3)
  %res3 = add <8 x i64> %res, %res1
  %res4 = add <8 x i64> %res2, %res3
  ret <8 x i64> %res4
}

declare <8 x i64> @llvm.x86.avx512.mask.punpckhqd.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_mask_punpckhqd_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhqd_q_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpunpckhqdq %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[1],k1[1],zmm2[3],k1[3],zmm2[5],k1[5],zmm2[7],k1[7]
; CHECK-NEXT:    vpunpckhqdq %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; CHECK-NEXT:    vpaddq %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.punpckhqd.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x3)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.punpckhqd.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.punpckhd.q.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_punpckhd_q_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckhd_q_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpunpckhdq %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[2],k1[2],zmm2[3],k1[3],zmm2[6],k1[6],zmm2[7],k1[7],zmm2[10],k1[10],zmm2[11],k1[11],zmm2[14],k1[14],zmm2[15],k1[15]
; CHECK-NEXT:    vpunpckhdq %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
; CHECK-NEXT:    vpaddd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.punpckhd.q.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.punpckhd.q.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.punpckld.q.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_punpckld_q_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_punpckld_q_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpunpckldq %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[0],k1[0],zmm2[1],k1[1],zmm2[4],k1[4],zmm2[5],k1[5],zmm2[8],k1[8],zmm2[9],k1[9],zmm2[12],k1[12],zmm2[13],k1[13]
; CHECK-NEXT:    vpunpckldq %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
; CHECK-NEXT:    vpaddd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.punpckld.q.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x3)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.punpckld.q.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.qb.512(<8 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmov_qb_512(<8 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qb_512:
; CHECK:       vpmovqb %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqb %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqb %zmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.512(<8 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.512(<8 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.qb.512(<8 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qb.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qb_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qb_mem_512:
; CHECK:  vpmovqb %zmm0, (%rdi)
; CHECK:  vpmovqb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qb.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qb.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.512(<8 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_qb_512(<8 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qb_512:
; CHECK:       vpmovsqb %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqb %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqb %zmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.512(<8 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.512(<8 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.qb.512(<8 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qb.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qb_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qb_mem_512:
; CHECK:  vpmovsqb %zmm0, (%rdi)
; CHECK:  vpmovsqb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qb.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qb.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.512(<8 x i64>, <16 x i8>, i8)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_qb_512(<8 x i64> %x0, <16 x i8> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qb_512:
; CHECK:       vpmovusqb %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqb %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqb %zmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.512(<8 x i64> %x0, <16 x i8> %x1, i8 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.512(<8 x i64> %x0, <16 x i8> %x1, i8 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.qb.512(<8 x i64> %x0, <16 x i8> zeroinitializer, i8 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qb.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qb_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qb_mem_512:
; CHECK:  vpmovusqb %zmm0, (%rdi)
; CHECK:  vpmovusqb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qb.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qb.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmov.qw.512(<8 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmov_qw_512(<8 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qw_512:
; CHECK:       vpmovqw %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovqw %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovqw %zmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.512(<8 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.512(<8 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.512(<8 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qw.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qw_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qw_mem_512:
; CHECK:  vpmovqw %zmm0, (%rdi)
; CHECK:  vpmovqw %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qw.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qw.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.512(<8 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovs_qw_512(<8 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qw_512:
; CHECK:       vpmovsqw %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsqw %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqw %zmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.512(<8 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.512(<8 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.512(<8 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qw.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qw_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qw_mem_512:
; CHECK:  vpmovsqw %zmm0, (%rdi)
; CHECK:  vpmovsqw %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qw.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qw.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.512(<8 x i64>, <8 x i16>, i8)

define <8 x i16>@test_int_x86_avx512_mask_pmovus_qw_512(<8 x i64> %x0, <8 x i16> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qw_512:
; CHECK:       vpmovusqw %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusqw %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqw %zmm0, %xmm0
    %res0 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.512(<8 x i64> %x0, <8 x i16> %x1, i8 -1)
    %res1 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.512(<8 x i64> %x0, <8 x i16> %x1, i8 %x2)
    %res2 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.512(<8 x i64> %x0, <8 x i16> zeroinitializer, i8 %x2)
    %res3 = add <8 x i16> %res0, %res1
    %res4 = add <8 x i16> %res3, %res2
    ret <8 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qw.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qw_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qw_mem_512:
; CHECK:  vpmovusqw %zmm0, (%rdi)
; CHECK:  vpmovusqw %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qw.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qw.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i32> @llvm.x86.avx512.mask.pmov.qd.512(<8 x i64>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmov_qd_512(<8 x i64> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qd_512:
; CHECK:       vpmovqd %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovqd %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovqd %zmm0, %ymm0
    %res0 = call <8 x i32> @llvm.x86.avx512.mask.pmov.qd.512(<8 x i64> %x0, <8 x i32> %x1, i8 -1)
    %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmov.qd.512(<8 x i64> %x0, <8 x i32> %x1, i8 %x2)
    %res2 = call <8 x i32> @llvm.x86.avx512.mask.pmov.qd.512(<8 x i64> %x0, <8 x i32> zeroinitializer, i8 %x2)
    %res3 = add <8 x i32> %res0, %res1
    %res4 = add <8 x i32> %res3, %res2
    ret <8 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmov.qd.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmov_qd_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_qd_mem_512:
; CHECK:  vpmovqd %zmm0, (%rdi)
; CHECK:  vpmovqd %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.qd.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmov.qd.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i32> @llvm.x86.avx512.mask.pmovs.qd.512(<8 x i64>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmovs_qd_512(<8 x i64> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qd_512:
; CHECK:       vpmovsqd %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovsqd %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovsqd %zmm0, %ymm0
    %res0 = call <8 x i32> @llvm.x86.avx512.mask.pmovs.qd.512(<8 x i64> %x0, <8 x i32> %x1, i8 -1)
    %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmovs.qd.512(<8 x i64> %x0, <8 x i32> %x1, i8 %x2)
    %res2 = call <8 x i32> @llvm.x86.avx512.mask.pmovs.qd.512(<8 x i64> %x0, <8 x i32> zeroinitializer, i8 %x2)
    %res3 = add <8 x i32> %res0, %res1
    %res4 = add <8 x i32> %res3, %res2
    ret <8 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.qd.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovs_qd_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_qd_mem_512:
; CHECK:  vpmovsqd %zmm0, (%rdi)
; CHECK:  vpmovsqd %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.qd.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovs.qd.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <8 x i32> @llvm.x86.avx512.mask.pmovus.qd.512(<8 x i64>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_pmovus_qd_512(<8 x i64> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qd_512:
; CHECK:       vpmovusqd %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovusqd %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovusqd %zmm0, %ymm0
    %res0 = call <8 x i32> @llvm.x86.avx512.mask.pmovus.qd.512(<8 x i64> %x0, <8 x i32> %x1, i8 -1)
    %res1 = call <8 x i32> @llvm.x86.avx512.mask.pmovus.qd.512(<8 x i64> %x0, <8 x i32> %x1, i8 %x2)
    %res2 = call <8 x i32> @llvm.x86.avx512.mask.pmovus.qd.512(<8 x i64> %x0, <8 x i32> zeroinitializer, i8 %x2)
    %res3 = add <8 x i32> %res0, %res1
    %res4 = add <8 x i32> %res3, %res2
    ret <8 x i32> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.qd.mem.512(i8* %ptr, <8 x i64>, i8)

define void @test_int_x86_avx512_mask_pmovus_qd_mem_512(i8* %ptr, <8 x i64> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_qd_mem_512:
; CHECK:  vpmovusqd %zmm0, (%rdi)
; CHECK:  vpmovusqd %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.qd.mem.512(i8* %ptr, <8 x i64> %x1, i8 -1)
    call void @llvm.x86.avx512.mask.pmovus.qd.mem.512(i8* %ptr, <8 x i64> %x1, i8 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmov.db.512(<16 x i32>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_pmov_db_512(<16 x i32> %x0, <16 x i8> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_db_512:
; CHECK:       vpmovdb %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovdb %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovdb %zmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.512(<16 x i32> %x0, <16 x i8> %x1, i16 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.512(<16 x i32> %x0, <16 x i8> %x1, i16 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.512(<16 x i32> %x0, <16 x i8> zeroinitializer, i16 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmov.db.mem.512(i8* %ptr, <16 x i32>, i16)

define void @test_int_x86_avx512_mask_pmov_db_mem_512(i8* %ptr, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_db_mem_512:
; CHECK:  vpmovdb %zmm0, (%rdi)
; CHECK:  vpmovdb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.db.mem.512(i8* %ptr, <16 x i32> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmov.db.mem.512(i8* %ptr, <16 x i32> %x1, i16 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_pmovs_db_512(<16 x i32> %x0, <16 x i8> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_db_512:
; CHECK:       vpmovsdb %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovsdb %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovsdb %zmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32> %x0, <16 x i8> %x1, i16 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32> %x0, <16 x i8> %x1, i16 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32> %x0, <16 x i8> zeroinitializer, i16 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.db.mem.512(i8* %ptr, <16 x i32>, i16)

define void @test_int_x86_avx512_mask_pmovs_db_mem_512(i8* %ptr, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_db_mem_512:
; CHECK:  vpmovsdb %zmm0, (%rdi)
; CHECK:  vpmovsdb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.db.mem.512(i8* %ptr, <16 x i32> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmovs.db.mem.512(i8* %ptr, <16 x i32> %x1, i16 %x2)
    ret void
}

declare <16 x i8> @llvm.x86.avx512.mask.pmovus.db.512(<16 x i32>, <16 x i8>, i16)

define <16 x i8>@test_int_x86_avx512_mask_pmovus_db_512(<16 x i32> %x0, <16 x i8> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_db_512:
; CHECK:       vpmovusdb %zmm0, %xmm1 {%k1}
; CHECK-NEXT:  vpmovusdb %zmm0, %xmm2 {%k1} {z}
; CHECK-NEXT:  vpmovusdb %zmm0, %xmm0
    %res0 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.512(<16 x i32> %x0, <16 x i8> %x1, i16 -1)
    %res1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.512(<16 x i32> %x0, <16 x i8> %x1, i16 %x2)
    %res2 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.512(<16 x i32> %x0, <16 x i8> zeroinitializer, i16 %x2)
    %res3 = add <16 x i8> %res0, %res1
    %res4 = add <16 x i8> %res3, %res2
    ret <16 x i8> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.db.mem.512(i8* %ptr, <16 x i32>, i16)

define void @test_int_x86_avx512_mask_pmovus_db_mem_512(i8* %ptr, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_db_mem_512:
; CHECK:  vpmovusdb %zmm0, (%rdi)
; CHECK:  vpmovusdb %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.db.mem.512(i8* %ptr, <16 x i32> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmovus.db.mem.512(i8* %ptr, <16 x i32> %x1, i16 %x2)
    ret void
}

declare <16 x i16> @llvm.x86.avx512.mask.pmov.dw.512(<16 x i32>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pmov_dw_512(<16 x i32> %x0, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_dw_512:
; CHECK:       vpmovdw %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovdw %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovdw %zmm0, %ymm0
    %res0 = call <16 x i16> @llvm.x86.avx512.mask.pmov.dw.512(<16 x i32> %x0, <16 x i16> %x1, i16 -1)
    %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmov.dw.512(<16 x i32> %x0, <16 x i16> %x1, i16 %x2)
    %res2 = call <16 x i16> @llvm.x86.avx512.mask.pmov.dw.512(<16 x i32> %x0, <16 x i16> zeroinitializer, i16 %x2)
    %res3 = add <16 x i16> %res0, %res1
    %res4 = add <16 x i16> %res3, %res2
    ret <16 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmov.dw.mem.512(i8* %ptr, <16 x i32>, i16)

define void @test_int_x86_avx512_mask_pmov_dw_mem_512(i8* %ptr, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmov_dw_mem_512:
; CHECK:  vpmovdw %zmm0, (%rdi)
; CHECK:  vpmovdw %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmov.dw.mem.512(i8* %ptr, <16 x i32> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmov.dw.mem.512(i8* %ptr, <16 x i32> %x1, i16 %x2)
    ret void
}

declare <16 x i16> @llvm.x86.avx512.mask.pmovs.dw.512(<16 x i32>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pmovs_dw_512(<16 x i32> %x0, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_dw_512:
; CHECK:       vpmovsdw %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovsdw %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovsdw %zmm0, %ymm0
    %res0 = call <16 x i16> @llvm.x86.avx512.mask.pmovs.dw.512(<16 x i32> %x0, <16 x i16> %x1, i16 -1)
    %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmovs.dw.512(<16 x i32> %x0, <16 x i16> %x1, i16 %x2)
    %res2 = call <16 x i16> @llvm.x86.avx512.mask.pmovs.dw.512(<16 x i32> %x0, <16 x i16> zeroinitializer, i16 %x2)
    %res3 = add <16 x i16> %res0, %res1
    %res4 = add <16 x i16> %res3, %res2
    ret <16 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovs.dw.mem.512(i8* %ptr, <16 x i32>, i16)

define void @test_int_x86_avx512_mask_pmovs_dw_mem_512(i8* %ptr, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovs_dw_mem_512:
; CHECK:  vpmovsdw %zmm0, (%rdi)
; CHECK:  vpmovsdw %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovs.dw.mem.512(i8* %ptr, <16 x i32> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmovs.dw.mem.512(i8* %ptr, <16 x i32> %x1, i16 %x2)
    ret void
}

declare <16 x i16> @llvm.x86.avx512.mask.pmovus.dw.512(<16 x i32>, <16 x i16>, i16)

define <16 x i16>@test_int_x86_avx512_mask_pmovus_dw_512(<16 x i32> %x0, <16 x i16> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_dw_512:
; CHECK:       vpmovusdw %zmm0, %ymm1 {%k1}
; CHECK-NEXT:  vpmovusdw %zmm0, %ymm2 {%k1} {z}
; CHECK-NEXT:  vpmovusdw %zmm0, %ymm0
    %res0 = call <16 x i16> @llvm.x86.avx512.mask.pmovus.dw.512(<16 x i32> %x0, <16 x i16> %x1, i16 -1)
    %res1 = call <16 x i16> @llvm.x86.avx512.mask.pmovus.dw.512(<16 x i32> %x0, <16 x i16> %x1, i16 %x2)
    %res2 = call <16 x i16> @llvm.x86.avx512.mask.pmovus.dw.512(<16 x i32> %x0, <16 x i16> zeroinitializer, i16 %x2)
    %res3 = add <16 x i16> %res0, %res1
    %res4 = add <16 x i16> %res3, %res2
    ret <16 x i16> %res4
}

declare void @llvm.x86.avx512.mask.pmovus.dw.mem.512(i8* %ptr, <16 x i32>, i16)

define void @test_int_x86_avx512_mask_pmovus_dw_mem_512(i8* %ptr, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_pmovus_dw_mem_512:
; CHECK:  vpmovusdw %zmm0, (%rdi)
; CHECK:  vpmovusdw %zmm0, (%rdi) {%k1}
    call void @llvm.x86.avx512.mask.pmovus.dw.mem.512(i8* %ptr, <16 x i32> %x1, i16 -1)
    call void @llvm.x86.avx512.mask.pmovus.dw.mem.512(i8* %ptr, <16 x i32> %x1, i16 %x2)
    ret void
}

declare <8 x double> @llvm.x86.avx512.mask.cvtdq2pd.512(<8 x i32>, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_cvt_dq2pd_512(<8 x i32> %x0, <8 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_dq2pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvtdq2pd %ymm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvtdq2pd %ymm0, %zmm0
; CHECK-NEXT:    vaddpd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.cvtdq2pd.512(<8 x i32> %x0, <8 x double> %x1, i8 %x2)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.cvtdq2pd.512(<8 x i32> %x0, <8 x double> %x1, i8 -1)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x float> @llvm.x86.avx512.mask.cvtdq2ps.512(<16 x i32>, <16 x float>, i16, i32)

define <16 x float>@test_int_x86_avx512_mask_cvt_dq2ps_512(<16 x i32> %x0, <16 x float> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_dq2ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtdq2ps %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvtdq2ps {rn-sae}, %zmm0, %zmm0
; CHECK-NEXT:    vaddps %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.cvtdq2ps.512(<16 x i32> %x0, <16 x float> %x1, i16 %x2, i32 4)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.cvtdq2ps.512(<16 x i32> %x0, <16 x float> %x1, i16 -1, i32 0)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvtpd2dq.512(<8 x double>, <8 x i32>, i8, i32)

define <8 x i32>@test_int_x86_avx512_mask_cvt_pd2dq_512(<8 x double> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2dq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvtpd2dq %zmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtpd2dq {rn-sae}, %zmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvtpd2dq.512(<8 x double> %x0, <8 x i32> %x1, i8 %x2, i32 4)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvtpd2dq.512(<8 x double> %x0, <8 x i32> %x1, i8 -1, i32 0)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <8 x float> @llvm.x86.avx512.mask.cvtpd2ps.512(<8 x double>, <8 x float>, i8, i32)

define <8 x float>@test_int_x86_avx512_mask_cvt_pd2ps_512(<8 x double> %x0, <8 x float> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvtpd2ps %zmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtpd2ps {ru-sae}, %zmm0, %ymm0
; CHECK-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x float> @llvm.x86.avx512.mask.cvtpd2ps.512(<8 x double> %x0, <8 x float> %x1, i8 %x2, i32 4)
  %res1 = call <8 x float> @llvm.x86.avx512.mask.cvtpd2ps.512(<8 x double> %x0, <8 x float> %x1, i8 -1, i32 2)
  %res2 = fadd <8 x float> %res, %res1
  ret <8 x float> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvtpd2udq.512(<8 x double>, <8 x i32>, i8, i32)

define <8 x i32>@test_int_x86_avx512_mask_cvt_pd2udq_512(<8 x double> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_pd2udq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvtpd2udq {ru-sae}, %zmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvtpd2udq {rn-sae}, %zmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvtpd2udq.512(<8 x double> %x0, <8 x i32> %x1, i8 %x2, i32 2)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvtpd2udq.512(<8 x double> %x0, <8 x i32> %x1, i8 -1, i32 0)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.cvtps2dq.512(<16 x float>, <16 x i32>, i16, i32)

define <16 x i32>@test_int_x86_avx512_mask_cvt_ps2dq_512(<16 x float> %x0, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2dq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2dq {ru-sae}, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvtps2dq {rn-sae}, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.cvtps2dq.512(<16 x float> %x0, <16 x i32> %x1, i16 %x2, i32 2)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.cvtps2dq.512(<16 x float> %x0, <16 x i32> %x1, i16 -1, i32 0)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.cvtps2pd.512(<8 x float>, <8 x double>, i8, i32)

define <8 x double>@test_int_x86_avx512_mask_cvt_ps2pd_512(<8 x float> %x0, <8 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvtps2pd %ymm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvtps2pd {sae}, %ymm0, %zmm0
; CHECK-NEXT:    vaddpd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.cvtps2pd.512(<8 x float> %x0, <8 x double> %x1, i8 %x2, i32 4)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.cvtps2pd.512(<8 x float> %x0, <8 x double> %x1, i8 -1, i32 8)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.cvtps2udq.512(<16 x float>, <16 x i32>, i16, i32)

define <16 x i32>@test_int_x86_avx512_mask_cvt_ps2udq_512(<16 x float> %x0, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ps2udq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtps2udq {ru-sae}, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvtps2udq {rn-sae}, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.cvtps2udq.512(<16 x float> %x0, <16 x i32> %x1, i16 %x2, i32 2)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.cvtps2udq.512(<16 x float> %x0, <16 x i32> %x1, i16 -1, i32 0)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvttpd2dq.512(<8 x double>, <8 x i32>, i8, i32)

define <8 x i32>@test_int_x86_avx512_mask_cvtt_pd2dq_512(<8 x double> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_pd2dq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvttpd2dq %zmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvttpd2dq {sae}, %zmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvttpd2dq.512(<8 x double> %x0, <8 x i32> %x1, i8 %x2, i32 4)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvttpd2dq.512(<8 x double> %x0, <8 x i32> %x1, i8 -1, i32 8)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.cvtudq2pd.512(<8 x i32>, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_cvt_udq2pd_512(<8 x i32> %x0, <8 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_udq2pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvtudq2pd %ymm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvtudq2pd %ymm0, %zmm0
; CHECK-NEXT:    vaddpd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.cvtudq2pd.512(<8 x i32> %x0, <8 x double> %x1, i8 %x2)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.cvtudq2pd.512(<8 x i32> %x0, <8 x double> %x1, i8 -1)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}


declare <16 x float> @llvm.x86.avx512.mask.cvtudq2ps.512(<16 x i32>, <16 x float>, i16, i32)

define <16 x float>@test_int_x86_avx512_mask_cvt_udq2ps_512(<16 x i32> %x0, <16 x float> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_udq2ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtudq2ps %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvtudq2ps {rn-sae}, %zmm0, %zmm0
; CHECK-NEXT:    vaddps %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.cvtudq2ps.512(<16 x i32> %x0, <16 x float> %x1, i16 %x2, i32 4)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.cvtudq2ps.512(<16 x i32> %x0, <16 x float> %x1, i16 -1, i32 0)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x i32> @llvm.x86.avx512.mask.cvttpd2udq.512(<8 x double>, <8 x i32>, i8, i32)

define <8 x i32>@test_int_x86_avx512_mask_cvtt_pd2udq_512(<8 x double> %x0, <8 x i32> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_pd2udq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vcvttpd2udq %zmm0, %ymm1 {%k1}
; CHECK-NEXT:    vcvttpd2udq {sae}, %zmm0, %ymm0
; CHECK-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvttpd2udq.512(<8 x double> %x0, <8 x i32> %x1, i8 %x2, i32 4)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.cvttpd2udq.512(<8 x double> %x0, <8 x i32> %x1, i8 -1, i32 8)
  %res2 = add <8 x i32> %res, %res1
  ret <8 x i32> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.cvttps2dq.512(<16 x float>, <16 x i32>, i16, i32)

define <16 x i32>@test_int_x86_avx512_mask_cvtt_ps2dq_512(<16 x float> %x0, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_ps2dq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttps2dq %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvttps2dq {sae}, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.cvttps2dq.512(<16 x float> %x0, <16 x i32> %x1, i16 %x2, i32 4)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.cvttps2dq.512(<16 x float> %x0, <16 x i32> %x1, i16 -1, i32 8)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.cvttps2udq.512(<16 x float>, <16 x i32>, i16, i32)

define <16 x i32>@test_int_x86_avx512_mask_cvtt_ps2udq_512(<16 x float> %x0, <16 x i32> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvtt_ps2udq_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcvttps2udq %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vcvttps2udq {sae}, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.cvttps2udq.512(<16 x float> %x0, <16 x i32> %x1, i16 %x2, i32 4)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.cvttps2udq.512(<16 x float> %x0, <16 x i32> %x1, i16 -1, i32 8)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}


declare <4 x float> @llvm.x86.avx512.mask.scalef.ss(<4 x float>, <4 x float>,<4 x float>, i8, i32)
; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_ss
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vscalefss {{.*}}{%k1} 
; CHECK: vscalefss	{rn-sae}
define <4 x float>@test_int_x86_avx512_mask_scalef_ss(<4 x float> %x0, <4 x float> %x1, <4 x float> %x3, i8 %x4) {
  %res = call <4 x float> @llvm.x86.avx512.mask.scalef.ss(<4 x float> %x0, <4 x float> %x1, <4 x float> %x3, i8 %x4, i32 4)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.scalef.ss(<4 x float> %x0, <4 x float> %x1, <4 x float> %x3, i8 -1, i32 8)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.scalef.sd(<2 x double>, <2 x double>,<2 x double>, i8, i32)
; CHECK-LABEL: @test_int_x86_avx512_mask_scalef_sd
; CHECK-NOT: call 
; CHECK: kmov 
; CHECK: vscalefsd {{.*}}{%k1} 
; CHECK: vscalefsd	{rn-sae}
define <2 x double>@test_int_x86_avx512_mask_scalef_sd(<2 x double> %x0, <2 x double> %x1, <2 x double> %x3, i8 %x4) {
  %res = call <2 x double> @llvm.x86.avx512.mask.scalef.sd(<2 x double> %x0, <2 x double> %x1, <2 x double> %x3, i8 %x4, i32 4)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.scalef.sd(<2 x double> %x0, <2 x double> %x1, <2 x double> %x3, i8 -1, i32 8)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.getexp.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_getexp_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
; CHECK-LABEL: test_getexp_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vgetexpss %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vgetexpss {sae}, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vgetexpss {sae}, %xmm1, %xmm0, %xmm4 {%k1} {z}
; CHECK-NEXT:    vgetexpss {sae}, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm2, %xmm3, %xmm1
; CHECK-NEXT:    vaddps %xmm0, %xmm4, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res0 = call <4 x float> @llvm.x86.avx512.mask.getexp.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 4)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.getexp.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> %a2, i8 %mask, i32 8)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.getexp.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %mask, i32 8)
  %res3 = call <4 x float> @llvm.x86.avx512.mask.getexp.ss(<4 x float>%a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 -1, i32 8)

  %res.1 = fadd <4 x float> %res0, %res1
  %res.2 = fadd <4 x float> %res2, %res3
  %res   = fadd <4 x float> %res.1, %res.2
  ret <4 x float> %res
}

declare <2 x double> @llvm.x86.avx512.mask.getexp.sd(<2 x double>, <2 x double>, <2 x double>, i8, i32) nounwind readnone

define <2 x double> @test_getexp_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
; CHECK-LABEL: test_getexp_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vgetexpsd %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vgetexpsd %xmm1, %xmm0, %xmm4
; CHECK-NEXT:    vgetexpsd {sae}, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vgetexpsd {sae}, %xmm1, %xmm0, %xmm0 {%k1} {z}
; CHECK-NEXT:    vaddpd %xmm2, %xmm3, %xmm1
; CHECK-NEXT:    vaddpd %xmm4, %xmm0, %xmm0
; CHECK-NEXT:    vaddpd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res0 = call <2 x double> @llvm.x86.avx512.mask.getexp.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 4)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.getexp.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> %a2, i8 %mask, i32 8)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.getexp.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 %mask, i32 8)
  %res3 = call <2 x double> @llvm.x86.avx512.mask.getexp.sd(<2 x double>%a0, <2 x double> %a1, <2 x double> zeroinitializer, i8 -1, i32 4)

  %res.1 = fadd <2 x double> %res0, %res1
  %res.2 = fadd <2 x double> %res2, %res3
  %res   = fadd <2 x double> %res.1, %res.2
  ret <2 x double> %res
}

declare i8 @llvm.x86.avx512.mask.cmp.sd(<2 x double>, <2 x double>, i32, i8, i32)

define i8@test_int_x86_avx512_mask_cmp_sd(<2 x double> %x0, <2 x double> %x1, i8 %x3, i32 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_cmp_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcmpnltsd {sae}, %xmm1, %xmm0, %k0 {%k1}
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    shlb $7, %al
; CHECK-NEXT:    sarb $7, %al
; CHECK-NEXT:    retq

  %res4 = call i8 @llvm.x86.avx512.mask.cmp.sd(<2 x double> %x0, <2 x double> %x1, i32 5, i8 %x3, i32 8)
  ret i8 %res4
}

define i8@test_int_x86_avx512_mask_cmp_sd_all(<2 x double> %x0, <2 x double> %x1, i8 %x3, i32 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_cmp_sd_all:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcmpunordsd {sae}, %xmm1, %xmm0, %k0
; CHECK-NEXT:    vcmplesd %xmm1, %xmm0, %k1
; CHECK-NEXT:    korw %k0, %k1, %k0
; CHECK-NEXT:    vcmpnltsd {sae}, %xmm1, %xmm0, %k1
; CHECK-NEXT:    vcmpneqsd %xmm1, %xmm0, %k2
; CHECK-NEXT:    korw %k1, %k2, %k1
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k2
; CHECK-NEXT:    kandw %k2, %k1, %k1
; CHECK-NEXT:    korw %k1, %k0, %k0
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    shlb $7, %al
; CHECK-NEXT:    sarb $7, %al
; CHECK-NEXT:    retq

  %res1 = call i8 @llvm.x86.avx512.mask.cmp.sd(<2 x double> %x0, <2 x double> %x1, i32 2, i8 -1, i32 4)
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.sd(<2 x double> %x0, <2 x double> %x1, i32 3, i8 -1, i32 8)
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.sd(<2 x double> %x0, <2 x double> %x1, i32 4, i8 %x3, i32 4)
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.sd(<2 x double> %x0, <2 x double> %x1, i32 5, i8 %x3, i32 8)

  %res11 = or i8 %res1, %res2
  %res12 = or i8 %res3, %res4
  %res13 = or i8 %res11, %res12
  ret i8 %res13
}

declare i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float>, <4 x float>, i32, i8, i32)

define i8@test_int_x86_avx512_mask_cmp_ss(<4 x float> %x0, <4 x float> %x1, i8 %x3, i32 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_cmp_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vcmpunordss %xmm1, %xmm0, %k0 {%k1}
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    shlb $7, %al
; CHECK-NEXT:    sarb $7, %al
; CHECK-NEXT:    retq

  %res2 = call i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float> %x0, <4 x float> %x1, i32 3, i8 %x3, i32 4)
  ret i8 %res2
}


define i8@test_int_x86_avx512_mask_cmp_ss_all(<4 x float> %x0, <4 x float> %x1, i8 %x3, i32 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_cmp_ss_all:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcmpless %xmm1, %xmm0, %k1
; CHECK-NEXT:    vcmpunordss {sae}, %xmm1, %xmm0, %k0 {%k1}
; CHECK-NEXT:    vcmpneqss %xmm1, %xmm0, %k1
; CHECK-NEXT:    vcmpnltss {sae}, %xmm1, %xmm0, %k1 {%k1}
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k2
; CHECK-NEXT:    kandw %k2, %k1, %k1
; CHECK-NEXT:    kandw %k1, %k0, %k0
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    shlb $7, %al
; CHECK-NEXT:    sarb $7, %al
; CHECK-NEXT:    retq
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float> %x0, <4 x float> %x1, i32 2, i8 -1, i32 4)
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float> %x0, <4 x float> %x1, i32 3, i8 -1, i32 8)
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float> %x0, <4 x float> %x1, i32 4, i8 %x3, i32 4)
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float> %x0, <4 x float> %x1, i32 5, i8 %x3, i32 8)

  %res11 = and i8 %res1, %res2
  %res12 = and i8 %res3, %res4
  %res13 = and i8 %res11, %res12
  ret i8 %res13
}

declare <16 x float> @llvm.x86.avx512.mask.shuf.f32x4(<16 x float>, <16 x float>, i32, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_shuf_f32x4(<16 x float> %x0, <16 x float> %x1, <16 x float> %x3, i16 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_f32x4:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshuff32x4 $22, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm0[8,9,10,11,4,5,6,7],zmm1[4,5,6,7,0,1,2,3]
; CHECK-NEXT:    vshuff32x4 $22, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[8,9,10,11,4,5,6,7],zmm1[4,5,6,7,0,1,2,3]
; CHECK-NEXT:    vaddps %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.shuf.f32x4(<16 x float> %x0, <16 x float> %x1, i32 22, <16 x float> %x3, i16 %x4)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.shuf.f32x4(<16 x float> %x0, <16 x float> %x1, i32 22, <16 x float> %x3, i16 -1)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.shuf.f64x2(<8 x double>, <8 x double>, i32, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_shuf_f64x2(<8 x double> %x0, <8 x double> %x1, <8 x double> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_f64x2:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vshuff64x2 $22, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm0[4,5,2,3],zmm1[2,3,0,1]
; CHECK-NEXT:    vshuff64x2 $22, %zmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    ## zmm3 = zmm0[4,5,2,3],zmm1[2,3,0,1]
; CHECK-NEXT:    vshuff64x2 $22, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[4,5,2,3],zmm1[2,3,0,1]
; CHECK-NEXT:    vaddpd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    vaddpd %zmm3, %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.shuf.f64x2(<8 x double> %x0, <8 x double> %x1, i32 22, <8 x double> %x3, i8 %x4)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.shuf.f64x2(<8 x double> %x0, <8 x double> %x1, i32 22, <8 x double> %x3, i8 -1)
  %res2 = call <8 x double> @llvm.x86.avx512.mask.shuf.f64x2(<8 x double> %x0, <8 x double> %x1, i32 22, <8 x double> zeroinitializer, i8 %x4)

  %res3 = fadd <8 x double> %res, %res1
  %res4 = fadd <8 x double> %res3, %res2
  ret <8 x double> %res4
}

declare <16 x i32> @llvm.x86.avx512.mask.shuf.i32x4(<16 x i32>, <16 x i32>, i32, <16 x i32>, i16)

define <16 x i32>@test_int_x86_avx512_mask_shuf_i32x4(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x3, i16 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_i32x4:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufi32x4 $22, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm0[8,9,10,11,4,5,6,7],zmm1[4,5,6,7,0,1,2,3]
; CHECK-NEXT:    vshufi32x4 $22, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[8,9,10,11,4,5,6,7],zmm1[4,5,6,7,0,1,2,3]
; CHECK-NEXT:    vpaddd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.shuf.i32x4(<16 x i32> %x0, <16 x i32> %x1, i32 22, <16 x i32> %x3, i16 %x4)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.shuf.i32x4(<16 x i32> %x0, <16 x i32> %x1, i32 22, <16 x i32> %x3, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.shuf.i64x2(<8 x i64>, <8 x i64>, i32, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_mask_shuf_i64x2(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_i64x2:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vshufi64x2 $22, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm0[4,5,2,3],zmm1[2,3,0,1]
; CHECK-NEXT:    vshufi64x2 $22, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[4,5,2,3],zmm1[2,3,0,1]
; CHECK-NEXT:    vpaddq %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.shuf.i64x2(<8 x i64> %x0, <8 x i64> %x1, i32 22, <8 x i64> %x3, i8 %x4)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.shuf.i64x2(<8 x i64> %x0, <8 x i64> %x1, i32 22, <8 x i64> %x3, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.getmant.pd.512(<8 x double>, i32, <8 x double>, i8, i32)

define <8 x double>@test_int_x86_avx512_mask_getmant_pd_512(<8 x double> %x0, <8 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vgetmantpd $11, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vgetmantpd $11,{sae}, %zmm0, %zmm0
; CHECK-NEXT:    vaddpd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.getmant.pd.512(<8 x double> %x0, i32 11, <8 x double> %x2, i8 %x3, i32 4)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.getmant.pd.512(<8 x double> %x0, i32 11, <8 x double> %x2, i8 -1, i32 8)
  %res2 = fadd <8 x double> %res, %res1
  ret <8 x double> %res2
}

declare <16 x float> @llvm.x86.avx512.mask.getmant.ps.512(<16 x float>, i32, <16 x float>, i16, i32)

define <16 x float>@test_int_x86_avx512_mask_getmant_ps_512(<16 x float> %x0, <16 x float> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vgetmantps $11, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vgetmantps $11,{sae}, %zmm0, %zmm0
; CHECK-NEXT:    vaddps %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.getmant.ps.512(<16 x float> %x0, i32 11, <16 x float> %x2, i16 %x3, i32 4)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.getmant.ps.512(<16 x float> %x0, i32 11, <16 x float> %x2, i16 -1, i32 8)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.getmant.sd(<2 x double>, <2 x double>, i32, <2 x double>, i8, i32)

define <2 x double>@test_int_x86_avx512_mask_getmant_sd(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm2, %zmm3
; CHECK-NEXT:    vgetmantsd $11, %xmm1, %xmm0, %xmm3 {%k1}
; CHECK-NEXT:    vgetmantsd $11, %xmm1, %xmm0, %xmm4 {%k1} {z}
; CHECK-NEXT:    vgetmantsd $11, %xmm1, %xmm0, %xmm5
; CHECK-NEXT:    vgetmantsd $11,{sae}, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vaddpd %xmm4, %xmm3, %xmm0
; CHECK-NEXT:    vaddpd %xmm5, %xmm2, %xmm1
; CHECK-NEXT:    vaddpd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res  = call <2 x double> @llvm.x86.avx512.mask.getmant.sd(<2 x double> %x0, <2 x double> %x1, i32 11, <2 x double> %x2, i8 %x3, i32 4)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.getmant.sd(<2 x double> %x0, <2 x double> %x1, i32 11, <2 x double> zeroinitializer, i8 %x3, i32 4)
  %res2 = call <2 x double> @llvm.x86.avx512.mask.getmant.sd(<2 x double> %x0, <2 x double> %x1, i32 11, <2 x double> %x2, i8 %x3, i32 8)
  %res3 = call <2 x double> @llvm.x86.avx512.mask.getmant.sd(<2 x double> %x0, <2 x double> %x1, i32 11, <2 x double> %x2, i8 -1, i32 4)
  %res11 = fadd <2 x double> %res, %res1
  %res12 = fadd <2 x double> %res2, %res3
  %res13 = fadd <2 x double> %res11, %res12
  ret <2 x double> %res13
}

declare <4 x float> @llvm.x86.avx512.mask.getmant.ss(<4 x float>, <4 x float>, i32, <4 x float>, i8, i32)

define <4 x float>@test_int_x86_avx512_mask_getmant_ss(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_getmant_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vgetmantss $11, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vgetmantss $11, %xmm1, %xmm0, %xmm3 {%k1} {z}
; CHECK-NEXT:    vgetmantss $11, %xmm1, %xmm0, %xmm4
; CHECK-NEXT:    vgetmantss $11,{sae}, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddps %xmm4, %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %res  = call <4 x float> @llvm.x86.avx512.mask.getmant.ss(<4 x float> %x0, <4 x float> %x1, i32 11, <4 x float> %x2, i8 %x3, i32 4)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.getmant.ss(<4 x float> %x0, <4 x float> %x1, i32 11, <4 x float> zeroinitializer, i8 %x3, i32 4)
  %res2 = call <4 x float> @llvm.x86.avx512.mask.getmant.ss(<4 x float> %x0, <4 x float> %x1, i32 11, <4 x float> %x2, i8 -1, i32 8)
  %res3 = call <4 x float> @llvm.x86.avx512.mask.getmant.ss(<4 x float> %x0, <4 x float> %x1, i32 11, <4 x float> %x2, i8 -1, i32 4)
  %res11 = fadd <4 x float> %res, %res1
  %res12 = fadd <4 x float> %res2, %res3
  %res13 = fadd <4 x float> %res11, %res12
  ret <4 x float> %res13
}

declare <8 x double> @llvm.x86.avx512.mask.shuf.pd.512(<8 x double>, <8 x double>, i32, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_shuf_pd_512(<8 x double> %x0, <8 x double> %x1, <8 x double> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vshufpd $22, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[0],k1[1],zmm2[3],k1[2],zmm2[5],k1[4],zmm2[6],k1[6]
; CHECK-NEXT:    vshufpd $22, %zmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    ## zmm3 = k1[0],zmm0[1],k1[3],zmm0[2],k1[5],zmm0[4],k1[6],zmm0[6]
; CHECK-NEXT:    vshufpd $22, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0],zmm1[1],zmm0[3],zmm1[2],zmm0[5],zmm1[4],zmm0[6],zmm1[6]
; CHECK-NEXT:    vaddpd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    vaddpd %zmm3, %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.shuf.pd.512(<8 x double> %x0, <8 x double> %x1, i32 22, <8 x double> %x3, i8 %x4)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.shuf.pd.512(<8 x double> %x0, <8 x double> %x1, i32 22, <8 x double> %x3, i8 -1)
  %res2 = call <8 x double> @llvm.x86.avx512.mask.shuf.pd.512(<8 x double> %x0, <8 x double> %x1, i32 22, <8 x double> zeroinitializer, i8 %x4)

  %res3 = fadd <8 x double> %res, %res1
  %res4 = fadd <8 x double> %res3, %res2
  ret <8 x double> %res4
}

declare <16 x float> @llvm.x86.avx512.mask.shuf.ps.512(<16 x float>, <16 x float>, i32, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_shuf_ps_512(<16 x float> %x0, <16 x float> %x1, <16 x float> %x3, i16 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_shuf_ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vshufps $22, %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    ## zmm2 = zmm2[2,1],k1[1,0],zmm2[6,5],k1[5,4],zmm2[10,9],k1[9,8],zmm2[14,13],k1[13,12]
; CHECK-NEXT:    vshufps $22, %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[2,1],zmm1[1,0],zmm0[6,5],zmm1[5,4],zmm0[10,9],zmm1[9,8],zmm0[14,13],zmm1[13,12]
; CHECK-NEXT:    vaddps %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.shuf.ps.512(<16 x float> %x0, <16 x float> %x1, i32 22, <16 x float> %x3, i16 %x4)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.shuf.ps.512(<16 x float> %x0, <16 x float> %x1, i32 22, <16 x float> %x3, i16 -1)
  %res2 = fadd <16 x float> %res, %res1
  ret <16 x float> %res2
}

declare <8 x double> @llvm.x86.avx512.mask.vpermil.pd.512(<8 x double>, i32, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_vpermil_pd_512(<8 x double> %x0, <8 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermil_pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpermilpd $22, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    ## zmm1 = zmm1[0,1,3,2,5,4,6,6]
; CHECK-NEXT:    vpermilpd $22, %zmm0, %zmm2 {%k1} {z}
; CHECK-NEXT:    ## zmm2 = k1[0,1,3,2,5,4,6,6]
; CHECK-NEXT:    vpermilpd $22, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0,1,3,2,5,4,6,6]
; CHECK-NEXT:    vaddpd %zmm2, %zmm1, %zmm1
; CHECK-NEXT:    vaddpd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.vpermil.pd.512(<8 x double> %x0, i32 22, <8 x double> %x2, i8 %x3)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.vpermil.pd.512(<8 x double> %x0, i32 22, <8 x double> zeroinitializer, i8 %x3)
  %res2 = call <8 x double> @llvm.x86.avx512.mask.vpermil.pd.512(<8 x double> %x0, i32 22, <8 x double> %x2, i8 -1)
  %res3 = fadd <8 x double> %res, %res1
  %res4 = fadd <8 x double> %res3, %res2
  ret <8 x double> %res4
}

declare <16 x float> @llvm.x86.avx512.mask.vpermil.ps.512(<16 x float>, i32, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_vpermil_ps_512(<16 x float> %x0, <16 x float> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermil_ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilps $22, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    ## zmm1 = zmm1[2,1,1,0,6,5,5,4,10,9,9,8,14,13,13,12]
; CHECK-NEXT:    vpermilps $22, %zmm0, %zmm2 {%k1} {z}
; CHECK-NEXT:    ## zmm2 = k1[2,1,1,0,6,5,5,4,10,9,9,8,14,13,13,12]
; CHECK-NEXT:    vpermilps $22, %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[2,1,1,0,6,5,5,4,10,9,9,8,14,13,13,12]
; CHECK-NEXT:    vaddps %zmm2, %zmm1, %zmm1
; CHECK-NEXT:    vaddps %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermil.ps.512(<16 x float> %x0, i32 22, <16 x float> %x2, i16 %x3)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.vpermil.ps.512(<16 x float> %x0, i32 22, <16 x float> zeroinitializer, i16 %x3)
  %res2 = call <16 x float> @llvm.x86.avx512.mask.vpermil.ps.512(<16 x float> %x0, i32 22, <16 x float> %x2, i16 -1)
  %res3 = fadd <16 x float> %res, %res1
  %res4 = fadd <16 x float> %res3, %res2
  ret <16 x float> %res4
}

declare <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double>, <8 x i64>, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_vpermilvar_pd_512(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermilvar_pd_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vpermilpd %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vpermilpd %zmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vpermilpd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vaddpd %zmm3, %zmm2, %zmm1
; CHECK-NEXT:    vaddpd %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2, i8 %x3)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double> %x0, <8 x i64> %x1, <8 x double> zeroinitializer, i8 %x3)
  %res2 = call <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2, i8 -1)
  %res3 = fadd <8 x double> %res, %res1
  %res4 = fadd <8 x double> %res2, %res3
  ret <8 x double> %res4
}

declare <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float>, <16 x i32>, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_vpermilvar_ps_512(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2, i16 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_vpermilvar_ps_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vpermilps %zmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vpermilps %zmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vpermilps %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    vaddps %zmm3, %zmm2, %zmm1
; CHECK-NEXT:    vaddps %zmm1, %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2, i16 %x3)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float> %x0, <16 x i32> %x1, <16 x float> zeroinitializer, i16 %x3)
  %res2 = call <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2, i16 -1)
  %res3 = fadd <16 x float> %res, %res1
  %res4 = fadd <16 x float> %res2, %res3
  ret <16 x float> %res4
}

declare <16 x float> @llvm.x86.avx512.mask.insertf32x4.512(<16 x float>, <4 x float>, i32, <16 x float>, i8)

define <16 x float>@test_int_x86_avx512_mask_insertf32x4_512(<16 x float> %x0, <4 x float> %x1, <16 x float> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_insertf32x4_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vinsertf32x4 $1, %xmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vinsertf32x4 $1, %xmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vinsertf32x4 $1, %xmm1, %zmm0, %zmm0
; CHECK-NEXT:    vaddps %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    vaddps %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x float> @llvm.x86.avx512.mask.insertf32x4.512(<16 x float> %x0, <4 x float> %x1, i32 1, <16 x float> %x3, i8 %x4)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.insertf32x4.512(<16 x float> %x0, <4 x float> %x1, i32 1, <16 x float> %x3, i8 -1)
  %res2 = call <16 x float> @llvm.x86.avx512.mask.insertf32x4.512(<16 x float> %x0, <4 x float> %x1, i32 1, <16 x float> zeroinitializer, i8 %x4)
  %res3 = fadd <16 x float> %res, %res1
  %res4 = fadd <16 x float> %res2, %res3
  ret <16 x float> %res4
}

declare <16 x i32> @llvm.x86.avx512.mask.inserti32x4.512(<16 x i32>, <4 x i32>, i32, <16 x i32>, i8)

define <16 x i32>@test_int_x86_avx512_mask_inserti32x4_512(<16 x i32> %x0, <4 x i32> %x1, <16 x i32> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_inserti32x4_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vinserti32x4 $1, %xmm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vinserti32x4 $1, %xmm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vinserti32x4 $1, %xmm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.inserti32x4.512(<16 x i32> %x0, <4 x i32> %x1, i32 1, <16 x i32> %x3, i8 %x4)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.inserti32x4.512(<16 x i32> %x0, <4 x i32> %x1, i32 1, <16 x i32> %x3, i8 -1)
  %res2 = call <16 x i32> @llvm.x86.avx512.mask.inserti32x4.512(<16 x i32> %x0, <4 x i32> %x1, i32 1, <16 x i32> zeroinitializer, i8 %x4)
  %res3 = add <16 x i32> %res, %res1
  %res4 = add <16 x i32> %res2, %res3
  ret <16 x i32> %res4
}

declare <8 x double> @llvm.x86.avx512.mask.insertf64x4.512(<8 x double>, <4 x double>, i32, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_insertf64x4_512(<8 x double> %x0, <4 x double> %x1, <8 x double> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_insertf64x4_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; CHECK-NEXT:    vaddpd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    vaddpd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.insertf64x4.512(<8 x double> %x0, <4 x double> %x1, i32 1, <8 x double> %x3, i8 %x4)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.insertf64x4.512(<8 x double> %x0, <4 x double> %x1, i32 1, <8 x double> %x3, i8 -1)
  %res2 = call <8 x double> @llvm.x86.avx512.mask.insertf64x4.512(<8 x double> %x0, <4 x double> %x1, i32 1, <8 x double> zeroinitializer, i8 %x4)
  %res3 = fadd <8 x double> %res, %res1
  %res4 = fadd <8 x double> %res2, %res3
  ret <8 x double> %res4
}

declare <8 x i64> @llvm.x86.avx512.mask.inserti64x4.512(<8 x i64>, <4 x i64>, i32, <8 x i64>, i8)

define <8 x i64>@test_int_x86_avx512_mask_inserti64x4_512(<8 x i64> %x0, <4 x i64> %x1, <8 x i64> %x3, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_inserti64x4_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm3 {%k1} {z}
; CHECK-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; CHECK-NEXT:    vpaddq %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    vpaddq %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.inserti64x4.512(<8 x i64> %x0, <4 x i64> %x1, i32 1, <8 x i64> %x3, i8 %x4)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.inserti64x4.512(<8 x i64> %x0, <4 x i64> %x1, i32 1, <8 x i64> %x3, i8 -1)
  %res2 = call <8 x i64> @llvm.x86.avx512.mask.inserti64x4.512(<8 x i64> %x0, <4 x i64> %x1, i32 1, <8 x i64> zeroinitializer, i8 %x4)
  %res3 = add <8 x i64> %res, %res1
  %res4 = add <8 x i64> %res2, %res3
  ret <8 x i64> %res4
}

declare <2 x double> @llvm.x86.avx512.mask.cvtss2sd.round(<4 x float>, <4 x float>, <2 x double>, i8, i32)

define <2 x double>@test_int_x86_avx512_mask_cvt_ss2sd_round(<4 x float> %x0,<4 x float> %x1, <2 x double> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_ss2sd_round:
; CHECK:    kmovw %edi, %k1    
; CHECK-NEXT:    vcvtss2sd %xmm1, %xmm0, %xmm2 {%k1} 
; CHECK-NEXT:    vcvtss2sd {sae}, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <2 x double> @llvm.x86.avx512.mask.cvtss2sd.round(<4 x float> %x0, <4 x float> %x1, <2 x double> %x2, i8 %x3, i32 4)
  %res1 = call <2 x double> @llvm.x86.avx512.mask.cvtss2sd.round(<4 x float> %x0, <4 x float> %x1, <2 x double> %x2, i8 -1, i32 8)
  %res2 = fadd <2 x double> %res, %res1
  ret <2 x double> %res2
}

declare <4 x float> @llvm.x86.avx512.mask.cvtsd2ss.round(<2 x double>, <2 x double>, <4 x float>, i8, i32)

define <4 x float>@test_int_x86_avx512_mask_cvt_sd2ss_round(<2 x double> %x0,<2 x double> %x1, <4 x float> %x2, i8 %x3) {
; CHECK-LABEL: test_int_x86_avx512_mask_cvt_sd2ss_round:
; CHECK:    kmovw %edi, %k1
; CHECK-NEXT:    vcvtsd2ss {rz-sae}, %xmm1, %xmm0, %xmm2 {%k1}
; CHECK-NEXT:    vcvtsd2ss {rn-sae}, %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddps %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %res = call <4 x float> @llvm.x86.avx512.mask.cvtsd2ss.round(<2 x double> %x0, <2 x double> %x1, <4 x float> %x2, i8 %x3, i32 3)
  %res1 = call <4 x float> @llvm.x86.avx512.mask.cvtsd2ss.round(<2 x double> %x0, <2 x double> %x1, <4 x float> %x2, i8 -1, i32 8)
  %res2 = fadd <4 x float> %res, %res1
  ret <4 x float> %res2
}

declare <16 x i32> @llvm.x86.avx512.mask.pternlog.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i32, i16)

define <16 x i32>@test_int_x86_avx512_mask_pternlog_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_pternlog_d_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogd $33, %zmm2, %zmm1, %zmm3 {%k1}
; CHECK-NEXT:    vpternlogd $33, %zmm2, %zmm1, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.mask.pternlog.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i32 33, i16 %x4)
  %res1 = call <16 x i32> @llvm.x86.avx512.mask.pternlog.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i32 33, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <16 x i32> @llvm.x86.avx512.maskz.pternlog.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i32, i16)

define <16 x i32>@test_int_x86_avx512_maskz_pternlog_d_512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i16 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_pternlog_d_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogd $33, %zmm2, %zmm1, %zmm3 {%k1} {z}
; CHECK-NEXT:    vpternlogd $33, %zmm2, %zmm1, %zmm0
; CHECK-NEXT:    vpaddd %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <16 x i32> @llvm.x86.avx512.maskz.pternlog.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i32 33, i16 %x4)
  %res1 = call <16 x i32> @llvm.x86.avx512.maskz.pternlog.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i32 33, i16 -1)
  %res2 = add <16 x i32> %res, %res1
  ret <16 x i32> %res2
}

declare <8 x i64> @llvm.x86.avx512.mask.pternlog.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i32, i8)

define <8 x i64>@test_int_x86_avx512_mask_pternlog_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_mask_pternlog_q_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogq $33, %zmm2, %zmm1, %zmm3 {%k1}
; CHECK-NEXT:    vpternlogq $33, %zmm2, %zmm1, %zmm0
; CHECK-NEXT:    vpaddq %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.mask.pternlog.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i32 33, i8 %x4)
  %res1 = call <8 x i64> @llvm.x86.avx512.mask.pternlog.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i32 33, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <8 x i64> @llvm.x86.avx512.maskz.pternlog.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i32, i8)

define <8 x i64>@test_int_x86_avx512_maskz_pternlog_q_512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i8 %x4) {
; CHECK-LABEL: test_int_x86_avx512_maskz_pternlog_q_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm3
; CHECK-NEXT:    vpternlogq $33, %zmm2, %zmm1, %zmm3 {%k1} {z}
; CHECK-NEXT:    vpternlogq $33, %zmm2, %zmm1, %zmm0
; CHECK-NEXT:    vpaddq %zmm0, %zmm3, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x i64> @llvm.x86.avx512.maskz.pternlog.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i32 33, i8 %x4)
  %res1 = call <8 x i64> @llvm.x86.avx512.maskz.pternlog.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i32 33, i8 -1)
  %res2 = add <8 x i64> %res, %res1
  ret <8 x i64> %res2
}

declare <16 x float> @llvm.x86.avx512.mask.movsldup.512(<16 x float>, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_movsldup_512(<16 x float> %x0, <16 x float> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movsldup_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vmovsldup %zmm0, %zmm1 {%k1} 
; CHECK-NEXT:    ## zmm1 = zmm0[0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14]
; CHECK-NEXT:    vmovsldup %zmm0, %zmm2 {%k1} {z} 
; CHECK-NEXT:    ## zmm2 = zmm0[0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14]
; CHECK-NEXT:    vmovsldup %zmm0, %zmm0 
; CHECK-NEXT:    ## zmm0 = zmm0[0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14]
; CHECK-NEXT:    vaddps %zmm0, %zmm1, %zmm0 
; CHECK-NEXT:    vaddps %zmm0, %zmm2, %zmm0 
; CHECK-NEXT:    retq 
  %res = call <16 x float> @llvm.x86.avx512.mask.movsldup.512(<16 x float> %x0, <16 x float> %x1, i16 %x2)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.movsldup.512(<16 x float> %x0, <16 x float> %x1, i16 -1)
  %res2 = call <16 x float> @llvm.x86.avx512.mask.movsldup.512(<16 x float> %x0, <16 x float> zeroinitializer, i16 %x2)
  %res3 = fadd <16 x float> %res, %res1
  %res4 = fadd <16 x float> %res2, %res3
  ret <16 x float> %res4
}

declare <16 x float> @llvm.x86.avx512.mask.movshdup.512(<16 x float>, <16 x float>, i16)

define <16 x float>@test_int_x86_avx512_mask_movshdup_512(<16 x float> %x0, <16 x float> %x1, i16 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movshdup_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %edi, %k1 
; CHECK-NEXT:    vmovshdup %zmm0, %zmm1 {%k1} 
; CHECK-NEXT:    ## zmm1 = zmm0[1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15]
; CHECK-NEXT:    vmovshdup %zmm0, %zmm2 {%k1} {z} 
; CHECK-NEXT:    ## zmm2 = zmm0[1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15]
; CHECK-NEXT:    vmovshdup %zmm0, %zmm0 
; CHECK-NEXT:    ## zmm0 = zmm0[1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15]
; CHECK-NEXT:    vaddps %zmm0, %zmm1, %zmm0 
; CHECK-NEXT:    vaddps %zmm0, %zmm2, %zmm0 
; CHECK-NEXT:    retq 
  %res = call <16 x float> @llvm.x86.avx512.mask.movshdup.512(<16 x float> %x0, <16 x float> %x1, i16 %x2)
  %res1 = call <16 x float> @llvm.x86.avx512.mask.movshdup.512(<16 x float> %x0, <16 x float> %x1, i16 -1)
  %res2 = call <16 x float> @llvm.x86.avx512.mask.movshdup.512(<16 x float> %x0, <16 x float> zeroinitializer, i16 %x2)
  %res3 = fadd <16 x float> %res, %res1
  %res4 = fadd <16 x float> %res2, %res3
  ret <16 x float> %res4
}

declare <8 x double> @llvm.x86.avx512.mask.movddup.512(<8 x double>, <8 x double>, i8)

define <8 x double>@test_int_x86_avx512_mask_movddup_512(<8 x double> %x0, <8 x double> %x1, i8 %x2) {
; CHECK-LABEL: test_int_x86_avx512_mask_movddup_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    kmovw %eax, %k1
; CHECK-NEXT:    vmovddup %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    ## zmm1 = zmm0[0,0,2,2,4,4,6,6]
; CHECK-NEXT:    vmovddup %zmm0, %zmm2 {%k1} {z}
; CHECK-NEXT:    ## zmm2 = zmm0[0,0,2,2,4,4,6,6]
; CHECK-NEXT:    vmovddup %zmm0, %zmm0
; CHECK-NEXT:    ## zmm0 = zmm0[0,0,2,2,4,4,6,6]
; CHECK-NEXT:    vaddpd %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    vaddpd %zmm0, %zmm2, %zmm0
; CHECK-NEXT:    retq
  %res = call <8 x double> @llvm.x86.avx512.mask.movddup.512(<8 x double> %x0, <8 x double> %x1, i8 %x2)
  %res1 = call <8 x double> @llvm.x86.avx512.mask.movddup.512(<8 x double> %x0, <8 x double> %x1, i8 -1)
  %res2 = call <8 x double> @llvm.x86.avx512.mask.movddup.512(<8 x double> %x0, <8 x double> zeroinitializer, i8 %x2)
  %res3 = fadd <8 x double> %res, %res1
  %res4 = fadd <8 x double> %res2, %res3
  ret <8 x double> %res4
}

