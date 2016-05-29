; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mattr=avx2 | FileCheck %s

define <16 x i16> @test_x86_avx2_pblendw(<16 x i16> %a0, <16 x i16> %a1) {
  ; CHECK: vpblendw
  %res = call <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16> %a0, <16 x i16> %a1, i32 7) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16>, <16 x i16>, i32) nounwind readnone


define <4 x i32> @test_x86_avx2_pblendd_128(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpblendd
  %res = call <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32> %a0, <4 x i32> %a1, i32 7) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32>, <4 x i32>, i32) nounwind readnone


define <8 x i32> @test_x86_avx2_pblendd_256(<8 x i32> %a0, <8 x i32> %a1) {
  ; CHECK: vpblendd
  %res = call <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32> %a0, <8 x i32> %a1, i32 7) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32>, <8 x i32>, i32) nounwind readnone


define <16 x i16> @test_x86_avx2_mpsadbw(<32 x i8> %a0, <32 x i8> %a1) {
  ; CHECK: vmpsadbw
  %res = call <16 x i16> @llvm.x86.avx2.mpsadbw(<32 x i8> %a0, <32 x i8> %a1, i32 7) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.mpsadbw(<32 x i8>, <32 x i8>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psll_dq_bs(<4 x i64> %a0) {
  ; CHECK: vpslldq {{.*#+}} ymm0 = zero,zero,zero,zero,zero,zero,zero,ymm0[0,1,2,3,4,5,6,7,8],zero,zero,zero,zero,zero,zero,zero,ymm0[16,17,18,19,20,21,22,23,24]
  %res = call <4 x i64> @llvm.x86.avx2.psll.dq.bs(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psll.dq.bs(<4 x i64>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psrl_dq_bs(<4 x i64> %a0) {
  ; CHECK: vpsrldq {{.*#+}} ymm0 = ymm0[7,8,9,10,11,12,13,14,15],zero,zero,zero,zero,zero,zero,zero,ymm0[23,24,25,26,27,28,29,30,31],zero,zero,zero,zero,zero,zero,zero
  %res = call <4 x i64> @llvm.x86.avx2.psrl.dq.bs(<4 x i64> %a0, i32 7) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psrl.dq.bs(<4 x i64>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psll_dq(<4 x i64> %a0) {
  ; CHECK: vpslldq {{.*#+}} ymm0 = zero,ymm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],zero,ymm0[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
  %res = call <4 x i64> @llvm.x86.avx2.psll.dq(<4 x i64> %a0, i32 8) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psll.dq(<4 x i64>, i32) nounwind readnone


define <4 x i64> @test_x86_avx2_psrl_dq(<4 x i64> %a0) {
  ; CHECK: vpsrldq {{.*#+}} ymm0 = ymm0[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],zero,ymm0[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],zero
  %res = call <4 x i64> @llvm.x86.avx2.psrl.dq(<4 x i64> %a0, i32 8) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.psrl.dq(<4 x i64>, i32) nounwind readnone


define <2 x i64> @test_x86_avx2_vextracti128(<4 x i64> %a0) {
; CHECK-LABEL:    test_x86_avx2_vextracti128:
; CHECK:          vextracti128

  %res = call <2 x i64> @llvm.x86.avx2.vextracti128(<4 x i64> %a0, i8 7)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx2.vextracti128(<4 x i64>, i8) nounwind readnone


define <4 x i64> @test_x86_avx2_vinserti128(<4 x i64> %a0, <2 x i64> %a1) {
; CHECK-LABEL:    test_x86_avx2_vinserti128:
; CHECK:          vinserti128

  %res = call <4 x i64> @llvm.x86.avx2.vinserti128(<4 x i64> %a0, <2 x i64> %a1, i8 7)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.vinserti128(<4 x i64>, <2 x i64>, i8) nounwind readnone


define <4 x double> @test_x86_avx2_vbroadcast_sd_pd_256(<2 x double> %a0) {
  ; CHECK-LABEL: test_x86_avx2_vbroadcast_sd_pd_256:
  ; CHECK:       ## BB#0:
  ; CHECK-NEXT:    vbroadcastsd %xmm0, %ymm0
  ; CHECK-NEXT:    retl
  %res = call <4 x double> @llvm.x86.avx2.vbroadcast.sd.pd.256(<2 x double> %a0)
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx2.vbroadcast.sd.pd.256(<2 x double>) nounwind readonly


define <4 x float> @test_x86_avx2_vbroadcast_ss_ps(<4 x float> %a0) {
  ; CHECK-LABEL: test_x86_avx2_vbroadcast_ss_ps:
  ; CHECK:       ## BB#0:
  ; CHECK-NEXT:    vbroadcastss %xmm0, %xmm0
  ; CHECK-NEXT:    retl
  %res = call <4 x float> @llvm.x86.avx2.vbroadcast.ss.ps(<4 x float> %a0)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx2.vbroadcast.ss.ps(<4 x float>) nounwind readonly


define <8 x float> @test_x86_avx2_vbroadcast_ss_ps_256(<4 x float> %a0) {
  ; CHECK-LABEL: test_x86_avx2_vbroadcast_ss_ps_256:
  ; CHECK:       ## BB#0:
  ; CHECK-NEXT:    vbroadcastss %xmm0, %ymm0
  ; CHECK-NEXT:    retl
  %res = call <8 x float> @llvm.x86.avx2.vbroadcast.ss.ps.256(<4 x float> %a0)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx2.vbroadcast.ss.ps.256(<4 x float>) nounwind readonly


define <16 x i8> @test_x86_avx2_pbroadcastb_128(<16 x i8> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastb_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpbroadcastb %xmm0, %xmm0
; CHECK-NEXT:    retl
  %res = call <16 x i8> @llvm.x86.avx2.pbroadcastb.128(<16 x i8> %a0)
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.avx2.pbroadcastb.128(<16 x i8>) nounwind readonly


define <32 x i8> @test_x86_avx2_pbroadcastb_256(<16 x i8> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastb_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpbroadcastb %xmm0, %ymm0
; CHECK-NEXT:    retl
  %res = call <32 x i8> @llvm.x86.avx2.pbroadcastb.256(<16 x i8> %a0)
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx2.pbroadcastb.256(<16 x i8>) nounwind readonly


define <8 x i16> @test_x86_avx2_pbroadcastw_128(<8 x i16> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastw_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpbroadcastw %xmm0, %xmm0
; CHECK-NEXT:    retl
  %res = call <8 x i16> @llvm.x86.avx2.pbroadcastw.128(<8 x i16> %a0)
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.avx2.pbroadcastw.128(<8 x i16>) nounwind readonly


define <16 x i16> @test_x86_avx2_pbroadcastw_256(<8 x i16> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastw_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpbroadcastw %xmm0, %ymm0
; CHECK-NEXT:    retl
  %res = call <16 x i16> @llvm.x86.avx2.pbroadcastw.256(<8 x i16> %a0)
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pbroadcastw.256(<8 x i16>) nounwind readonly


define <4 x i32> @test_x86_avx2_pbroadcastd_128(<4 x i32> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastd_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vbroadcastss %xmm0, %xmm0
; CHECK-NEXT:    retl
  %res = call <4 x i32> @llvm.x86.avx2.pbroadcastd.128(<4 x i32> %a0)
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx2.pbroadcastd.128(<4 x i32>) nounwind readonly


define <8 x i32> @test_x86_avx2_pbroadcastd_256(<4 x i32> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastd_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vbroadcastss %xmm0, %ymm0
; CHECK-NEXT:    retl
  %res = call <8 x i32> @llvm.x86.avx2.pbroadcastd.256(<4 x i32> %a0)
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pbroadcastd.256(<4 x i32>) nounwind readonly


define <2 x i64> @test_x86_avx2_pbroadcastq_128(<2 x i64> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastq_128:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpbroadcastq %xmm0, %xmm0
; CHECK-NEXT:    retl
  %res = call <2 x i64> @llvm.x86.avx2.pbroadcastq.128(<2 x i64> %a0)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx2.pbroadcastq.128(<2 x i64>) nounwind readonly


define <4 x i64> @test_x86_avx2_pbroadcastq_256(<2 x i64> %a0) {
; CHECK-LABEL: test_x86_avx2_pbroadcastq_256:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vbroadcastsd %xmm0, %ymm0
; CHECK-NEXT:    retl
  %res = call <4 x i64> @llvm.x86.avx2.pbroadcastq.256(<2 x i64> %a0)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pbroadcastq.256(<2 x i64>) nounwind readonly


define <8 x i32> @test_x86_avx2_pmovsxbd(<16 x i8> %a0) {
; CHECK: vpmovsxbd
  %res = call <8 x i32> @llvm.x86.avx2.pmovsxbd(<16 x i8> %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pmovsxbd(<16 x i8>) nounwind readnone


define <4 x i64> @test_x86_avx2_pmovsxbq(<16 x i8> %a0) {
; CHECK: vpmovsxbq
  %res = call <4 x i64> @llvm.x86.avx2.pmovsxbq(<16 x i8> %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pmovsxbq(<16 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_pmovsxbw(<16 x i8> %a0) {
; CHECK: vpmovsxbw
  %res = call <16 x i16> @llvm.x86.avx2.pmovsxbw(<16 x i8> %a0) ; <<8 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pmovsxbw(<16 x i8>) nounwind readnone


define <4 x i64> @test_x86_avx2_pmovsxdq(<4 x i32> %a0) {
; CHECK: vpmovsxdq
  %res = call <4 x i64> @llvm.x86.avx2.pmovsxdq(<4 x i32> %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pmovsxdq(<4 x i32>) nounwind readnone


define <8 x i32> @test_x86_avx2_pmovsxwd(<8 x i16> %a0) {
; CHECK: vpmovsxwd
  %res = call <8 x i32> @llvm.x86.avx2.pmovsxwd(<8 x i16> %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pmovsxwd(<8 x i16>) nounwind readnone


define <4 x i64> @test_x86_avx2_pmovsxwq(<8 x i16> %a0) {
; CHECK: vpmovsxwq
  %res = call <4 x i64> @llvm.x86.avx2.pmovsxwq(<8 x i16> %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pmovsxwq(<8 x i16>) nounwind readnone


define <8 x i32> @test_x86_avx2_pmovzxbd(<16 x i8> %a0) {
; CHECK: vpmovzxbd
  %res = call <8 x i32> @llvm.x86.avx2.pmovzxbd(<16 x i8> %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pmovzxbd(<16 x i8>) nounwind readnone


define <4 x i64> @test_x86_avx2_pmovzxbq(<16 x i8> %a0) {
; CHECK: vpmovzxbq
  %res = call <4 x i64> @llvm.x86.avx2.pmovzxbq(<16 x i8> %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pmovzxbq(<16 x i8>) nounwind readnone


define <16 x i16> @test_x86_avx2_pmovzxbw(<16 x i8> %a0) {
; CHECK: vpmovzxbw
  %res = call <16 x i16> @llvm.x86.avx2.pmovzxbw(<16 x i8> %a0) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx2.pmovzxbw(<16 x i8>) nounwind readnone


define <4 x i64> @test_x86_avx2_pmovzxdq(<4 x i32> %a0) {
; CHECK: vpmovzxdq
  %res = call <4 x i64> @llvm.x86.avx2.pmovzxdq(<4 x i32> %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pmovzxdq(<4 x i32>) nounwind readnone


define <8 x i32> @test_x86_avx2_pmovzxwd(<8 x i16> %a0) {
; CHECK: vpmovzxwd
  %res = call <8 x i32> @llvm.x86.avx2.pmovzxwd(<8 x i16> %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx2.pmovzxwd(<8 x i16>) nounwind readnone


define <4 x i64> @test_x86_avx2_pmovzxwq(<8 x i16> %a0) {
; CHECK: vpmovzxwq
  %res = call <4 x i64> @llvm.x86.avx2.pmovzxwq(<8 x i16> %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx2.pmovzxwq(<8 x i16>) nounwind readnone
