; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=corei7-avx | FileCheck %s

define <2 x i64> @test_x86_aesni_aesdec(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vaesdec
  %res = call <2 x i64> @llvm.x86.aesni.aesdec(<2 x i64> %a0, <2 x i64> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesdec(<2 x i64>, <2 x i64>) nounwind readnone


define <2 x i64> @test_x86_aesni_aesdeclast(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vaesdeclast
  %res = call <2 x i64> @llvm.x86.aesni.aesdeclast(<2 x i64> %a0, <2 x i64> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesdeclast(<2 x i64>, <2 x i64>) nounwind readnone


define <2 x i64> @test_x86_aesni_aesenc(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vaesenc
  %res = call <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64> %a0, <2 x i64> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64>, <2 x i64>) nounwind readnone


define <2 x i64> @test_x86_aesni_aesenclast(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vaesenclast
  %res = call <2 x i64> @llvm.x86.aesni.aesenclast(<2 x i64> %a0, <2 x i64> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesenclast(<2 x i64>, <2 x i64>) nounwind readnone


define <2 x i64> @test_x86_aesni_aesimc(<2 x i64> %a0) {
  ; CHECK: vaesimc
  %res = call <2 x i64> @llvm.x86.aesni.aesimc(<2 x i64> %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesimc(<2 x i64>) nounwind readnone


define <2 x i64> @test_x86_aesni_aeskeygenassist(<2 x i64> %a0) {
  ; CHECK: vaeskeygenassist
  %res = call <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64> %a0, i8 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64>, i8) nounwind readnone


define <2 x double> @test_x86_sse2_add_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vaddsd
  %res = call <2 x double> @llvm.x86.sse2.add.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.add.sd(<2 x double>, <2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_cmp_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcmpordpd
  %res = call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %a0, <2 x double> %a1, i8 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double>, <2 x double>, i8) nounwind readnone


define <2 x double> @test_x86_sse2_cmp_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcmpordsd
  %res = call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %a0, <2 x double> %a1, i8 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8) nounwind readnone


define i32 @test_x86_sse2_comieq_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcomisd
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.comieq.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.comieq.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_comige_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcomisd
  ; CHECK: setae
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.comige.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.comige.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_comigt_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcomisd
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.comigt.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.comigt.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_comile_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcomisd
  ; CHECK: setbe
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.comile.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.comile.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_comilt_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcomisd
  ; CHECK: sbbl    %eax, %eax
  ; CHECK: andl    $1, %eax
  %res = call i32 @llvm.x86.sse2.comilt.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.comilt.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_comineq_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vcomisd
  ; CHECK: setne
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.comineq.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.comineq.sd(<2 x double>, <2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_cvtdq2pd(<4 x i32> %a0) {
  ; CHECK: vcvtdq2pd
  %res = call <2 x double> @llvm.x86.sse2.cvtdq2pd(<4 x i32> %a0) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtdq2pd(<4 x i32>) nounwind readnone


define <4 x float> @test_x86_sse2_cvtdq2ps(<4 x i32> %a0) {
  ; CHECK: vcvtdq2ps
  %res = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone


define <4 x i32> @test_x86_sse2_cvtpd2dq(<2 x double> %a0) {
  ; CHECK: vcvtpd2dq
  %res = call <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double>) nounwind readnone


define <4 x float> @test_x86_sse2_cvtpd2ps(<2 x double> %a0) {
  ; CHECK: vcvtpd2ps
  %res = call <4 x float> @llvm.x86.sse2.cvtpd2ps(<2 x double> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse2.cvtpd2ps(<2 x double>) nounwind readnone


define <4 x i32> @test_x86_sse2_cvtps2dq(<4 x float> %a0) {
  ; CHECK: vcvtps2dq
  %res = call <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float>) nounwind readnone


define <2 x double> @test_x86_sse2_cvtps2pd(<4 x float> %a0) {
  ; CHECK: vcvtps2pd
  %res = call <2 x double> @llvm.x86.sse2.cvtps2pd(<4 x float> %a0) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtps2pd(<4 x float>) nounwind readnone


define i32 @test_x86_sse2_cvtsd2si(<2 x double> %a0) {
  ; CHECK: vcvtsd2si
  %res = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone


define <4 x float> @test_x86_sse2_cvtsd2ss(<4 x float> %a0, <2 x double> %a1) {
  ; CHECK: vcvtsd2ss
  %res = call <4 x float> @llvm.x86.sse2.cvtsd2ss(<4 x float> %a0, <2 x double> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse2.cvtsd2ss(<4 x float>, <2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_cvtsi2sd(<2 x double> %a0) {
  ; CHECK: movl
  ; CHECK: vcvtsi2sd
  %res = call <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double> %a0, i32 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double>, i32) nounwind readnone


define <2 x double> @test_x86_sse2_cvtss2sd(<2 x double> %a0, <4 x float> %a1) {
  ; CHECK: vcvtss2sd
  %res = call <2 x double> @llvm.x86.sse2.cvtss2sd(<2 x double> %a0, <4 x float> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtss2sd(<2 x double>, <4 x float>) nounwind readnone


define <4 x i32> @test_x86_sse2_cvttpd2dq(<2 x double> %a0) {
  ; CHECK: vcvttpd2dq
  %res = call <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double>) nounwind readnone


define <4 x i32> @test_x86_sse2_cvttps2dq(<4 x float> %a0) {
  ; CHECK: vcvttps2dq
  %res = call <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float>) nounwind readnone


define i32 @test_x86_sse2_cvttsd2si(<2 x double> %a0) {
  ; CHECK: vcvttsd2si
  %res = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_div_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vdivsd
  %res = call <2 x double> @llvm.x86.sse2.div.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.div.sd(<2 x double>, <2 x double>) nounwind readnone



define <2 x double> @test_x86_sse2_max_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vmaxpd
  %res = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_max_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vmaxsd
  %res = call <2 x double> @llvm.x86.sse2.max.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_min_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vminpd
  %res = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_min_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vminsd
  %res = call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_movmsk_pd(<2 x double> %a0) {
  ; CHECK: vmovmskpd
  %res = call i32 @llvm.x86.sse2.movmsk.pd(<2 x double> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.movmsk.pd(<2 x double>) nounwind readnone




define <2 x double> @test_x86_sse2_mul_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: test_x86_sse2_mul_sd
  ; CHECK: vmulsd
  %res = call <2 x double> @llvm.x86.sse2.mul.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.mul.sd(<2 x double>, <2 x double>) nounwind readnone


define <8 x i16> @test_x86_sse2_packssdw_128(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpackssdw
  %res = call <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32> %a0, <4 x i32> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32>, <4 x i32>) nounwind readnone


define <16 x i8> @test_x86_sse2_packsswb_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpacksswb
  %res = call <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16> %a0, <8 x i16> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse2_packuswb_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpackuswb
  %res = call <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16> %a0, <8 x i16> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse2_padds_b(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpaddsb
  %res = call <16 x i8> @llvm.x86.sse2.padds.b(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.padds.b(<16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse2_padds_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpaddsw
  %res = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse2_paddus_b(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpaddusb
  %res = call <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse2_paddus_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpaddusw
  %res = call <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse2_pavg_b(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpavgb
  %res = call <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse2_pavg_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpavgw
  %res = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16>, <8 x i16>) nounwind readnone


define <4 x i32> @test_x86_sse2_pmadd_wd(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpmaddwd
  %res = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %a0, <8 x i16> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16>, <8 x i16>) nounwind readnone


define <8 x i16> @test_x86_sse2_pmaxs_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpmaxsw
  %res = call <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse2_pmaxu_b(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpmaxub
  %res = call <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse2_pmins_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpminsw
  %res = call <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse2_pminu_b(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpminub
  %res = call <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8>, <16 x i8>) nounwind readnone


define i32 @test_x86_sse2_pmovmskb_128(<16 x i8> %a0) {
  ; CHECK: vpmovmskb
  %res = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse2_pmulh_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpmulhw
  %res = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16>, <8 x i16>) nounwind readnone


define <8 x i16> @test_x86_sse2_pmulhu_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpmulhuw
  %res = call <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16>, <8 x i16>) nounwind readnone


define <2 x i64> @test_x86_sse2_pmulu_dq(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpmuludq
  %res = call <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32> %a0, <4 x i32> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32>, <4 x i32>) nounwind readnone


define <2 x i64> @test_x86_sse2_psad_bw(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpsadbw
  %res = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %a0, <16 x i8> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone


define <4 x i32> @test_x86_sse2_psll_d(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpslld
  %res = call <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32>, <4 x i32>) nounwind readnone


define <2 x i64> @test_x86_sse2_psll_dq(<2 x i64> %a0) {
  ; CHECK: vpslldq
  %res = call <2 x i64> @llvm.x86.sse2.psll.dq(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psll.dq(<2 x i64>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_psll_dq_bs(<2 x i64> %a0) {
  ; CHECK: vpslldq
  %res = call <2 x i64> @llvm.x86.sse2.psll.dq.bs(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psll.dq.bs(<2 x i64>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_psll_q(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpsllq
  %res = call <2 x i64> @llvm.x86.sse2.psll.q(<2 x i64> %a0, <2 x i64> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psll.q(<2 x i64>, <2 x i64>) nounwind readnone


define <8 x i16> @test_x86_sse2_psll_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsllw
  %res = call <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16>, <8 x i16>) nounwind readnone


define <4 x i32> @test_x86_sse2_pslli_d(<4 x i32> %a0) {
  ; CHECK: vpslld
  %res = call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %a0, i32 7) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_pslli_q(<2 x i64> %a0) {
  ; CHECK: vpsllq
  %res = call <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64>, i32) nounwind readnone


define <8 x i16> @test_x86_sse2_pslli_w(<8 x i16> %a0) {
  ; CHECK: vpsllw
  %res = call <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16> %a0, i32 7) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16>, i32) nounwind readnone


define <4 x i32> @test_x86_sse2_psra_d(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpsrad
  %res = call <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32>, <4 x i32>) nounwind readnone


define <8 x i16> @test_x86_sse2_psra_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsraw
  %res = call <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16>, <8 x i16>) nounwind readnone


define <4 x i32> @test_x86_sse2_psrai_d(<4 x i32> %a0) {
  ; CHECK: vpsrad
  %res = call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %a0, i32 7) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32>, i32) nounwind readnone


define <8 x i16> @test_x86_sse2_psrai_w(<8 x i16> %a0) {
  ; CHECK: vpsraw
  %res = call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %a0, i32 7) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16>, i32) nounwind readnone


define <4 x i32> @test_x86_sse2_psrl_d(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpsrld
  %res = call <4 x i32> @llvm.x86.sse2.psrl.d(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.psrl.d(<4 x i32>, <4 x i32>) nounwind readnone


define <2 x i64> @test_x86_sse2_psrl_dq(<2 x i64> %a0) {
  ; CHECK: vpsrldq
  %res = call <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_psrl_dq_bs(<2 x i64> %a0) {
  ; CHECK: vpsrldq
  %res = call <2 x i64> @llvm.x86.sse2.psrl.dq.bs(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psrl.dq.bs(<2 x i64>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_psrl_q(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpsrlq
  %res = call <2 x i64> @llvm.x86.sse2.psrl.q(<2 x i64> %a0, <2 x i64> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psrl.q(<2 x i64>, <2 x i64>) nounwind readnone


define <8 x i16> @test_x86_sse2_psrl_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsrlw
  %res = call <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16>, <8 x i16>) nounwind readnone


define <4 x i32> @test_x86_sse2_psrli_d(<4 x i32> %a0) {
  ; CHECK: vpsrld
  %res = call <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32> %a0, i32 7) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_psrli_q(<2 x i64> %a0) {
  ; CHECK: vpsrlq
  %res = call <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64>, i32) nounwind readnone


define <8 x i16> @test_x86_sse2_psrli_w(<8 x i16> %a0) {
  ; CHECK: vpsrlw
  %res = call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> %a0, i32 7) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16>, i32) nounwind readnone


define <16 x i8> @test_x86_sse2_psubs_b(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpsubsb
  %res = call <16 x i8> @llvm.x86.sse2.psubs.b(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.psubs.b(<16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse2_psubs_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsubsw
  %res = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse2_psubus_b(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpsubusb
  %res = call <16 x i8> @llvm.x86.sse2.psubus.b(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse2.psubus.b(<16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse2_psubus_w(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsubusw
  %res = call <8 x i16> @llvm.x86.sse2.psubus.w(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse2.psubus.w(<8 x i16>, <8 x i16>) nounwind readnone


define <2 x double> @test_x86_sse2_sqrt_pd(<2 x double> %a0) {
  ; CHECK: vsqrtpd
  %res = call <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double> %a0) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_sqrt_sd(<2 x double> %a0) {
  ; CHECK: vsqrtsd
  %res = call <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double> %a0) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone


define void @test_x86_sse2_storel_dq(i8* %a0, <4 x i32> %a1) {
  ; CHECK: test_x86_sse2_storel_dq
  ; CHECK: movl
  ; CHECK: vmovq
  call void @llvm.x86.sse2.storel.dq(i8* %a0, <4 x i32> %a1)
  ret void
}
declare void @llvm.x86.sse2.storel.dq(i8*, <4 x i32>) nounwind


define void @test_x86_sse2_storeu_dq(i8* %a0, <16 x i8> %a1) {
  ; CHECK: test_x86_sse2_storeu_dq
  ; CHECK: movl
  ; CHECK: vmovdqu
  call void @llvm.x86.sse2.storeu.dq(i8* %a0, <16 x i8> %a1)
  ret void
}
declare void @llvm.x86.sse2.storeu.dq(i8*, <16 x i8>) nounwind


define void @test_x86_sse2_storeu_pd(i8* %a0, <2 x double> %a1) {
  ; CHECK: test_x86_sse2_storeu_pd
  ; CHECK: movl
  ; CHECK: vmovupd
  %a2 = fadd <2 x double> %a1, <double 0x0, double 0x4200000000000000>
  call void @llvm.x86.sse2.storeu.pd(i8* %a0, <2 x double> %a2)
  ret void
}
declare void @llvm.x86.sse2.storeu.pd(i8*, <2 x double>) nounwind


define <2 x double> @test_x86_sse2_sub_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: test_x86_sse2_sub_sd
  ; CHECK: vsubsd
  %res = call <2 x double> @llvm.x86.sse2.sub.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.sub.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_ucomieq_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vucomisd
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.ucomieq.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.ucomieq.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_ucomige_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vucomisd
  ; CHECK: setae
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.ucomige.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.ucomige.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_ucomigt_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vucomisd
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.ucomigt.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.ucomigt.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_ucomile_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vucomisd
  ; CHECK: setbe
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.ucomile.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.ucomile.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_ucomilt_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vucomisd
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse2.ucomilt.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.ucomilt.sd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_sse2_ucomineq_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vucomisd
  ; CHECK: setne
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse2.ucomineq.sd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse2.ucomineq.sd(<2 x double>, <2 x double>) nounwind readnone


define <2 x double> @test_x86_sse3_addsub_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vaddsubpd
  %res = call <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double>, <2 x double>) nounwind readnone


define <4 x float> @test_x86_sse3_addsub_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vaddsubps
  %res = call <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float>, <4 x float>) nounwind readnone


define <2 x double> @test_x86_sse3_hadd_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vhaddpd
  %res = call <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double>, <2 x double>) nounwind readnone


define <4 x float> @test_x86_sse3_hadd_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vhaddps
  %res = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone


define <2 x double> @test_x86_sse3_hsub_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vhsubpd
  %res = call <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double>, <2 x double>) nounwind readnone


define <4 x float> @test_x86_sse3_hsub_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vhsubps
  %res = call <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float>, <4 x float>) nounwind readnone


define <16 x i8> @test_x86_sse3_ldu_dq(i8* %a0) {
  ; CHECK: movl
  ; CHECK: vlddqu
  %res = call <16 x i8> @llvm.x86.sse3.ldu.dq(i8* %a0) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse3.ldu.dq(i8*) nounwind readonly


define <2 x double> @test_x86_sse41_blendpd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vblendpd
  %res = call <2 x double> @llvm.x86.sse41.blendpd(<2 x double> %a0, <2 x double> %a1, i32 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse41.blendpd(<2 x double>, <2 x double>, i32) nounwind readnone


define <4 x float> @test_x86_sse41_blendps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vblendps
  %res = call <4 x float> @llvm.x86.sse41.blendps(<4 x float> %a0, <4 x float> %a1, i32 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse41.blendps(<4 x float>, <4 x float>, i32) nounwind readnone


define <2 x double> @test_x86_sse41_blendvpd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: vblendvpd
  %res = call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone


define <4 x float> @test_x86_sse41_blendvps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: vblendvps
  %res = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone


define <2 x double> @test_x86_sse41_dppd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vdppd
  %res = call <2 x double> @llvm.x86.sse41.dppd(<2 x double> %a0, <2 x double> %a1, i32 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse41.dppd(<2 x double>, <2 x double>, i32) nounwind readnone


define <4 x float> @test_x86_sse41_dpps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vdpps
  %res = call <4 x float> @llvm.x86.sse41.dpps(<4 x float> %a0, <4 x float> %a1, i32 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse41.dpps(<4 x float>, <4 x float>, i32) nounwind readnone


define <4 x float> @test_x86_sse41_insertps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vinsertps
  %res = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a0, <4 x float> %a1, i32 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i32) nounwind readnone



define <8 x i16> @test_x86_sse41_mpsadbw(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vmpsadbw
  %res = call <8 x i16> @llvm.x86.sse41.mpsadbw(<16 x i8> %a0, <16 x i8> %a1, i32 7) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.mpsadbw(<16 x i8>, <16 x i8>, i32) nounwind readnone


define <8 x i16> @test_x86_sse41_packusdw(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpackusdw
  %res = call <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32> %a0, <4 x i32> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32>, <4 x i32>) nounwind readnone


define <16 x i8> @test_x86_sse41_pblendvb(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2) {
  ; CHECK: vpblendvb
  %res = call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse41_pblendw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpblendw
  %res = call <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16> %a0, <8 x i16> %a1, i32 7) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16>, <8 x i16>, i32) nounwind readnone


define <8 x i16> @test_x86_sse41_phminposuw(<8 x i16> %a0) {
  ; CHECK: vphminposuw
  %res = call <8 x i16> @llvm.x86.sse41.phminposuw(<8 x i16> %a0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.phminposuw(<8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse41_pmaxsb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpmaxsb
  %res = call <16 x i8> @llvm.x86.sse41.pmaxsb(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse41.pmaxsb(<16 x i8>, <16 x i8>) nounwind readnone


define <4 x i32> @test_x86_sse41_pmaxsd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpmaxsd
  %res = call <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone


define <4 x i32> @test_x86_sse41_pmaxud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpmaxud
  %res = call <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readnone


define <8 x i16> @test_x86_sse41_pmaxuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpmaxuw
  %res = call <8 x i16> @llvm.x86.sse41.pmaxuw(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.pmaxuw(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_sse41_pminsb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpminsb
  %res = call <16 x i8> @llvm.x86.sse41.pminsb(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse41.pminsb(<16 x i8>, <16 x i8>) nounwind readnone


define <4 x i32> @test_x86_sse41_pminsd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpminsd
  %res = call <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone


define <4 x i32> @test_x86_sse41_pminud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpminud
  %res = call <4 x i32> @llvm.x86.sse41.pminud(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readnone


define <8 x i16> @test_x86_sse41_pminuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpminuw
  %res = call <8 x i16> @llvm.x86.sse41.pminuw(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.pminuw(<8 x i16>, <8 x i16>) nounwind readnone


define <4 x i32> @test_x86_sse41_pmovsxbd(<16 x i8> %a0) {
  ; CHECK: vpmovsxbd
  %res = call <4 x i32> @llvm.x86.sse41.pmovsxbd(<16 x i8> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pmovsxbd(<16 x i8>) nounwind readnone


define <2 x i64> @test_x86_sse41_pmovsxbq(<16 x i8> %a0) {
  ; CHECK: vpmovsxbq
  %res = call <2 x i64> @llvm.x86.sse41.pmovsxbq(<16 x i8> %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse41.pmovsxbq(<16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse41_pmovsxbw(<16 x i8> %a0) {
  ; CHECK: vpmovsxbw
  %res = call <8 x i16> @llvm.x86.sse41.pmovsxbw(<16 x i8> %a0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.pmovsxbw(<16 x i8>) nounwind readnone


define <2 x i64> @test_x86_sse41_pmovsxdq(<4 x i32> %a0) {
  ; CHECK: vpmovsxdq
  %res = call <2 x i64> @llvm.x86.sse41.pmovsxdq(<4 x i32> %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse41.pmovsxdq(<4 x i32>) nounwind readnone


define <4 x i32> @test_x86_sse41_pmovsxwd(<8 x i16> %a0) {
  ; CHECK: vpmovsxwd
  %res = call <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16>) nounwind readnone


define <2 x i64> @test_x86_sse41_pmovsxwq(<8 x i16> %a0) {
  ; CHECK: vpmovsxwq
  %res = call <2 x i64> @llvm.x86.sse41.pmovsxwq(<8 x i16> %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse41.pmovsxwq(<8 x i16>) nounwind readnone


define <4 x i32> @test_x86_sse41_pmovzxbd(<16 x i8> %a0) {
  ; CHECK: vpmovzxbd
  %res = call <4 x i32> @llvm.x86.sse41.pmovzxbd(<16 x i8> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pmovzxbd(<16 x i8>) nounwind readnone


define <2 x i64> @test_x86_sse41_pmovzxbq(<16 x i8> %a0) {
  ; CHECK: vpmovzxbq
  %res = call <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8> %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8>) nounwind readnone


define <8 x i16> @test_x86_sse41_pmovzxbw(<16 x i8> %a0) {
  ; CHECK: vpmovzxbw
  %res = call <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8> %a0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8>) nounwind readnone


define <2 x i64> @test_x86_sse41_pmovzxdq(<4 x i32> %a0) {
  ; CHECK: vpmovzxdq
  %res = call <2 x i64> @llvm.x86.sse41.pmovzxdq(<4 x i32> %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse41.pmovzxdq(<4 x i32>) nounwind readnone


define <4 x i32> @test_x86_sse41_pmovzxwd(<8 x i16> %a0) {
  ; CHECK: vpmovzxwd
  %res = call <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16>) nounwind readnone


define <2 x i64> @test_x86_sse41_pmovzxwq(<8 x i16> %a0) {
  ; CHECK: vpmovzxwq
  %res = call <2 x i64> @llvm.x86.sse41.pmovzxwq(<8 x i16> %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse41.pmovzxwq(<8 x i16>) nounwind readnone


define <2 x i64> @test_x86_sse41_pmuldq(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpmuldq
  %res = call <2 x i64> @llvm.x86.sse41.pmuldq(<4 x i32> %a0, <4 x i32> %a1) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse41.pmuldq(<4 x i32>, <4 x i32>) nounwind readnone


define i32 @test_x86_sse41_ptestc(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vptest 
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse41.ptestc(<2 x i64> %a0, <2 x i64> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse41.ptestc(<2 x i64>, <2 x i64>) nounwind readnone


define i32 @test_x86_sse41_ptestnzc(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vptest 
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse41.ptestnzc(<2 x i64> %a0, <2 x i64> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse41.ptestnzc(<2 x i64>, <2 x i64>) nounwind readnone


define i32 @test_x86_sse41_ptestz(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vptest 
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse41.ptestz(<2 x i64> %a0, <2 x i64> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse41.ptestz(<2 x i64>, <2 x i64>) nounwind readnone


define <2 x double> @test_x86_sse41_round_pd(<2 x double> %a0) {
  ; CHECK: vroundpd
  %res = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %a0, i32 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) nounwind readnone


define <4 x float> @test_x86_sse41_round_ps(<4 x float> %a0) {
  ; CHECK: vroundps
  %res = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %a0, i32 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone


define <2 x double> @test_x86_sse41_round_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vroundsd
  %res = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %a0, <2 x double> %a1, i32 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone


define <4 x float> @test_x86_sse41_round_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vroundss
  %res = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %a0, <4 x float> %a1, i32 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone


define i32 @test_x86_sse42_pcmpestri128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl $7
  ; CHECK: movl $7
  ; CHECK: vpcmpestri $7
  ; CHECK: movl
  %res = call i32 @llvm.x86.sse42.pcmpestri128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestri128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestri128_load(<16 x i8>* %a0, <16 x i8>* %a2) {
  ; CHECK: movl $7
  ; CHECK: movl $7
  ; CHECK: vpcmpestri $7, (
  ; CHECK: movl
  %1 = load <16 x i8>* %a0
  %2 = load <16 x i8>* %a2
  %res = call i32 @llvm.x86.sse42.pcmpestri128(<16 x i8> %1, i32 7, <16 x i8> %2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}


define i32 @test_x86_sse42_pcmpestria128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: vpcmpestri
  ; CHECK: seta
  %res = call i32 @llvm.x86.sse42.pcmpestria128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestria128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestric128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: vpcmpestri
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse42.pcmpestric128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestric128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestrio128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: vpcmpestri
  ; CHECK: seto
  %res = call i32 @llvm.x86.sse42.pcmpestrio128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestrio128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestris128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: vpcmpestri
  ; CHECK: sets
  %res = call i32 @llvm.x86.sse42.pcmpestris128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestris128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestriz128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: vpcmpestri
  ; CHECK: sete
  %res = call i32 @llvm.x86.sse42.pcmpestriz128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestriz128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpestrm128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: vpcmpestrm
  ; CHECK-NOT: vmov
  %res = call <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpestrm128_load(<16 x i8> %a0, <16 x i8>* %a2) {
  ; CHECK: movl $7
  ; CHECK: movl $7
  ; CHECK: vpcmpestrm $7,
  ; CHECK-NOT: vmov
  %1 = load <16 x i8>* %a2
  %res = call <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8> %a0, i32 7, <16 x i8> %1, i32 7, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}


define i32 @test_x86_sse42_pcmpistri128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcmpistri $7
  ; CHECK: movl
  %res = call i32 @llvm.x86.sse42.pcmpistri128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistri128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistri128_load(<16 x i8>* %a0, <16 x i8>* %a1) {
  ; CHECK: vpcmpistri $7, (
  ; CHECK: movl
  %1 = load <16 x i8>* %a0
  %2 = load <16 x i8>* %a1
  %res = call i32 @llvm.x86.sse42.pcmpistri128(<16 x i8> %1, <16 x i8> %2, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}


define i32 @test_x86_sse42_pcmpistria128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcmpistri
  ; CHECK: seta
  %res = call i32 @llvm.x86.sse42.pcmpistria128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistria128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistric128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcmpistri
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse42.pcmpistric128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistric128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistrio128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcmpistri
  ; CHECK: seto
  %res = call i32 @llvm.x86.sse42.pcmpistrio128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistrio128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistris128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcmpistri
  ; CHECK: sets
  %res = call i32 @llvm.x86.sse42.pcmpistris128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistris128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistriz128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcmpistri
  ; CHECK: sete
  %res = call i32 @llvm.x86.sse42.pcmpistriz128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistriz128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpistrm128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcmpistrm $7
  ; CHECK-NOT: vmov
  %res = call <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpistrm128_load(<16 x i8> %a0, <16 x i8>* %a1) {
  ; CHECK: vpcmpistrm $7, (
  ; CHECK-NOT: vmov
  %1 = load <16 x i8>* %a1
  %res = call <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8> %a0, <16 x i8> %1, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}


define <4 x float> @test_x86_sse_add_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vaddss
  %res = call <4 x float> @llvm.x86.sse.add.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.add.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cmp_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcmpordps
  %res = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %a0, <4 x float> %a1, i8 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone


define <4 x float> @test_x86_sse_cmp_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcmpordss
  %res = call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %a0, <4 x float> %a1, i8 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cmp.ss(<4 x float>, <4 x float>, i8) nounwind readnone


define i32 @test_x86_sse_comieq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcomiss
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comieq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comieq.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comige_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcomiss
  ; CHECK: setae
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comige.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comige.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comigt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcomiss
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comigt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comigt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comile_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcomiss
  ; CHECK: setbe
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comile.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comile.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comilt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcomiss
  ; CHECK: sbb
  %res = call i32 @llvm.x86.sse.comilt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comilt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comineq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vcomiss
  ; CHECK: setne
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comineq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comineq.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cvtsi2ss(<4 x float> %a0) {
  ; CHECK: movl
  ; CHECK: vcvtsi2ss
  %res = call <4 x float> @llvm.x86.sse.cvtsi2ss(<4 x float> %a0, i32 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cvtsi2ss(<4 x float>, i32) nounwind readnone


define i32 @test_x86_sse_cvtss2si(<4 x float> %a0) {
  ; CHECK: vcvtss2si
  %res = call i32 @llvm.x86.sse.cvtss2si(<4 x float> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone


define i32 @test_x86_sse_cvttss2si(<4 x float> %a0) {
  ; CHECK: vcvttss2si
  %res = call i32 @llvm.x86.sse.cvttss2si(<4 x float> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_div_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vdivss
  %res = call <4 x float> @llvm.x86.sse.div.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.div.ss(<4 x float>, <4 x float>) nounwind readnone


define void @test_x86_sse_ldmxcsr(i8* %a0) {
  ; CHECK: movl
  ; CHECK: vldmxcsr
  call void @llvm.x86.sse.ldmxcsr(i8* %a0)
  ret void
}
declare void @llvm.x86.sse.ldmxcsr(i8*) nounwind



define <4 x float> @test_x86_sse_max_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vmaxps
  %res = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_max_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vmaxss
  %res = call <4 x float> @llvm.x86.sse.max.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_min_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vminps
  %res = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_min_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vminss
  %res = call <4 x float> @llvm.x86.sse.min.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_movmsk_ps(<4 x float> %a0) {
  ; CHECK: vmovmskps
  %res = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone



define <4 x float> @test_x86_sse_mul_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vmulss
  %res = call <4 x float> @llvm.x86.sse.mul.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rcp_ps(<4 x float> %a0) {
  ; CHECK: vrcpps
  %res = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rcp_ss(<4 x float> %a0) {
  ; CHECK: vrcpss
  %res = call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rsqrt_ps(<4 x float> %a0) {
  ; CHECK: vrsqrtps
  %res = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rsqrt_ss(<4 x float> %a0) {
  ; CHECK: vrsqrtss
  %res = call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_sqrt_ps(<4 x float> %a0) {
  ; CHECK: vsqrtps
  %res = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_sqrt_ss(<4 x float> %a0) {
  ; CHECK: vsqrtss
  %res = call <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone


define void @test_x86_sse_stmxcsr(i8* %a0) {
  ; CHECK: movl
  ; CHECK: vstmxcsr
  call void @llvm.x86.sse.stmxcsr(i8* %a0)
  ret void
}
declare void @llvm.x86.sse.stmxcsr(i8*) nounwind


define void @test_x86_sse_storeu_ps(i8* %a0, <4 x float> %a1) {
  ; CHECK: movl
  ; CHECK: vmovups
  call void @llvm.x86.sse.storeu.ps(i8* %a0, <4 x float> %a1)
  ret void
}
declare void @llvm.x86.sse.storeu.ps(i8*, <4 x float>) nounwind


define <4 x float> @test_x86_sse_sub_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vsubss
  %res = call <4 x float> @llvm.x86.sse.sub.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomieq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vucomiss
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomieq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomieq.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomige_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vucomiss
  ; CHECK: setae
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomige.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomige.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomigt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vucomiss
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomigt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomigt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomile_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vucomiss
  ; CHECK: setbe
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomile.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomile.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomilt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vucomiss
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse.ucomilt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomilt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomineq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vucomiss
  ; CHECK: setne
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomineq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomineq.ss(<4 x float>, <4 x float>) nounwind readnone


define <16 x i8> @test_x86_ssse3_pabs_b_128(<16 x i8> %a0) {
  ; CHECK: vpabsb
  %res = call <16 x i8> @llvm.x86.ssse3.pabs.b.128(<16 x i8> %a0) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.ssse3.pabs.b.128(<16 x i8>) nounwind readnone


define <4 x i32> @test_x86_ssse3_pabs_d_128(<4 x i32> %a0) {
  ; CHECK: vpabsd
  %res = call <4 x i32> @llvm.x86.ssse3.pabs.d.128(<4 x i32> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.ssse3.pabs.d.128(<4 x i32>) nounwind readnone


define <8 x i16> @test_x86_ssse3_pabs_w_128(<8 x i16> %a0) {
  ; CHECK: vpabsw
  %res = call <8 x i16> @llvm.x86.ssse3.pabs.w.128(<8 x i16> %a0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.pabs.w.128(<8 x i16>) nounwind readnone


define <4 x i32> @test_x86_ssse3_phadd_d_128(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vphaddd
  %res = call <4 x i32> @llvm.x86.ssse3.phadd.d.128(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.ssse3.phadd.d.128(<4 x i32>, <4 x i32>) nounwind readnone


define <8 x i16> @test_x86_ssse3_phadd_sw_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vphaddsw
  %res = call <8 x i16> @llvm.x86.ssse3.phadd.sw.128(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.phadd.sw.128(<8 x i16>, <8 x i16>) nounwind readnone


define <8 x i16> @test_x86_ssse3_phadd_w_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vphaddw
  %res = call <8 x i16> @llvm.x86.ssse3.phadd.w.128(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.phadd.w.128(<8 x i16>, <8 x i16>) nounwind readnone


define <4 x i32> @test_x86_ssse3_phsub_d_128(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vphsubd
  %res = call <4 x i32> @llvm.x86.ssse3.phsub.d.128(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.ssse3.phsub.d.128(<4 x i32>, <4 x i32>) nounwind readnone


define <8 x i16> @test_x86_ssse3_phsub_sw_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vphsubsw
  %res = call <8 x i16> @llvm.x86.ssse3.phsub.sw.128(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.phsub.sw.128(<8 x i16>, <8 x i16>) nounwind readnone


define <8 x i16> @test_x86_ssse3_phsub_w_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vphsubw
  %res = call <8 x i16> @llvm.x86.ssse3.phsub.w.128(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.phsub.w.128(<8 x i16>, <8 x i16>) nounwind readnone


define <8 x i16> @test_x86_ssse3_pmadd_ub_sw_128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpmaddubsw
  %res = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %a0, <16 x i8> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8>, <16 x i8>) nounwind readnone


define <8 x i16> @test_x86_ssse3_pmul_hr_sw_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpmulhrsw
  %res = call <8 x i16> @llvm.x86.ssse3.pmul.hr.sw.128(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.pmul.hr.sw.128(<8 x i16>, <8 x i16>) nounwind readnone


define <16 x i8> @test_x86_ssse3_pshuf_b_128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpshufb
  %res = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>) nounwind readnone


define <16 x i8> @test_x86_ssse3_psign_b_128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpsignb
  %res = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %a0, <16 x i8> %a1) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8>, <16 x i8>) nounwind readnone


define <4 x i32> @test_x86_ssse3_psign_d_128(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpsignd
  %res = call <4 x i32> @llvm.x86.ssse3.psign.d.128(<4 x i32> %a0, <4 x i32> %a1) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.ssse3.psign.d.128(<4 x i32>, <4 x i32>) nounwind readnone


define <8 x i16> @test_x86_ssse3_psign_w_128(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpsignw
  %res = call <8 x i16> @llvm.x86.ssse3.psign.w.128(<8 x i16> %a0, <8 x i16> %a1) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.ssse3.psign.w.128(<8 x i16>, <8 x i16>) nounwind readnone


define <4 x double> @test_x86_avx_addsub_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vaddsubpd
  %res = call <4 x double> @llvm.x86.avx.addsub.pd.256(<4 x double> %a0, <4 x double> %a1) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.addsub.pd.256(<4 x double>, <4 x double>) nounwind readnone


define <8 x float> @test_x86_avx_addsub_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vaddsubps
  %res = call <8 x float> @llvm.x86.avx.addsub.ps.256(<8 x float> %a0, <8 x float> %a1) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.addsub.ps.256(<8 x float>, <8 x float>) nounwind readnone


define <4 x double> @test_x86_avx_blend_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vblendpd
  %res = call <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double> %a0, <4 x double> %a1, i32 7) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double>, <4 x double>, i32) nounwind readnone


define <8 x float> @test_x86_avx_blend_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vblendps
  %res = call <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float> %a0, <8 x float> %a1, i32 7) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float>, <8 x float>, i32) nounwind readnone


define <4 x double> @test_x86_avx_blendv_pd_256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK: vblendvpd
  %res = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>) nounwind readnone


define <8 x float> @test_x86_avx_blendv_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: vblendvps
  %res = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone


define <4 x double> @test_x86_avx_cmp_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vcmpordpd
  %res = call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %a0, <4 x double> %a1, i8 7) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone


define <8 x float> @test_x86_avx_cmp_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vcmpordps
  %res = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a1, i8 7) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}

define <8 x float> @test_x86_avx_cmp_ps_256_pseudo_op(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vcmpeqps
  %a2 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a1, i8 0) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpltps
  %a3 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a2, i8 1) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpleps
  %a4 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a3, i8 2) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpunordps
  %a5 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a4, i8 3) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpneqps
  %a6 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a5, i8 4) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpnltps
  %a7 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a6, i8 5) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpnleps
  %a8 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a7, i8 6) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpordps
  %a9 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a8, i8 7) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpeq_uqps
  %a10 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a9, i8 8) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpngeps
  %a11 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a10, i8 9) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpngtps
  %a12 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a11, i8 10) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpfalseps
  %a13 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a12, i8 11) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpneq_oqps
  %a14 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a13, i8 12) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpgeps
  %a15 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a14, i8 13) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpgtps
  %a16 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a15, i8 14) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmptrueps
  %a17 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a16, i8 15) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpeq_osps
  %a18 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a17, i8 16) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmplt_oqps
  %a19 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a18, i8 17) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmple_oqps
  %a20 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a19, i8 18) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpunord_sps
  %a21 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a20, i8 19) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpneq_usps
  %a22 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a21, i8 20) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpnlt_uqps
  %a23 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a22, i8 21) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpnle_uqps
  %a24 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a23, i8 22) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpord_sps
  %a25 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a24, i8 23) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpeq_usps
  %a26 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a25, i8 24) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpnge_uqps
  %a27 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a26, i8 25) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpngt_uqps
  %a28 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a27, i8 26) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpfalse_osps
  %a29 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a28, i8 27) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpneq_osps
  %a30 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a29, i8 28) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpge_oqps
  %a31 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a30, i8 29) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmpgt_oqps
  %a32 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a31, i8 30) ; <<8 x float>> [#uses=1]
  ; CHECK: vcmptrue_usps
  %res = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a32, i8 31) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone


define <4 x float> @test_x86_avx_cvt_pd2_ps_256(<4 x double> %a0) {
  ; CHECK: vcvtpd2psy
  %res = call <4 x float> @llvm.x86.avx.cvt.pd2.ps.256(<4 x double> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx.cvt.pd2.ps.256(<4 x double>) nounwind readnone


define <4 x i32> @test_x86_avx_cvt_pd2dq_256(<4 x double> %a0) {
  ; CHECK: vcvtpd2dqy
  %res = call <4 x i32> @llvm.x86.avx.cvt.pd2dq.256(<4 x double> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx.cvt.pd2dq.256(<4 x double>) nounwind readnone


define <4 x double> @test_x86_avx_cvt_ps2_pd_256(<4 x float> %a0) {
  ; CHECK: vcvtps2pd
  %res = call <4 x double> @llvm.x86.avx.cvt.ps2.pd.256(<4 x float> %a0) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.cvt.ps2.pd.256(<4 x float>) nounwind readnone


define <8 x i32> @test_x86_avx_cvt_ps2dq_256(<8 x float> %a0) {
  ; CHECK: vcvtps2dq
  %res = call <8 x i32> @llvm.x86.avx.cvt.ps2dq.256(<8 x float> %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx.cvt.ps2dq.256(<8 x float>) nounwind readnone


define <4 x double> @test_x86_avx_cvtdq2_pd_256(<4 x i32> %a0) {
  ; CHECK: vcvtdq2pd
  %res = call <4 x double> @llvm.x86.avx.cvtdq2.pd.256(<4 x i32> %a0) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.cvtdq2.pd.256(<4 x i32>) nounwind readnone


define <8 x float> @test_x86_avx_cvtdq2_ps_256(<8 x i32> %a0) {
  ; CHECK: vcvtdq2ps
  %res = call <8 x float> @llvm.x86.avx.cvtdq2.ps.256(<8 x i32> %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.cvtdq2.ps.256(<8 x i32>) nounwind readnone


define <4 x i32> @test_x86_avx_cvtt_pd2dq_256(<4 x double> %a0) {
  ; CHECK: vcvttpd2dqy
  %res = call <4 x i32> @llvm.x86.avx.cvtt.pd2dq.256(<4 x double> %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx.cvtt.pd2dq.256(<4 x double>) nounwind readnone


define <8 x i32> @test_x86_avx_cvtt_ps2dq_256(<8 x float> %a0) {
  ; CHECK: vcvttps2dq
  %res = call <8 x i32> @llvm.x86.avx.cvtt.ps2dq.256(<8 x float> %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx.cvtt.ps2dq.256(<8 x float>) nounwind readnone


define <8 x float> @test_x86_avx_dp_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vdpps
  %res = call <8 x float> @llvm.x86.avx.dp.ps.256(<8 x float> %a0, <8 x float> %a1, i32 7) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.dp.ps.256(<8 x float>, <8 x float>, i32) nounwind readnone


define <4 x double> @test_x86_avx_hadd_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vhaddpd
  %res = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %a0, <4 x double> %a1) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone


define <8 x float> @test_x86_avx_hadd_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vhaddps
  %res = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %a0, <8 x float> %a1) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone


define <4 x double> @test_x86_avx_hsub_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vhsubpd
  %res = call <4 x double> @llvm.x86.avx.hsub.pd.256(<4 x double> %a0, <4 x double> %a1) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.hsub.pd.256(<4 x double>, <4 x double>) nounwind readnone


define <8 x float> @test_x86_avx_hsub_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vhsubps
  %res = call <8 x float> @llvm.x86.avx.hsub.ps.256(<8 x float> %a0, <8 x float> %a1) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.hsub.ps.256(<8 x float>, <8 x float>) nounwind readnone


define <32 x i8> @test_x86_avx_ldu_dq_256(i8* %a0) {
  ; CHECK: vlddqu
  %res = call <32 x i8> @llvm.x86.avx.ldu.dq.256(i8* %a0) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx.ldu.dq.256(i8*) nounwind readonly


define <2 x double> @test_x86_avx_maskload_pd(i8* %a0, <2 x double> %a1) {
  ; CHECK: vmaskmovpd
  %res = call <2 x double> @llvm.x86.avx.maskload.pd(i8* %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx.maskload.pd(i8*, <2 x double>) nounwind readonly


define <4 x double> @test_x86_avx_maskload_pd_256(i8* %a0, <4 x double> %a1) {
  ; CHECK: vmaskmovpd
  %res = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %a0, <4 x double> %a1) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.maskload.pd.256(i8*, <4 x double>) nounwind readonly


define <4 x float> @test_x86_avx_maskload_ps(i8* %a0, <4 x float> %a1) {
  ; CHECK: vmaskmovps
  %res = call <4 x float> @llvm.x86.avx.maskload.ps(i8* %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx.maskload.ps(i8*, <4 x float>) nounwind readonly


define <8 x float> @test_x86_avx_maskload_ps_256(i8* %a0, <8 x float> %a1) {
  ; CHECK: vmaskmovps
  %res = call <8 x float> @llvm.x86.avx.maskload.ps.256(i8* %a0, <8 x float> %a1) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.maskload.ps.256(i8*, <8 x float>) nounwind readonly


define void @test_x86_avx_maskstore_pd(i8* %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: vmaskmovpd
  call void @llvm.x86.avx.maskstore.pd(i8* %a0, <2 x double> %a1, <2 x double> %a2)
  ret void
}
declare void @llvm.x86.avx.maskstore.pd(i8*, <2 x double>, <2 x double>) nounwind


define void @test_x86_avx_maskstore_pd_256(i8* %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK: vmaskmovpd
  call void @llvm.x86.avx.maskstore.pd.256(i8* %a0, <4 x double> %a1, <4 x double> %a2)
  ret void
}
declare void @llvm.x86.avx.maskstore.pd.256(i8*, <4 x double>, <4 x double>) nounwind


define void @test_x86_avx_maskstore_ps(i8* %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: vmaskmovps
  call void @llvm.x86.avx.maskstore.ps(i8* %a0, <4 x float> %a1, <4 x float> %a2)
  ret void
}
declare void @llvm.x86.avx.maskstore.ps(i8*, <4 x float>, <4 x float>) nounwind


define void @test_x86_avx_maskstore_ps_256(i8* %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: vmaskmovps
  call void @llvm.x86.avx.maskstore.ps.256(i8* %a0, <8 x float> %a1, <8 x float> %a2)
  ret void
}
declare void @llvm.x86.avx.maskstore.ps.256(i8*, <8 x float>, <8 x float>) nounwind


define <4 x double> @test_x86_avx_max_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vmaxpd
  %res = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %a0, <4 x double> %a1) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone


define <8 x float> @test_x86_avx_max_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vmaxps
  %res = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %a0, <8 x float> %a1) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone


define <4 x double> @test_x86_avx_min_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vminpd
  %res = call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %a0, <4 x double> %a1) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone


define <8 x float> @test_x86_avx_min_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vminps
  %res = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %a0, <8 x float> %a1) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone


define i32 @test_x86_avx_movmsk_pd_256(<4 x double> %a0) {
  ; CHECK: vmovmskpd
  %res = call i32 @llvm.x86.avx.movmsk.pd.256(<4 x double> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.movmsk.pd.256(<4 x double>) nounwind readnone


define i32 @test_x86_avx_movmsk_ps_256(<8 x float> %a0) {
  ; CHECK: vmovmskps
  %res = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) nounwind readnone







define i32 @test_x86_avx_ptestc_256(<4 x i64> %a0, <4 x i64> %a1) {
  ; CHECK: vptest
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.avx.ptestc.256(<4 x i64> %a0, <4 x i64> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.ptestc.256(<4 x i64>, <4 x i64>) nounwind readnone


define i32 @test_x86_avx_ptestnzc_256(<4 x i64> %a0, <4 x i64> %a1) {
  ; CHECK: vptest
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.ptestnzc.256(<4 x i64> %a0, <4 x i64> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.ptestnzc.256(<4 x i64>, <4 x i64>) nounwind readnone


define i32 @test_x86_avx_ptestz_256(<4 x i64> %a0, <4 x i64> %a1) {
  ; CHECK: vptest
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.ptestz.256(<4 x i64> %a0, <4 x i64> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.ptestz.256(<4 x i64>, <4 x i64>) nounwind readnone


define <8 x float> @test_x86_avx_rcp_ps_256(<8 x float> %a0) {
  ; CHECK: vrcpps
  %res = call <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float> %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone


define <4 x double> @test_x86_avx_round_pd_256(<4 x double> %a0) {
  ; CHECK: vroundpd
  %res = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %a0, i32 7) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) nounwind readnone


define <8 x float> @test_x86_avx_round_ps_256(<8 x float> %a0) {
  ; CHECK: vroundps
  %res = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %a0, i32 7) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.round.ps.256(<8 x float>, i32) nounwind readnone


define <8 x float> @test_x86_avx_rsqrt_ps_256(<8 x float> %a0) {
  ; CHECK: vrsqrtps
  %res = call <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float> %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>) nounwind readnone


define <4 x double> @test_x86_avx_sqrt_pd_256(<4 x double> %a0) {
  ; CHECK: vsqrtpd
  %res = call <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double> %a0) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double>) nounwind readnone


define <8 x float> @test_x86_avx_sqrt_ps_256(<8 x float> %a0) {
  ; CHECK: vsqrtps
  %res = call <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float> %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float>) nounwind readnone


define void @test_x86_avx_storeu_dq_256(i8* %a0, <32 x i8> %a1) {
  ; FIXME: unfortunately the execution domain fix pass changes this to vmovups and its hard to force with no 256-bit integer instructions
  ; CHECK: vmovups
  ; add operation forces the execution domain.
  %a2 = add <32 x i8> %a1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  call void @llvm.x86.avx.storeu.dq.256(i8* %a0, <32 x i8> %a2)
  ret void
}
declare void @llvm.x86.avx.storeu.dq.256(i8*, <32 x i8>) nounwind


define void @test_x86_avx_storeu_pd_256(i8* %a0, <4 x double> %a1) {
  ; CHECK: vmovupd
  ; add operation forces the execution domain.
  %a2 = fadd <4 x double> %a1, <double 0x0, double 0x0, double 0x0, double 0x0>
  call void @llvm.x86.avx.storeu.pd.256(i8* %a0, <4 x double> %a2)
  ret void
}
declare void @llvm.x86.avx.storeu.pd.256(i8*, <4 x double>) nounwind


define void @test_x86_avx_storeu_ps_256(i8* %a0, <8 x float> %a1) {
  ; CHECK: vmovups
  call void @llvm.x86.avx.storeu.ps.256(i8* %a0, <8 x float> %a1)
  ret void
}
declare void @llvm.x86.avx.storeu.ps.256(i8*, <8 x float>) nounwind


define <4 x double> @test_x86_avx_vbroadcast_sd_256(i8* %a0) {
  ; CHECK: vbroadcastsd
  %res = call <4 x double> @llvm.x86.avx.vbroadcast.sd.256(i8* %a0) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.vbroadcast.sd.256(i8*) nounwind readonly


define <4 x double> @test_x86_avx_vbroadcastf128_pd_256(i8* %a0) {
  ; CHECK: vbroadcastf128
  %res = call <4 x double> @llvm.x86.avx.vbroadcastf128.pd.256(i8* %a0) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.vbroadcastf128.pd.256(i8*) nounwind readonly


define <8 x float> @test_x86_avx_vbroadcastf128_ps_256(i8* %a0) {
  ; CHECK: vbroadcastf128
  %res = call <8 x float> @llvm.x86.avx.vbroadcastf128.ps.256(i8* %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.vbroadcastf128.ps.256(i8*) nounwind readonly


define <4 x float> @test_x86_avx_vbroadcast_ss(i8* %a0) {
  ; CHECK: vbroadcastss
  %res = call <4 x float> @llvm.x86.avx.vbroadcast.ss(i8* %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx.vbroadcast.ss(i8*) nounwind readonly


define <8 x float> @test_x86_avx_vbroadcast_ss_256(i8* %a0) {
  ; CHECK: vbroadcastss
  %res = call <8 x float> @llvm.x86.avx.vbroadcast.ss.256(i8* %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.vbroadcast.ss.256(i8*) nounwind readonly


define <2 x double> @test_x86_avx_vextractf128_pd_256(<4 x double> %a0) {
  ; CHECK: vextractf128
  %res = call <2 x double> @llvm.x86.avx.vextractf128.pd.256(<4 x double> %a0, i8 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx.vextractf128.pd.256(<4 x double>, i8) nounwind readnone


define <4 x float> @test_x86_avx_vextractf128_ps_256(<8 x float> %a0) {
  ; CHECK: vextractf128
  %res = call <4 x float> @llvm.x86.avx.vextractf128.ps.256(<8 x float> %a0, i8 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx.vextractf128.ps.256(<8 x float>, i8) nounwind readnone


define <4 x i32> @test_x86_avx_vextractf128_si_256(<8 x i32> %a0) {
  ; CHECK: vextractf128
  %res = call <4 x i32> @llvm.x86.avx.vextractf128.si.256(<8 x i32> %a0, i8 7) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx.vextractf128.si.256(<8 x i32>, i8) nounwind readnone


define <4 x double> @test_x86_avx_vinsertf128_pd_256(<4 x double> %a0, <2 x double> %a1) {
  ; CHECK: vinsertf128
  %res = call <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double> %a0, <2 x double> %a1, i8 7) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double>, <2 x double>, i8) nounwind readnone


define <8 x float> @test_x86_avx_vinsertf128_ps_256(<8 x float> %a0, <4 x float> %a1) {
  ; CHECK: vinsertf128
  %res = call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> %a0, <4 x float> %a1, i8 7) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float>, <4 x float>, i8) nounwind readnone


define <8 x i32> @test_x86_avx_vinsertf128_si_256(<8 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vinsertf128
  %res = call <8 x i32> @llvm.x86.avx.vinsertf128.si.256(<8 x i32> %a0, <4 x i32> %a1, i8 7) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx.vinsertf128.si.256(<8 x i32>, <4 x i32>, i8) nounwind readnone


define <4 x double> @test_x86_avx_vperm2f128_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vperm2f128
  %res = call <4 x double> @llvm.x86.avx.vperm2f128.pd.256(<4 x double> %a0, <4 x double> %a1, i8 7) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.vperm2f128.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone


define <8 x float> @test_x86_avx_vperm2f128_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vperm2f128
  %res = call <8 x float> @llvm.x86.avx.vperm2f128.ps.256(<8 x float> %a0, <8 x float> %a1, i8 7) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.vperm2f128.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone


define <8 x i32> @test_x86_avx_vperm2f128_si_256(<8 x i32> %a0, <8 x i32> %a1) {
  ; CHECK: vperm2f128
  %res = call <8 x i32> @llvm.x86.avx.vperm2f128.si.256(<8 x i32> %a0, <8 x i32> %a1, i8 7) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx.vperm2f128.si.256(<8 x i32>, <8 x i32>, i8) nounwind readnone


define <2 x double> @test_x86_avx_vpermil_pd(<2 x double> %a0) {
  ; CHECK: vpermilpd
  %res = call <2 x double> @llvm.x86.avx.vpermil.pd(<2 x double> %a0, i8 7) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx.vpermil.pd(<2 x double>, i8) nounwind readnone


define <4 x double> @test_x86_avx_vpermil_pd_256(<4 x double> %a0) {
  ; CHECK: vpermilpd
  %res = call <4 x double> @llvm.x86.avx.vpermil.pd.256(<4 x double> %a0, i8 7) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.vpermil.pd.256(<4 x double>, i8) nounwind readnone


define <4 x float> @test_x86_avx_vpermil_ps(<4 x float> %a0) {
  ; CHECK: vpermilps
  %res = call <4 x float> @llvm.x86.avx.vpermil.ps(<4 x float> %a0, i8 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx.vpermil.ps(<4 x float>, i8) nounwind readnone


define <8 x float> @test_x86_avx_vpermil_ps_256(<8 x float> %a0) {
  ; CHECK: vpermilps
  %res = call <8 x float> @llvm.x86.avx.vpermil.ps.256(<8 x float> %a0, i8 7) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.vpermil.ps.256(<8 x float>, i8) nounwind readnone


define <2 x double> @test_x86_avx_vpermilvar_pd(<2 x double> %a0, <2 x i64> %a1) {
  ; CHECK: vpermilpd
  %res = call <2 x double> @llvm.x86.avx.vpermilvar.pd(<2 x double> %a0, <2 x i64> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx.vpermilvar.pd(<2 x double>, <2 x i64>) nounwind readnone


define <4 x double> @test_x86_avx_vpermilvar_pd_256(<4 x double> %a0, <4 x i64> %a1) {
  ; CHECK: vpermilpd
  %res = call <4 x double> @llvm.x86.avx.vpermilvar.pd.256(<4 x double> %a0, <4 x i64> %a1) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.vpermilvar.pd.256(<4 x double>, <4 x i64>) nounwind readnone


define <4 x float> @test_x86_avx_vpermilvar_ps(<4 x float> %a0, <4 x i32> %a1) {
  ; CHECK: vpermilps
  %res = call <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float> %a0, <4 x i32> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
define <4 x float> @test_x86_avx_vpermilvar_ps_load(<4 x float> %a0, <4 x i32>* %a1) {
  ; CHECK: vpermilps
  %a2 = load <4 x i32>* %a1
  %res = call <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float> %a0, <4 x i32> %a2) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float>, <4 x i32>) nounwind readnone


define <8 x float> @test_x86_avx_vpermilvar_ps_256(<8 x float> %a0, <8 x i32> %a1) {
  ; CHECK: vpermilps
  %res = call <8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float> %a0, <8 x i32> %a1) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float>, <8 x i32>) nounwind readnone


define i32 @test_x86_avx_vtestc_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vtestpd
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.avx.vtestc.pd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestc.pd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_avx_vtestc_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vtestpd
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.avx.vtestc.pd.256(<4 x double> %a0, <4 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestc.pd.256(<4 x double>, <4 x double>) nounwind readnone


define i32 @test_x86_avx_vtestc_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vtestps
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.avx.vtestc.ps(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestc.ps(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_avx_vtestc_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vtestps
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.avx.vtestc.ps.256(<8 x float> %a0, <8 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestc.ps.256(<8 x float>, <8 x float>) nounwind readnone


define i32 @test_x86_avx_vtestnzc_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vtestpd
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestnzc.pd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestnzc.pd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_avx_vtestnzc_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vtestpd
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestnzc.pd.256(<4 x double> %a0, <4 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestnzc.pd.256(<4 x double>, <4 x double>) nounwind readnone


define i32 @test_x86_avx_vtestnzc_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vtestps
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestnzc.ps(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestnzc.ps(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_avx_vtestnzc_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vtestps
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestnzc.ps.256(<8 x float> %a0, <8 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestnzc.ps.256(<8 x float>, <8 x float>) nounwind readnone


define i32 @test_x86_avx_vtestz_pd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vtestpd
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestz.pd(<2 x double> %a0, <2 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestz.pd(<2 x double>, <2 x double>) nounwind readnone


define i32 @test_x86_avx_vtestz_pd_256(<4 x double> %a0, <4 x double> %a1) {
  ; CHECK: vtestpd
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestz.pd.256(<4 x double> %a0, <4 x double> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestz.pd.256(<4 x double>, <4 x double>) nounwind readnone


define i32 @test_x86_avx_vtestz_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vtestps
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestz.ps(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestz.ps(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_avx_vtestz_ps_256(<8 x float> %a0, <8 x float> %a1) {
  ; CHECK: vtestps
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.avx.vtestz.ps.256(<8 x float> %a0, <8 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.avx.vtestz.ps.256(<8 x float>, <8 x float>) nounwind readnone


define void @test_x86_avx_vzeroall() {
  ; CHECK: vzeroall
  call void @llvm.x86.avx.vzeroall()
  ret void
}
declare void @llvm.x86.avx.vzeroall() nounwind


define void @test_x86_avx_vzeroupper() {
  ; CHECK: vzeroupper
  call void @llvm.x86.avx.vzeroupper()
  ret void
}
declare void @llvm.x86.avx.vzeroupper() nounwind

; Make sure instructions with no AVX equivalents, but are associated with SSEX feature flags still work

; CHECK: monitor
define void @monitor(i8* %P, i32 %E, i32 %H) nounwind {
entry:
  tail call void @llvm.x86.sse3.monitor(i8* %P, i32 %E, i32 %H)
  ret void
}
declare void @llvm.x86.sse3.monitor(i8*, i32, i32) nounwind

; CHECK: mwait
define void @mwait(i32 %E, i32 %H) nounwind {
entry:
  tail call void @llvm.x86.sse3.mwait(i32 %E, i32 %H)
  ret void
}
declare void @llvm.x86.sse3.mwait(i32, i32) nounwind

; CHECK: sfence
define void @sfence() nounwind {
entry:
  tail call void @llvm.x86.sse.sfence()
  ret void
}
declare void @llvm.x86.sse.sfence() nounwind

; CHECK: lfence
define void @lfence() nounwind {
entry:
  tail call void @llvm.x86.sse2.lfence()
  ret void
}
declare void @llvm.x86.sse2.lfence() nounwind

; CHECK: mfence
define void @mfence() nounwind {
entry:
  tail call void @llvm.x86.sse2.mfence()
  ret void
}
declare void @llvm.x86.sse2.mfence() nounwind

; CHECK: clflush
define void @clflush(i8* %p) nounwind {
entry:
  tail call void @llvm.x86.sse2.clflush(i8* %p)
  ret void
}
declare void @llvm.x86.sse2.clflush(i8*) nounwind

; CHECK: crc32b
define i32 @crc32_32_8(i32 %a, i8 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.32.8(i32 %a, i8 %b)
  ret i32 %tmp
}
declare i32 @llvm.x86.sse42.crc32.32.8(i32, i8) nounwind

; CHECK: crc32w
define i32 @crc32_32_16(i32 %a, i16 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.32.16(i32 %a, i16 %b)
  ret i32 %tmp
}
declare i32 @llvm.x86.sse42.crc32.32.16(i32, i16) nounwind

; CHECK: crc32l
define i32 @crc32_32_32(i32 %a, i32 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.32.32(i32 %a, i32 %b)
  ret i32 %tmp
}
declare i32 @llvm.x86.sse42.crc32.32.32(i32, i32) nounwind

; CHECK: movntdq
define void @movnt_dq(i8* %p, <4 x i64> %a1) nounwind {
  %a2 = add <4 x i64> %a1, <i64 1, i64 1, i64 1, i64 1>
  tail call void @llvm.x86.avx.movnt.dq.256(i8* %p, <4 x i64> %a2) nounwind
  ret void
}
declare void @llvm.x86.avx.movnt.dq.256(i8*, <4 x i64>) nounwind

; CHECK: movntps
define void @movnt_ps(i8* %p, <8 x float> %a) nounwind {
  tail call void @llvm.x86.avx.movnt.ps.256(i8* %p, <8 x float> %a) nounwind
  ret void
}
declare void @llvm.x86.avx.movnt.ps.256(i8*, <8 x float>) nounwind

; CHECK: movntpd
define void @movnt_pd(i8* %p, <4 x double> %a1) nounwind {
  ; add operation forces the execution domain.
  %a2 = fadd <4 x double> %a1, <double 0x0, double 0x0, double 0x0, double 0x0>
  tail call void @llvm.x86.avx.movnt.pd.256(i8* %p, <4 x double> %a2) nounwind
  ret void
}
declare void @llvm.x86.avx.movnt.pd.256(i8*, <4 x double>) nounwind


; Check for pclmulqdq
define <2 x i64> @test_x86_pclmulqdq(<2 x i64> %a0, <2 x i64> %a1) {
; CHECK: vpclmulqdq
  %res = call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %a0, <2 x i64> %a1, i8 0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.pclmulqdq(<2 x i64>, <2 x i64>, i8) nounwind readnone
