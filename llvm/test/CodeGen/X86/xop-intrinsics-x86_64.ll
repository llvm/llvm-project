; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mattr=+avx,+fma4,+xop | FileCheck %s

define <2 x double> @test_int_x86_xop_vpermil2pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: vpermil2pd
  %res = call <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 1) ;  [#uses=1]
  ret <2 x double> %res
}
define <2 x double> @test_int_x86_xop_vpermil2pd_mr(<2 x double> %a0, <2 x double>* %a1, <2 x double> %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpermil2pd
  %vec = load <2 x double>* %a1
  %res = call <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double> %a0, <2 x double> %vec, <2 x double> %a2, i8 1) ;  [#uses=1]
  ret <2 x double> %res
}
define <2 x double> @test_int_x86_xop_vpermil2pd_rm(<2 x double> %a0, <2 x double> %a1, <2 x double>* %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpermil2pd
  %vec = load <2 x double>* %a2
  %res = call <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %vec, i8 1) ;  [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <4 x double> @test_int_x86_xop_vpermil2pd_256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK: vpermil2pd
  ; CHECK: ymm
  %res = call <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 2) ;
  ret <4 x double> %res
}
define <4 x double> @test_int_x86_xop_vpermil2pd_256_mr(<4 x double> %a0, <4 x double>* %a1, <4 x double> %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpermil2pd
  ; CHECK: ymm
  %vec = load <4 x double>* %a1
  %res = call <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double> %a0, <4 x double> %vec, <4 x double> %a2, i8 2) ;
  ret <4 x double> %res
}
define <4 x double> @test_int_x86_xop_vpermil2pd_256_rm(<4 x double> %a0, <4 x double> %a1, <4 x double>* %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpermil2pd
  ; CHECK: ymm
  %vec = load <4 x double>* %a2
  %res = call <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %vec, i8 2) ;
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x float> @test_int_x86_xop_vpermil2ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: vpermil2ps
  %res = call <4 x float> @llvm.x86.xop.vpermil2ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 3) ;
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.xop.vpermil2ps(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <8 x float> @test_int_x86_xop_vpermil2ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: vpermil2ps
  ; CHECK: ymm
  %res = call <8 x float> @llvm.x86.xop.vpermil2ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 4) ;
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.xop.vpermil2ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcmov(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> %a2) {
  ; CHECK: vpcmov
  %res = call <2 x i64> @llvm.x86.xop.vpcmov(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> %a2) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcmov(<2 x i64>, <2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @test_int_x86_xop_vpcmov_256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> %a2) {
  ; CHECK: vpcmov
  ; CHECK: ymm
  %res = call <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> %a2) ;
  ret <4 x i64> %res
}
define <4 x i64> @test_int_x86_xop_vpcmov_256_mr(<4 x i64> %a0, <4 x i64>* %a1, <4 x i64> %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpcmov
  ; CHECK: ymm
  %vec = load <4 x i64>* %a1
  %res = call <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64> %a0, <4 x i64> %vec, <4 x i64> %a2) ;
  ret <4 x i64> %res
}
define <4 x i64> @test_int_x86_xop_vpcmov_256_rm(<4 x i64> %a0, <4 x i64> %a1, <4 x i64>* %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpcmov
  ; CHECK: ymm
 %vec = load <4 x i64>* %a2
 %res = call <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> %vec) ;
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64>, <4 x i64>, <4 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomeqb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK:vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomeqb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
define <16 x i8> @test_int_x86_xop_vpcomeqb_mem(<16 x i8> %a0, <16 x i8>* %a1) {
  ; CHECK-NOT: vmovaps
  ; CHECK:vpcomb
  %vec = load <16 x i8>* %a1
  %res = call <16 x i8> @llvm.x86.xop.vpcomeqb(<16 x i8> %a0, <16 x i8> %vec) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomeqb(<16 x i8>, <16 x i8>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomeqw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomeqw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomeqw(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomeqd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomeqd(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomeqd(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomeqq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomeqq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomeqq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomequb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomequb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomequb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomequd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomequd(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomequd(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomequq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomequq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomequq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomequw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomequw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomequw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomfalseb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomfalseb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomfalseb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomfalsed(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomfalsed(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomfalsed(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomfalseq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomfalseq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomfalseq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomfalseub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomfalseub(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomfalseub(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomfalseud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomfalseud(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomfalseud(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomfalseuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomfalseuq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomfalseuq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomfalseuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomfalseuw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomfalseuw(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomfalsew(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomfalsew(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomfalsew(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomgeb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomgeb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomgeb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomged(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomged(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomged(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomgeq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomgeq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomgeq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomgeub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomgeub(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomgeub(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomgeud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomgeud(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomgeud(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomgeuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomgeuq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomgeuq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomgeuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomgeuw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomgeuw(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomgew(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomgew(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomgew(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomgtb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomgtb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomgtb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomgtd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomgtd(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomgtd(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomgtq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomgtq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomgtq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomgtub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomgtub(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomgtub(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomgtud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomgtud(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomgtud(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomgtuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomgtuq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomgtuq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomgtuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomgtuw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomgtuw(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomgtw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomgtw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomgtw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomleb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomleb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomleb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomled(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomled(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomled(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomleq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomleq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomleq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomleub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomleub(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomleub(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomleud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomleud(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomleud(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomleuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomleuq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomleuq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomleuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomleuw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomleuw(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomlew(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomlew(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomlew(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomltb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomltb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomltb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomltd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomltd(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomltd(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomltq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomltq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomltq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomltub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomltub(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomltub(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomltud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomltud(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomltud(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomltuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomltuq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomltuq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomltuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomltuw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomltuw(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomltw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomltw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomltw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomneb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomneb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomneb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomned(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomned(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomned(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomneq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomneq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomneq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomneub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomneub(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomneub(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomneud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomneud(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomneud(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomneuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomneuq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomneuq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomneuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomneuw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomneuw(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomnew(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomnew(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomnew(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomtrueb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomtrueb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomtrueb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomtrued(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomtrued(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomtrued(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomtrueq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomtrueq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomtrueq(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomtrueub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomtrueub(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomtrueub(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomtrueud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomtrueud(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomtrueud(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomtrueuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomtrueuq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomtrueuq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomtrueuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomtrueuw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomtrueuw(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomtruew(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomtruew(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomtruew(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vphaddbd(<16 x i8> %a0) {
  ; CHECK: vphaddbd
  %res = call <4 x i32> @llvm.x86.xop.vphaddbd(<16 x i8> %a0) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vphaddbd(<16 x i8>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vphaddbq(<16 x i8> %a0) {
  ; CHECK: vphaddbq
  %res = call <2 x i64> @llvm.x86.xop.vphaddbq(<16 x i8> %a0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vphaddbq(<16 x i8>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vphaddbw(<16 x i8> %a0) {
  ; CHECK: vphaddbw
  %res = call <8 x i16> @llvm.x86.xop.vphaddbw(<16 x i8> %a0) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vphaddbw(<16 x i8>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vphadddq(<4 x i32> %a0) {
  ; CHECK: vphadddq
  %res = call <2 x i64> @llvm.x86.xop.vphadddq(<4 x i32> %a0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vphadddq(<4 x i32>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vphaddubd(<16 x i8> %a0) {
  ; CHECK: vphaddubd
  %res = call <4 x i32> @llvm.x86.xop.vphaddubd(<16 x i8> %a0) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vphaddubd(<16 x i8>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vphaddubq(<16 x i8> %a0) {
  ; CHECK: vphaddubq
  %res = call <2 x i64> @llvm.x86.xop.vphaddubq(<16 x i8> %a0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vphaddubq(<16 x i8>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vphaddubw(<16 x i8> %a0) {
  ; CHECK: vphaddubw
  %res = call <8 x i16> @llvm.x86.xop.vphaddubw(<16 x i8> %a0) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vphaddubw(<16 x i8>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vphaddudq(<4 x i32> %a0) {
  ; CHECK: vphaddudq
  %res = call <2 x i64> @llvm.x86.xop.vphaddudq(<4 x i32> %a0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vphaddudq(<4 x i32>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vphadduwd(<8 x i16> %a0) {
  ; CHECK: vphadduwd
  %res = call <4 x i32> @llvm.x86.xop.vphadduwd(<8 x i16> %a0) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vphadduwd(<8 x i16>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vphadduwq(<8 x i16> %a0) {
  ; CHECK: vphadduwq
  %res = call <2 x i64> @llvm.x86.xop.vphadduwq(<8 x i16> %a0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vphadduwq(<8 x i16>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vphaddwd(<8 x i16> %a0) {
  ; CHECK: vphaddwd
  %res = call <4 x i32> @llvm.x86.xop.vphaddwd(<8 x i16> %a0) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vphaddwd(<8 x i16>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vphaddwq(<8 x i16> %a0) {
  ; CHECK: vphaddwq
  %res = call <2 x i64> @llvm.x86.xop.vphaddwq(<8 x i16> %a0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vphaddwq(<8 x i16>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vphsubbw(<16 x i8> %a0) {
  ; CHECK: vphsubbw
  %res = call <8 x i16> @llvm.x86.xop.vphsubbw(<16 x i8> %a0) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vphsubbw(<16 x i8>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vphsubdq(<4 x i32> %a0) {
  ; CHECK: vphsubdq
  %res = call <2 x i64> @llvm.x86.xop.vphsubdq(<4 x i32> %a0) ;
  ret <2 x i64> %res
}
define <2 x i64> @test_int_x86_xop_vphsubdq_mem(<4 x i32>* %a0) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vphsubdq
  %vec = load <4 x i32>* %a0
  %res = call <2 x i64> @llvm.x86.xop.vphsubdq(<4 x i32> %vec) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vphsubdq(<4 x i32>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vphsubwd(<8 x i16> %a0) {
  ; CHECK: vphsubwd
  %res = call <4 x i32> @llvm.x86.xop.vphsubwd(<8 x i16> %a0) ;
  ret <4 x i32> %res
}
define <4 x i32> @test_int_x86_xop_vphsubwd_mem(<8 x i16>* %a0) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vphsubwd
  %vec = load <8 x i16>* %a0
  %res = call <4 x i32> @llvm.x86.xop.vphsubwd(<8 x i16> %vec) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vphsubwd(<8 x i16>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpmacsdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2) {
  ; CHECK: vpmacsdd
  %res = call <4 x i32> @llvm.x86.xop.vpmacsdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpmacsdd(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpmacsdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ; CHECK: vpmacsdqh
  %res = call <2 x i64> @llvm.x86.xop.vpmacsdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpmacsdqh(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpmacsdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ; CHECK: vpmacsdql
  %res = call <2 x i64> @llvm.x86.xop.vpmacsdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpmacsdql(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpmacssdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2) {
  ; CHECK: vpmacssdd
  %res = call <4 x i32> @llvm.x86.xop.vpmacssdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpmacssdd(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpmacssdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ; CHECK: vpmacssdqh
  %res = call <2 x i64> @llvm.x86.xop.vpmacssdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpmacssdqh(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpmacssdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ; CHECK: vpmacssdql
  %res = call <2 x i64> @llvm.x86.xop.vpmacssdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpmacssdql(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpmacsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ; CHECK: vpmacsswd
  %res = call <4 x i32> @llvm.x86.xop.vpmacsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpmacsswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpmacssww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2) {
  ; CHECK: vpmacssww
  %res = call <8 x i16> @llvm.x86.xop.vpmacssww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpmacssww(<8 x i16>, <8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpmacswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ; CHECK: vpmacswd
  %res = call <4 x i32> @llvm.x86.xop.vpmacswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpmacswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpmacsww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2) {
  ; CHECK: vpmacsww
  %res = call <8 x i16> @llvm.x86.xop.vpmacsww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpmacsww(<8 x i16>, <8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpmadcsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ; CHECK: vpmadcsswd
  %res = call <4 x i32> @llvm.x86.xop.vpmadcsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpmadcsswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpmadcswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ; CHECK: vpmadcswd
  %res = call <4 x i32> @llvm.x86.xop.vpmadcswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) ;
  ret <4 x i32> %res
}
define <4 x i32> @test_int_x86_xop_vpmadcswd_mem(<8 x i16> %a0, <8 x i16>* %a1, <4 x i32> %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpmadcswd
  %vec = load <8 x i16>* %a1
  %res = call <4 x i32> @llvm.x86.xop.vpmadcswd(<8 x i16> %a0, <8 x i16> %vec, <4 x i32> %a2) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpmadcswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpperm(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2) {
  ; CHECK: vpperm
  %res = call <16 x i8> @llvm.x86.xop.vpperm(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2) ;
  ret <16 x i8> %res
}
define <16 x i8> @test_int_x86_xop_vpperm_rm(<16 x i8> %a0, <16 x i8> %a1, <16 x i8>* %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpperm
  %vec = load <16 x i8>* %a2
  %res = call <16 x i8> @llvm.x86.xop.vpperm(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %vec) ;
  ret <16 x i8> %res
}
define <16 x i8> @test_int_x86_xop_vpperm_mr(<16 x i8> %a0, <16 x i8>* %a1, <16 x i8> %a2) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpperm
  %vec = load <16 x i8>* %a1
  %res = call <16 x i8> @llvm.x86.xop.vpperm(<16 x i8> %a0, <16 x i8> %vec, <16 x i8> %a2) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpperm(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vprotb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vprotb
  %res = call <16 x i8> @llvm.x86.xop.vprotb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vprotb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vprotd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vprotd
  %res = call <4 x i32> @llvm.x86.xop.vprotd(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vprotd(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vprotq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vprotq
  %res = call <2 x i64> @llvm.x86.xop.vprotq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vprotq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vprotw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vprotw
  %res = call <8 x i16> @llvm.x86.xop.vprotw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vprotw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpshab(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpshab
  %res = call <16 x i8> @llvm.x86.xop.vpshab(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpshab(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpshad(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpshad
  %res = call <4 x i32> @llvm.x86.xop.vpshad(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpshad(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpshaq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpshaq
  %res = call <2 x i64> @llvm.x86.xop.vpshaq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpshaq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpshaw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpshaw
  %res = call <8 x i16> @llvm.x86.xop.vpshaw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpshaw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpshlb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: vpshlb
  %res = call <16 x i8> @llvm.x86.xop.vpshlb(<16 x i8> %a0, <16 x i8> %a1) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpshlb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpshld(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpshld
  %res = call <4 x i32> @llvm.x86.xop.vpshld(<4 x i32> %a0, <4 x i32> %a1) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpshld(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpshlq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpshlq
  %res = call <2 x i64> @llvm.x86.xop.vpshlq(<2 x i64> %a0, <2 x i64> %a1) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpshlq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpshlw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpshlw
  %res = call <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16> %a0, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
define <8 x i16> @test_int_x86_xop_vpshlw_rm(<8 x i16> %a0, <8 x i16>* %a1) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpshlw
  %vec = load <8 x i16>* %a1
  %res = call <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16> %a0, <8 x i16> %vec) ;
  ret <8 x i16> %res
}
define <8 x i16> @test_int_x86_xop_vpshlw_mr(<8 x i16>* %a0, <8 x i16> %a1) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vpshlw
  %vec = load <8 x i16>* %a0
  %res = call <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16> %vec, <8 x i16> %a1) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x float> @test_int_x86_xop_vfrcz_ss(<4 x float> %a0) {
  ; CHECK-NOT: mov
  ; CHECK: vfrczss
  %res = call <4 x float> @llvm.x86.xop.vfrcz.ss(<4 x float> %a0) ;
  ret <4 x float> %res
}
define <4 x float> @test_int_x86_xop_vfrcz_ss_mem(float* %a0) {
  ; CHECK-NOT: mov
  ; CHECK: vfrczss
  %elem = load float* %a0
  %vec = insertelement <4 x float> undef, float %elem, i32 0
  %res = call <4 x float> @llvm.x86.xop.vfrcz.ss(<4 x float> %vec) ;
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.xop.vfrcz.ss(<4 x float>) nounwind readnone

define <2 x double> @test_int_x86_xop_vfrcz_sd(<2 x double> %a0) {
  ; CHECK-NOT: mov
  ; CHECK: vfrczsd
  %res = call <2 x double> @llvm.x86.xop.vfrcz.sd(<2 x double> %a0) ;
  ret <2 x double> %res
}
define <2 x double> @test_int_x86_xop_vfrcz_sd_mem(double* %a0) {
  ; CHECK-NOT: mov
  ; CHECK: vfrczsd
  %elem = load double* %a0
  %vec = insertelement <2 x double> undef, double %elem, i32 0
  %res = call <2 x double> @llvm.x86.xop.vfrcz.sd(<2 x double> %vec) ;
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.xop.vfrcz.sd(<2 x double>) nounwind readnone

define <2 x double> @test_int_x86_xop_vfrcz_pd(<2 x double> %a0) {
  ; CHECK: vfrczpd
  %res = call <2 x double> @llvm.x86.xop.vfrcz.pd(<2 x double> %a0) ;
  ret <2 x double> %res
}
define <2 x double> @test_int_x86_xop_vfrcz_pd_mem(<2 x double>* %a0) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vfrczpd
  %vec = load <2 x double>* %a0
  %res = call <2 x double> @llvm.x86.xop.vfrcz.pd(<2 x double> %vec) ;
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.xop.vfrcz.pd(<2 x double>) nounwind readnone

define <4 x double> @test_int_x86_xop_vfrcz_pd_256(<4 x double> %a0) {
  ; CHECK: vfrczpd
  ; CHECK: ymm
  %res = call <4 x double> @llvm.x86.xop.vfrcz.pd.256(<4 x double> %a0) ;
  ret <4 x double> %res
}
define <4 x double> @test_int_x86_xop_vfrcz_pd_256_mem(<4 x double>* %a0) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vfrczpd
  ; CHECK: ymm
  %vec = load <4 x double>* %a0
  %res = call <4 x double> @llvm.x86.xop.vfrcz.pd.256(<4 x double> %vec) ;
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.xop.vfrcz.pd.256(<4 x double>) nounwind readnone

define <4 x float> @test_int_x86_xop_vfrcz_ps(<4 x float> %a0) {
  ; CHECK: vfrczps
  %res = call <4 x float> @llvm.x86.xop.vfrcz.ps(<4 x float> %a0) ;
  ret <4 x float> %res
}
define <4 x float> @test_int_x86_xop_vfrcz_ps_mem(<4 x float>* %a0) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vfrczps
  %vec = load <4 x float>* %a0
  %res = call <4 x float> @llvm.x86.xop.vfrcz.ps(<4 x float> %vec) ;
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.xop.vfrcz.ps(<4 x float>) nounwind readnone

define <8 x float> @test_int_x86_xop_vfrcz_ps_256(<8 x float> %a0) {
  ; CHECK: vfrczps
  ; CHECK: ymm
  %res = call <8 x float> @llvm.x86.xop.vfrcz.ps.256(<8 x float> %a0) ;
  ret <8 x float> %res
}
define <8 x float> @test_int_x86_xop_vfrcz_ps_256_mem(<8 x float>* %a0) {
  ; CHECK-NOT: vmovaps
  ; CHECK: vfrczps
  ; CHECK: ymm
  %vec = load <8 x float>* %a0
  %res = call <8 x float> @llvm.x86.xop.vfrcz.ps.256(<8 x float> %vec) ;
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.xop.vfrcz.ps.256(<8 x float>) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomb(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK:vpcomb
  %res = call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %a0, <16 x i8> %a1, i8 0) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8>, <16 x i8>, i8) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomw
  %res = call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %a0, <8 x i16> %a1, i8 0) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16>, <8 x i16>, i8) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomd(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomd
  %res = call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %a0, <4 x i32> %a1, i8 0) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32>, <4 x i32>, i8) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomq
  %res = call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %a0, <2 x i64> %a1, i8 0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <16 x i8> @test_int_x86_xop_vpcomub(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK:vpcomub
  %res = call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %a0, <16 x i8> %a1, i8 0) ;
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8>, <16 x i8>, i8) nounwind readnone

define <8 x i16> @test_int_x86_xop_vpcomuw(<8 x i16> %a0, <8 x i16> %a1) {
  ; CHECK: vpcomuw
  %res = call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %a0, <8 x i16> %a1, i8 0) ;
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16>, <8 x i16>, i8) nounwind readnone

define <4 x i32> @test_int_x86_xop_vpcomud(<4 x i32> %a0, <4 x i32> %a1) {
  ; CHECK: vpcomud
  %res = call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %a0, <4 x i32> %a1, i8 0) ;
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32>, <4 x i32>, i8) nounwind readnone

define <2 x i64> @test_int_x86_xop_vpcomuq(<2 x i64> %a0, <2 x i64> %a1) {
  ; CHECK: vpcomuq
  %res = call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %a0, <2 x i64> %a1, i8 0) ;
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64>, <2 x i64>, i8) nounwind readnone

