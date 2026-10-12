//===- XtensaS3DSPInstrInfo.td - Xtensa Target Description -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Xtensa S3 DSP instructions in TableGen format.
//
// These definitions are generated
// This file is generated
// AI_S6_V2.h
//
//===----------------------------------------------------------------------===//

// This file is generated


def EE_ANDQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.andq\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-12} = 0x3;
  let Inst{11-10} = qy{2-1};
  let Inst{9-8} = 0x0;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

def EE_BITREV:  EE_Inst24<(outs QR:$qa, AR:$axr), (ins AR:$ax),
                   "ee.bitrev\t $qa, $ax", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<4> ax;


  let Constraints = "$axr = $ax";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-8} = 0x7b;
  let Inst{7-4} = ax{3-0};
  let Inst{3-0} = 0x4;
}

def EE_CMUL_S16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy, select_4:$sel4),
                   "ee.cmul.s16\t $qz, $qx, $qy, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-6} = 0x0;
  let Inst{5-4} = sel4{1-0};
  let Inst{3-0} = 0x4;
}

def EE_CMUL_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy, select_4:$sel4),
                   "ee.cmul.s16.ld.incp\t $qu, $as, $qz, $qx, $qy, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{9-8} = sel4{1-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_CMUL_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy, select_4:$sel4),
                   "ee.cmul.s16.st.incp\t $qv, $as, $qz, $qx, $qy, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{9-8} = sel4{1-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_FFT_AMS_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz, QR:$qz1), (ins AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.ld.incp\t $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qz1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = sel2{0};
  let Inst{24} = qz1{2};
  let Inst{23} = qz{0};
  let Inst{22-20} = qy{2-0};
  let Inst{19} = qz1{1};
  let Inst{18-16} = qm{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qz{2-1};
  let Inst{11} = qz1{0};
  let Inst{10-8} = qu{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_FFT_AMS_S16_LD_INCP_UAUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz, QR:$qz1), (ins AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.ld.incp.uaup\t $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qz1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = sel2{0};
  let Inst{24} = qz1{2};
  let Inst{23} = qz{0};
  let Inst{22-20} = qy{2-0};
  let Inst{19} = qz1{1};
  let Inst{18-16} = qm{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qz{2-1};
  let Inst{11} = qz1{0};
  let Inst{10-8} = qu{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_FFT_AMS_S16_LD_R32_DECP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz, QR:$qz1), (ins AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.ld.r32.decp\t $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qz1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = sel2{0};
  let Inst{24} = qz1{2};
  let Inst{23} = qz{0};
  let Inst{22-20} = qy{2-0};
  let Inst{19} = qz1{1};
  let Inst{18-16} = qm{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qz{2-1};
  let Inst{11} = qz1{0};
  let Inst{10-8} = qu{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_FFT_AMS_S16_ST_INCP:  EE_Inst32<(outs QR:$qz1, AR:$as0r, AR:$asr), (ins QR:$qv, AR:$as0, AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.st.incp\t $qv, $qz1, $as0, $as, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<3> qz1;
  bits<4> as0;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;

  let mayStore = 1;

  let Constraints = "$as0r = $as0, $asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = sel2{0};
  let Inst{25-24} = qz1{2-1};
  let Inst{23} = qm{0};
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qz1{0};
  let Inst{18-16} = qx{2-0};
  let Inst{15-14} = qy{1-0};
  let Inst{13-12} = qm{2-1};
  let Inst{11-8} = as{3-0};
  let Inst{7-4} = as0{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qy{2};
}

def EE_FFT_CMUL_S16_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, select_8:$sel8),
                   "ee.fft.cmul.s16.ld.xp\t $qu, $as, $ad, $qz, $qx, $qy, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<3> sel8;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25-24} = sel8{2-1};
  let Inst{23} = qu{0};
  let Inst{22-20} = qy{2-0};
  let Inst{19} = sel8{0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qu{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_FFT_CMUL_S16_ST_XP:  EE_Inst32<(outs AR:$asr), (ins QR:$qx, QR:$qy, QR:$qv, AR:$as, AR:$ad, select_8:$sel8, select_4:$upd4, select_4:$sar4),
                   "ee.fft.cmul.s16.st.xp\t $qx, $qy, $qv, $as, $ad, $sel8, $upd4, $sar4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  bits<4> as;
  bits<4> ad;
  bits<3> sel8;
  bits<2> upd4;
  bits<2> sar4;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26-25} = sar4{1-0};
  let Inst{24} = upd4{1};
  let Inst{23} = sel8{0};
  let Inst{22-20} = qy{2-0};
  let Inst{19} = upd4{0};
  let Inst{18-16} = qv{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = sel8{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_FFT_R2BF_S16:  EE_Inst24<(outs QR:$qa0, QR:$qa1), (ins QR:$qx, QR:$qy, select_2:$sel2),
                   "ee.fft.r2bf.s16\t $qa0, $qa1, $qx, $qy, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa0;
  bits<3> qa1;
  bits<3> qx;
  bits<3> qy;
  bits<1> sel2;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa0{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qa0{0};
  let Inst{14-12} = qa1{2-0};
  let Inst{11-10} = qy{2-1};
  let Inst{9} = 0x0;
  let Inst{8} = sel2{0};
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

def EE_FFT_R2BF_S16_ST_INCP:  EE_Inst32<(outs QR:$qa0, AR:$asr), (ins QR:$qx, QR:$qy, AR:$as, select_4:$sar4),
                   "ee.fft.r2bf.s16.st.incp\t $qa0, $qx, $qy, $as, $sar4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa0;
  bits<3> qx;
  bits<3> qy;
  bits<4> as;
  bits<2> sar4;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = sar4{0};
  let Inst{19} = 1;
  let Inst{22-20} = qy{2-0};
  let Inst{18-16} = qa0{2-0};
  let Inst{13} = 0;
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{12} = sar4{1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_FFT_VST_R32_DECP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, select_2:$sar2),
                   "ee.fft.vst.r32.decp\t $qv, $as, $sar2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<1> sar2;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-11} = 0x6;
  let Inst{10} = sar2{0};
  let Inst{9-8} = 0x3;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDF_128_IP:  EE_Inst32<(outs FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, offset_16_16:$imm16f),
                   "ee.ldf.128.ip\t $fu3, $fu2, $fu1, $fu0, $as, $imm16f", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu3;
  bits<4> fu2;
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<4> imm16f;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26-24} = fu3{3-1};
  let Inst{23-20} = fu0{3-0};
  let Inst{19} = fu3{0};
  let Inst{18-16} = fu2{3-1};
  let Inst{15-12} = fu1{3-0};
  let Inst{11-8} = imm16f{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = fu2{0};
}

def EE_LDF_128_XP:  EE_Inst32<(outs FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldf.128.xp\t $fu3, $fu2, $fu1, $fu0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu3;
  bits<4> fu2;
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26-24} = fu3{3-1};
  let Inst{23-20} = fu0{3-0};
  let Inst{19} = fu3{0};
  let Inst{18-16} = fu2{3-1};
  let Inst{15-12} = fu1{3-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = fu2{0};
}

def EE_LDF_64_IP:  EE_Inst32<(outs FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.ldf.64.ip\t $fu1, $fu0, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<8> imm8;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = imm8{7-6};
  let Inst{23-20} = fu0{3-0};
  let Inst{19-16} = imm8{5-2};
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{15-12} = fu1{3-0};
  let Inst{8} = imm8{0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = imm8{1};
}

def EE_LDF_64_XP:  EE_Inst24<(outs FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldf.64.xp\t $fu1, $fu0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-20} = fu0{3-0};
  let Inst{19-16} = 0x6;
  let Inst{15-12} = fu1{3-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x0;
}

def EE_LDQA_S16_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.s16.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x2;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDQA_S16_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.s16.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-12} = 0x7e4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDQA_S8_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.s8.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x22;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDQA_S8_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.s8.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-12} = 0x714;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDQA_U16_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.u16.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0xa;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDQA_U16_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.u16.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-12} = 0x7a4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDQA_U8_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.u8.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x2a;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDQA_U8_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.u8.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-12} = 0x704;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LDXQ_32:  EE_Inst32<(outs QR:$qu), (ins QR:$qs, AR:$as, select_4:$sel4, select_8:$sel8),
                   "ee.ldxq.32\t $qu, $qs, $as, $sel4, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<3> qs;
  bits<4> as;
  bits<2> sel4;
  bits<3> sel8;

  let mayLoad = 1;

  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = sel4{1};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{23} = sel8{0};
  let Inst{19} = sel4{0};
  let Inst{18-16} = qu{2-0};
  let Inst{15-14} = qs{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = sel8{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qs{2};
}

def EE_LD_128_USAR_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.128.usar.ip\t $qu, $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm16{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x1;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LD_128_USAR_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ld.128.usar.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x0;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LD_ACCX_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.ld.accx.ip\t $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm8;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm8{7};
  let Inst{21-15} = 0x1c;
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LD_QACC_H_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.ld.qacc_h.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x3c;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LD_QACC_H_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.qacc_h.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0xc;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LD_QACC_L_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.ld.qacc_l.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x2c;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LD_QACC_L_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.qacc_l.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x0;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_LD_UA_STATE_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.ua_state.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x20;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_MOVI_32_A:  EE_Inst24<(outs AR:$au), (ins QR:$qs, select_4:$sel4),
                   "ee.movi.32.a\t $qs, $au, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  bits<4> au;
  bits<2> sel4;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-12} = 0x7;
  let Inst{11-10} = sel4{1-0};
  let Inst{9-8} = 0x1;
  let Inst{7-4} = au{3-0};
  let Inst{3-0} = 0x4;
}

def EE_MOVI_32_Q:  EE_Inst24<(outs QR:$qu), (ins AR:$as, select_4:$sel4),
                   "ee.movi.32.q\t $qu, $as, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<2> sel4;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x3;
  let Inst{11-10} = sel4{1-0};
  let Inst{9-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_MOV_S16_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.s16.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f24;
}

def EE_MOV_S8_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.s8.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f34;
}

def EE_MOV_U16_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.u16.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f64;
}

def EE_MOV_U8_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.u8.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f74;
}

def EE_NOTQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx),
                   "ee.notq\t $qa, $qx", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-8} = 0x7f;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = 0x0;
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

def EE_ORQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.orq\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-12} = 0x7;
  let Inst{11-10} = qy{2-1};
  let Inst{9-8} = 0x0;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

def EE_SLCI_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r), (ins QR:$qs1, QR:$qs0, select_16:$sar16),
                   "ee.slci.2q\t $qs1, $qs0, $sar16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> sar16;


  let Constraints = "$qs1r = $qs1, $qs0r = $qs0";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = 0x6;
  let Inst{7-4} = sar16{3-0};
  let Inst{3-0} = 0x4;
}

def EE_SLCXXP_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r, AR:$asr), (ins QR:$qs1, QR:$qs0, AR:$as, AR:$ad),
                   "ee.slcxxp.2q\t $qs1, $qs0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> as;
  bits<4> ad;


  let Constraints = "$qs1r = $qs1, $qs0r = $qs0, $asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0x6;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_SRCI_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r), (ins QR:$qs1, QR:$qs0, select_16:$sar16),
                   "ee.srci.2q\t $qs1, $qs0, $sar16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> sar16;


  let Constraints = "$qs1r = $qs1, $qs0r = $qs0";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = 0xa;
  let Inst{7-4} = sar16{3-0};
  let Inst{3-0} = 0x4;
}

def EE_SRCMB_S16_QACC:  EE_Inst24<(outs QR:$qu), (ins AR:$as, select_2:$sel2),
                   "ee.srcmb.s16.qacc\t $qu, $as, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<1> sel2;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-11} = 0xe;
  let Inst{10} = sel2{0};
  let Inst{9-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_SRCMB_S8_QACC:  EE_Inst24<(outs QR:$qu), (ins AR:$as, select_2:$sel2),
                   "ee.srcmb.s8.qacc\t $qu, $as, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<1> sel2;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-11} = 0xf;
  let Inst{10} = sel2{0};
  let Inst{9-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_SRCQ_128_ST_INCP:  EE_Inst24<(outs AR:$asr), (ins QR:$qs0, QR:$qs1, AR:$as),
                   "ee.srcq.128.st.incp\t $qs0, $qs1, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  bits<4> as;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = 0xe;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_SRCXXP_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r, AR:$asr), (ins QR:$qs1, QR:$qs0, AR:$as, AR:$ad),
                   "ee.srcxxp.2q\t $qs1, $qs0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> as;
  bits<4> ad;


  let Constraints = "$qs1r = $qs1, $qs0r = $qs0, $asr = $as";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0x6;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_SRC_Q:  EE_Inst24<(outs QR:$qa), (ins QR:$qs0, QR:$qs1),
                   "ee.src.q\t $qa, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs0;
  bits<3> qs1;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-7} = 0x6;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

def EE_SRC_Q_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_256_16:$imm16, QR:$qs0, QR:$qs1),
                   "ee.src.q.ld.ip\t $qu, $as, $imm16, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = imm16{7-6};
  let Inst{23} = imm16{2};
  let Inst{22-20} = qs1{2-0};
  let Inst{19} = imm16{5};
  let Inst{18-16} = qu{2-0};
  let Inst{15-14} = qs0{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{13-12} = imm16{4-3};
  let Inst{9-8} = imm16{1-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qs0{2};
}

def EE_SRC_Q_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qs0, QR:$qs1),
                   "ee.src.q.ld.xp\t $qu, $as, $ad, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{19} = 0;
  let Inst{22-20} = qs1{2-0};
  let Inst{18-16} = qu{2-0};
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{15-14} = qs0{1-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qs0{2};
}

def EE_SRC_Q_QUP:  EE_Inst24<(outs QR:$qa, QR:$qs0r), (ins QR:$qs0, QR:$qs1),
                   "ee.src.q.qup\t $qa, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs0;
  bits<3> qs1;


  let Constraints = "$qs0r = $qs0";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-7} = 0xe;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

def EE_SRS_ACCX:  EE_Inst24<(outs AR:$au), (ins AR:$as, select_2:$sel2),
                   "ee.srs.accx\t $au, $as, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> au;
  bits<4> as;
  bits<1> sel2;


  let Inst{23-15} = 0xfc;
  let Inst{14} = sel2{0};
  let Inst{13-12} = 0x1;
  let Inst{11-8} = au{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_STF_128_IP:  EE_Inst32<(outs AR:$asr), (ins FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, offset_16_16:$imm16f),
                   "ee.stf.128.ip\t $fv3, $fv2, $fv1, $fv0, $as, $imm16f", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv3;
  bits<4> fv2;
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<4> imm16f;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26-24} = fv3{3-1};
  let Inst{23-20} = fv0{3-0};
  let Inst{19} = fv3{0};
  let Inst{18-16} = fv2{3-1};
  let Inst{15-12} = fv1{3-0};
  let Inst{11-8} = imm16f{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = fv2{0};
}

def EE_STF_128_XP:  EE_Inst32<(outs AR:$asr), (ins FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad),
                   "ee.stf.128.xp\t $fv3, $fv2, $fv1, $fv0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv3;
  bits<4> fv2;
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<4> ad;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26-24} = fv3{3-1};
  let Inst{23-20} = fv0{3-0};
  let Inst{19} = fv3{0};
  let Inst{18-16} = fv2{3-1};
  let Inst{15-12} = fv1{3-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = fv2{0};
}

def EE_STF_64_IP:  EE_Inst32<(outs AR:$asr), (ins FPR:$fv1, FPR:$fv0, AR:$as, offset_256_8:$imm8),
                   "ee.stf.64.ip\t $fv1, $fv0, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<8> imm8;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = imm8{7-6};
  let Inst{23-20} = fv0{3-0};
  let Inst{19-16} = imm8{5-2};
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{15-12} = fv1{3-0};
  let Inst{8} = imm8{0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = imm8{1};
}

def EE_STF_64_XP:  EE_Inst24<(outs AR:$asr), (ins FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad),
                   "ee.stf.64.xp\t $fv1, $fv0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<4> ad;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23-20} = fv0{3-0};
  let Inst{19-16} = 0x7;
  let Inst{15-12} = fv1{3-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x0;
}

def EE_STXQ_32:  EE_Inst32<(outs), (ins QR:$qv, QR:$qs, AR:$as, select_4:$sel4, select_8:$sel8),
                   "ee.stxq.32\t $qv, $qs, $as, $sel4, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<3> qs;
  bits<4> as;
  bits<2> sel4;
  bits<3> sel8;

  let mayStore = 1;

  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = sel4{1};
  let Inst{23} = sel8{0};
  let Inst{22-20} = qv{2-0};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{19} = sel4{0};
  let Inst{15-14} = qs{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = sel8{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qs{2};
}

def EE_ST_ACCX_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.st.accx.ip\t $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm8;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm8{7};
  let Inst{21-15} = 0x4;
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_ST_QACC_H_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.st.qacc_h.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x24;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_ST_QACC_H_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.st.qacc_h.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x1a;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_ST_QACC_L_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.st.qacc_l.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x3a;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_ST_QACC_L_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.st.qacc_l.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x18;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_ST_UA_STATE_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.st.ua_state.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x38;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VADDS_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vadds.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x64;
}

def EE_VADDS_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VADDS_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VADDS_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vadds.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x74;
}

def EE_VADDS_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VADDS_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VADDS_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vadds.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

def EE_VADDS_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VADDS_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VCMP_EQ_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.eq.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x94;
}

def EE_VCMP_EQ_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.eq.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xa4;
}

def EE_VCMP_EQ_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.eq.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xb4;
}

def EE_VCMP_GT_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.gt.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

def EE_VCMP_GT_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.gt.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xd4;
}

def EE_VCMP_GT_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.gt.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xe4;
}

def EE_VCMP_LT_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.lt.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xf4;
}

def EE_VCMP_LT_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.lt.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x4;
}

def EE_VCMP_LT_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.lt.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x14;
}

def EE_VLDBC_16:  EE_Inst24<(outs QR:$qu), (ins AR:$as),
                   "ee.vldbc.16\t $qu, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;

  let mayLoad = 1;

  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-8} = 0x73;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_16_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_128_2:$imm2),
                   "ee.vldbc.16.ip\t $qu, $as, $imm2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<7> imm2;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x5;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm2{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_16_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vldbc.16.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_32:  EE_Inst24<(outs QR:$qu), (ins AR:$as),
                   "ee.vldbc.32\t $qu, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;

  let mayLoad = 1;

  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-8} = 0x77;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_32_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.vldbc.32.ip\t $qu, $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm4;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm4{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x2;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_32_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vldbc.32.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x1;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_8:  EE_Inst24<(outs QR:$qu), (ins AR:$as),
                   "ee.vldbc.8\t $qu, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;

  let mayLoad = 1;

  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-8} = 0x3b;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_8_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_128_1:$imm1),
                   "ee.vldbc.8.ip\t $qu, $as, $imm1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<7> imm1;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x5;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm1{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDBC_8_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vldbc.8.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x5;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLDHBC_16_INCP:  EE_Inst24<(outs QR:$qu, QR:$qu1, AR:$asr), (ins AR:$as),
                   "ee.vldhbc.16.incp\t $qu, $qu1, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<3> qu1;
  bits<4> as;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qu{0};
  let Inst{14-12} = qu1{2-0};
  let Inst{11-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLD_128_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.vld.128.ip\t $qu, $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm16{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x3;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLD_128_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vld.128.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x2;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLD_H_64_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.vld.h.64.ip\t $qu, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm8;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x8;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLD_H_64_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vld.h.64.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x6;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLD_L_64_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.vld.l.64.ip\t $qu, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm8;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x9;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VLD_L_64_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vld.l.64.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x3;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VMAX_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmax.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x24;
}

def EE_VMAX_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMAX_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMAX_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmax.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x34;
}

def EE_VMAX_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMAX_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMAX_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmax.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x44;
}

def EE_VMAX_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMAX_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMIN_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmin.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x54;
}

def EE_VMIN_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMIN_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMIN_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmin.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x64;
}

def EE_VMIN_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMIN_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMIN_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmin.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x74;
}

def EE_VMIN_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMIN_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s16.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

def EE_VMULAS_S16_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

def EE_VMULAS_S16_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-21} = 0x4;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VMULAS_S16_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S16_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s8.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

def EE_VMULAS_S8_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

def EE_VMULAS_S8_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-21} = 0x5;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VMULAS_S8_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_S8_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u16.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

def EE_VMULAS_U16_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

def EE_VMULAS_U16_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-21} = 0x6;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VMULAS_U16_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U16_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u8.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

def EE_VMULAS_U8_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;


  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

def EE_VMULAS_U8_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{23-21} = 0x7;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VMULAS_U8_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27-26} = imm16{5-4};
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = imm16{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{19} = qu{0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMULAS_U8_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;

  let mayLoad = 1;

  let Constraints = "$asr = $as, $qs0r = $qs0";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25-24} = qu{2-1};
  let Inst{23} = qy{0};
  let Inst{22-20} = qs0{2-0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qs1{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{13-12} = qy{2-1};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_S16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.s16\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

def EE_VMUL_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s16.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s16.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_S8:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.s8\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x94;
}

def EE_VMUL_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s8.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s8.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_U16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.u16\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xa4;
}

def EE_VMUL_U16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u16.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_U16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u16.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_U8:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.u8\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xb4;
}

def EE_VMUL_U8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u8.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VMUL_U8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u8.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qz{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VPRELU_S16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy, AR:$ay),
                   "ee.vprelu.s16\t $qz, $qx, $qy, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<4> ay;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VPRELU_S8:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy, AR:$ay),
                   "ee.vprelu.s8\t $qz, $qx, $qy, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<4> ay;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VRELU_S16:  EE_Inst24<(outs QR:$qsr), (ins QR:$qs, AR:$ax, AR:$ay),
                   "ee.vrelu.s16\t $qs, $ax, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  bits<4> ax;
  bits<4> ay;


  let Constraints = "$qsr = $qs";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-12} = 0x1;
  let Inst{11-8} = ax{3-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VRELU_S8:  EE_Inst24<(outs QR:$qsr), (ins QR:$qs, AR:$ax, AR:$ay),
                   "ee.vrelu.s8\t $qs, $ax, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  bits<4> ax;
  bits<4> ay;


  let Constraints = "$qsr = $qs";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-12} = 0x5;
  let Inst{11-8} = ax{3-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VSL_32:  EE_Inst24<(outs QR:$qa), (ins QR:$qs),
                   "ee.vsl.32\t $qa, $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-7} = 0x7e;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

def EE_VSMULAS_S16_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy, select_8:$sel8),
                   "ee.vsmulas.s16.qacc\t $qx, $qy, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> sel8;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = sel8{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = sel8{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

def EE_VSMULAS_S16_QACC_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy, select_8:$sel8),
                   "ee.vsmulas.s16.qacc.ld.incp\t $qu, $as, $qx, $qy, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> sel8;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = sel8{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VSMULAS_S8_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy, select_16:$sel16),
                   "ee.vsmulas.s8.qacc\t $qx, $qy, $sel16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = sel16{3-2};
  let Inst{19-16} = 0xe;
  let Inst{15} = sel16{1};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-5} = 0x2;
  let Inst{4} = sel16{0};
  let Inst{3-0} = 0x4;
}

def EE_VSMULAS_S8_QACC_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy, select_16:$sel16),
                   "ee.vsmulas.s8.qacc.ld.incp\t $qu, $as, $qx, $qy, $sel16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{23} = qy{0};
  let Inst{20} = sel16{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = sel16{3-1};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VSR_32:  EE_Inst24<(outs QR:$qa), (ins QR:$qs),
                   "ee.vsr.32\t $qa, $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-7} = 0x7f;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

def EE_VST_128_IP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, offset_256_16:$imm16),
                   "ee.vst.128.ip\t $qv, $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<8> imm16;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm16{7};
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xa;
  let Inst{15} = qv{0};
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VST_128_XP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, AR:$ad),
                   "ee.vst.128.xp\t $qv, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<4> ad;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-12} = 0x7;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VST_H_64_IP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, offset_256_8:$imm8),
                   "ee.vst.h.64.ip\t $qv, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<8> imm8;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xb;
  let Inst{15} = qv{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VST_H_64_XP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, AR:$ad),
                   "ee.vst.h.64.xp\t $qv, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<4> ad;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-12} = 0x0;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VST_L_64_IP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, offset_256_8:$imm8),
                   "ee.vst.l.64.ip\t $qv, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<8> imm8;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0x4;
  let Inst{15} = qv{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VST_L_64_XP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, AR:$ad),
                   "ee.vst.l.64.xp\t $qv, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<4> ad;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-12} = 0x4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

def EE_VSUBS_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vsubs.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xd4;
}

def EE_VSUBS_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VSUBS_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VSUBS_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vsubs.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xe4;
}

def EE_VSUBS_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VSUBS_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VSUBS_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vsubs.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xf4;
}

def EE_VSUBS_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayLoad = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qu{2-1};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = qu{0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VSUBS_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;

  let mayStore = 1;

  let Constraints = "$asr = $as";
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = qy{0};
  let Inst{19} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{18-16} = qa{2-0};
  let Inst{15-14} = qx{1-0};
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{13-12} = qy{2-1};
  let Inst{7-4} = as{3-0};
  let Inst{3-1} = 0b111;
  let Inst{0} = qx{2};
}

def EE_VUNZIP_16:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vunzip.16\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;


  let Constraints = "$qs0r = $qs0, $qs1r = $qs1";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x384;
}

def EE_VUNZIP_32:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vunzip.32\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;


  let Constraints = "$qs0r = $qs0, $qs1r = $qs1";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x394;
}

def EE_VUNZIP_8:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vunzip.8\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;


  let Constraints = "$qs0r = $qs0, $qs1r = $qs1";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3a4;
}

def EE_VZIP_16:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vzip.16\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;


  let Constraints = "$qs0r = $qs0, $qs1r = $qs1";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3b4;
}

def EE_VZIP_32:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vzip.32\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;


  let Constraints = "$qs0r = $qs0, $qs1r = $qs1";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3c4;
}

def EE_VZIP_8:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vzip.8\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;


  let Constraints = "$qs0r = $qs0, $qs1r = $qs1";
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3d4;
}

def EE_XORQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.xorq\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-12} = 0x3;
  let Inst{11-10} = qy{2-1};
  let Inst{9-8} = 0x1;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

def EE_ZERO_ACCX:  EE_Inst24<(outs), (ins),
                   "ee.zero.accx\t", []>, Requires<[HasESP32S3Ops]>
{


  let Inst{23-0} = 0x250804;
}

def EE_ZERO_Q:  EE_Inst24<(outs QR:$qa), (ins),
                   "ee.zero.q\t $qa", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;


  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-0} = 0x7fa4;
}

def EE_ZERO_QACC:  EE_Inst24<(outs), (ins),
                   "ee.zero.qacc\t", []>, Requires<[HasESP32S3Ops]>
{


  let Inst{23-0} = 0x250844;
}

def RUR_ACCX_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.accx_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x0;
}

def RUR_ACCX_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.accx_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x10;
}

def RUR_FFT_BIT_WIDTH:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.fft_bit_width\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xe0;
}

def RUR_GPIO_OUT:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.gpio_out\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xc0;
}

def RUR_QACC_H_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x20;
}

def RUR_QACC_H_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x30;
}

def RUR_QACC_H_2:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_2\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x40;
}

def RUR_QACC_H_3:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_3\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x50;
}

def RUR_QACC_H_4:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_4\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x60;
}

def RUR_QACC_L_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x70;
}

def RUR_QACC_L_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x80;
}

def RUR_QACC_L_2:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_2\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x90;
}

def RUR_QACC_L_3:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_3\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xa0;
}

def RUR_QACC_L_4:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_4\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xb0;
}

def RUR_SAR_BYTE:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.sar_byte\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xd0;
}

def RUR_UA_STATE_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xf0;
}

def RUR_UA_STATE_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x100;
}

def RUR_UA_STATE_2:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_2\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x110;
}

def RUR_UA_STATE_3:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_3\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;


  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x120;
}

def WUR_ACCX_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.accx_0\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf300;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_ACCX_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.accx_1\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf301;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_FFT_BIT_WIDTH:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.fft_bit_width\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf30e;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_GPIO_OUT:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.gpio_out\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf30c;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_0\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf302;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_1\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf303;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_2:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_2\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf304;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_3:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_3\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf305;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_4:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_4\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf306;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_0\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf307;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_1\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf308;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_2:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_2\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf309;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_3:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_3\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf30a;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_4:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_4\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf30b;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_SAR_BYTE:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.sar_byte\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf30d;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_0\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf30f;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_1\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf310;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_2:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_2\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf311;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_3:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_3\t $art", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;


  let Inst{23-8} = 0xf312;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def mv_QR:  EE_Inst24<(outs QR:$a), (ins QR:$b),
                   "mv.qr\t $a, $b", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> a;
  bits<3> b;


  let Inst{23-22} = 0x2;
  let Inst{21-20} = a{2-1};
  let Inst{19-16} = 0xf;
  let Inst{15} = a{0};
  let Inst{14-12} = 0x0;
  let Inst{11-10} = b{2-1};
  let Inst{9-6} = 0x0;
  let Inst{5} = b{0};
  let Inst{4-0} = 0x4;
}
