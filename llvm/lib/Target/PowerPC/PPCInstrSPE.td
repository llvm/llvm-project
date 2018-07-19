//=======-- PPCInstrSPE.td - The PowerPC SPE Extension -*- tablegen -*-=======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Signal Processing Engine extension to
// the PowerPC instruction set.
//
//===----------------------------------------------------------------------===//

class EVXForm_1<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;

  let Pattern = [];
  
  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-31} = xo;
}

class EVXForm_2<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : EVXForm_1<xo, OOL, IOL, asmstr, itin> {
  let RB = 0;
}

class EVXForm_3<bits<11> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin> : I<4, OOL, IOL, asmstr, itin> {
  bits<3> crD;
  bits<5> RA;
  bits<5> RB;

  let Pattern = [];
  
  let Inst{6-8}  = crD;
  let Inst{9-10}  = 0;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-31} = xo;
}

class EVXForm_D<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<21> D;

  let Pattern = [];

  let Inst{6-10}  = RT;
  let Inst{20} = D{0};
  let Inst{19} = D{1};
  let Inst{18} = D{2};
  let Inst{17} = D{3};
  let Inst{16} = D{4};
  let Inst{15} = D{5};
  let Inst{14} = D{6};
  let Inst{13} = D{7};
  let Inst{12} = D{8};
  let Inst{11} = D{9};
  let Inst{11-20} = D{0-9};
  let Inst{21-31} = xo;
}

let Predicates = [HasSPE], isAsmParserOnly = 1 in {

def EVLDD          : EVXForm_D<769, (outs gprc:$RT), (ins spe8dis:$dst),
                               "evldd $RT, $dst", IIC_VecFP>;
def EVLDW          : EVXForm_D<771, (outs gprc:$RT), (ins spe8dis:$dst),
                               "evldw $RT, $dst", IIC_VecFP>;
def EVLDH          : EVXForm_D<773, (outs gprc:$RT), (ins spe8dis:$dst),
                               "evldh $RT, $dst", IIC_VecFP>;
def EVLHHESPLAT    : EVXForm_D<777, (outs gprc:$RT), (ins spe2dis:$dst),
                               "evlhhesplat $RT, $dst", IIC_VecFP>;
def EVLHHOUSPLAT   : EVXForm_D<781, (outs gprc:$RT), (ins spe2dis:$dst),
                               "evlhhousplat $RT, $dst", IIC_VecFP>;
def EVLHHOSSPLAT   : EVXForm_D<783, (outs gprc:$RT), (ins spe2dis:$dst),
                               "evlhhossplat $RT, $dst", IIC_VecFP>;
def EVLWHE         : EVXForm_D<785, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhe $RT, $dst", IIC_VecFP>;
def EVLWHOU        : EVXForm_D<789, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhou $RT, $dst", IIC_VecFP>;
def EVLWHOS        : EVXForm_D<791, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhos $RT, $dst", IIC_VecFP>;
def EVLWWSPLAT     : EVXForm_D<793, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwwsplat $RT, $dst", IIC_VecFP>;
def EVLWHSPLAT     : EVXForm_D<797, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhsplat $RT, $dst", IIC_VecFP>;

def EVSTDD         : EVXForm_D<801, (outs), (ins gprc:$RT, spe8dis:$dst),
                               "evstdd $RT, $dst", IIC_VecFP>;
def EVSTDH         : EVXForm_D<805, (outs), (ins gprc:$RT, spe8dis:$dst),
                               "evstdh $RT, $dst", IIC_VecFP>;
def EVSTDW         : EVXForm_D<803, (outs), (ins gprc:$RT, spe8dis:$dst),
                               "evstdw $RT, $dst", IIC_VecFP>;
def EVSTWHE        : EVXForm_D<817, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwhe $RT, $dst", IIC_VecFP>;
def EVSTWHO        : EVXForm_D<821, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwho $RT, $dst", IIC_VecFP>;
def EVSTWWE        : EVXForm_D<825, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwwe $RT, $dst", IIC_VecFP>;
def EVSTWWO        : EVXForm_D<829, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwwo $RT, $dst", IIC_VecFP>;

def EVMRA : EVXForm_1<1220, (outs gprc:$RT), (ins gprc:$RA),
                      "evmra $RT, $RA", IIC_VecFP> {
  let RB = 0;
}

def BRINC          : EVXForm_1<527, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "brinc $RT, $RA, $RB", IIC_VecFP>;
def EVABS          : EVXForm_2<520, (outs gprc:$RT), (ins gprc:$RA),
                               "evabs $RT, $RA", IIC_VecFP>;

def EVADDIW        : EVXForm_1<514, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evaddiw $RT, $RB, $RA", IIC_VecFP>;
def EVADDSMIAAW    : EVXForm_2<1225, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddsmiaaw $RT, $RA", IIC_VecFP>;
def EVADDSSIAAW    : EVXForm_2<1217, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddssiaaw $RT, $RA", IIC_VecFP>;
def EVADDUSIAAW    : EVXForm_2<1216, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddusiaaw $RT, $RA", IIC_VecFP>;
def EVADDUMIAAW    : EVXForm_2<1224, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddumiaaw $RT, $RA", IIC_VecFP>;
def EVADDW         : EVXForm_1<512, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evaddw $RT, $RA, $RB", IIC_VecFP>;

def EVAND          : EVXForm_1<529, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evand $RT, $RA, $RB", IIC_VecFP>;
def EVANDC         : EVXForm_1<530, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evandc $RT, $RA, $RB", IIC_VecFP>;

def EVCMPEQ        : EVXForm_3<564, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpeq $crD, $RA, $RB", IIC_VecFP>;
def EVCMPGTS       : EVXForm_3<561, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpgts $crD, $RA, $RB", IIC_VecFP>;
def EVCMPGTU       : EVXForm_3<560, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpgtu $crD, $RA, $RB", IIC_VecFP>;
def EVCMPLTS       : EVXForm_3<563, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmplts $crD, $RA, $RB", IIC_VecFP>;
def EVCMPLTU       : EVXForm_3<562, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpltu $crD, $RA, $RB", IIC_VecFP>;

def EVCNTLSW       : EVXForm_2<526, (outs gprc:$RT), (ins gprc:$RA),
                               "evcntlsw $RT, $RA", IIC_VecFP>;
def EVCNTLZW       : EVXForm_2<525, (outs gprc:$RT), (ins gprc:$RA),
                               "evcntlzw $RT, $RA", IIC_VecFP>;

def EVDIVWS        : EVXForm_1<1222, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evdivws $RT, $RA, $RB", IIC_VecFP>;
def EVDIVWU        : EVXForm_1<1223, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evdivwu $RT, $RA, $RB", IIC_VecFP>;

def EVEQV          : EVXForm_1<537, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "eveqv $RT, $RA, $RB", IIC_VecFP>;

def EVEXTSB        : EVXForm_2<522, (outs gprc:$RT), (ins gprc:$RA),
                               "evextsb $RT, $RA", IIC_VecFP>;
def EVEXTSH        : EVXForm_2<523, (outs gprc:$RT), (ins gprc:$RA),
                               "evextsh $RT, $RA", IIC_VecFP>;

def EVLDDX         : EVXForm_1<768, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlddx $RT, $RA, $RB", IIC_VecFP>;
def EVLDWX         : EVXForm_1<770, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evldwx $RT, $RA, $RB", IIC_VecFP>;
def EVLDHX         : EVXForm_1<772, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evldhx $RT, $RA, $RB", IIC_VecFP>;
def EVLHHESPLATX   : EVXForm_1<776, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlhhesplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLHHOUSPLATX  : EVXForm_1<780, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlhhousplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLHHOSSPLATX  : EVXForm_1<782, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlhhossplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLWHEX        : EVXForm_1<784, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhex $RT, $RA, $RB", IIC_VecFP>;
def EVLWHOUX       : EVXForm_1<788, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhoux $RT, $RA, $RB", IIC_VecFP>;
def EVLWHOSX       : EVXForm_1<790, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhosx $RT, $RA, $RB", IIC_VecFP>;
def EVLWWSPLATX    : EVXForm_1<792, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwwsplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLWHSPLATX    : EVXForm_1<796, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhsplatx $RT, $RA, $RB", IIC_VecFP>;

def EVMERGEHI      : EVXForm_1<556, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergehi $RT, $RA, $RB", IIC_VecFP>;
def EVMERGELO      : EVXForm_1<557, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergelo $RT, $RA, $RB", IIC_VecFP>;
def EVMERGEHILO    : EVXForm_1<558, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergehilo $RT, $RA, $RB", IIC_VecFP>;
def EVMERGELOHI    : EVXForm_1<559, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergelohi $RT, $RA, $RB", IIC_VecFP>;

def EVMHEGSMFAA    : EVXForm_1<1323, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmfaa $RT, $RA, $RB", IIC_VecFP>;
def EVMHEGSMFAN    : EVXForm_1<1451, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmfan $RT, $RA, $RB", IIC_VecFP>;
def EVMHEGSMIAA    : EVXForm_1<1321, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmiaa $RT, $RA, $RB", IIC_VecFP>;
def EVMHEGSMIAN    : EVXForm_1<1449, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmian $RT, $RA, $RB", IIC_VecFP>;
def EVMHEGUMIAA    : EVXForm_1<1320, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegumiaa $RT, $RA, $RB", IIC_VecFP>;
def EVMHEGUMIAN    : EVXForm_1<1448, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegumian $RT, $RA, $RB", IIC_VecFP>;

def EVMHESMF       : EVXForm_1<1035, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmf $RT, $RA, $RB", IIC_VecFP>;
def EVMHESMFA      : EVXForm_1<1067, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmfa $RT, $RA, $RB", IIC_VecFP>;
def EVMHESMFAAW    : EVXForm_1<1291, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmfaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHESMFANW    : EVXForm_1<1419, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmfanw $RT, $RA, $RB", IIC_VecFP>;
def EVMHESMI       : EVXForm_1<1033, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmi $RT, $RA, $RB", IIC_VecFP>;
def EVMHESMIA      : EVXForm_1<1065, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmia $RT, $RA, $RB", IIC_VecFP>;
def EVMHESMIAAW    : EVXForm_1<1289, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHESMIANW    : EVXForm_1<1417, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmianw $RT, $RA, $RB", IIC_VecFP>;
def EVMHESSF       : EVXForm_1<1027, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessf $RT, $RA, $RB", IIC_VecFP>;
def EVMHESSFA      : EVXForm_1<1059, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessfa $RT, $RA, $RB", IIC_VecFP>;
def EVMHESSFAAW    : EVXForm_1<1283, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessfaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHESSFANW    : EVXForm_1<1411, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessfanw $RT, $RA, $RB", IIC_VecFP>;
def EVMHESSIAAW    : EVXForm_1<1281, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHESSIANW    : EVXForm_1<1409, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessianw $RT, $RA, $RB", IIC_VecFP>;
def EVMHEUMI       : EVXForm_1<1032, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumi $RT, $RA, $RB", IIC_VecFP>;
def EVMHEUMIA      : EVXForm_1<1064, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumia $RT, $RA, $RB", IIC_VecFP>;
def EVMHEUMIAAW    : EVXForm_1<1288, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHEUMIANW    : EVXForm_1<1416, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumianw $RT, $RA, $RB", IIC_VecFP>;
def EVMHEUSIAAW    : EVXForm_1<1280, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheusiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHEUSIANW    : EVXForm_1<1408, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheusianw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOGSMFAA    : EVXForm_1<1327, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmfaa $RT, $RA, $RB", IIC_VecFP>;
def EVMHOGSMFAN    : EVXForm_1<1455, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmfan $RT, $RA, $RB", IIC_VecFP>;
def EVMHOGSMIAA    : EVXForm_1<1325, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmiaa $RT, $RA, $RB", IIC_VecFP>;
def EVMHOGSMIAN    : EVXForm_1<1453, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmian $RT, $RA, $RB", IIC_VecFP>;
def EVMHOGUMIAA    : EVXForm_1<1324, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogumiaa $RT, $RA, $RB", IIC_VecFP>;
def EVMHOGUMIAN    : EVXForm_1<1452, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogumian $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMF       : EVXForm_1<1039, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmf $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMFA      : EVXForm_1<1071, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmfa $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMFAAW    : EVXForm_1<1295, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmfaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMFANW    : EVXForm_1<1423, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmfanw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMI       : EVXForm_1<1037, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmi $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMIA      : EVXForm_1<1069, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmia $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMIAAW    : EVXForm_1<1293, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSMIANW    : EVXForm_1<1421, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmianw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSSF       : EVXForm_1<1031, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossf $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSSFA      : EVXForm_1<1063, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossfa $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSSFAAW    : EVXForm_1<1287, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossfaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSSFANW    : EVXForm_1<1415, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossfanw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSSIAAW    : EVXForm_1<1285, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOSSIANW    : EVXForm_1<1413, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossianw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOUMI       : EVXForm_1<1036, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumi $RT, $RA, $RB", IIC_VecFP>;
def EVMHOUMIA      : EVXForm_1<1068, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumia $RT, $RA, $RB", IIC_VecFP>;
def EVMHOUMIAAW    : EVXForm_1<1292, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOUMIANW    : EVXForm_1<1420, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumianw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOUSIAAW    : EVXForm_1<1284, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhousiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMHOUSIANW    : EVXForm_1<1412, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhousianw $RT, $RA, $RB", IIC_VecFP>;


def EVMWHSMF       : EVXForm_1<1103, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmf $RT, $RA, $RB", IIC_VecFP>;
def EVMWHSMFA      : EVXForm_1<1135, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmfa $RT, $RA, $RB", IIC_VecFP>;
def EVMWHSMI       : EVXForm_1<1101, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmi $RT, $RA, $RB", IIC_VecFP>;
def EVMWHSMIA      : EVXForm_1<1133, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmia $RT, $RA, $RB", IIC_VecFP>;
def EVMWHSSF       : EVXForm_1<1095, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhssf $RT, $RA, $RB", IIC_VecFP>;
def EVMWHSSFA      : EVXForm_1<1127, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhssfa $RT, $RA, $RB", IIC_VecFP>;
def EVMWHUMI       : EVXForm_1<1100, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhumi $RT, $RA, $RB", IIC_VecFP>;
def EVMWHUMIA      : EVXForm_1<1132, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhumia $RT, $RA, $RB", IIC_VecFP>;
def EVMWLSMIAAW    : EVXForm_1<1353, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlsmiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMWLSMIANW    : EVXForm_1<1481, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlsmianw $RT, $RA, $RB", IIC_VecFP>;
def EVMWLSSIAAW    : EVXForm_1<1345, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlssiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMWLSSIANW    : EVXForm_1<1473, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlssianw $RT, $RA, $RB", IIC_VecFP>;
def EVMWLUMI       : EVXForm_1<1096, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumi $RT, $RA, $RB", IIC_VecFP>;
def EVMWLUMIA      : EVXForm_1<1128, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumia $RT, $RA, $RB", IIC_VecFP>;
def EVMWLUMIAAW    : EVXForm_1<1352, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMWLUMIANW    : EVXForm_1<1480, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumianw $RT, $RA, $RB", IIC_VecFP>;
def EVMWLUSIAAW    : EVXForm_1<1344, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlusiaaw $RT, $RA, $RB", IIC_VecFP>;
def EVMWLUSIANW    : EVXForm_1<1472, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlusianw $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMF        : EVXForm_1<1115, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmf $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMFA       : EVXForm_1<1147, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmfa $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMFAA      : EVXForm_1<1371, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmfaa $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMFAN      : EVXForm_1<1499, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmfan $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMI        : EVXForm_1<1113, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmi $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMIA       : EVXForm_1<1145, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmia $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMIAA      : EVXForm_1<1369, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmiaa $RT, $RA, $RB", IIC_VecFP>;
def EVMWSMIAN      : EVXForm_1<1497, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmian $RT, $RA, $RB", IIC_VecFP>;
def EVMWSSF        : EVXForm_1<1107, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssf $RT, $RA, $RB", IIC_VecFP>;
def EVMWSSFA       : EVXForm_1<1139, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssfa $RT, $RA, $RB", IIC_VecFP>;
def EVMWSSFAA      : EVXForm_1<1363, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssfaa $RT, $RA, $RB", IIC_VecFP>;
def EVMWSSFAN      : EVXForm_1<1491, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssfan $RT, $RA, $RB", IIC_VecFP>;
def EVMWUMI        : EVXForm_1<1112, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumi $RT, $RA, $RB", IIC_VecFP>;
def EVMWUMIA       : EVXForm_1<1144, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumia $RT, $RA, $RB", IIC_VecFP>;
def EVMWUMIAA      : EVXForm_1<1368, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumiaa $RT, $RA, $RB", IIC_VecFP>;
def EVMWUMIAN      : EVXForm_1<1496, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumian $RT, $RA, $RB", IIC_VecFP>;


def EVNAND         : EVXForm_1<542, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evnand $RT, $RA, $RB", IIC_VecFP>;

def EVNEG          : EVXForm_2<521, (outs gprc:$RT), (ins gprc:$RA),
                               "evneg $RT, $RA", IIC_VecFP>;

def EVNOR          : EVXForm_1<536, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evnor $RT, $RA, $RB", IIC_VecFP>;
def EVOR           : EVXForm_1<535, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evor $RT, $RA, $RB", IIC_VecFP>;
def EVORC          : EVXForm_1<539, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evorc $RT, $RA, $RB", IIC_VecFP>;

def EVRLWI         : EVXForm_1<554, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evrlwi $RT, $RA, $RB", IIC_VecFP>;
def EVRLW          : EVXForm_1<552, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evrlw $RT, $RA, $RB", IIC_VecFP>;

def EVRNDW         : EVXForm_2<524, (outs gprc:$RT), (ins gprc:$RA),
                               "evrndw $RT, $RA", IIC_VecFP>;

def EVSLWI         : EVXForm_1<550, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evslwi $RT, $RA, $RB", IIC_VecFP>;
def EVSLW          : EVXForm_1<548, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evslw $RT, $RA, $RB", IIC_VecFP>;

def EVSPLATFI      : EVXForm_2<555, (outs gprc:$RT), (ins i32imm:$RA),
                               "evsplatfi $RT, $RA", IIC_VecFP>;
def EVSPLATI       : EVXForm_2<553, (outs gprc:$RT), (ins i32imm:$RA),
                               "evsplati $RT, $RA", IIC_VecFP>;

def EVSRWIS        : EVXForm_1<547, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evsrwis $RT, $RA, $RB", IIC_VecFP>;
def EVSRWIU        : EVXForm_1<546, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evsrwiu $RT, $RA, $RB", IIC_VecFP>;
def EVSRWS         : EVXForm_1<545, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evsrws $RT, $RA, $RB", IIC_VecFP>;
def EVSRWU         : EVXForm_1<544, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evsrwu $RT, $RA, $RB", IIC_VecFP>;

def EVSTDDX        : EVXForm_1<800, (outs), (ins gprc:$RT, gprc:$RA, gprc:$RB),
                               "evstddx $RT, $RA, $RB", IIC_VecFP>;
def EVSTDHX        : EVXForm_1<804, (outs), (ins gprc:$RT, gprc:$RA, gprc:$RB),
                               "evstdhx $RT, $RA, $RB", IIC_VecFP>;
def EVSTDWX        : EVXForm_1<802, (outs), (ins gprc:$RT, gprc:$RA, gprc:$RB),
                               "evstdwx $RT, $RA, $RB", IIC_VecFP>;
def EVSTWHEX       : EVXForm_1<816, (outs), (ins gprc:$RT, gprc:$RA, gprc:$RB),
                               "evstwhex $RT, $RA, $RB", IIC_VecFP>;
def EVSTWHOX       : EVXForm_1<820, (outs), (ins gprc:$RT, gprc:$RA, gprc:$RB),
                               "evstwhox $RT, $RA, $RB", IIC_VecFP>;
def EVSTWWEX       : EVXForm_1<824, (outs), (ins gprc:$RT, gprc:$RA, gprc:$RB),
                               "evstwwex $RT, $RA, $RB", IIC_VecFP>;
def EVSTWWOX       : EVXForm_1<828, (outs), (ins gprc:$RT, gprc:$RA, gprc:$RB),
                               "evstwwox $RT, $RA, $RB", IIC_VecFP>;

def EVSUBFSSIAAW   : EVXForm_2<1219, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfssiaaw $RT, $RA", IIC_VecFP>;
def EVSUBFSMIAAW   : EVXForm_2<1227, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfsmiaaw $RT, $RA", IIC_VecFP>;
def EVSUBFUMIAAW   : EVXForm_2<1226, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfumiaaw $RT, $RA", IIC_VecFP>;
def EVSUBFUSIAAW   : EVXForm_2<1218, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfusiaaw $RT, $RA", IIC_VecFP>;
def EVSUBFW        : EVXForm_1<516, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evsubfw $RT, $RA, $RB", IIC_VecFP>;
def EVSUBIFW       : EVXForm_1<518, (outs gprc:$RT), (ins u5imm:$RA, gprc:$RB),
                               "evsubifw $RT, $RA, $RB", IIC_VecFP>;
def EVXOR          : EVXForm_1<534, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evxor $RT, $RA, $RB", IIC_VecFP>;

} // HasSPE
