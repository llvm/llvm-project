//===-- MBlazeInstrFSL.td - MBlaze FSL Instruction defs ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FSL Instruction Formats
//===----------------------------------------------------------------------===//
class FSLGet<bits<6> op, bits<5> flags, string instr_asm, Intrinsic OpNode> :
             MBlazeInst<op, FRCX, (outs GPR:$dst), (ins fslimm:$b),
                        !strconcat(instr_asm, " $dst, $b"),
                        [(set GPR:$dst, (OpNode immZExt4:$b))],IIC_FSLg>
{
    bits<5> rd;
    bits<4> fslno;

    let Inst{6-10}  = rd;
    let Inst{11-15} = 0x0;
    let Inst{16}    = 0x0;
    let Inst{17-21} = flags; // NCTAE
    let Inst{22-27} = 0x0;
    let Inst{28-31} = fslno;
}

class FSLGetD<bits<6> op, bits<5> flags, string instr_asm, Intrinsic OpNode> :
              MBlazeInst<op, FRCR, (outs GPR:$dst), (ins GPR:$b),
                         !strconcat(instr_asm, " $dst, $b"),
                         [(set GPR:$dst, (OpNode GPR:$b))], IIC_FSLg>
{
    bits<5> rd;
    bits<5> rb;

    let Inst{6-10}  = rd;
    let Inst{11-15} = 0x0;
    let Inst{16-20} = rb;
    let Inst{21}    = 0x0;
    let Inst{22-26} = flags; // NCTAE
    let Inst{27-31} = 0x0;
}

class FSLPut<bits<6> op, bits<4> flags, string instr_asm, Intrinsic OpNode> :
             MBlazeInst<op, FCRCX, (outs), (ins GPR:$v, fslimm:$b),
                        !strconcat(instr_asm, " $v, $b"),
                        [(OpNode GPR:$v, immZExt4:$b)], IIC_FSLp>
{
    bits<5> ra;
    bits<4> fslno;

    let Inst{6-10}  = 0x0;
    let Inst{11-15} = ra;
    let Inst{16}    = 0x1;
    let Inst{17-20} = flags; // NCTA
    let Inst{21-27} = 0x0;
    let Inst{28-31} = fslno;
}

class FSLPutD<bits<6> op, bits<4> flags, string instr_asm, Intrinsic OpNode> :
              MBlazeInst<op, FCRR, (outs), (ins GPR:$v, GPR:$b),
                         !strconcat(instr_asm, " $v, $b"),
                         [(OpNode GPR:$v, GPR:$b)], IIC_FSLp>
{
    bits<5> ra;
    bits<5> rb;

    let Inst{6-10}  = 0x0;
    let Inst{11-15} = ra;
    let Inst{16-20} = rb;
    let Inst{21}    = 0x1;
    let Inst{22-25} = flags; // NCTA
    let Inst{26-31} = 0x0;
}

class FSLPutT<bits<6> op, bits<4> flags, string instr_asm, Intrinsic OpNode> :
              MBlazeInst<op, FCX, (outs), (ins fslimm:$b),
                         !strconcat(instr_asm, " $b"),
                         [(OpNode immZExt4:$b)], IIC_FSLp>
{
    bits<4> fslno;

    let Inst{6-10}  = 0x0;
    let Inst{11-15} = 0x0;
    let Inst{16}    = 0x1;
    let Inst{17-20} = flags; // NCTA
    let Inst{21-27} = 0x0;
    let Inst{28-31} = fslno;
}

class FSLPutTD<bits<6> op, bits<4> flags, string instr_asm, Intrinsic OpNode> :
               MBlazeInst<op, FCR, (outs), (ins GPR:$b),
                          !strconcat(instr_asm, " $b"),
                          [(OpNode GPR:$b)], IIC_FSLp>
{
    bits<5> rb;

    let Inst{6-10}  = 0x0;
    let Inst{11-15} = 0x0;
    let Inst{16-20} = rb;
    let Inst{21}    = 0x1;
    let Inst{22-25} = flags; // NCTA
    let Inst{26-31} = 0x0;
}

//===----------------------------------------------------------------------===//
// FSL Get Instructions
//===----------------------------------------------------------------------===//
def GET      : FSLGet<0x1B, 0x00, "get      ", int_mblaze_fsl_get>;
def AGET     : FSLGet<0x1B, 0x02, "aget     ", int_mblaze_fsl_aget>;
def CGET     : FSLGet<0x1B, 0x08, "cget     ", int_mblaze_fsl_cget>;
def CAGET    : FSLGet<0x1B, 0x0A, "caget    ", int_mblaze_fsl_caget>;
def EGET     : FSLGet<0x1B, 0x01, "eget     ", int_mblaze_fsl_eget>;
def EAGET    : FSLGet<0x1B, 0x03, "eaget    ", int_mblaze_fsl_eaget>;
def ECGET    : FSLGet<0x1B, 0x09, "ecget    ", int_mblaze_fsl_ecget>;
def ECAGET   : FSLGet<0x1B, 0x0B, "ecaget   ", int_mblaze_fsl_ecaget>;
def TGET     : FSLGet<0x1B, 0x04, "tget     ", int_mblaze_fsl_tget>;
def TAGET    : FSLGet<0x1B, 0x06, "taget    ", int_mblaze_fsl_taget>;
def TCGET    : FSLGet<0x1B, 0x0C, "tcget    ", int_mblaze_fsl_tcget>;
def TCAGET   : FSLGet<0x1B, 0x0E, "tcaget   ", int_mblaze_fsl_tcaget>;
def TEGET    : FSLGet<0x1B, 0x05, "teget    ", int_mblaze_fsl_teget>;
def TEAGET   : FSLGet<0x1B, 0x07, "teaget   ", int_mblaze_fsl_teaget>;
def TECGET   : FSLGet<0x1B, 0x0D, "tecget   ", int_mblaze_fsl_tecget>;
def TECAGET  : FSLGet<0x1B, 0x0F, "tecaget  ", int_mblaze_fsl_tecaget>;

let Defs = [CARRY] in {
  def NGET     : FSLGet<0x1B, 0x10, "nget     ", int_mblaze_fsl_nget>;
  def NAGET    : FSLGet<0x1B, 0x12, "naget    ", int_mblaze_fsl_naget>;
  def NCGET    : FSLGet<0x1B, 0x18, "ncget    ", int_mblaze_fsl_ncget>;
  def NCAGET   : FSLGet<0x1B, 0x1A, "ncaget   ", int_mblaze_fsl_ncaget>;
  def NEGET    : FSLGet<0x1B, 0x11, "neget    ", int_mblaze_fsl_neget>;
  def NEAGET   : FSLGet<0x1B, 0x13, "neaget   ", int_mblaze_fsl_neaget>;
  def NECGET   : FSLGet<0x1B, 0x19, "necget   ", int_mblaze_fsl_necget>;
  def NECAGET  : FSLGet<0x1B, 0x1B, "necaget  ", int_mblaze_fsl_necaget>;
  def TNGET    : FSLGet<0x1B, 0x14, "tnget    ", int_mblaze_fsl_tnget>;
  def TNAGET   : FSLGet<0x1B, 0x16, "tnaget   ", int_mblaze_fsl_tnaget>;
  def TNCGET   : FSLGet<0x1B, 0x1C, "tncget   ", int_mblaze_fsl_tncget>;
  def TNCAGET  : FSLGet<0x1B, 0x1E, "tncaget  ", int_mblaze_fsl_tncaget>;
  def TNEGET   : FSLGet<0x1B, 0x15, "tneget   ", int_mblaze_fsl_tneget>;
  def TNEAGET  : FSLGet<0x1B, 0x17, "tneaget  ", int_mblaze_fsl_tneaget>;
  def TNECGET  : FSLGet<0x1B, 0x1D, "tnecget  ", int_mblaze_fsl_tnecget>;
  def TNECAGET : FSLGet<0x1B, 0x1F, "tnecaget ", int_mblaze_fsl_tnecaget>;
}

//===----------------------------------------------------------------------===//
// FSL Dynamic Get Instructions
//===----------------------------------------------------------------------===//
def GETD      : FSLGetD<0x13, 0x00, "getd     ", int_mblaze_fsl_get>;
def AGETD     : FSLGetD<0x13, 0x02, "agetd    ", int_mblaze_fsl_aget>;
def CGETD     : FSLGetD<0x13, 0x08, "cgetd    ", int_mblaze_fsl_cget>;
def CAGETD    : FSLGetD<0x13, 0x0A, "cagetd   ", int_mblaze_fsl_caget>;
def EGETD     : FSLGetD<0x13, 0x01, "egetd    ", int_mblaze_fsl_eget>;
def EAGETD    : FSLGetD<0x13, 0x03, "eagetd   ", int_mblaze_fsl_eaget>;
def ECGETD    : FSLGetD<0x13, 0x09, "ecgetd   ", int_mblaze_fsl_ecget>;
def ECAGETD   : FSLGetD<0x13, 0x0B, "ecagetd  ", int_mblaze_fsl_ecaget>;
def TGETD     : FSLGetD<0x13, 0x04, "tgetd    ", int_mblaze_fsl_tget>;
def TAGETD    : FSLGetD<0x13, 0x06, "tagetd   ", int_mblaze_fsl_taget>;
def TCGETD    : FSLGetD<0x13, 0x0C, "tcgetd   ", int_mblaze_fsl_tcget>;
def TCAGETD   : FSLGetD<0x13, 0x0E, "tcagetd  ", int_mblaze_fsl_tcaget>;
def TEGETD    : FSLGetD<0x13, 0x05, "tegetd   ", int_mblaze_fsl_teget>;
def TEAGETD   : FSLGetD<0x13, 0x07, "teagetd  ", int_mblaze_fsl_teaget>;
def TECGETD   : FSLGetD<0x13, 0x0D, "tecgetd  ", int_mblaze_fsl_tecget>;
def TECAGETD  : FSLGetD<0x13, 0x0F, "tecagetd ", int_mblaze_fsl_tecaget>;

let Defs = [CARRY] in {
  def NGETD     : FSLGetD<0x13, 0x10, "ngetd    ", int_mblaze_fsl_nget>;
  def NAGETD    : FSLGetD<0x13, 0x12, "nagetd   ", int_mblaze_fsl_naget>;
  def NCGETD    : FSLGetD<0x13, 0x18, "ncgetd   ", int_mblaze_fsl_ncget>;
  def NCAGETD   : FSLGetD<0x13, 0x1A, "ncagetd  ", int_mblaze_fsl_ncaget>;
  def NEGETD    : FSLGetD<0x13, 0x11, "negetd   ", int_mblaze_fsl_neget>;
  def NEAGETD   : FSLGetD<0x13, 0x13, "neagetd  ", int_mblaze_fsl_neaget>;
  def NECGETD   : FSLGetD<0x13, 0x19, "necgetd  ", int_mblaze_fsl_necget>;
  def NECAGETD  : FSLGetD<0x13, 0x1B, "necagetd ", int_mblaze_fsl_necaget>;
  def TNGETD    : FSLGetD<0x13, 0x14, "tngetd   ", int_mblaze_fsl_tnget>;
  def TNAGETD   : FSLGetD<0x13, 0x16, "tnagetd  ", int_mblaze_fsl_tnaget>;
  def TNCGETD   : FSLGetD<0x13, 0x1C, "tncgetd  ", int_mblaze_fsl_tncget>;
  def TNCAGETD  : FSLGetD<0x13, 0x1E, "tncagetd ", int_mblaze_fsl_tncaget>;
  def TNEGETD   : FSLGetD<0x13, 0x15, "tnegetd  ", int_mblaze_fsl_tneget>;
  def TNEAGETD  : FSLGetD<0x13, 0x17, "tneagetd ", int_mblaze_fsl_tneaget>;
  def TNECGETD  : FSLGetD<0x13, 0x1D, "tnecgetd ", int_mblaze_fsl_tnecget>;
  def TNECAGETD : FSLGetD<0x13, 0x1F, "tnecagetd", int_mblaze_fsl_tnecaget>;
}

//===----------------------------------------------------------------------===//
// FSL Put Instructions
//===----------------------------------------------------------------------===//
def PUT     :  FSLPut<0x1B, 0x0, "put      ", int_mblaze_fsl_put>;
def APUT    :  FSLPut<0x1B, 0x1, "aput     ", int_mblaze_fsl_aput>;
def CPUT    :  FSLPut<0x1B, 0x4, "cput     ", int_mblaze_fsl_cput>;
def CAPUT   :  FSLPut<0x1B, 0x5, "caput    ", int_mblaze_fsl_caput>;
def TPUT    : FSLPutT<0x1B, 0x2, "tput     ", int_mblaze_fsl_tput>;
def TAPUT   : FSLPutT<0x1B, 0x3, "taput    ", int_mblaze_fsl_taput>;
def TCPUT   : FSLPutT<0x1B, 0x6, "tcput    ", int_mblaze_fsl_tcput>;
def TCAPUT  : FSLPutT<0x1B, 0x7, "tcaput   ", int_mblaze_fsl_tcaput>;

let Defs = [CARRY] in {
  def NPUT    :  FSLPut<0x1B, 0x8, "nput     ", int_mblaze_fsl_nput>;
  def NAPUT   :  FSLPut<0x1B, 0x9, "naput    ", int_mblaze_fsl_naput>;
  def NCPUT   :  FSLPut<0x1B, 0xC, "ncput    ", int_mblaze_fsl_ncput>;
  def NCAPUT  :  FSLPut<0x1B, 0xD, "ncaput   ", int_mblaze_fsl_ncaput>;
  def TNPUT   : FSLPutT<0x1B, 0xA, "tnput    ", int_mblaze_fsl_tnput>;
  def TNAPUT  : FSLPutT<0x1B, 0xB, "tnaput   ", int_mblaze_fsl_tnaput>;
  def TNCPUT  : FSLPutT<0x1B, 0xE, "tncput   ", int_mblaze_fsl_tncput>;
  def TNCAPUT : FSLPutT<0x1B, 0xF, "tncaput  ", int_mblaze_fsl_tncaput>;
}

//===----------------------------------------------------------------------===//
// FSL Dynamic Put Instructions
//===----------------------------------------------------------------------===//
def PUTD     :  FSLPutD<0x13, 0x0, "putd     ", int_mblaze_fsl_put>;
def APUTD    :  FSLPutD<0x13, 0x1, "aputd    ", int_mblaze_fsl_aput>;
def CPUTD    :  FSLPutD<0x13, 0x4, "cputd    ", int_mblaze_fsl_cput>;
def CAPUTD   :  FSLPutD<0x13, 0x5, "caputd   ", int_mblaze_fsl_caput>;
def TPUTD    : FSLPutTD<0x13, 0x2, "tputd    ", int_mblaze_fsl_tput>;
def TAPUTD   : FSLPutTD<0x13, 0x3, "taputd   ", int_mblaze_fsl_taput>;
def TCPUTD   : FSLPutTD<0x13, 0x6, "tcputd   ", int_mblaze_fsl_tcput>;
def TCAPUTD  : FSLPutTD<0x13, 0x7, "tcaputd  ", int_mblaze_fsl_tcaput>;

let Defs = [CARRY] in {
  def NPUTD    :  FSLPutD<0x13, 0x8, "nputd    ", int_mblaze_fsl_nput>;
  def NAPUTD   :  FSLPutD<0x13, 0x9, "naputd   ", int_mblaze_fsl_naput>;
  def NCPUTD   :  FSLPutD<0x13, 0xC, "ncputd   ", int_mblaze_fsl_ncput>;
  def NCAPUTD  :  FSLPutD<0x13, 0xD, "ncaputd  ", int_mblaze_fsl_ncaput>;
  def TNPUTD   : FSLPutTD<0x13, 0xA, "tnputd   ", int_mblaze_fsl_tnput>;
  def TNAPUTD  : FSLPutTD<0x13, 0xB, "tnaputd  ", int_mblaze_fsl_tnaput>;
  def TNCPUTD  : FSLPutTD<0x13, 0xE, "tncputd  ", int_mblaze_fsl_tncput>;
  def TNCAPUTD : FSLPutTD<0x13, 0xF, "tncaputd ", int_mblaze_fsl_tncaput>;
}
