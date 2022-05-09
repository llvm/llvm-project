//==- LoongArchInstrFormatsF.td - LoongArch FP Instr Formats -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe LoongArch floating-point instructions format
//
//  opcode       - operation code.
//  fd           - destination register operand.
//  {c/f}{j/k/a} - source register operand.
//  immN         - immediate data operand.
//
//===----------------------------------------------------------------------===//

// 2R-type
// <opcode | fj | fd>
class FPFmt2R<bits<22> op, dag outs, dag ins, string opcstr, string opnstr,
              list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<5> fj;
  bits<5> fd;

  let Inst{31-10} = op;
  let Inst{9-5} = fj;
  let Inst{4-0} = fd;
}

// 3R-type
// <opcode | fk | fj | fd>
class FPFmt3R<bits<17> op, dag outs, dag ins, string opcstr, string opnstr,
              list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<5> fk;
  bits<5> fj;
  bits<5> fd;

  let Inst{31-15} = op;
  let Inst{14-10} = fk;
  let Inst{9-5} = fj;
  let Inst{4-0} = fd;
}

// 4R-type
// <opcode | fa | fk | fj | fd>
class FPFmt4R<bits<12> op, dag outs, dag ins, string opcstr, string opnstr,
              list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<5> fa;
  bits<5> fk;
  bits<5> fj;
  bits<5> fd;

  let Inst{31-20} = op;
  let Inst{19-15} = fa;
  let Inst{14-10} = fk;
  let Inst{9-5} = fj;
  let Inst{4-0} = fd;
}

// 2RI12-type
// <opcode | I12 | rj | fd>
class FPFmt2RI12<bits<10> op, dag outs, dag ins, string opcstr, string opnstr,
                 list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<12> imm12;
  bits<5> rj;
  bits<5> fd;

  let Inst{31-22} = op;
  let Inst{21-10} = imm12;
  let Inst{9-5} = rj;
  let Inst{4-0} = fd;
}

// FmtFCMP
// <opcode | cond | fk | fj | 0b00 | cd>
class FPFmtFCMP<bits<12> op, bits<5> cond, dag outs, dag ins, string opcstr,
                string opnstr, list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<5> fk;
  bits<5> fj;
  bits<3> cd;

  let Inst{31-20} = op;
  let Inst{19-15} = cond;
  let Inst{14-10} = fk;
  let Inst{9-5} = fj;
  let Inst{4-3} = 0b00;
  let Inst{2-0} = cd;
}

// FPFmtBR
// <opcode[7:2] | I21[15:0] | opcode[1:0] | cj | I21[20:16]>
class FPFmtBR<bits<8> opcode, dag outs, dag ins, string opcstr,
              string opnstr, list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<21> imm21;
  bits<3> cj;

  let Inst{31-26} = opcode{7-2};
  let Inst{25-10} = imm21{15-0};
  let Inst{9-8} = opcode{1-0};
  let Inst{7-5} = cj;
  let Inst{4-0} = imm21{20-16};
}

// FmtFSEL
// <opcode | ca | fk | fj | fd>
class FPFmtFSEL<bits<14> op, dag outs, dag ins, string opcstr, string opnstr,
                list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<3> ca;
  bits<5> fk;
  bits<5> fj;
  bits<5> fd;

  let Inst{31-18} = op;
  let Inst{17-15} = ca;
  let Inst{14-10} = fk;
  let Inst{9-5} = fj;
  let Inst{4-0} = fd;
}

// FPFmtMOV
// <opcode | src | dst>
class FPFmtMOV<bits<22> op, dag outs, dag ins, string opcstr, string opnstr,
               list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<5> src;
  bits<5> dst;

  let Inst{31-10} = op;
  let Inst{9-5} = src;
  let Inst{4-0} = dst;
}

// FPFmtMEM
// <opcode | rk | rj | fd>
class FPFmtMEM<bits<17> op, dag outs, dag ins, string opcstr, string opnstr,
               list<dag> pattern = []>
    : LAInst<outs, ins, opcstr, opnstr, pattern> {
  bits<5> rk;
  bits<5> rj;
  bits<5> fd;

  let Inst{31-15} = op;
  let Inst{14-10} = rk;
  let Inst{9-5} = rj;
  let Inst{4-0} = fd;
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

class FP_ALU_2R<bits<22> op, string opstr, RegisterClass rc>
    : FPFmt2R<op, (outs rc:$fd), (ins rc:$fj), opstr, "$fd, $fj">;

class FP_ALU_3R<bits<17> op, string opstr, RegisterClass rc>
    : FPFmt3R<op, (outs rc:$fd), (ins rc:$fj, rc:$fk), opstr, "$fd, $fj, $fk">;

class FP_ALU_4R<bits<12> op, string opstr, RegisterClass rc>
    : FPFmt4R<op, (outs rc:$fd), (ins rc:$fj, rc:$fk, rc:$fa), opstr,
              "$fd, $fj, $fk, $fa">;

class FPCMPOpc<bits<12> value> {
  bits<12> val = value;
}

class FPCMPCond<bits<5> value> {
  bits<5> val = value;
}

class FP_CMP<FPCMPOpc op, FPCMPCond cond, string opstr, RegisterClass rc>
    : FPFmtFCMP<op.val, cond.val, (outs CFR:$cd), (ins rc:$fj, rc:$fk), opstr,
                "$cd, $fj, $fk">;

class FP_CONV<bits<22> op, string opstr, RegisterClass rcd, RegisterClass rcs>
    : FPFmt2R<op, (outs rcd:$fd), (ins rcs:$fj), opstr, "$fd, $fj">;

class FP_MOV<bits<22> op, string opstr, RegisterClass rcd, RegisterClass rcs>
    : FPFmtMOV<op, (outs rcd:$dst), (ins rcs:$src), opstr, "$dst, $src">;

class FP_SEL<bits<14> op, string opstr, RegisterClass rc>
    : FPFmtFSEL<op, (outs rc:$fd), (ins rc:$fj, rc:$fk, CFR:$ca), opstr,
                "$fd, $fj, $fk, $ca">;

class FP_BRANCH<bits<8> opcode, string opstr>
    : FPFmtBR<opcode, (outs), (ins CFR:$cj, simm21_lsl2:$imm21), opstr,
              "$cj, $imm21"> {
  let isBranch = 1;
  let isTerminator = 1;
}

let mayLoad = 1 in {
class FP_LOAD_3R<bits<17> op, string opstr, RegisterClass rc>
    : FPFmtMEM<op, (outs rc:$fd), (ins GPR:$rj, GPR:$rk), opstr,
               "$fd, $rj, $rk">;
class FP_LOAD_2RI12<bits<10> op, string opstr, RegisterClass rc>
    : FPFmt2RI12<op, (outs rc:$fd), (ins GPR:$rj, simm12:$imm12), opstr,
                 "$fd, $rj, $imm12">;
} // mayLoad = 1

let mayStore = 1 in {
class FP_STORE_3R<bits<17> op, string opstr, RegisterClass rc>
    : FPFmtMEM<op, (outs), (ins rc:$fd, GPR:$rj, GPR:$rk), opstr,
               "$fd, $rj, $rk">;
class FP_STORE_2RI12<bits<10> op, string opstr, RegisterClass rc>
    : FPFmt2RI12<op, (outs), (ins rc:$fd, GPR:$rj, simm12:$imm12), opstr,
                 "$fd, $rj, $imm12">;
} // mayStore = 1

def FPCMP_OPC_S : FPCMPOpc<0b000011000001>;
def FPCMP_OPC_D : FPCMPOpc<0b000011000010>;

def FPCMP_COND_CAF  : FPCMPCond<0x0>;
def FPCMP_COND_CUN  : FPCMPCond<0x8>;
def FPCMP_COND_CEQ  : FPCMPCond<0x4>;
def FPCMP_COND_CUEQ : FPCMPCond<0xC>;
def FPCMP_COND_CLT  : FPCMPCond<0x2>;
def FPCMP_COND_CULT : FPCMPCond<0xA>;
def FPCMP_COND_CLE  : FPCMPCond<0x6>;
def FPCMP_COND_CULE : FPCMPCond<0xE>;
def FPCMP_COND_CNE  : FPCMPCond<0x10>;
def FPCMP_COND_COR  : FPCMPCond<0x14>;
def FPCMP_COND_CUNE : FPCMPCond<0x18>;
def FPCMP_COND_SAF  : FPCMPCond<0x1>;
def FPCMP_COND_SUN  : FPCMPCond<0x9>;
def FPCMP_COND_SEQ  : FPCMPCond<0x5>;
def FPCMP_COND_SUEQ : FPCMPCond<0xD>;
def FPCMP_COND_SLT  : FPCMPCond<0x3>;
def FPCMP_COND_SULT : FPCMPCond<0xB>;
def FPCMP_COND_SLE  : FPCMPCond<0x7>;
def FPCMP_COND_SULE : FPCMPCond<0xF>;
def FPCMP_COND_SNE  : FPCMPCond<0x11>;
def FPCMP_COND_SOR  : FPCMPCond<0x15>;
def FPCMP_COND_SUNE : FPCMPCond<0x19>;
