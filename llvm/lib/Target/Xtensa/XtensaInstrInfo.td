//===- XtensaInstrInfo.td - Xtensa Target Description ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Xtensa instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "XtensaInstrFormats.td"
include "XtensaOperands.td"
include "XtensaOperators.td"

//===----------------------------------------------------------------------===//
// Arithmetic & Logical instructions
//===----------------------------------------------------------------------===//

class ArithLogic_RRR<bits<4> oper2, bits<4> oper1, string instrAsm, 
      SDPatternOperator opNode, bit isComm = 0>
  : RRR_Inst<0x00, oper1, oper2, (outs AR:$r), (ins AR:$s, AR:$t),
             instrAsm#"\t$r, $s, $t",
            [(set AR:$r, (opNode AR:$s, AR:$t))]> 
{
  let isCommutable = isComm;
  let isReMaterializable = 0;
}

def ADD: ArithLogic_RRR<0x08, 0x00, "add", add, 1>;
def SUB: ArithLogic_RRR<0x0C, 0x00, "sub", sub>;
def AND: ArithLogic_RRR<0x01, 0x00, "and", and, 1>;
def OR: ArithLogic_RRR<0x02, 0x00, "or", or, 1>;
def XOR: ArithLogic_RRR<0x03, 0x00, "xor", xor, 1>;

class ADDX<bits<4> oper, string instrAsm, list<dag> pattern>
    : RRR_Inst<0x00, 0x00, oper, (outs AR:$r), (ins AR:$s, AR:$t),
               instrAsm#"\t$r, $s, $t", pattern>; 

def ADDX2: ADDX<0x09, "addx2", [(set AR:$r, (add AR:$t, (shl AR:$s, (i32 1))))]>;
def ADDX4: ADDX<0x0A, "addx4", [(set AR:$r, (add AR:$t, (shl AR:$s, (i32 2))))]>;
def ADDX8: ADDX<0x0B, "addx8", [(set AR:$r, (add AR:$t, (shl AR:$s, (i32 3))))]>;

class SUBX<bits<4> oper, string instrAsm, list<dag> pattern>
    : RRR_Inst<0x00, 0x00, oper, (outs AR:$r), (ins AR:$s, AR:$t),
               instrAsm#"\t$r, $s, $t", pattern>; 

def SUBX2: SUBX<0x0D, "subx2", [(set AR:$r, (sub (shl AR:$s, (i32 1)), AR:$t))]>;
def SUBX4: SUBX<0x0E, "subx4", [(set AR:$r, (sub (shl AR:$s, (i32 2)), AR:$t))]>;
def SUBX8: SUBX<0x0F, "subx8", [(set AR:$r, (sub (shl AR:$s, (i32 3)), AR:$t))]>;

def ABS: RRR_Inst<0x00, 0x00, 0x06, (outs AR:$r), (ins AR:$t),
                 "abs\t$r, $t", []>
{
  let s = 0x1;
}

def ADDI: RRI8_Inst<0x02, (outs AR:$t), (ins AR:$s, imm8:$imm8),
                   "addi\t$t, $s, $imm8",
                   [(set AR:$t, (add AR:$s, imm8:$imm8))]> 
{
  let r = 0x0C;
}

def ADDMI: RRI8_Inst<0x02, (outs AR:$t), (ins AR:$s, imm8_sh8:$imm_sh8),
                    "addmi\t$t, $s, $imm_sh8",
                    [(set AR:$t, (add AR:$s, imm8_sh8:$imm_sh8))]> 
{
  bits<16> imm_sh8;

  let r = 0x0D;
  let imm8 = imm_sh8{15-8};
}

def NEG: RRR_Inst<0x00, 0x00, 0x06, (outs AR:$r), (ins AR:$t),
                 "neg\t$r, $t",
                 [(set AR:$r, (ineg AR:$t))]>
{
  let s = 0x00;
}

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//
def MOVI: RRI8_Inst<0x02, (outs AR:$t), (ins imm12m:$imm),
                   "movi\t$t, $imm",
                   [(set AR:$t, imm12m:$imm)]>
{
  bits<12> imm;

  let imm8{7-0} = imm{7-0}; 
  let s{3-0} = imm{11-8}; 
  let r = 0xa;
}

def MOVEQZ : RRR_Inst<0x00, 0x03, 0x08, (outs AR:$r), (ins AR:$s, AR:$t),
                     "moveqz\t$r, $s, $t", []>;
def MOVNEZ : RRR_Inst<0x00, 0x03, 0x09, (outs AR:$r), (ins AR:$s, AR:$t),
                     "movnez\t$r, $s, $t", []>;
def MOVLTZ : RRR_Inst<0x00, 0x03, 0x0A, (outs AR:$r), (ins AR:$s, AR:$t),
                     "movltz\t$r, $s, $t", []>;
def MOVGEZ : RRR_Inst<0x00, 0x03, 0x0B, (outs AR:$r), (ins AR:$s, AR:$t),
                     "movgez\t$r, $s, $t", []>;

//===----------------------------------------------------------------------===//
// Shift instructions
//===----------------------------------------------------------------------===//
let Uses = [SAR] in
{
  def SLL: RRR_Inst<0x00, 0x01, 0x0A, (outs AR:$r), (ins AR:$s),
                   "sll\t$r, $s",
                   [(set AR:$r, (Xtensa_shl AR:$s))]>
  {
    let t = 0x00;
  }

  def SRA: RRR_Inst<0x00, 0x01, 0x0B, (outs AR:$r), (ins AR:$t),
                   "sra\t$r, $t",
                   [(set AR:$r, (Xtensa_sra AR:$t))]>
  {
    let s = 0x00;
  }

  def SRC: RRR_Inst<0x00, 0x01, 0x08, (outs AR:$r), (ins AR:$s, AR:$t),
                   "src\t$r, $s, $t",
                   [(set AR:$r, (Xtensa_src AR:$s, AR:$t))]>;

  def SRL: RRR_Inst<0x00, 0x01, 0x09, (outs AR:$r), (ins AR:$t),
                   "srl\t$r, $t",
                   [(set AR:$r, (Xtensa_srl AR:$t))]>
  {
    let s = 0x00;
  }
}

let Defs = [SAR] in
{
  def SSL: RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$s),
                   "ssl\t$s",
                   [(Xtensa_ssl AR:$s)]>
  {
    let r = 0x01;
    let t = 0x00;
  }

  def SSR: RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$s),
                   "ssr\t$s", 
                   [(Xtensa_ssr AR:$s)]>
  {
    let r = 0x00;
    let t = 0x00;
  }
}


def EXTUI: RRR_Inst<0x00, 0x04, 0x00, (outs AR:$r), (ins AR:$t, uimm5:$imm1, imm1_16:$imm2),
                   "extui\t$r, $t, $imm1, $imm2",
                   []>
{
  bits<5> imm1;
  bits<4> imm2;

  let s = imm1{3-0};
  let Inst{16} = imm1{4};
  let Inst{23-20} = imm2;
}

def SRAI: RRR_Inst<0x00, 0x01, 0x02, (outs AR:$r), (ins AR:$t, uimm5:$sa),
                  "srai\t$r, $t, $sa",
                  [(set AR:$r, (sra AR:$t, uimm5:$sa))]> 
{
  bits<5> sa;

  let Inst{20} = sa{4};
  let s = sa{3-0};
}

def SRLI: RRR_Inst<0x00, 0x01, 0x04, (outs AR:$r), (ins AR:$t, uimm4:$sa),
                  "srli\t$r, $t, $sa",
                  [(set AR:$r, (srl AR:$t, uimm4:$sa))]> 
{
  bits<4> sa; 

  let s = sa;
}

def SLLI: RRR_Inst<0x00, 0x01, 0x00, (outs AR:$r), (ins AR:$s, shimm1_31:$sa),
                  "slli\t$r, $s, $sa",
                  [(set AR:$r, (shl AR:$s, shimm1_31:$sa))]> 
{
  bits<5> sa;

  let Inst{20} = sa{4};
  let t = sa{3-0};
}

def SSA8L : RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$s),
     "ssa8l\t$s", []>
{
  let r = 0x2;
  let t = 0x0;
} 

def SSAI: RRR_Inst<0x00, 0x00, 0x04, (outs), (ins uimm5:$imm),
                  "ssai\t$imm", []>
{
  bits<5> imm;

  let r = 0x04;
  let s = imm{3-0};
  let t{3-1} = 0;
  let t{0} = imm{4};
} 

//===----------------------------------------------------------------------===//
// Load and store instructions
//===----------------------------------------------------------------------===//

// Load instructions
let mayLoad = 1, usesCustomInserter = 1 in
{

  class Load_RRI8<bits<4> oper, string instrAsm, SDPatternOperator opNode, 
        ComplexPattern addrOp, Operand memOp>
	  : RRI8_Inst<0x02, (outs AR:$t), (ins memOp:$addr), 
                  instrAsm#"\t$t, $addr", 
                 [(set AR:$t, (opNode addrOp:$addr))]>
  {
    bits<12> addr;

    let r = oper;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

def L8UI: Load_RRI8<0x00, "l8ui", zextloadi8, addr_ish1, mem8>;
def L16SI: Load_RRI8<0x09, "l16si", sextloadi16, addr_ish2, mem16>;
def L16UI: Load_RRI8<0x01, "l16ui", zextloadi16, addr_ish2, mem16>;
def L32I: Load_RRI8<0x02, "l32i", load, addr_ish4, mem32>;

// Store instructions
let mayStore = 1, usesCustomInserter = 1 in
{
  class Store_II8<bits<4> oper, string instrAsm, SDPatternOperator opNode, 
        ComplexPattern addrOp, Operand memOp>
	  : RRI8_Inst<0x02, (outs), (ins AR:$t, memOp:$addr),
                  instrAsm#"\t$t, $addr", 
                 [(opNode AR:$t, addrOp:$addr)]>
  {
    bits<12> addr;

    let r = oper;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

def S8I: Store_II8<0x04, "s8i", truncstorei8, addr_ish1, mem8>;
def S16I: Store_II8<0x05, "s16i", truncstorei16, addr_ish2, mem16>;
def S32I: Store_II8<0x06, "s32i", store, addr_ish4, mem32>;

def L32R: RI16_Inst<0x01, 
                   (outs AR:$t), (ins L32Rtarget:$label),
				   "l32r\t$t, $label", []>
{
  bits<16> label;	
  let imm16 = label;
}

//pcrel addr loading using L32R
def : Pat<(Xtensa_pcrel_wrapper tconstpool:$in), (L32R tconstpool:$in)>;

//===----------------------------------------------------------------------===//
// Conditional branch instructions
//===----------------------------------------------------------------------===//
let isBranch = 1, isTerminator = 1 in 
{
  class Branch_RR<bits<4> oper, string instrAsm, CondCode CC>
      : RRI8_Inst<0x07, (outs), 
                 (ins AR:$s, AR:$t, brtarget:$target), 
                  instrAsm#"\t$s, $t, $target", 
                 [(brcc CC, AR:$s, AR:$t,  bb:$target)]>
  {
    bits<8> target;

    let r = oper;
    let imm8 = target;
  }

  class Branch_RI<bits<4> oper, string instrAsm, CondCode CC>
      : RRI8_Inst<0x06, (outs), 
                 (ins AR:$s, b4const:$imm, brtarget:$target), 
                  instrAsm#"\t$s, $imm, $target", 
                 [(brcc CC, AR:$s, b4const:$imm,  bb:$target)]>
  {
    bits<4> imm;
    bits<8> target;

    let t = oper;
    let r = imm;
    let imm8 = target;
  }

  class Branch_RIU<bits<4> oper, string instrAsm, CondCode CC>
      : RRI8_Inst<0x06, (outs), 
                 (ins AR:$s, b4constu:$imm, brtarget:$target), 
                  instrAsm#"\t$s, $imm, $target", 
                 [(brcc CC, AR:$s, b4constu:$imm,  bb:$target)]>
  {
    bits<4> imm;
    bits<8> target;

    let t = oper;
    let r = imm;
    let imm8 = target;
  }

  class Branch_RZ<bits<2> n, bits<2> m, string instrAsm, CondCode CC>
      : BRI12_Inst<0x06, n, m, (outs), 
                  (ins AR:$s, brtarget:$target), 
                   instrAsm#"\t$s, $target", 
                  [(brcc CC, AR:$s, (i32 0),  bb:$target)]>
  {
    bits<12> target;

    let imm12 = target;
  }
}

def BEQ: Branch_RR<0x01, "beq", SETEQ>;
def BNE: Branch_RR<0x09, "bne", SETNE>;
def BGE: Branch_RR<0x0A, "bge", SETGE>;
def BLT: Branch_RR<0x02, "blt", SETLT>;
def BGEU: Branch_RR<0x0B, "bgeu", SETUGE>;
def BLTU: Branch_RR<0x03, "bltu", SETULT>;

def BEQI: Branch_RI<0x02, "beqi", SETEQ>;
def BNEI: Branch_RI<0x06, "bnei", SETNE>;
def BGEI: Branch_RI<0x0E, "bgei", SETGE>;
def BLTI: Branch_RI<0x0A, "blti", SETLT>;
def BGEUI: Branch_RIU<0x0F, "bgeui", SETUGE>;
def BLTUI: Branch_RIU<0x0B, "bltui", SETULT>;

def BEQZ: Branch_RZ<0x01, 0x00, "beqz", SETEQ>;
def BNEZ: Branch_RZ<0x01, 0x01, "bnez", SETNE>;
def BGEZ: Branch_RZ<0x01, 0x03, "bgez", SETGE>;
def BLTZ: Branch_RZ<0x01, 0x02, "bltz", SETLT>; 

def BALL: RRI8_Inst<0x07, (outs), 
                   (ins AR:$s, AR:$t, brtarget:$target), 
                   "ball\t$s, $t, $target", []>
{
  bits<8> target;

  let r = 0x04;
  let imm8 = target;
}

def BANY: RRI8_Inst<0x07, (outs), 
                   (ins AR:$s, AR:$t, brtarget:$target), 
                   "bany\t$s, $t, $target", []>
{
  bits<8> target;

  let r = 0x08;
  let imm8 = target;
}

def BBC: RRI8_Inst<0x07, (outs), 
                  (ins AR:$s, AR:$t, brtarget:$target), 
                  "bbc\t$s, $t, $target", []>
{
  bits<8> target;

  let r = 0x05;
  let imm8 = target;
}

def BBS: RRI8_Inst<0x07, (outs), 
                  (ins AR:$s, AR:$t, brtarget:$target), 
                  "bbs\t$s, $t, $target", []>
{
  bits<8> target;

  let r = 0x0d;
  let imm8 = target;
}

def BNALL: RRI8_Inst<0x07, (outs), 
                   (ins AR:$s, AR:$t, brtarget:$target), 
                   "bnall\t$s, $t, $target", []>
{
  bits<8> target;

  let r = 0x0c;
  let imm8 = target;
}

def BNONE: RRI8_Inst<0x07, (outs), 
                   (ins AR:$s, AR:$t, brtarget:$target), 
                   "bnone\t$s, $t, $target", []>
{
  bits<8> target;

  let r = 0x00;
  let imm8 = target;
}

def BBCI: RRI8_Inst<0x07, (outs), 
                   (ins AR:$s, uimm5:$imm, brtarget:$target), 
                   "bbci\t$s, $imm, $target", []>
{
  bits<8> target;
  bits<5> imm;

  let r{3-1} = 0x3;
  let r{0} = imm{4};
  let t{3-0} = imm{3-0};
  let imm8 = target;
}

def BBSI: RRI8_Inst<0x07, (outs), 
                   (ins AR:$s, uimm5:$imm, brtarget:$target), 
                   "bbsi\t$s, $imm, $target", []>
{
  bits<8> target;
  bits<5> imm;

  let r{3-1} = 0x7;
  let r{0} = imm{4};
  let t{3-0} = imm{3-0};
  let imm8 = target;
} 

def : Pat<(brcc SETGT, AR:$s, AR:$t, bb:$target),
          (BLT AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcc SETUGT, AR:$s, AR:$t, bb:$target),
          (BLTU AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcc SETLE, AR:$s, AR:$t, bb:$target),
          (BGE AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcc SETULE, AR:$s, AR:$t, bb:$target),
          (BGEU AR:$t, AR:$s, bb:$target)>;

def : Pat<(brcond (i32 (seteq AR:$s, AR:$t)), bb:$target),
          (BEQ AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setne AR:$s, AR:$t)), bb:$target),
          (BNE AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setge AR:$s, AR:$t)), bb:$target),
          (BGE AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setle AR:$s, AR:$t)), bb:$target),
          (BLT AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setuge AR:$s, AR:$t)), bb:$target),
          (BGEU AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setult AR:$s, AR:$t)), bb:$target),
          (BLTU AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setgt AR:$s, AR:$t)), bb:$target),
          (BLT AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcond (i32 (setugt AR:$s, AR:$t)), bb:$target),
          (BLTU AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcond (i32 (setle AR:$s, AR:$t)), bb:$target),
          (BGE AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcond (i32 (setule AR:$s, AR:$t)), bb:$target),
          (BGEU AR:$t, AR:$s, bb:$target)>;

//===----------------------------------------------------------------------===//
// Call and jump instructions
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1, isBarrier = 1 in 
{
  def J: CALL_Inst<0x06, (outs), (ins jumptarget:$offset), 
                  "j\t$offset", 
                  [(br bb:$offset)]>
  {
    let n = 0x0;
  }

  def JX: CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                    "jx\t$s",
                    [(brind AR:$s)]>
  {
    let m = 0x2;
    let n = 0x2;
    let r = 0;
    let isIndirectBranch = 1;
  }
}

let isCall = 1, Defs = [A0] in 
{
  def CALL0: CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                      "call0\t$offset", []>
  {
    let n = 0;
  }
  
  let isIndirectBranch = 1 in
  {
    def CALLX0: CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                          "callx0\t$s", [(Xtensa_call AR:$s)]>
    {
      let m = 0x3;
      let n = 0x0;
      let r = 0;
    }
  }
} 

let isReturn = 1, isTerminator = 1,
    isBarrier = 1, hasCtrlDep = 1, Uses = [A0] in
{
  
  def RET: CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins), 
                     "ret", [(Xtensa_retflag)]>
  {
    let m = 0x2;
    let n = 0x0;
    let s = 0;
    let r = 0;
  } 
}

// calls
def : Pat<(Xtensa_call AR:$dst),
          (CALLX0 AR:$dst)>;

//===----------------------------------------------------------------------===//
// Mem barrier instructions
//===----------------------------------------------------------------------===//

def MEMW:  RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "memw", [(Xtensa_mem_barrier)]>
{
  let r = 0x2;
  let t = 0x0c;
  let s = 0x0;
  let hasSideEffects = 1;
} 

def EXTW : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
     "extw", []>
{
  let r = 0x2;
  let s = 0x0;
  let t = 0xd;
}

//===----------------------------------------------------------------------===//
// Processor control instructions
//===----------------------------------------------------------------------===//

def DSYNC: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "dsync", []>
{
  let r = 0x2;
  let s = 0x0;
  let t = 0x3;
}

def ISYNC: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "isync", []>
{
  let r = 0x2;
  let s = 0x0;
  let t = 0x0;
}

def RSYNC: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "rsync", []>
{
  let r = 0x2;
  let s = 0x0;
  let t = 0x1;
}

def ESYNC: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "esync", []>
{
  let r = 0x2;
  let s = 0x0;
  let t = 0x2;
}

def NOP: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                 "nop",
                 []>
{
  let r = 0x02;
  let s = 0x00;
  let t = 0x0f;
}

def WSR: RSR_Inst<0x00, 0x03, 0x01, (outs SR:$sr), (ins AR:$t),
                 "wsr\t$t, $sr", []>;

def RSR: RSR_Inst<0x00, 0x03, 0x00, (outs AR:$t), (ins SR:$sr),
                 "rsr\t$t, $sr", []>;

def XSR: RSR_Inst<0x00, 0x01, 0x06, (outs), (ins AR:$t, SR:$sr),
                 "xsr\t$t, $sr", []>;

def WUR: RRR_Inst<0x00, 0x03, 0x0F, (outs UR:$ur), (ins AR:$t),
                 "wur\t$t, $ur", []>
{
  bits<8> ur;

  let r = ur{7-4};
  let s = ur{3-0};
}

def RUR: RRR_Inst<0x00, 0x03, 0x0E, (outs AR:$r), (ins UR:$ur),
                 "rur\t$r, $ur", [(set AR:$r, (Xtensa_rur UR:$ur))]>
{
  bits<8> ur;

  let s = ur{7-4};
  let t = ur{3-0};
}

def RER: RRR_Inst<0x00, 0x00, 0x04, (outs AR:$t), (ins AR:$s),
                 "rer\t$t, $s", []>
{
  let r = 0x6;
}

def WER: RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$t, AR:$s),
                 "wer\t$t, $s", []>
{
  let r = 0x7;
  let hasSideEffects = 1;
}

//===----------------------------------------------------------------------===//
// Stack allocation
//===----------------------------------------------------------------------===//

// ADJCALLSTACKDOWN/UP implicitly use/def SP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
let Defs = [SP], Uses = [SP] in 
{
  def ADJCALLSTACKDOWN: Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2), 
                              "#ADJCALLSTACKDOWN",
                              [(Xtensa_callseq_start timm:$amt1, timm:$amt2)]>;
  def ADJCALLSTACKUP  : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              "#ADJCALLSTACKUP",
                              [(Xtensa_callseq_end timm:$amt1, timm:$amt2)]>;
}

//===----------------------------------------------------------------------===//
// Generic select instruction
//===----------------------------------------------------------------------===//
let usesCustomInserter = 1 in
{
  def SELECT: Pseudo<(outs AR:$dst), (ins AR:$lhs, AR:$rhs, AR:$t, AR:$f, i32imm:$cond),
                    "!select $dst, $lhs, $rhs, $t, $f, $cond",
                    [(set AR:$dst, (Xtensa_select_cc AR:$lhs, AR:$rhs, AR:$t, AR:$f, imm:$cond))]>;
}

//===----------------------------------------------------------------------===//
// Code Density instructions
//===----------------------------------------------------------------------===//

class ArithLogic_RRRN<bits<4> oper0, string instrAsm, 
      SDPatternOperator opNode, bit isComm = 0>
  : RRRN_Inst<oper0, (outs AR:$r), (ins AR:$s, AR:$t),
              instrAsm#"\t$r, $s, $t",
             [(set AR:$r, (opNode AR:$s, AR:$t))]>, Requires<[HasDensity]> 
{
  let isCommutable = isComm;
  let isReMaterializable = 0;
}

def ADD_N: ArithLogic_RRRN<0x0a, "add.n", add, 1>;

def ADDI_N: RRRN_Inst<0x0B, (outs AR:$r), (ins AR:$s, imm1n_15:$imm),
                     "addi.n\t$r, $s, $imm",
                     [(set AR:$r, (add AR:$s, imm1n_15:$imm))]>, Requires<[HasDensity]>
{
  bits<4> imm;

  let t = imm;
}

def MOV_N: RRRN_Inst<0x0D, (outs AR:$t), (ins AR:$s),
                    "mov.n\t$t, $s", []>, Requires<[HasDensity]>
{
  let r = 0;
}

def : InstAlias<"mov\t $t, $s", (OR AR:$t, AR:$s, AR:$s)>;

def MOVI_N: RI7_Inst<0xc, 0x0, (outs AR:$s), (ins imm32n_95:$imm7),
                    "movi.n\t$s, $imm7",
                    [(set AR:$s, imm32n_95:$imm7)]>, Requires<[HasDensity]>;

// Load instruction
let mayLoad = 1, usesCustomInserter = 1 in
{
  def L32I_N: RRRN_Inst<0x8, (outs AR:$t), (ins mem32n:$addr), 
                       "l32i.n\t$t, $addr", []>, Requires<[HasDensity]>
  {
    bits<8> addr;

    let r{3-0} = addr{7-4};
    let s{3-0} = addr{3-0};
  }
}

// Store instruction
let mayStore = 1, usesCustomInserter = 1 in
{
  def S32I_N: RRRN_Inst<0x9, (outs), (ins  AR:$t, mem32n:$addr),
                       "s32i.n\t$t, $addr", []>, Requires<[HasDensity]>
  {
    bits<8> addr;

    let r{3-0} = addr{7-4};
    let s{3-0} = addr{3-0};
  }
}

//Return instruction
let isReturn = 1, isTerminator = 1,
    isBarrier = 1, hasCtrlDep = 1, Uses = [A0] in
{
  def RET_N: RRRN_Inst<0x0D, (outs), (ins),
                "ret.n", [(Xtensa_retflag)]>, Requires<[HasDensity]>
  {
    let r = 0x0F;
    let s = 0;
    let t = 0;
  }
}

//===----------------------------------------------------------------------===//
// Windowed instructions
//===----------------------------------------------------------------------===//

def ENTRY: BRI12_Inst<0x06, 0x3, 0x0, (outs), (ins AR:$s, entry_imm12:$imm), 
                     "entry\t$s, $imm", []>, Requires<[HasWindowed]>
{
  bits<15> imm;

  let imm12{11-0} = imm{14-3};
  let Defs = [SP];
}

//Call instructions
let isCall = 1, Defs = [A0] in 
{
  def CALL4: CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                      "call4\t$offset", []>, Requires<[HasWindowed]>
  {
    let n = 1;
  }

  def CALL8: CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                      "call8\t$offset", []>, Requires<[HasWindowed]>
  {
    let n = 2;
  }

  def CALL12: CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                       "call12\t$offset", []>, Requires<[HasWindowed]>
  {
    let n = 3;
  }

  let isIndirectBranch = 1 in
  {
    def CALLX4: CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                          "callx4\t$s", []>, Requires<[HasWindowed]>
    {
      let m = 0x3;
      let n = 0x1;
      let r = 0;
    }

    def CALLX8: CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                          "callx8\t$s", []>, Requires<[HasWindowed]>
    {
      let m = 0x3;
      let n = 0x2;
      let r = 0;
    }

    def CALLX12: CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                           "callx12\t$s", []>, Requires<[HasWindowed]>
    {
      let m = 0x3;
      let n = 0x3;
      let r = 0;
    }
  }
}

def MOVSP: RRR_Inst<0x00, 0x00, 0x00, (outs AR:$t), (ins AR:$s),
                   "movsp\t$t, $s",
                   [(set AR:$t, (Xtensa_movsp AR:$s))]>, Requires<[HasWindowed]>
{
  let r = 0x01;
}

//Return instructions
let isReturn = 1, isTerminator = 1,
    isBarrier = 1, hasCtrlDep = 1, Uses = [A0] in
{
  def RETW_N: RRRN_Inst<0x0D, (outs), (ins),
                "retw.n", [(Xtensa_retWflag)]>, Requires<[HasWindowed, HasDensity]>
  {
    let r = 0x0F;
    let s = 0;
    let t = 1;
  }

  def RETW: CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins), 
            "retw", [(Xtensa_retWflag)]>, Requires<[HasWindowed]>
  {
    let m = 0x2;
    let n = 0x1;
    let s = 0;
    let r = 0;
  }
}

//Store 32-bit for Window Exceptions
def S32E: RRI4_Inst<0x00, 0x09, (outs), (ins AR:$t, AR:$s, imm64n_4n:$imm),
					"s32e\t$t $s $imm", []>, Requires<[HasWindowed]>
{
  bits<6> imm;

  let r = imm{5-2};
  let imm4 = 0x4;
  let mayStore = 1;
}

def L32E: RRI4_Inst<0x00, 0x09, (outs), (ins AR:$t, AR:$s, imm64n_4n:$imm),
					"l32e\t$t $s $imm", []>, Requires<[HasWindowed]>
{
  bits<6> imm;

  let r = imm{5-2};
  let imm4 = 0x0;
  let mayLoad = 1;
}

//Return from window
def RFWU: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                  "rfwu", []>, Requires<[HasWindowed]>
{
  bits<4> imm;

  let r = 0x3;
  let s = 0x5;
  let t = 0x0;
}

def RFWO: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                  "rfwo", []>, Requires<[HasWindowed]>
{
  bits<4> imm;

  let r = 0x3;
  let s = 0x4;
  let t = 0x0;
}

//Rotate window
def ROTW: RRR_Inst<0x00, 0x00, 0x04, (outs), (ins imm8n_7:$imm),
                  "rotw\t$imm", []>, Requires<[HasWindowed]>
{
  bits<4> imm;

  let r = 0x8;
  let s = 0x0;
  let t = imm{3-0};
}

//===----------------------------------------------------------------------===//
// Floating-Point Instructions
//===----------------------------------------------------------------------===//

class FPArith_RRR<bits<4> oper2, bits<4> oper1, string instrAsm, 
                 SDPatternOperator opNode, bit isComm = 0>
  : RRR_Inst<0x00, oper1, oper2, (outs FPR:$r), (ins FPR:$s, FPR:$t),
             instrAsm#"\t$r, $s, $t",
            [(set FPR:$r, (opNode FPR:$s, FPR:$t))]> 
{
  let isCommutable = isComm;
  let isReMaterializable = 0;
  let Predicates = [HasSingleFloat];
}

def ADD_S: FPArith_RRR<0x00, 0x0A, "add.s", fadd, 1>;
def SUB_S: FPArith_RRR<0x01, 0x0A, "sub.s", fsub>;
def MUL_S: FPArith_RRR<0x02, 0x0A, "mul.s", fmul, 1>;

def ABS_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                   "abs.s\t$r, $s",
                   [(set FPR:$r, (fabs FPR:$s))]> 
{
  let t = 0x01;
} 

def NEG_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                   "neg.s\t$r, $s",
                   [(set FPR:$r, (fneg FPR:$s))]> 
{
  let t = 0x06;
} 

def TRUNC_S: RRR_Inst<0x00, 0x0A, 0x09, (outs AR:$r), (ins FPR:$s),
                     "trunc.s\t$r, $s, 0",
                     [(set AR:$r, (fp_to_sint FPR:$s))]> 
{
  let t = 0x00;
} 

def UTRUNC_S: RRR_Inst<0x00, 0x0A, 0x0e, (outs AR:$r), (ins FPR:$s),
                      "utrunc.s\t$r, $s, 0",
                      [(set AR:$r, (fp_to_uint FPR:$s))]> 
{
  let t = 0x00;
} 

def FLOAT_S: RRR_Inst<0x00, 0x0A, 0x0c, (outs FPR:$r), (ins AR:$s),
                     "float.s\t$r, $s, 0",
                     [(set FPR:$r, (sint_to_fp AR:$s))]> 
{
  let t = 0x00;
} 

def UFLOAT_S: RRR_Inst<0x00, 0x0A, 0x0D, (outs FPR:$r), (ins AR:$s),
                      "ufloat.s\t$r, $s, 0",
                      [(set FPR:$r, (uint_to_fp AR:$s))]> 
{
  let t = 0x00;
} 

def RFR: RRR_Inst<0x00, 0x0A, 0x0f, (outs AR:$r), (ins FPR:$s),
                 "rfr\t$r, $s",
                 [(set AR:$r, (bitconvert FPR:$s))]> 
{
  let t = 0x04;
} 

def WFR: RRR_Inst<0x00, 0x0A, 0x0f, (outs FPR:$r), (ins AR:$s),
                 "wfr\t$r, $s",
                 [(set FPR:$r, (bitconvert AR:$s))]> 
{
  let t = 0x05;
} 

// FP load instructions
let mayLoad = 1, usesCustomInserter = 1, Predicates = [HasSingleFloat] in
{
  class LoadF_RRI8<bits<4> oper, string instrAsm, SDPatternOperator opNode, 
                  ComplexPattern addrOp,Operand memOp>: RRI8_Inst<0x03, (outs FPR:$t), (ins memOp:$addr), 
                   instrAsm#"\t$t, $addr", 
                  [(set FPR:$t, (opNode addrOp:$addr))]>
  {
    bits<12> addr;

    let r = oper;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

def L32F: LoadF_RRI8<0x00, "lsi", load, addr_ish4, mem32>, Requires<[]>;

// FP store instructions
let mayStore = 1, usesCustomInserter = 1, Predicates = [HasSingleFloat] in
{
  class StoreF_RRI8<bits<4> oper, string instrAsm, SDPatternOperator opNode, 
                   ComplexPattern addrOp, Operand memOp>: RRI8_Inst<0x03, (outs), (ins FPR:$t, memOp:$addr),
                    instrAsm#"\t$t, $addr", 
                   [(opNode FPR:$t, addrOp:$addr)]>
  {
    bits<12> addr;

    let r = oper;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

def S32F: StoreF_RRI8<0x04, "ssi", store, addr_ish4, mem32>;

// FP compare instructions
let isCompare = 1, Predicates = [HasSingleFloat] in 
{
  class FCompare <bits<4> oper2, bits<4> oper1, string instrAsm, 
                 SDPatternOperator opNode, bit isComm = 0>
    : RRR_Inst<0x00, oper1, oper2, (outs BR:$b), (ins FPR:$s, FPR:$t),
               instrAsm#"\t$b, $s, $t",
              [(set BR:$b, (opNode FPR:$s, FPR:$t))]> 
  {
    let isCommutable = isComm;
    let isReMaterializable = 0;
	let Predicates = [HasSingleFloat];
  }
}

def OEQ_S:  FCompare<0x02, 0x0b, "oeq.s", Xtensa_cmpoeq, 1>;
def OLT_S:  FCompare<0x04, 0x0b, "olt.s", Xtensa_cmpolt, 1>;
def OLE_S:  FCompare<0x06, 0x0b, "ole.s", Xtensa_cmpole, 1>;

def UEQ_S:  FCompare<0x03, 0x0b, "ueq.s", Xtensa_cmpueq, 1>;
def ULT_S:  FCompare<0x05, 0x0b, "ult.s", Xtensa_cmpult, 1>;
def ULE_S:  FCompare<0x07, 0x0b, "ule.s", Xtensa_cmpule, 1>;
def UN_S:   FCompare<0x01, 0x0b, "un.s",  Xtensa_cmpuo, 1>;

//FP complex operations
def MADD_S: RRR_Inst<0x00, 0x0A, 0x04, (outs FPR:$r), (ins FPR:$a, FPR:$s, FPR:$t),
                    "madd.s\t$r, $s, $t",
                    [(set FPR:$r, (Xtensa_madd FPR:$a, FPR:$s, FPR:$t))]>, Requires<[HasSingleFloat]>
{
  let isCommutable = 0;
  let isReMaterializable = 0;
  let Constraints = "$r = $a";
}

def MSUB_S: RRR_Inst<0x00, 0x0A, 0x05, (outs FPR:$r), (ins FPR:$a, FPR:$s, FPR:$t),
                    "msub.s\t$r, $s, $t",
                    [(set FPR:$r, (Xtensa_msub FPR:$a, FPR:$s, FPR:$t))]>, Requires<[HasSingleFloat]> 
{
  let isCommutable = 0;
  let isReMaterializable = 0;
  let Constraints = "$r = $a";
}

//FP move operations
def MOV_S: RRR_Inst<0x00, 0x0A, 0x0f, (outs FPR:$r), (ins FPR:$s),
                   "mov.s\t$r, $s",
                   [(set FPR:$r, (Xtensa_movs FPR:$s))]>, Requires<[HasSingleFloat]>
{
  let t = 0x00;
}

def CONST_S: RRR_Inst<0x00, 0x0a, 0x0f, (outs FPR:$r), (ins uimm4:$imm),
					"const.s\t$r, $imm", []>, Requires<[HasSingleFloat]> 
{
  bits<4> imm;

  let t = 0x03;
  let s = imm{3-0};
}

def DIV0_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
					"div0.s\t$r, $s", []>, Requires<[HasSingleFloat]> 
{
  let t = 0x7;
}

def MADDN_S: RRR_Inst<0x00, 0x0A, 0x06, (outs FPR:$r), (ins FPR:$s, FPR:$t),
					"maddn.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]> 
{
  let isCommutable = 0;
}

def MKDADJ_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
					"mkdadj.s\t$r, $s", []>, Requires<[HasSingleFloat]> 
{
  let t = 0x0D;
}

def MKSADJ_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
					"mksadj.s\t$r, $s", []>, Requires<[HasSingleFloat]> 
{
  let t = 0x0C;
}

def ADDEXP_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
					"addexp.s\t$r, $s", []>, Requires<[HasSingleFloat]> 
{
  let t = 0x0E;
}

def ADDEXPM_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
					"addexpm.s\t$r, $s", []>, Requires<[HasSingleFloat]> 
{
  let t = 0x0F;
}

def DIVN_S: RRR_Inst<0x00, 0x0A, 0x07, (outs FPR:$r), (ins FPR:$s, FPR:$t),
					"divn.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]> 
{
}

def NEXP01_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
					"nexp01.s\t$r, $s", []>, Requires<[HasSingleFloat]> 
{
  let t = 0x0B;
}

def SQRT0_S: RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
					"sqrt0.s\t$r, $s", []>, Requires<[HasSingleFloat]> 
{
  let t = 0x09;
}

//===----------------------------------------------------------------------===//
// Boolean branch Instructions
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1, Predicates = [HasBoolean] in 
{
  def BT: RRI8_Inst<0x06, (outs), (ins BR:$b, brtarget:$target), 
                   "bt\t$b, $target", 
                   [(Xtensa_brcc_t BR:$b, bb:$target)]>
  {
    bits<8> target;
    bits<4> b;

    let r = 0x1;
    let s = b;
    let t = 0x7; 
    let imm8 = target;
  }

  def BF: RRI8_Inst<0x06, (outs), (ins BR:$b, brtarget:$target), 
                   "bf\t$b, $target",
                   [(Xtensa_brcc_f BR:$b, bb:$target)]>
  {
    bits<8> target;
    bits<4> b;

    let r = 0x0;
    let s = b;
    let t = 0x7; 
    let imm8 = target;
  }
}


//===----------------------------------------------------------------------===//
// FP select operations
let usesCustomInserter = 1 in
{
  def SELECT_CC_FP_INT: Pseudo<(outs AR:$dst), (ins FPR:$lhs, FPR:$rhs, AR:$t, AR:$f, i32imm:$cond),
                                    "!select_cc_fp_int $dst, $lhs, $rhs, $t, $f, $cond",
                                    [(set AR:$dst, (Xtensa_select_cc_fp FPR:$lhs, FPR:$rhs, AR:$t, AR:$f, imm:$cond))]>;
  def SELECT_CC_INT_FP: Pseudo<(outs FPR:$dst), (ins AR:$lhs, AR:$rhs, FPR:$t, FPR:$f, i32imm:$cond),
                                    "!select_cc_int_fp $dst, $lhs, $rhs, $t, $f, $cond",
                                    [(set FPR:$dst, (Xtensa_select_cc_fp AR:$lhs, AR:$rhs, FPR:$t, FPR:$f, imm:$cond))]>;
  def SELECT_CC_FP_FP: Pseudo<(outs FPR:$dst), (ins FPR:$lhs, FPR:$rhs, FPR:$t, FPR:$f, i32imm:$cond),
                                    "!select_cc_fp_fp $dst, $lhs, $rhs, $t, $f, $cond",
                                    [(set FPR:$dst, (Xtensa_select_cc_fp FPR:$lhs, FPR:$rhs, FPR:$t, FPR:$f, imm:$cond))]>;
}

// Shift Pseudo instructions:
// SSL/SSR + Shift combination
let usesCustomInserter = 1 in
{
  def SLL_P: Pseudo<(outs AR:$r), (ins AR:$s, AR:$sa),
                       "# SLL_P $r, $s, $sa",
                       [(set AR:$r, (shl AR:$s, AR:$sa))]>;
 
  def SRA_P: Pseudo<(outs AR:$r), (ins AR:$t, AR:$sa),
                       "# SRA_P $r, $t, $sa",
                       [(set AR:$r, (sra AR:$t, AR:$sa))]>;

  def SRL_P: Pseudo<(outs AR:$r), (ins AR:$t, AR:$sa),
                       "# SRL_P $r, $t, $sa",
                       [(set AR:$r, (srl AR:$t, AR:$sa))]>;
}

// Xtensa missed L8I load operation, use pseudo operation
let usesCustomInserter = 1 in
def L8I_P: Pseudo<(outs AR:$t), (ins mem8:$addr),
               "!L8I_P $t, $addr",
                [(set AR:$t, (sextloadi8 
				addr_ish1:$addr))]>;

def SEXT: RRR_Inst<0x00, 0x03, 0x02, (outs AR:$r), (ins AR:$s, seimm7_22:$imm),
     "sext\t$r, $s, $imm",
     []>,Requires<[HasSEXT]>
{
  bits<4> imm;

  let t = imm;
}

// FrameIndexes are legalized when they are operands from load/store
// instructions. The same not happens for stack address copies, so an
// add op with mem ComplexPattern is used and the stack address copy
// can be matched.
// Setting of attribute mayLoad is trick to process instruction operands
// in function XtensaRegisterInfo::eliminateFI

let isCodeGenOnly = 1, mayLoad = 1 in
{

  def LEA_ADD : RRI8_Inst<0x02, (outs AR:$t), (ins mem32:$addr),
       "addi\t$t, $addr",
       [(set AR:$t, addr_ish4:$addr)]>
  {
   bits<12> addr;

   let r = 0x0C;
   let imm8{7-0} = addr{11-4};
   let s{3-0} = addr{3-0};
  }
}

def MULL: ArithLogic_RRR<0x08, 0x02, "mull", mul, 1>, Requires<[HasMul32]>;
def MULUH: ArithLogic_RRR<0x0A, 0x02, "muluh", mulhu, 1>, Requires<[HasMul32High]>;
def MULSH: ArithLogic_RRR<0x0B, 0x02, "mulsh", mulhs, 1>, Requires<[HasMul32High]>;
def QUOS: ArithLogic_RRR<0x0D, 0x02, "quos", sdiv>, Requires<[HasDiv32]>;
def QUOU: ArithLogic_RRR<0x0C, 0x02, "quou", udiv>, Requires<[HasDiv32]>;
def REMS: ArithLogic_RRR<0x0F, 0x02, "rems", srem>, Requires<[HasDiv32]>;
def REMU: ArithLogic_RRR<0x0E, 0x02, "remu", urem>, Requires<[HasDiv32]>;

let Predicates = [HasNSA] in {
  def NSA : RRR_Inst<0x00, 0x00, 0x04, (outs AR:$t), (ins AR:$s),
       "nsa\t$t, $s",
       []>
  {
    let r = 0xE;
  }

  def NSAU : RRR_Inst<0x00, 0x00, 0x04, (outs AR:$t), (ins AR:$s),
       "nsau\t$t, $s",
       []>
  {
    let r = 0xF;
  }
}

def : Pat<(brcond AR:$s, bb:$target),
          (BNEZ AR:$s, bb:$target)>;

// It's a trick for redundant branches with -O0 option
// like br i1 true, label ...
def : Pat<(brcond (i32 1), bb:$target),
          (J bb:$target)>;

let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1, Size = 3 in 
{
  def BR_JT: Pseudo<(outs), (ins AR:$s, i32imm:$jt),
                     "!br_jt_p, $s, $jt",
                    [(Xtensa_brjt AR:$s, tjumptable:$jt)]>;
}

//extended loads
def : Pat<(i32 (extloadi1  addr_ish1:$addr)), (L8UI addr_ish1:$addr)>;
def : Pat<(i32 (extloadi8  addr_ish1:$addr)), (L8UI addr_ish1:$addr)>;
def : Pat<(i32 (extloadi16 addr_ish2:$addr)), (L16UI addr_ish2:$addr)>;

//break
let isBarrier = 1, isTerminator = 1 in
{
  def BREAK: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins i32imm:$t, i32imm:$s),
                     "break\t$t, $s", []>
  {
    let r = 0x04;
  }
}

def: Pat<(trap), (BREAK (i32 1), (i32 15))>;

// calls
def : Pat<(Xtensa_call (i32 tglobaladdr:$dst)),
          (CALL0 tglobaladdr:$dst)>;
def : Pat<(Xtensa_callw (i32 tglobaladdr:$dst)),
          (CALL8 tglobaladdr:$dst)>;
def : Pat<(Xtensa_call (i32 texternalsym:$dst)),
          (CALL0 texternalsym:$dst)>;
def : Pat<(Xtensa_callw (i32 texternalsym:$dst)),
          (CALL8 texternalsym:$dst)>;
def : Pat<(Xtensa_call AR:$dst),
          (CALLX0 AR:$dst)>;
def : Pat<(Xtensa_callw AR:$dst),
          (CALLX8 AR:$dst)>;

def RSIL: RRR_Inst<0x00, 0x00, 0x00, (outs AR:$t), (ins uimm4:$imm),
					"rsil\t$t, $imm", []>
{
  bits<4> imm;

  let r = 0x6;
  let s = imm{3-0};
}

def WAITI: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins uimm4:$imm),
					"waiti\t$imm", []>
{
  bits<4> imm;

  let r = 0x7;
  let s = imm{3-0};
  let t = 0;
}

def WDTLB: RRR_Inst<0x00, 0x00, 0x05, (outs AR:$t), (ins AR:$s),
					"wdtlb\t$t, $s", []>
{
  let r = 0xE;
}

def WITLB: RRR_Inst<0x00, 0x00, 0x05, (outs AR:$t), (ins AR:$s),
					"witlb\t$t, $s", []>
{
  let r = 0x6;
}

def RFI: RRR_Inst<0x00, 0x00, 0x00, (outs), (ins uimm4:$imm),
					"rfi\t$imm", []>
{
  bits<4> imm;

  let r = 0x3;
  let s = imm{3-0};
  let t = 0x1;
}

let mayStore = 1, mayLoad = 1, usesCustomInserter = 1, Predicates = [HasS32C1I] in
{
  def S32C1I: RRI8_Inst<0x02, (outs AR:$a), (ins AR:$t, mem32:$addr),
                       "s32c1i\t$t, $addr", []>
  {
    bits<12> addr;

    let r = 0x0e;
    let Uses = [SCOMPARE1];
    let Constraints = "$a = $t";
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

//===----------------------------------------------------------------------===//
// Atomic patterns
//===----------------------------------------------------------------------===//

def : Pat<(i32 (atomic_load_8  addr_ish1:$addr)), (L8UI addr_ish1:$addr)>;
def : Pat<(i32 (atomic_load_16 addr_ish2:$addr)), (L16UI addr_ish2:$addr)>;
def : Pat<(i32 (atomic_load_32 addr_ish4:$addr)), (L32I addr_ish4:$addr)>;

def : Pat<(atomic_store_8  addr_ish1:$addr, AR:$t), (S8I AR:$t, addr_ish1:$addr)>;
def : Pat<(atomic_store_16 addr_ish2:$addr, AR:$t), (S16I AR:$t, addr_ish2:$addr)>;
def : Pat<(atomic_store_32 addr_ish4:$addr, AR:$t), (S32I AR:$t, addr_ish4:$addr)>;

let usesCustomInserter = 1, Predicates = [HasS32C1I] in
{
	def ATOMIC_CMP_SWAP_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$cmp, AR:$swap),
										"!atomic_cmp_swap_8_p, $dst, $ptr, $cmp, $swap",
										[(set AR:$dst, (atomic_cmp_swap_8 AR:$ptr, AR:$cmp, AR:$swap))]>;
	def ATOMIC_CMP_SWAP_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$cmp, AR:$swap),
										"!atomic_cmp_swap_16_p, $dst, $ptr, $cmp, $swap",
										[(set AR:$dst, (atomic_cmp_swap_16 AR:$ptr, AR:$cmp, AR:$swap))]>;
	def ATOMIC_CMP_SWAP_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$cmp, AR:$swap),
										"!atomic_cmp_swap_32_p, $dst, $ptr, $cmp, $swap",
										[(set AR:$dst, (atomic_cmp_swap_32 AR:$ptr, AR:$cmp, AR:$swap))]>;

	def ATOMIC_LOAD_ADD_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_add_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_add_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_ADD_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_add_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_add_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_ADD_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_add_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_add_32 AR:$ptr, AR:$arg))]>;

	def ATOMIC_LOAD_SUB_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_sub_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_sub_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_SUB_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_sub_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_sub_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_SUB_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_sub_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_sub_32 AR:$ptr, AR:$arg))]>;

	def ATOMIC_LOAD_AND_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_and_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_and_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_AND_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_and_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_and_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_AND_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_and_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_and_32 AR:$ptr, AR:$arg))]>;

	def ATOMIC_LOAD_OR_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_or_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_or_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_OR_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_or_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_or_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_OR_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_or_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_or_32 AR:$ptr, AR:$arg))]>;

	def ATOMIC_LOAD_XOR_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_xor_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_xor_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_XOR_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_xor_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_xor_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_XOR_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_xor_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_xor_32 AR:$ptr, AR:$arg))]>;

	def ATOMIC_LOAD_NAND_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_nand_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_nand_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_NAND_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_nand_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_nand_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_NAND_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_nand_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_nand_32 AR:$ptr, AR:$arg))]>;

	def ATOMIC_LOAD_MIN_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_min_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_min_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_MIN_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_min_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_min_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_MIN_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_min_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_min_32 AR:$ptr, AR:$arg))]>;			
										
	def ATOMIC_LOAD_MAX_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_max_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_max_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_MAX_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_max_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_max_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_MAX_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_max_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_max_32 AR:$ptr, AR:$arg))]>;	

	def ATOMIC_LOAD_UMIN_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_umin_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_umin_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_UMIN_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_umin_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_umin_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_UMIN_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_umin_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_umin_32 AR:$ptr, AR:$arg))]>;			
										
	def ATOMIC_LOAD_UMAX_8_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_umax_8_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_umax_8 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_UMAX_16_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_umax_16_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_umax_16 AR:$ptr, AR:$arg))]>;
	def ATOMIC_LOAD_UMAX_32_P: Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
										"!atomic_load_umax_32_p, $dst, $ptr, $arg",
										[(set AR:$dst, (atomic_load_umax_32 AR:$ptr, AR:$arg))]>;	
}

