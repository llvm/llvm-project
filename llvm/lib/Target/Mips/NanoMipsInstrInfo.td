//===-- NanoMipsInstrInfo.td - nanoMIPS Instr Information --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes NanoMips instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips profiles and nodes
//===----------------------------------------------------------------------===//

// TODO: Add nodes

//===----------------------------------------------------------------------===//
// nanoMIPS Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

def uimm2_nm : Operand<i32>;
def uimm2code_nm : Operand<i32>;
def uimm3_nm : Operand<i32>;
def uimm3shift_nm : Operand<i32>;
def uimm4mask_nm : Operand<i32>;
def uimm3plus1_nm : Operand<i32>;
def uimm5s2_nm : Operand<i32>;
def uimm6_nm : Operand<i32>;
def uimm8s2_nm : Operand<i32>;
def uimm8s4_nm : Operand<i32>;
def uimm12_nm : Operand<i32>;
def uimm12mask_nm : Operand<i32>;
def uimm12s3_nm : Operand<i32>;
def uimm16_nm : Operand<i32>;
def uimm18_nm : Operand<i32>;
def uimm19_nm : Operand<i32>;
def uimm5s1_nm : Operand<i32>;
def simm22s1_nm : Operand<i32>;
def simm32_nm : Operand<i32>;
def simm32s12_nm : Operand<i32>;
def simm32s12_pcrel_nm : Operand<i32>;
def nimm7_nm : Operand<i32>;
def nimm12_nm : Operand<i32>;

def mem_nm_sp5s2 : Operand<iPTR>;
def mem_nm_gp7s2 : Operand<iPTR>;
def mem_nm_6s2 : Operand<iPTR>;
def mem_nm_3s1 : Operand<iPTR>;
def mem_nm_2 : Operand<iPTR>;
def mem_nm_gp19s2 : Operand<iPTR>;
def mem_nm_gp18 : Operand<iPTR>;
def mem_nm_gp17s1 : Operand<iPTR>;
def mem_nm_u12 : Operand<iPTR>;
def mem_nm_s9 : Operand<iPTR>;
def mem_nm_rx : Operand<iPTR>;
def mem_nm_rx16 : Operand<iPTR>;
def mem_nm_4s2 : Operand<iPTR>;
def mem_nm_7s2 : Operand<iPTR>;
def mem_nm_0 : Operand<iPTR>;
def mem_nm_0_no_offset : Operand<iPTR>;

def reglist16_nm : Operand<OtherVT>;
def reglist_nm : Operand<OtherVT>;

foreach I = {7, 10, 11, 14, 25} in
  def brtarget # I # _nm : Operand<OtherVT>;

def brtarget4s1_nm : Operand<OtherVT>;

def calltarget21_nm : Operand<iPTR>;
def calltarget10_nm : Operand<iPTR>;
def calltarget25_nm : Operand<iPTR>;

def sym32_pc_nm : Operand<pAny>;
def sym32_pc22_nm : Operand<pAny>;
def sym32_gp_nm : Operand<pAny>;
def sym32_gp18_nm : Operand<pAny>;
def sym32_gp19s2_nm : Operand<pAny>;
def sym32_nm : Operand<pAny>;

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

// 16-bit arithmetic and logical instructions with 4x4 register operands
// P16_4X4
class Arith4x4_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$dst), (ins RO:$rt, RO:$rs),
           !strconcat(opstr, "\t$dst, $rt, $rs")>;

// 32-bit arithmetic and logical instructions with 3 register operands.
class ArithLogicR32_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rd), (ins RO:$rs, RO:$rt),
           !strconcat(opstr, "\t$rd, $rs, $rt")>;

// 16-bit arithmetic and logical instructions with 2 register operands
// POOL16C_0
class ArithLogicR16_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rd), (ins RO:$rs, RO:$rt),
           !strconcat(opstr, "\t$rd, $rs, $rt")>;

class ArithLogicR16U_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rd), (ins RO:$rs),
           !strconcat(opstr, "\t$rd, $rs")>;

// 16-bit arithmetic instructions with 2 register operands, POOL16C_0
class AddSubR16_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rd), (ins RO:$rs, RO:$rt),
           !strconcat(opstr, "\t$rd, $rs, $rt")>;

class ArithLogicINM2Reg_Desc<string opstr, Operand Od, RegisterOperand DRO,
                             RegisterOperand SRO>
  : InstNM<(outs DRO:$rt), (ins SRO:$rs, Od:$imm),
           !strconcat(opstr, "\t$rt, $rs, $imm")>;

// Arithmetic and logical instructions with 2 register operands and immediate.
class ArithLogicINM_Desc<string opstr, Operand Od, RegisterOperand RO>
  : ArithLogicINM2Reg_Desc<opstr, Od, RO, RO>;

class ShiftI_Desc<string opstr, Operand Od, RegisterOperand RO>
  : ArithLogicINM_Desc<opstr, Od, RO>;

class ExtBase_Desc<RegisterOperand RO = GPRNM32Opnd>
  : InstNM<(outs RO:$rt), (ins RO:$rs, uimm5:$pos, uimm5_plus1:$size),
           "ext\t$rt, $rs, $pos, $size">;

class InsBase_Desc<RegisterOperand RO = GPRNM32Opnd>
  : InstNM<(outs RO:$rt),
           (ins RO:$rs, uimm5:$pos, uimm5_inssize_plus1:$size, RO:$src),
           "ins \t$rt, $rs, $pos, $size">, RegConstraint<"$src = $rt">;

class AddImmediate_Desc<string opstr, Operand Od,
                        RegisterOperand DRO, RegisterOperand SRO>
  : ArithLogicINM2Reg_Desc<opstr, Od, DRO, SRO>;

class CondLogicI32_Desc<string opstr, Operand Od, RegisterOperand RO>
  : ArithLogicINM_Desc<opstr, Od, RO>;

let isMoveReg = 1 in {
  class MoveBase_Desc
    : InstNM<(outs GPRNM32NZOpnd:$rt), (ins GPRNM32Opnd:$rs),
            "move\t$rt, $rs">;

  class MovePBase_Desc<RegisterOperand SR1O, RegisterOperand SR2O,
                       RegisterOperand DR1O, RegisterOperand DR2O>
    : InstNM<(outs DR1O:$dst1, DR2O:$dst2), (ins SR1O:$src1, SR2O:$src2),
            "movep\t$dst1, $dst2, $src1, $src2">;
}

class Call_Desc<string opstr, DAGOperand calltarget>
  : InstNM<(outs), (ins calltarget:$addr), !strconcat(opstr, "\t$addr")> {
  let isCall = 1;
  let isCTI = 1;
  let Defs = [RA_NM];
}

// Indirect branch instruction.
class IndirectBranch_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs), (ins RO:$rs), !strconcat(opstr, "\t$rs")> {
  let isTerminator = 1;
  let isBarrier = 1;
  let isCTI = 1;
  let isBranch = 1;
  let isIndirectBranch = 1;
}

class BCBase_Desc<DAGOperand brtarget>
  : InstNM<(outs), (ins brtarget:$addr), "bc\t$addr"> {
  let isTerminator = 1;
  let isBarrier = 1;
  let isBranch = 1;
  let isCTI = 1;
}

class LoadImmAsm_Desc<string opstr, RegisterOperand RO, Operand Od>
  : InstNM<(outs RO:$rt), (ins Od:$eu), !strconcat(opstr, "\t$rt, $eu")>;

class LoadAddressAbs_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs RO:$rt), (ins MO:$imm), !strconcat(opstr, "\t$rt, $imm")>;

//===----------------------------------------------------------------------===//
// Instruction Definitions
//===----------------------------------------------------------------------===//

// 16-bit register operations
def NOT16_NM : ArithLogicR16U_Desc<"not", GPRNM16R3Opnd>,
               ArithLogicR16_Enc<0b00>;
let isCommutable = 1, TwoOperandAliasConstraint = "$rt = $rd" in {
  def XOR16_NM : ArithLogicR16_Desc<"xor", GPRNM16R3Opnd>,
                 ArithLogicR16_Enc<0b01>;
  def AND16_NM : ArithLogicR16_Desc<"and", GPRNM16R3Opnd>,
                 ArithLogicR16_Enc<0b10>;
  def OR16_NM  : ArithLogicR16_Desc<"or", GPRNM16R3Opnd>,
                 ArithLogicR16_Enc<0b11>;
}

let TwoOperandAliasConstraint = "$rs = $rt" in
def ANDI16_NM : ArithLogicINM2Reg_Desc<"andi", uimm4mask_nm,
                                       GPRNM16R3Opnd, GPRNM16R3Opnd>,
                ANDI16_Enc<0b11100>;

// _POOL32A0_0 pool of instructions.
let TwoOperandAliasConstraint = "$rs = $rd" in {
  def SLLV_NM : ArithLogicR32_Desc<"sllv", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b000010>;
  def SRLV_NM : ArithLogicR32_Desc<"srlv", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b000110>;
  def SRAV_NM : ArithLogicR32_Desc<"srav", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b001010>;
  def ROTRV_NM: ArithLogicR32_Desc<"rotrv", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b001110>;
  def ADD_NM  : ArithLogicR32_Desc<"add", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b010010>;
  def ADDu_NM : ArithLogicR32_Desc<"addu", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b010110>;
  def SUB_NM  : ArithLogicR32_Desc<"sub", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b011010>;
  def SUBu_NM : ArithLogicR32_Desc<"subu", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b011110>;
  def AND_NM  : ArithLogicR32_Desc<"and", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b100110>;
  def OR_NM   : ArithLogicR32_Desc<"or", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b101010>;
  def NOR_NM  : ArithLogicR32_Desc<"nor", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b101110>;
  def XOR_NM  : ArithLogicR32_Desc<"xor", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b110010>;
  def SLT_NM  : ArithLogicR32_Desc<"slt", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b110110>;
  def SLTU_NM : ArithLogicR32_Desc<"sltu", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b111010>;
  def MUL_NM  : ArithLogicR32_Desc<"mul", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b000011>;
  def MUH_NM  : ArithLogicR32_Desc<"muh", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b000111>;
  def MULU_NM : ArithLogicR32_Desc<"mulu", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b001011>;
  def MUHU_NM : ArithLogicR32_Desc<"muhu", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b001111>;
  def DIV_NM  : ArithLogicR32_Desc<"div", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b010011>;
  def MOD_NM  : ArithLogicR32_Desc<"mod", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b010111>;
  def DIVU_NM : ArithLogicR32_Desc<"divu", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b011011>;
  def MODU_NM : ArithLogicR32_Desc<"modu", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b011111>;
  def SOV_NM  : ArithLogicR32_Desc<"sov", GPRNM32Opnd>,
                ArithLogicR32_Enc<0b111110>;
}

// 16-bit register operations
def SUBu16_NM  : AddSubR16_Desc<"subu", GPRNM16R3Opnd>,
                 AddSubR16_Enc<0b1>;
let isCommutable = 1 in {
  def ADDu16_NM  : AddSubR16_Desc<"addu", GPRNM16R3Opnd>,
                   AddSubR16_Enc<0b0>;
  def ADDu4x4_NM : Arith4x4_Desc<"addu", GPRNM4Opnd>,
                   Arith4x4_Enc<0b00>;
  def MUL4x4_NM  : Arith4x4_Desc<"mul", GPRNM4Opnd>,
                   Arith4x4_Enc<0b01>;
}

let TwoOperandAliasConstraint = "$rs = $rt" in
def ORI_NM : CondLogicI32_Desc<"ori", uimm12mask_nm, GPRNM32Opnd>,
             CondLogicI32_Enc<0b0000>;

def XORI_NM  : CondLogicI32_Desc<"xori", uimm12mask_nm, GPRNM32Opnd>,
               CondLogicI32_Enc<0b0001>;
def ANDI_NM  : CondLogicI32_Desc<"andi", uimm12mask_nm, GPRNM32Opnd>,
               CondLogicI32_Enc<0b0010>;
def SLTI_NM  : CondLogicI32_Desc<"slti", uimm12_nm, GPRNM32Opnd>,
               CondLogicI32_Enc<0b0100>;
def SLTIU_NM : CondLogicI32_Desc<"sltiu", uimm12_nm, GPRNM32Opnd>,
               CondLogicI32_Enc<0b0101>;
def SEQI_NM  : CondLogicI32_Desc<"seqi", uimm12_nm, GPRNM32Opnd>,
               CondLogicI32_Enc<0b0110>;

// P.SHIFT pool of instructions
def SLL16_NM : ArithLogicINM2Reg_Desc<"sll", uimm3shift_nm,
                                      GPRNM16R3Opnd, GPRNM16R3Opnd>,
               ShiftI16_Enc<0b0>;
def SLL_NM   : ShiftI_Desc<"sll", uimm5, GPRNM32Opnd>, ShiftI32_Enc<0b0000>;
def SRL16_NM : ArithLogicINM2Reg_Desc<"srl", uimm3shift_nm,
                                      GPRNM16R3Opnd, GPRNM16R3Opnd>,
               ShiftI16_Enc<0b1>;
def SRL_NM   : ShiftI_Desc<"srl", uimm5, GPRNM32Opnd>, ShiftI32_Enc<0b0010>;
def SRA_NM   : ShiftI_Desc<"sra", uimm5, GPRNM32Opnd>, ShiftI32_Enc<0b0100>;
def ROTR_NM  : ShiftI_Desc<"rotr", uimm5, GPRNM32Opnd>, ShiftI32_Enc<0b0110>;

def EXT_NM : ExtBase_Desc, ExtInsBase_Enc<0b1111>;
def INS_NM : InsBase_Desc, ExtInsBase_Enc<0b1110>;

let TwoOperandAliasConstraint = "$rs = $rt" in
def ADDIU_NM : AddImmediate_Desc<"addiu", uimm16_nm, GPRNM32Opnd, GPRNM32Opnd>,
               AddIU_Enc;

def ADDIUNEG_NM  : AddImmediate_Desc<"addiu", nimm12_nm,
                                     GPRNM32Opnd, GPRNM32Opnd>,
                   CondLogicI32_Enc<0b1000>;

def ADDIUR2_NM   : ArithLogicINM2Reg_Desc<"addiu", uimm5s2_nm,
                                          GPRNM16R3Opnd, GPRNM16R3Opnd>,
                   ADDIUR2_Enc;

def ADDIURS5_NM  : AddImmediate_Desc<"addiu", simm4,
                                     GPRNM16R5NZOpnd, GPRNM16R5NZOpnd>,
                   ADDIURS5_Enc, RegConstraint<"$rs = $rt">;

def ADDIUR1SP_NM : ArithLogicINM2Reg_Desc<"addiu", uimm8s2_nm,
                                          GPRNM16R3Opnd, GPRNMSPOpnd>,
                   ADDIUR1SP_Enc;

def ADDIU48_NM   : ArithLogicINM_Desc<"addiu", sym32_nm, GPRNM48Opnd>,
                   Op48_Imm_Enc<0b00001>, RegConstraint<"$rs = $rt">;

class AddImmediateGP_Desc<string opstr, Operand Od, RegisterOperand RO>
  : InstNM<(outs RO:$rt), (ins GPRNMGPOpnd:$rs, Od:$offset),
           !strconcat(opstr, "\t$rt, $rs, $offset")>;

def ADDIUGPB_NM : AddImmediateGP_Desc<"addiu", sym32_gp18_nm, GPRNM32Opnd>,
                  ADDIUGPB_Enc;

def ADDIUGPW_NM : AddImmediateGP_Desc<"addiu", sym32_gp19s2_nm, GPRNM32Opnd>,
                  ADDIUGPW_Enc;

class LoadAddressPCRel_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs RO:$rt), (ins MO:$addr),
           !strconcat(opstr, "\t$rt, $addr")>;

let isReMaterializable = 1 in {
  def LAPC32_NM : LoadAddressPCRel_Desc<"lapc.h", GPRNM32Opnd, sym32_pc22_nm>,
                  Op32_Addr_Enc<0b00001>;
  def LAPC48_NM : LoadAddressPCRel_Desc<"lapc.b", GPRNM48Opnd, sym32_pc_nm>,
                  Op48_Addr_Enc<0b00011>;
}

class LoadAddressPCRelImm_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs RO:$rt), (ins MO:$addr), !strconcat(opstr, "\t$rt, $addr")>;

let isAsmParserOnly = 1 in {
  def ADDIUPC_NM : LoadAddressPCRelImm_Desc<"addiupc", GPRNM32Opnd,
                                            simm22s1_nm>,
                   Op32_Addr_Enc<0b00001>;
  def ADDIUPC48_NM : LoadAddressPCRelImm_Desc<"addiupc", GPRNM48Opnd,
                                              simm32_nm>,
                     Op48_Addr_Enc<0b00011>;
}

class LoadScaledAddress_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rd), (ins RO:$rs, RO:$rt, uimm2:$shift),
           !strconcat(opstr, "\t$rd, $rs, $rt, $shift")>;

def LSA_NM : LoadScaledAddress_Desc<"lsa", GPRNM32Opnd>, LoadScaledAddress_Enc;

class SignExtInReg_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rt), (ins RO:$rs), !strconcat(opstr, "\t$rt, $rs")>;

def SEB_NM : SignExtInReg_Desc<"seb", GPRNM32Opnd>, LogicU32_Enc<0b000001>;
def SEH_NM : SignExtInReg_Desc<"seh", GPRNM32Opnd>, LogicU32_Enc<0b000101>;

class CountLeading0_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rt), (ins RO:$rs), !strconcat(opstr, "\t$rt, $rs")>;

class CountLeading1_Desc<string opstr, RegisterOperand RO>
  : InstNM<(outs RO:$rt), (ins RO:$rs), !strconcat(opstr, "\t$rt, $rs")>;

def CLZ_NM : CountLeading0_Desc<"clz", GPRNM32Opnd>, CLZO_Enc<0b0101101>;
def CLO_NM : CountLeading1_Desc<"clo", GPRNM32Opnd>, CLZO_Enc<0b0100101>;

// P.TRAP pool of instructions.
class Trap_Desc<string opstr, RegisterOperand RO, Operand ImmOp>
  : InstNM<(outs), (ins RO:$rs, RO:$rt, ImmOp:$imm),
           !strconcat(opstr, "\t$rs, $rt, $imm")> {
  let hasSideEffects = 1;
  let isCTI = 1;
}

def TEQ_NM : Trap_Desc<"teq", GPRNM32Opnd, uimm5>, Trap_Enc<0b0>;
def TNE_NM : Trap_Desc<"tne", GPRNM32Opnd, uimm5>, Trap_Enc<0b1>;

class MOVEBALC_Desc : InstNM<(outs GPRNM1R1Opnd:$rd),
                             (ins GPRNM4ZOpnd:$rt, calltarget21_nm:$addr),
                             "move.balc\t$rd, $rt, $addr">  {
  let isCall = 1;
  let isCTI = 1;
  let Defs = [RA_NM];
}

def MOVEBALC_NM : MOVEBALC_Desc, MOVEBALC_Enc;

def LI16_NM : LoadImmAsm_Desc<"li", GPRNM16R3Opnd, nimm7_nm>,
              LI16_Enc<0b11000>;

let isReMaterializable = 1 in {
  def LI48_NM : LoadAddressAbs_Desc<"li", GPRNM48Opnd, sym32_nm>,
                Op48_Imm_Enc<0b00000>;
}

class LoadAddressGPRel_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs RO:$rt), (ins RO:$rs, MO:$addr),
           !strconcat(opstr, "\t$rt, $rs, $addr")>;

def ADDIUGP48_NM : LoadAddressGPRel_Desc<"addiu.b32", GPRNM48Opnd,
                                         sym32_gp_nm>,
                   Op48_Addr_Enc<0b00010>;

// Instructions with 1 register (dest) and 1 immediate operand.
class RegImm_Desc<string opstr, Operand Od, RegisterOperand RO>
  : InstNM<(outs RO:$rt), (ins Od:$imm), !strconcat(opstr, "\t$rt, $imm")>;

let isReMaterializable = 1 in
def LUI_NM : RegImm_Desc<"lui", simm32s12_nm, GPRNM32Opnd>,
             LUI_Enc<0b0>;

def ALUIPC_NM : RegImm_Desc<"aluipc", simm32s12_pcrel_nm, GPRNM32Opnd>,
                LUI_Enc<0b1>;

class ReadHardware_Desc<RegisterOperand CPURegOperand, RegisterOperand RO>
  : InstNM<(outs CPURegOperand:$rt), (ins RO:$hs, uimm5:$sel),
           "rdhwr\t$rt, $hs, $sel">;

def RDHWR_NM : ReadHardware_Desc<GPRNM32Opnd, COP0Opnd>,
               RDHWR_Enc;

def JRC_NM : IndirectBranch_Desc<"jrc", GPRNM32Opnd>, IndirectBranch16_Enc<0>;

class IndirectCall_Desc<string opstr, RegisterOperand DRO, RegisterOperand SRO>
  : InstNM<(outs DRO:$rt), (ins SRO:$rs), !strconcat(opstr, "\t$rt, $rs")> {
  let isCall = 1;
  let isCTI = 1;
}

// TODO: check let ... in, isCall and isCTI is already in IndirectCall_Desc
let Defs = [RA_NM] in {
  def JALRC16_NM : IndirectCall_Desc<"jalrc", GPRNMRAOpnd, GPRNM32Opnd>,
                   IndirectCall16_Enc;
  def JALRC_NM   : IndirectCall_Desc<"jalrc", GPRNM32Opnd, GPRNM32Opnd>,
                   IndirectCall32_Enc<0b0000>;
  def JALRCHB_NM : IndirectCall_Desc<"jalrc.hb", GPRNM32Opnd, GPRNM32Opnd>,
                   IndirectCall32_Enc<0b0001>;
}

def BC_NM     : BCBase_Desc<brtarget25_nm>, BC32_Enc<0b0>;
def BALC_NM   : Call_Desc<"balc", calltarget25_nm>, BC32_Enc<0b1>;
def BC16_NM   : BCBase_Desc<brtarget10_nm>, BC16_Enc<0b00010>;
def BALC16_NM : Call_Desc<"balc", calltarget10_nm>, BC16_Enc<0b00110>;

// This class is missing instruction size, because these instructions can be
// encoded in formats of different size (eg. BEQC16 and BEQC32).
class CBranch_Desc<string opstr, DAGOperand opnd, RegisterOperand RO>
  : InstNM<(outs), (ins RO:$rs, RO:$rt, opnd:$offset),
           !strconcat(opstr, "\t$rs, $rt, $offset")> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 0;
  let isCTI = 1;
}

class CBranchAsm_Desc<string opstr, DAGOperand opnd,
                      RegisterOperand RS, RegisterOperand RT>
  : InstNM<(outs), (ins RS:$rs, RT:$rt, opnd:$offset),
           !strconcat(opstr, "\t$rs, $rt, $offset")> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 0;
  let isCTI = 1;
}

// NOTE: BEQC[16] / BNEC[16] are intentionally *not* defined.
//
// Both instructions share the same 16-bit encoding:
//
//   110110   rt3   rs3   u[4:1]
//
// BEQC[16]: rs3 < rt3 && u != 0
// BNEC[16]: rs3 >= rt3 && u != 0
//
// Adding both BEQC[16] and BNEC[16] with this encoding causes a decoding
// conflict: the two instructions differ only by semantic conditions on
// (rs3, rt3, u), which the decoder cannot represent. Until a special
// mechanism is added, these instructions are intentionally omitted from
// the nanoMIPS backend.

def BEQC_NM : CBranch_Desc<"beqc", brtarget14_nm, GPRNM32Opnd>, B14_Enc<0b000>;
// def BEQC16_NM : CBranchAsm_Desc<"beqc", brtarget4s1_nm, GPRNM16R3Opnd,
//                                 GPRNM16R3Opnd>, B4_Enc;
def BGEC_NM  : CBranch_Desc<"bgec", brtarget14_nm, GPRNM32Opnd>,
               B14_Enc<0b010>;
def BGEUC_NM : CBranch_Desc<"bgeuc", brtarget14_nm, GPRNM32Opnd>,
               B14_Enc<0b011>;

def BNEC_NM   : CBranch_Desc<"bnec", brtarget14_nm, GPRNM32Opnd>,
                B14_Enc<0b100>;
// def BNEC16_NM : CBranchAsm_Desc<"bnec", brtarget4s1_nm,
//                                 GPRNM16R3Opnd, GPRNM16R3Opnd>, B4_Enc;

def BLTC_NM  : CBranch_Desc<"bltc", brtarget14_nm, GPRNM32Opnd>,
               B14_Enc<0b110>;
def BLTUC_NM : CBranch_Desc<"bltuc", brtarget14_nm, GPRNM32Opnd>,
               B14_Enc<0b111>;

class CBranchZeroAsm_Desc<string opstr, DAGOperand opnd, RegisterOperand RO>
  : InstNM<(outs), (ins RO:$rs, opnd:$offset),
           !strconcat(opstr, "\t$rs, $offset")> {
  let isBranch = 1;
  let isTerminator = 1;
  let isCTI = 1;
}

def BEQZC16_NM : CBranchZeroAsm_Desc<"beqzc", brtarget7_nm, GPRNM16R3Opnd>,
                 BZ16_Enc<0b10010>;
def BNEZC16_NM : CBranchZeroAsm_Desc<"bnezc", brtarget7_nm, GPRNM16R3Opnd>,
                 BZ16_Enc<0b10110>;

class CBranchImm_Desc<string opstr, DAGOperand opnd,
                      RegisterOperand RO, Operand imm_type>
  : InstNM<(outs), (ins RO:$rt, imm_type:$u, opnd:$offset),
           !strconcat(opstr, "\t$rt, $u, $offset")> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 0;
  let isCTI = 1;
}

class CBranchRegScaled_Desc<string opstr>
  : InstNM<(outs), (ins GPRNM32Opnd:$rs, i32imm:$jti),
           !strconcat(opstr, "\t$rs")> {
  let hasNoSchedulingInfo = 1;
  let hasDelaySlot = 0;
  let isBranch = 1;
  let isBarrier = 1;
  let isTerminator = 1;
  let isCTI = 1;
}

def BEQIC_NM  : CBranchImm_Desc<"beqic", brtarget11_nm, GPRNM32Opnd, uimm7>,
                BRI_Enc<0b000>;
def BGEIC_NM  : CBranchImm_Desc<"bgeic", brtarget11_nm, GPRNM32Opnd, uimm7>,
                BRI_Enc<0b010>;
def BLTIC_NM  : CBranchImm_Desc<"bltic", brtarget11_nm, GPRNM32Opnd, uimm7>,
                BRI_Enc<0b110>;
def BNEIC_NM  : CBranchImm_Desc<"bneic", brtarget11_nm, GPRNM32Opnd, uimm7>,
                BRI_Enc<0b100>;
def BGEIUC_NM : CBranchImm_Desc<"bgeiuc", brtarget11_nm, GPRNM32Opnd, uimm7>,
                BRI_Enc<0b011>;
def BLTIUC_NM : CBranchImm_Desc<"bltiuc", brtarget11_nm, GPRNM32Opnd, uimm7>,
                BRI_Enc<0b111>;

def BRSC_NM : CBranchRegScaled_Desc<"brsc">, IndirectBranch32_Enc;
def BALRSC_NM : IndirectCall_Desc<"balrsc", GPRNM32NZOpnd, GPRNM32Opnd>,
                IndirectCall32_Enc<0b1000>;

class CBranchBit_Desc<string opstr, DAGOperand opnd,
                      RegisterOperand RO, Operand imm_type>
  : InstNM<(outs), (ins RO:$rt, imm_type:$u, opnd:$offset),
           !strconcat(opstr, "\t$rt, $u, $offset")> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 0;
  let isCTI = 1;
}

// Bit-test branches
let hasNoSchedulingInfo = 1, hasDelaySlot = 0 in {
  def BBNEZC_NM : CBranchBit_Desc<"bbnezc", brtarget11_nm,
                                  GPRNM32Opnd, uimm6_nm>,
                  BBit_Enc<0b101>;
  def BBEQZC_NM : CBranchBit_Desc<"bbeqzc", brtarget11_nm,
                                  GPRNM32Opnd, uimm6_nm>,
                  BBit_Enc<0b001>;
}

//===----------------------------------------------------------------------===//
//
// Load / store instructions
//
//===----------------------------------------------------------------------===//
class LoadAsm_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs RO:$rt), (ins MO:$addr), !strconcat(opstr, "\t$rt, $addr")>;

class StoreAsm_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs), (ins RO:$rt, MO:$addr), !strconcat(opstr, "\t$rt, $addr")>;

def LWSP16_NM : LoadAsm_Desc<"lw", GPRNM32Opnd, mem_nm_sp5s2>,
                LSSPRegMem16_Enc<0b00101>;
def SWSP16_NM : StoreAsm_Desc<"sw", GPRNM32Opnd, mem_nm_sp5s2>,
                LSSPRegMem16_Enc<0b10101>;
def LWGP16_NM : LoadAsm_Desc<"lw", GPRNM16R3Opnd, mem_nm_gp7s2>,
                LSGPRegMem16_Enc<0b01001>;
def SWGP16_NM : StoreAsm_Desc<"sw", GPRNM16R3ZOpnd, mem_nm_gp7s2>,
                LSGPRegMem16_Enc<0b11001>;

def LW16_NM  : LoadAsm_Desc<"lw", GPRNM16R3Opnd, mem_nm_6s2>,
               LSMem16_Enc<0b00001>;
def SW16_NM  : StoreAsm_Desc<"sw", GPRNM16R3ZOpnd, mem_nm_6s2>,
               LSMem16_Enc<0b10001>;
def LH16_NM  : LoadAsm_Desc<"lh", GPRNM16R3Opnd, mem_nm_3s1>,
               LHMem16_Enc<0b00>;
def LHU16_NM : LoadAsm_Desc<"lhu", GPRNM16R3Opnd, mem_nm_3s1>,
               LHMem16_Enc<0b10>;
def SH16_NM  : StoreAsm_Desc<"sh", GPRNM16R3ZOpnd, mem_nm_3s1>,
               LHMem16_Enc<0b01>;

def LB16_NM  : LoadAsm_Desc<"lb", GPRNM16R3Opnd, mem_nm_2>,
               LBMem16_Enc<0b00>;
def LBU16_NM : LoadAsm_Desc<"lbu", GPRNM16R3Opnd, mem_nm_2>,
               LBMem16_Enc<0b10>;
def SB16_NM  : StoreAsm_Desc<"sb", GPRNM16R3ZOpnd, mem_nm_2>,
               LBMem16_Enc<0b01>;

class LoadMemory_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs RO:$rt), (ins MO:$addr), !strconcat(opstr, "\t$rt, $addr")> {
  let canFoldAsLoad = 1;
  let mayLoad = 1;
  string BaseOpcode = opstr;
}

class StoreMemory_Desc<string opstr, RegisterOperand RO, DAGOperand MO>
  : InstNM<(outs), (ins RO:$rt, MO:$addr), !strconcat(opstr, "\t$rt, $addr")> {
  let mayStore = 1;
  string BaseOpcode = opstr;
}

def LWGP_NM : LoadMemory_Desc<"lw", GPRNM32Opnd, mem_nm_gp19s2>,
              LSMemGP19S2_Enc<0b10>;
def SWGP_NM : StoreMemory_Desc<"sw", GPRNM32Opnd, mem_nm_gp19s2>,
              LSMemGP19S2_Enc<0b11>;

def LBGP_NM  : LoadMemory_Desc<"lb", GPRNM32Opnd, mem_nm_gp18>,
               LSMemGP18_Enc<0b000>;
def SBGP_NM  : StoreMemory_Desc<"sb", GPRNM32Opnd, mem_nm_gp18>,
               LSMemGP18_Enc<0b001>;
def LBUGP_NM : LoadMemory_Desc<"lbu", GPRNM32Opnd, mem_nm_gp18>,
               LSMemGP18_Enc<0b010>;

def LHGP_NM  : LoadMemory_Desc<"lh", GPRNM32Opnd, mem_nm_gp17s1>,
               LSMemGP17S1_Enc<0b1000>;
def SHGP_NM  : StoreMemory_Desc<"sh", GPRNM32Opnd, mem_nm_gp17s1>,
               LSMemGP17S1_Enc<0b1010>;
def LHUGP_NM : LoadMemory_Desc<"lhu", GPRNM32Opnd, mem_nm_gp17s1>,
               LSMemGP17S1_Enc<0b1001>;

def LB_NM : LoadMemory_Desc<"lb", GPRNM32Opnd, mem_nm_u12>,
            LSMem_U12_Enc<0b0000>;
def SB_NM : StoreMemory_Desc<"sb", GPRNM32Opnd, mem_nm_u12>,
            LSMem_U12_Enc<0b0001>;
def LH_NM : LoadMemory_Desc<"lh", GPRNM32Opnd, mem_nm_u12>,
            LSMem_U12_Enc<0b0100>;
def SH_NM : StoreMemory_Desc<"sh", GPRNM32Opnd, mem_nm_u12>,
            LSMem_U12_Enc<0b0101>;
def LW_NM : LoadMemory_Desc<"lw", GPRNM32Opnd, mem_nm_u12>,
            LSMem_U12_Enc<0b1000>;
def SW_NM : StoreMemory_Desc<"sw", GPRNM32Opnd, mem_nm_u12>,
            LSMem_U12_Enc<0b1001>;
def LBU_NM : LoadMemory_Desc<"lbu", GPRNM32Opnd, mem_nm_u12>,
             LSMem_U12_Enc<0b0010>;
def LHU_NM : LoadMemory_Desc<"lhu", GPRNM32Opnd, mem_nm_u12>,
             LSMem_U12_Enc<0b0110>;

def LBs9_NM : LoadMemory_Desc<"lb", GPRNM32Opnd, mem_nm_s9>,
              LSMem_S9_Enc<0b000, 0b0000>;
def SBs9_NM : StoreMemory_Desc<"sb", GPRNM32Opnd, mem_nm_s9>,
              LSMem_S9_Enc<0b000, 0b0001>;
def LHs9_NM : LoadMemory_Desc<"lh", GPRNM32Opnd, mem_nm_s9>,
              LSMem_S9_Enc<0b000, 0b0100>;
def SHs9_NM : StoreMemory_Desc<"sh", GPRNM32Opnd, mem_nm_s9>,
              LSMem_S9_Enc<0b000, 0b0101>;
def LWs9_NM : LoadMemory_Desc<"lw", GPRNM32Opnd, mem_nm_s9>,
              LSMem_S9_Enc<0b000, 0b1000>;
def SWs9_NM : StoreMemory_Desc<"sw", GPRNM32Opnd, mem_nm_s9>,
              LSMem_S9_Enc<0b000, 0b1001>;
def LBUs9_NM : LoadMemory_Desc<"lbu", GPRNM32Opnd, mem_nm_s9>,
               LSMem_S9_Enc<0b000, 0b0010>;
def LHUs9_NM : LoadMemory_Desc<"lhu", GPRNM32Opnd, mem_nm_s9>,
               LSMem_S9_Enc<0b000, 0b0110>;

def LBX_NM : LoadMemory_Desc<"lbx", GPRNM32Opnd, mem_nm_rx>,
             LSMemX_Enc<0, 0b0000>;
def SBX_NM : StoreMemory_Desc<"sbx", GPRNM32Opnd, mem_nm_rx>,
             LSMemX_Enc<0, 0b0001>;
def LHX_NM : LoadMemory_Desc<"lhx", GPRNM32Opnd, mem_nm_rx>,
             LSMemX_Enc<0, 0b0100>;
def SHX_NM : StoreMemory_Desc<"shx", GPRNM32Opnd, mem_nm_rx>,
             LSMemX_Enc<0, 0b0101>;
def LWX_NM : LoadMemory_Desc<"lwx", GPRNM32Opnd, mem_nm_rx>,
             LSMemX_Enc<0, 0b1000>;
def SWX_NM : StoreMemory_Desc<"swx", GPRNM32Opnd, mem_nm_rx>,
             LSMemX_Enc<0, 0b1001>;
def LBUX_NM : LoadMemory_Desc<"lbux", GPRNM32Opnd, mem_nm_rx>,
              LSMemX_Enc<0, 0b0010>;
def LHUX_NM : LoadMemory_Desc<"lhux", GPRNM32Opnd, mem_nm_rx>,
              LSMemX_Enc<0, 0b0110>;

def LHXS_NM : LoadMemory_Desc<"lhxs", GPRNM32Opnd, mem_nm_rx>,
              LSMemX_Enc<0b1, 0b0100>;
def SHXS_NM : StoreMemory_Desc<"shxs", GPRNM32Opnd, mem_nm_rx>,
              LSMemX_Enc<0b1, 0b0101>;
def LHUXS_NM : LoadMemory_Desc<"lhuxs", GPRNM32Opnd, mem_nm_rx>,
               LSMemX_Enc<0b1, 0b0110>;
def LWXS16_NM : LoadAsm_Desc<"lwxs", GPRNM16R3Opnd, mem_nm_rx16>,
                LSMemXS16_Enc;
def LWXS_NM : LoadMemory_Desc<"lwxs", GPRNM32Opnd, mem_nm_rx>,
              LSMemX_Enc<0b1, 0b1000>;
def SWXS_NM : StoreMemory_Desc<"swxs", GPRNM32Opnd, mem_nm_rx>,
              LSMemX_Enc<0b1, 0b1001>;

class UnalignedLoad_Desc<string opstr, RegisterOperand RO = GPRNM32Opnd>
  : InstNM<(outs RO:$rt), (ins mem_nm_s9:$addr, RO:$src),
           !strconcat(opstr, "\t$rt, $addr")>,
    RegConstraint<"$src = $rt"> {
  let mayLoad = 1;
  let canFoldAsLoad = 1;
}

class UnalignedStore_Desc<string opstr, RegisterOperand RO = GPRNM32Opnd>
  : InstNM<(outs), (ins RO:$rt, mem_nm_s9:$addr),
           !strconcat(opstr, "\t$rt, $addr")> {
  let mayStore = 1;
}

class SaveRestore_Desc<string opstr, Operand Od, Operand RL>
  : InstNM<(outs), (ins Od:$adj, RL:$regs, variable_ops),
           !strconcat(opstr, "\t$adj$regs")> {
  let Defs = [SP_NM];
  let Uses = [SP_NM];
  let hasSideEffects = 1;
}

def SAVE16_NM : SaveRestore_Desc<"save", uimm8s4_nm, reglist16_nm>,
                SaveRestore16_Enc<0b0>;
def RESTOREJRC16_NM : SaveRestore_Desc<"restore.jrc", uimm8s4_nm,
                                       reglist16_nm>,
                      SaveRestore16_Enc<0b1>;

def SAVE_NM : SaveRestore_Desc<"save", uimm12s3_nm, reglist_nm>,
              SaveRestore32_Enc<0b00>;
def RESTORE_NM : SaveRestore_Desc<"restore", uimm12s3_nm, reglist_nm>,
                 SaveRestore32_Enc<0b10>;
def RESTOREJRC_NM : SaveRestore_Desc<"restore.jrc", uimm12s3_nm, reglist_nm>,
                    SaveRestore32_Enc<0b11>;

class LoadMultiple_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$rt),
           (ins mem_nm_s9:$addr, uimm3plus1_nm:$rcount),
           !strconcat(opstr, "\t$rt, $addr, $rcount")> {
  let mayLoad = 1;
}

class StoreMultiple_Desc<string opstr>
  : InstNM<(outs),
           (ins GPRNM32Opnd:$rt, mem_nm_s9:$addr, uimm3plus1_nm:$rcount),
           !strconcat(opstr, "\t$rt, $addr, $rcount")> {
  let mayStore = 1;
}

def LWM_NM   : LoadMultiple_Desc<"lwm">, LSMem_S9_Mult_Enc<0b0100>;
def SWM_NM   : StoreMultiple_Desc<"swm">, LSMem_S9_Mult_Enc<0b1100>;
def UALWM_NM : LoadMultiple_Desc<"ualwm">, LSMem_S9_Mult_Enc<0b0101>;
def UASWM_NM : StoreMultiple_Desc<"uaswm">, LSMem_S9_Mult_Enc<0b1101>;

def UALH_NM  : UnalignedLoad_Desc<"ualh">, LSMem_S9_Enc<0b001, 0b0100>;
def UASH_NM  : UnalignedStore_Desc<"uash">, LSMem_S9_Enc<0b001, 0b0101>;
def UALW_NM : UnalignedLoad_Desc<"ualw">, LSMem_S9_Enc<0b101, 0b0010>;
def UASW_NM : UnalignedStore_Desc<"uasw">, LSMem_S9_Enc<0b101, 0b0011>;

class LoadPCBase_Desc<DAGOperand RO, DAGOperand MO>
  : InstNM<(outs RO:$rt), (ins MO:$addr), "lwpc\t$rt, $addr"> {
  let mayLoad = 1;
}

class StorePCBase_Desc<DAGOperand RO, DAGOperand MO>
  : InstNM<(outs), (ins RO:$rt, MO:$addr), "swpc\t$rt, $addr"> {
  let mayStore = 1;
}

def SWPC_NM : StorePCBase_Desc<GPRNM32Opnd, sym32_pc_nm>,
              Op48_Addr_Enc<0b01111>;
def LWPC_NM : LoadPCBase_Desc<GPRNM32Opnd, sym32_pc_nm>,
              Op48_Addr_Enc<0b01011>;

def LW4x4_NM : LoadAsm_Desc<"lw", GPRNM4Opnd, mem_nm_4s2>,
               LSMem4x4_Enc<0b01101>;
def SW4x4_NM : StoreAsm_Desc<"sw", GPRNM4ZOpnd, mem_nm_4s2>,
               LSMem4x4_Enc<0b11101>;

def MOVE_NM : MoveBase_Desc, MOVE_Enc;

def MOVEP_NM    : MovePBase_Desc<GPRNM4ZOpnd, GPRNM4ZOpnd,
                                 GPRNM2R1Opnd, GPRNM2R2Opnd>,
                  MOVEP_Enc<0b10111>;
def MOVEPREV_NM : MovePBase_Desc<GPRNM2R1Opnd, GPRNM2R2Opnd,
                                 GPRNM4Opnd, GPRNM4Opnd>,
                  MOVEPREV_Enc<0b11111>;

class SC_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$dst), (ins GPRNM32Opnd:$rt, mem_nm_7s2:$addr),
           !strconcat(opstr, "\t$rt, $addr")>,
    RegConstraint<"$rt = $dst"> {
  let mayStore = 1;
}

let hasNoSchedulingInfo = 1 in {
  def LL_NM  : LoadMemory_Desc<"ll", GPRNM32Opnd, mem_nm_7s2>,
               LLSC_Enc<0b001, 0b1010>;
  def SC_NM  : SC_Desc<"sc">, LLSC_Enc<0b001, 0b1011>;
}

class LLWP_Desc<string opstr, DAGOperand MO>
  : InstNM<(outs GPRNM32Opnd:$rt, GPRNM32Opnd:$ru), (ins MO:$addr),
           !strconcat(opstr, "\t$rt, $ru, $addr")> {
  let canFoldAsLoad = 1;
  let mayLoad = 1;
  string BaseOpcode = "llwp";
}

def LLWP_NM : LLWP_Desc<"llwp", mem_nm_0_no_offset>, LLSCWP_Enc<0b001, 0b1010>;

class SCWP_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$dst),
           (ins GPRNM32Opnd:$rt, GPRNM32Opnd:$ru, mem_nm_0:$addr),
           !strconcat(opstr, "\t$rt, $ru, $addr")>,
    RegConstraint<"$dst = $rt"> {
  let mayStore = 1;
  string BaseOpcode = "scwp";
}

def SCWP_NM : SCWP_Desc<"scwp">, LLSCWP_Enc<0b001, 0b1011>;

class CondMove_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$rd),
           (ins GPRNM32Opnd:$rs, GPRNM32Opnd:$rt, GPRNM32Opnd:$F),
           !strconcat(opstr, "\t$rd, $rs, $rt")>,
    RegConstraint<"$F = $rd">, IsCommutable;

def MOVZ_NM : CondMove_Desc<"movz">, CondMove_Enc<0b0>;
def MOVN_NM : CondMove_Desc<"movn">, CondMove_Enc<0b1>;

class ToggleInterrupts_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$rt), (ins), !strconcat(opstr, "\t$rt")>;

def DI_NM : ToggleInterrupts_Desc<"di">, ToggleInterrupts_Enc<0b00011>;
def EI_NM : ToggleInterrupts_Desc<"ei">, ToggleInterrupts_Enc<0b01011>;

// Instruction with no operands
class MnemonicOnly_Desc<string opstr> : InstNM<(outs), (ins), opstr>;

def ERET_NM   : MnemonicOnly_Desc<"eret">, ERET_Enc<0b11001, 0b0>;
def ERETNC_NM : MnemonicOnly_Desc<"eretnc">, ERET_Enc<0b11001, 0b1>;
def DERET_NM  : MnemonicOnly_Desc<"deret">, ERET_Enc<0b10001, 0b0>;

def EHB_NM : MnemonicOnly_Desc<"ehb">, SLL_VAR_Enc<3>;

def PAUSE_NM : MnemonicOnly_Desc<"pause">, SLL_VAR_Enc<5>;

def NOP32_NM : MnemonicOnly_Desc<"nop32">, SLL_VAR_Enc<0>;
def NOP_NM   : MnemonicOnly_Desc<"nop">, NOP_Enc;

class ImmOnly_Desc<string opstr, Operand Od>
  : InstNM<(outs), (ins Od:$imm), !strconcat(opstr, "\t$imm")>;

def SIGRIE_NM : ImmOnly_Desc<"sigrie", uimm19_nm>, IntExcpFlow_Enc<0b00>;

def BREAK_NM   : ImmOnly_Desc<"break", uimm19_nm>, IntExcpFlow_Enc<0b10>;
def BREAK16_NM : ImmOnly_Desc<"break", uimm3_nm>, IntExcpFlow16_Enc<0b10>;

def SDBBP_NM   : ImmOnly_Desc<"sdbbp", uimm19_nm>, IntExcpFlow_Enc<0b11>;
def SDBBP16_NM : ImmOnly_Desc<"sdbbp", uimm3_nm>, IntExcpFlow16_Enc<0b11>;

def SYSCALL_NM   : ImmOnly_Desc<"syscall", uimm18_nm>,
                   IntExcpFlow_Enc<0b01, 18>;
def SYSCALL16_NM : ImmOnly_Desc<"syscall", uimm2code_nm>,
                   IntExcpFlow16_Enc<0b01, 2>;

class ShadowRW_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$rt), (ins GPRNM32Opnd:$rs),
           !strconcat(opstr, "\t$rt, $rs")>;

def WRPGPR_NM : ShadowRW_Desc<"wrpgpr">, ShadowRW_Enc<0b11000>;
def RDPGPR_NM : ShadowRW_Desc<"rdpgpr">, ShadowRW_Enc<0b10000>;

class EXTW_Desc
  : InstNM<(outs GPRNM32Opnd:$rd),
           (ins GPRNM32Opnd:$rs, GPRNM32Opnd:$rt, uimm5:$shift),
           "extw\t$rd, $rs, $rt, $shift">;

def EXTW_NM : EXTW_Desc, EXTW_Enc;

// TODO: Define ALIGN_NM.
// NOTE: ALIGN will be added later as an assembly pseudo-instruction.
// It is intended to expand into EXTW/MOVE instructions (depending on bp).

class RotxAlias_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$rt), (ins GPRNM32Opnd:$rs),
           !strconcat(opstr, "\t$rt, $rs")>;

class ROTX_Desc
  : InstNM<(outs GPRNM32Opnd:$rt),
           (ins GPRNM32Opnd:$rs, uimm5:$shift,
            uimm5s1_nm:$shiftx, uimm1:$stripe),
           "rotx\t$rt, $rs, $shift, $shiftx, $stripe">;

def BITREVW_NM  : RotxAlias_Desc<"bitrevw">, RotxAlias_Enc<31, 0, 0>;
def BYTEREVW_NM : RotxAlias_Desc<"byterevw">, RotxAlias_Enc<24, 8, 0>;

def ROTX_NM : ROTX_Desc, ROTX_Enc;

class CacheOp_Desc<string opstr, Operand Od, DAGOperand MO>
  : InstNM<(outs), (ins Od:$op, MO:$addr), !strconcat(opstr, "\t$op, $addr")>;

def CACHE_NM  : CacheOp_Desc<"cache", uimm5, mem_nm_s9>,
                CACHE_PREFs9_Enc<0b001, 0b0111>;

def PREFs9_NM : CacheOp_Desc<"pref", uimm5, mem_nm_s9>,
                CACHE_PREFs9_Enc<0b000, 0b0011>;

def PREF_NM   : CacheOp_Desc<"pref", uimm5, mem_nm_u12>, PREF_Enc;

class SyncOp_Desc<string opstr, DAGOperand MO>
  : InstNM<(outs), (ins MO:$stype), !strconcat(opstr, "\t$stype")>;

class SyncIOp_Desc<string opstr, DAGOperand MO>
  : InstNM<(outs), (ins MO:$addr), !strconcat(opstr, "\t$addr")>;

def SYNC_NM : SyncOp_Desc<"sync", uimm5>, SYNC_Enc;

def SYNCIs9_NM : SyncIOp_Desc<"synci", mem_nm_s9>,
                 SYNCIs9_SYNCIE_Enc<0b000, 0b0011>;

def SYNCI_NM   : SyncIOp_Desc<"synci", mem_nm_u12>, SYNCI_Enc;

class WaitOp_Desc<string opstr, DAGOperand MO>
  : InstNM<(outs), (ins MO:$cd), !strconcat(opstr, "\t$cd")>;

def WAIT_NM : WaitOp_Desc<"wait", uimm10>, WAIT_Enc;

class WriteToCP0_Desc<string opstr, RegisterOperand RO, RegisterOperand CO>
  : InstNM<(outs), (ins RO:$rt, CO:$c0s, uimm5:$sel),
           !strconcat(opstr, "\t$rt, $c0s, $sel")>;

class WriteToCP0Sel_Desc<string opstr, RegisterOperand RO, RegisterOperand CO>
  : InstNM<(outs), (ins RO:$rt, CO:$c0s),
           !strconcat(opstr, "\t$rt, $c0s")>;

class ReadFromCP0_Desc<string opstr, RegisterOperand RO, RegisterOperand CO>
  : InstNM<(outs RO:$rt), (ins CO:$c0s, uimm5:$sel),
           !strconcat(opstr, "\t$rt, $c0s, $sel")>;

class ReadFromCP0Sel_Desc<string opstr, RegisterOperand RO, RegisterOperand CO>
  : InstNM<(outs RO:$rt), (ins CO:$c0s),
           !strconcat(opstr, "\t$rt, $c0s")>;

def MFC0Sel_NM  : ReadFromCP0Sel_Desc<"mfc0", GPRNM32Opnd, COP0SelOpnd>,
                  ReadWriteCP0Sel_Enc<0b000010>;
def MFHC0Sel_NM : ReadFromCP0Sel_Desc<"mfhc0", GPRNM32Opnd, COP0SelOpnd>,
                  ReadWriteCP0Sel_Enc<0b000011>;
def MTC0Sel_NM  : WriteToCP0Sel_Desc<"mtc0", GPRNM32Opnd, COP0SelOpnd>,
                  ReadWriteCP0Sel_Enc<0b000110>;
def MTHC0Sel_NM : WriteToCP0Sel_Desc<"mthc0", GPRNM32Opnd, COP0SelOpnd>,
                  ReadWriteCP0Sel_Enc<0b000111>;

let isAsmParserOnly = 1 in {
  def MFC0_NM  : ReadFromCP0_Desc<"mfc0", GPRNM32Opnd, COP0Opnd>,
                 ReadWriteCP0_Enc<0b000010>;
  def MFHC0_NM : ReadFromCP0_Desc<"mfhc0", GPRNM32Opnd, COP0Opnd>,
                 ReadWriteCP0_Enc<0b000011>;
  def MTC0_NM  : WriteToCP0_Desc<"mtc0", GPRNM32Opnd, COP0Opnd>,
                 ReadWriteCP0_Enc<0b000110>;
  def MTHC0_NM : WriteToCP0_Desc<"mthc0", GPRNM32Opnd, COP0Opnd>,
                 ReadWriteCP0_Enc<0b000111>;
}

class GINVI_Desc : InstNM<(outs), (ins GPRNM32Opnd:$rs), "ginvi\t$rs">;
class GINVT_Desc : InstNM<(outs), (ins GPRNM32Opnd:$rs, uimm2:$type),
                          "ginvt\t$rs, $type">;

def GINVI_NM : GINVI_Desc, GINVI_Enc<0b01111>, ASE_GINV;
def GINVT_NM : GINVT_Desc, GINVT_Enc<0b00111>, ASE_GINV;

class CRC_Op_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$rt), (ins GPRNM32Opnd:$val, GPRNM32Opnd:$rs),
           !strconcat(opstr, "\t$rt, $rs")>, RegConstraint<"$val = $rt">;

def CRC32B_NM  : CRC_Op_Desc<"crc32b">, CRC_Enc<0b000>, ASE_CRC;
def CRC32H_NM  : CRC_Op_Desc<"crc32h">, CRC_Enc<0b001>, ASE_CRC;
def CRC32W_NM  : CRC_Op_Desc<"crc32w">, CRC_Enc<0b010>, ASE_CRC;
def CRC32CB_NM : CRC_Op_Desc<"crc32cb">, CRC_Enc<0b100>, ASE_CRC;
def CRC32CH_NM : CRC_Op_Desc<"crc32ch">, CRC_Enc<0b101>, ASE_CRC;
def CRC32CW_NM : CRC_Op_Desc<"crc32cw">, CRC_Enc<0b110>, ASE_CRC;

class VP_Desc<string opstr>
  : InstNM<(outs GPRNM32Opnd:$rt), (ins), !strconcat(opstr, "\t$rt")>;

def EVP_NM : VP_Desc<"evp">, VP_Enc<0b1>;
def DVP_NM : VP_Desc<"dvp">, VP_Enc<0b0>;
