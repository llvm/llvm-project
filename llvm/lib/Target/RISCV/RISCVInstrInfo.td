//===-- RISCVInstrInfo.td - Target Description for RISCV ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormats.td"

class ImmAsmOperand<string prefix, int width, string suffix> : AsmOperandClass {
  let Name = prefix # "Imm" # width # suffix;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}

class SImmAsmOperand<int width, string suffix = "">
  : ImmAsmOperand<"S", width, suffix> {
}

class UImmAsmOperand<int width, string suffix = "">
  : ImmAsmOperand<"U", width, suffix> {
}

def FenceArg : AsmOperandClass {
  let Name = "FenceArg";
  let RenderMethod = "addFenceArgOperands";
  let DiagnosticType = "InvalidFenceArg";
}

def fencearg : Operand<i32> {
  let ParserMatchClass = FenceArg;
  let PrintMethod = "printFenceArg";
  let DecoderMethod = "decodeUImmOperand<4>";
}

def uimm5 : Operand<i32> {
  let ParserMatchClass = UImmAsmOperand<5>;
  let DecoderMethod = "decodeUImmOperand<5>";
}

def simm12 : Operand<i32> {
  let ParserMatchClass = SImmAsmOperand<12>;
  let DecoderMethod = "decodeSImmOperand<12>";
}

def uimm12 : Operand<i32> {
  let ParserMatchClass = UImmAsmOperand<12>;
  let DecoderMethod = "decodeUImmOperand<12>";
}

// A 13-bit signed immediate where the least significant bit is zero.
def simm13_lsb0 : Operand<i32> {
  let ParserMatchClass = SImmAsmOperand<13, "Lsb0">;
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<13>";
}

def uimm20 : Operand<i32> {
  let ParserMatchClass = UImmAsmOperand<20>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<20>";
}

// A 21-bit signed immediate where the least significant bit is zero.
def simm21_lsb0 : Operand<i32> {
  let ParserMatchClass = SImmAsmOperand<21, "Lsb0">;
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<21>";
}

// As noted in RISCVRegisterInfo.td, the hope is that support for
// variable-sized register classes will mean that instruction definitions do
// not need to be duplicated for 32-bit and 64-bit register classes. For now
// we use 'GPR', which is 32-bit. When codegen for both RV32 and RV64 is
// added, we will need to duplicate instruction definitions unless a proposal
// like <http://lists.llvm.org/pipermail/llvm-dev/2016-September/105027.html>
// is adopted.

def LUI : FU<0b0110111, (outs GPR:$rd), (ins uimm20:$imm20),
             "lui\t$rd, $imm20", []>;

def AUIPC : FU<0b0010111, (outs GPR:$rd), (ins uimm20:$imm20),
             "auipc\t$rd, $imm20", []>;

def JAL : FUJ<0b1101111, (outs GPR:$rd), (ins simm21_lsb0:$imm20),
              "jal\t$rd, $imm20", []>;

def JALR : FI<0b000, 0b1100111, (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
              "jalr\t$rd, $rs1, $imm12", []>;

class Bcc<bits<3> funct3, string OpcodeStr> :
      FSB<funct3, 0b1100011, (outs), (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12),
         OpcodeStr#"\t$rs1, $rs2, $imm12", []> {
}

def BEQ   : Bcc<0b000, "beq">;
def BNE   : Bcc<0b001, "bne">;
def BLT   : Bcc<0b100, "blt">;
def BGE   : Bcc<0b101, "bge">;
def BLTU  : Bcc<0b110, "bltu">;
def BGEU  : Bcc<0b111, "bgeu">;

class LD_ri<bits<3> funct3, string OpcodeStr> :
      FI<funct3, 0b0000011, (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
         OpcodeStr#"\t$rd, ${imm12}(${rs1})", []> {
  let mayLoad = 1;
}

def LB   : LD_ri<0b000, "lb">;
def LH   : LD_ri<0b001, "lh">;
def LW   : LD_ri<0b010, "lw">;
def LBU  : LD_ri<0b100, "lbu">;
def LHU  : LD_ri<0b101, "lhu">;

class ST_ri<bits<3> funct3, string OpcodeStr> :
      FS<funct3, 0b0100011, (outs), (ins GPR:$rs1, GPR:$rs2, simm12:$imm12),
         OpcodeStr#"\t$rs2, ${imm12}(${rs1})", []> {
  let mayStore = 1;
}

def SB  : ST_ri<0b000, "sb">;
def SH  : ST_ri<0b001, "sh">;
def SW  : ST_ri<0b010, "sw">;

class ALU_ri<bits<3> funct3, string OpcodeStr> :
      FI<funct3, 0b0010011, (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
         OpcodeStr#"\t$rd, $rs1, $imm12", []>
{
}

def ADDI  : ALU_ri<0b000, "addi">;
def SLTI  : ALU_ri<0b010, "slti">;
def SLTIU : ALU_ri<0b011, "sltiu">;
def XORI  : ALU_ri<0b100, "xori">;
def ORI   : ALU_ri<0b110, "ori">;
def ANDI  : ALU_ri<0b111, "andi">;

class SHIFT32_ri<bit arithshift, bits<3> funct3, string OpcodeStr> :
      FI32Shift<arithshift, funct3, 0b0010011, (outs GPR:$rd), (ins GPR:$rs1, uimm5:$shamt),
         OpcodeStr#"\t$rd, $rs1, $shamt", []>
{
}

def SLLI : SHIFT32_ri<0, 0b001, "slli">;
def SRLI : SHIFT32_ri<0, 0b101, "srli">;
def SRAI : SHIFT32_ri<1, 0b101, "srai">;

class ALU_rr<bits<7> funct7, bits<3> funct3, string OpcodeStr> :
      FR<funct7, funct3, 0b0110011, (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
         OpcodeStr#"\t$rd, $rs1, $rs2", []>
{
}

def ADD  : ALU_rr<0b0000000, 0b000, "add">;
def SUB  : ALU_rr<0b0100000, 0b000, "sub">;
def SLL  : ALU_rr<0b0000000, 0b001, "sll">;
def SLT  : ALU_rr<0b0000000, 0b010, "slt">;
def SLTU : ALU_rr<0b0000000, 0b011, "sltu">;
def XOR  : ALU_rr<0b0000000, 0b100, "xor">;
def SRL  : ALU_rr<0b0000000, 0b101, "srl">;
def SRA  : ALU_rr<0b0100000, 0b101, "sra">;
def OR   : ALU_rr<0b0000000, 0b110, "or">;
def AND  : ALU_rr<0b0000000, 0b111, "and">;

def FENCE : FI<0b000, 0b0001111, (outs), (ins fencearg:$pred, fencearg:$succ),
               "fence\t$pred, $succ", []> {
  bits<4> pred;
  bits<4> succ;

  let rs1 = 0;
  let rd = 0;
  let imm12 = {0b0000,pred,succ};
}

def FENCEI : FI<0b001, 0b0001111, (outs), (ins), "fence.i", []> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 0;
}

let rs1=0, rd=0 in {
  def ECALL  : FI<0b000, 0b1110011, (outs), (ins), "ecall", []> {
    let imm12=0;
  }
  def EBREAK : FI<0b000, 0b1110011, (outs), (ins), "ebreak", []> {
    let imm12=1;
  }
}

class CSR_rr<bits<3> funct3, string OpcodeStr> :
      FI<funct3, 0b1110011, (outs GPR:$rd), (ins uimm12:$imm12, GPR:$rs1),
         OpcodeStr#"\t$rd, $imm12, $rs1", []>
{
}

def CSRRW : CSR_rr<0b001, "csrrw">;
def CSRRS : CSR_rr<0b010, "csrrs">;
def CSRRC : CSR_rr<0b011, "csrrc">;

class CSR_ri<bits<3> funct3, string OpcodeStr> :
      FI<funct3, 0b1110011, (outs GPR:$rd), (ins uimm12:$imm12, uimm5:$rs1),
         OpcodeStr#"\t$rd, $imm12, $rs1", []>
{
}

def CSRRWI : CSR_ri<0b101, "csrrwi">;
def CSRRSI : CSR_ri<0b110, "csrrsi">;
def CSRRCI : CSR_ri<0b111, "csrrci">;
