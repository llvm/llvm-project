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

def simm12 : Operand<i32>;

// As noted in RISCVRegisterInfo.td, the hope is that support for
// variable-sized register classes will mean that instruction definitions do
// not need to be duplicated for 32-bit and 64-bit register classes. For now
// we use 'GPR', which is 32-bit. When codegen for both RV32 and RV64 is
// added, we will need to duplicate instruction definitions unless a proposal
// like <http://lists.llvm.org/pipermail/llvm-dev/2016-September/105027.html>
// is adopted.

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

