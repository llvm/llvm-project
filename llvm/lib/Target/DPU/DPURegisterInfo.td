//===-- DPURegisterInfo.td - DPU Register defs -------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Description of the DPU registers.
//===----------------------------------------------------------------------===//

// Co-locate the registers in the same dedicated namespace
// Double registers are aliases of pairs of longword registers, expressed
// by the subregs optional field.
class DPUReg<bits<16> HWIndex, string n, list<Register> subregs = [], list<Register> aliases = []> : Register<n> {
  let Namespace = "DPU";
  let HWEncoding = HWIndex;
  let SubRegs = subregs;
  let Aliases = aliases;
}

// Subregister indices.
let Namespace = "DPU" in {
  def sub_32bit   : SubRegIndex<32>;
  def sub_32bit_hi: SubRegIndex<32, 32>;
  def sub_64bit   : SubRegIndex<64>;
}

// Thread data registers
def R0 : DPUReg<0, "r0">, DwarfRegNum<[0]>;
def R1 : DPUReg<1, "r1">, DwarfRegNum<[1]>;
def R2 : DPUReg<2, "r2">, DwarfRegNum<[2]>;
def R3 : DPUReg<3, "r3">, DwarfRegNum<[3]>;
def R4 : DPUReg<4, "r4">, DwarfRegNum<[4]>;
def R5 : DPUReg<5, "r5">, DwarfRegNum<[5]>;
def R6 : DPUReg<6, "r6">, DwarfRegNum<[6]>;
def R7 : DPUReg<7, "r7">, DwarfRegNum<[7]>;
def R8 : DPUReg<8, "r8">, DwarfRegNum<[8]>;
def R9 : DPUReg<9, "r9">, DwarfRegNum<[9]>;
def R10 : DPUReg<10, "r10">, DwarfRegNum<[10]>;
def R11 : DPUReg<11, "r11">, DwarfRegNum<[11]>;
def R12 : DPUReg<12, "r12">, DwarfRegNum<[12]>;
def R13 : DPUReg<13, "r13">, DwarfRegNum<[13]>;
def R14 : DPUReg<14, "r14">, DwarfRegNum<[14]>;
def R15 : DPUReg<15, "r15">, DwarfRegNum<[15]>;
def R16 : DPUReg<16, "r16">, DwarfRegNum<[16]>;
def R17 : DPUReg<17, "r17">, DwarfRegNum<[17]>;
def R18 : DPUReg<18, "r18">, DwarfRegNum<[18]>;
def R19 : DPUReg<19, "r19">, DwarfRegNum<[19]>;
def R20 : DPUReg<20, "r20">, DwarfRegNum<[20]>;
def R21 : DPUReg<21, "r21">, DwarfRegNum<[21]>;
// R22: reserved as the stack pointer, SP
def R22 : DPUReg<22, "r22">, DwarfRegNum<[22]>;
// R23: reserved as the return address for functions
def R23 : DPUReg<23, "r23">, DwarfRegNum<[23]>;

// Thread data registers
def MAJ_R0 : DPUReg<0, "R0", [], [R0]>, DwarfRegNum<[0]>;
def MAJ_R1 : DPUReg<1, "R1", [], [R1]>, DwarfRegNum<[1]>;
def MAJ_R2 : DPUReg<2, "R2", [], [R2]>, DwarfRegNum<[2]>;
def MAJ_R3 : DPUReg<3, "R3", [], [R3]>, DwarfRegNum<[3]>;
def MAJ_R4 : DPUReg<4, "R4", [], [R4]>, DwarfRegNum<[4]>;
def MAJ_R5 : DPUReg<5, "R5", [], [R5]>, DwarfRegNum<[5]>;
def MAJ_R6 : DPUReg<6, "R6", [], [R6]>, DwarfRegNum<[6]>;
def MAJ_R7 : DPUReg<7, "R7", [], [R7]>, DwarfRegNum<[7]>;
def MAJ_R8 : DPUReg<8, "R8", [], [R8]>, DwarfRegNum<[8]>;
def MAJ_R9 : DPUReg<9, "R9", [], [R9]>, DwarfRegNum<[9]>;
def MAJ_R10 : DPUReg<10, "R10", [], [R10]>, DwarfRegNum<[10]>;
def MAJ_R11 : DPUReg<11, "R11", [], [R11]>, DwarfRegNum<[11]>;
def MAJ_R12 : DPUReg<12, "R12", [], [R12]>, DwarfRegNum<[12]>;
def MAJ_R13 : DPUReg<13, "R13", [], [R13]>, DwarfRegNum<[13]>;
def MAJ_R14 : DPUReg<14, "R14", [], [R14]>, DwarfRegNum<[14]>;
def MAJ_R15 : DPUReg<15, "R15", [], [R15]>, DwarfRegNum<[15]>;
def MAJ_R16 : DPUReg<16, "R16", [], [R16]>, DwarfRegNum<[16]>;
def MAJ_R17 : DPUReg<17, "R17", [], [R17]>, DwarfRegNum<[17]>;
def MAJ_R18 : DPUReg<18, "R18", [], [R18]>, DwarfRegNum<[18]>;
def MAJ_R19 : DPUReg<19, "R19", [], [R19]>, DwarfRegNum<[19]>;
def MAJ_R20 : DPUReg<20, "R20", [], [R20]>, DwarfRegNum<[20]>;
def MAJ_R21 : DPUReg<21, "R21", [], [R21]>, DwarfRegNum<[21]>;
def MAJ_R22 : DPUReg<22, "R22", [], [R22]>, DwarfRegNum<[22]>;
def MAJ_R23 : DPUReg<23, "R23", [], [R23]>, DwarfRegNum<[23]>;

// Thread data registers, extended to 64 bits.
let SubRegIndices = [sub_32bit, sub_32bit_hi], CoveredBySubRegs = 1 in {
  def D0 : DPUReg<0, "d0", [R1, R0]>;
  def D2 : DPUReg<2, "d2", [R3, R2]>;
  def D4 : DPUReg<4, "d4", [R5, R4]>;
  def D6 : DPUReg<6, "d6", [R7, R6]>;
  def D8 : DPUReg<8, "d8", [R9, R8]>;
  def D10 : DPUReg<10, "d10", [R11, R10]>;
  def D12 : DPUReg<12, "d12", [R13, R12]>;
  def D14 : DPUReg<14, "d14", [R15, R14]>;
  def D16 : DPUReg<16, "d16", [R17, R16]>;
  def D18 : DPUReg<18, "d18", [R19, R18]>;
  def D20 : DPUReg<20, "d20", [R21, R20]>;
  def D22 : DPUReg<22, "d22", [R23, R22]>;

  def MAJ_D0 : DPUReg<0, "D0", [MAJ_R1, MAJ_R0], [D0]>;
  def MAJ_D2 : DPUReg<2, "D2", [MAJ_R3, MAJ_R2], [D2]>;
  def MAJ_D4 : DPUReg<4, "D4", [MAJ_R5, MAJ_R4], [D4]>;
  def MAJ_D6 : DPUReg<6, "D6", [MAJ_R7, MAJ_R6], [D6]>;
  def MAJ_D8 : DPUReg<8, "D8", [MAJ_R9, MAJ_R8], [D8]>;
  def MAJ_D10 : DPUReg<10, "D10", [MAJ_R11, MAJ_R10], [D10]>;
  def MAJ_D12 : DPUReg<12, "D12", [MAJ_R13, MAJ_R12], [D12]>;
  def MAJ_D14 : DPUReg<14, "D14", [MAJ_R15, MAJ_R14], [D14]>;
  def MAJ_D16 : DPUReg<16, "D16", [MAJ_R17, MAJ_R16], [D16]>;
  def MAJ_D18 : DPUReg<18, "D18", [MAJ_R19, MAJ_R18], [D18]>;
  def MAJ_D20 : DPUReg<20, "D20", [MAJ_R21, MAJ_R20], [D20]>;
  def MAJ_D22 : DPUReg<22, "D22", [MAJ_R23, MAJ_R22], [D22]>;
}

// Constant registers.
def ZERO: DPUReg<24, "zero">;
def MAJ_ZERO: DPUReg<24, "ZERO", [], [ZERO]>;
def ONE: DPUReg<25, "one">;
def MAJ_ONE: DPUReg<25, "ONE", [], [ONE]>;
def LNEG: DPUReg<26, "lneg">;
def MAJ_LNEG: DPUReg<26, "LNEG", [], [LNEG]>;
def MNEG:  DPUReg<27, "mneg">;
def MAJ_MNEG: DPUReg<27, "MNEG", [], [MNEG]>;
// Thread id registers. Return the thread identification for the
// current thread, times 1, 2, 4, 8.
def ID:    DPUReg<28, "id">;
def ID2:   DPUReg<29, "id2">;
def ID4:   DPUReg<30, "id4">;
def ID8:   DPUReg<31, "id8">;
def MAJ_ID:    DPUReg<28, "ID", [], [ID]>;
def MAJ_ID2:   DPUReg<29, "ID2", [], [ID2]>;
def MAJ_ID4:   DPUReg<30, "ID4", [], [ID4]>;
def MAJ_ID8:   DPUReg<31, "ID8", [], [ID8]>;

// Define the register class representing this bank of general
// purpose registers used by ONE thread.
// The general purpose registers (GP_REG) are readable and writable.
// The constant registers (CONST_REG) are read only registers.
// The operand registers (OP_REG) are the collection of any readable register
// that can be used as an instruction operand.
// Hide the reserved registers, so that we are very sure that the compiler will
// not do anything with them.
def GP_REG    : RegisterClass<"DPU", [i32], 32, (add (sequence "R%u", 0, 23), (sequence "MAJ_R%u", 0, 23))>;
def CONST_REG : RegisterClass<"DPU", [i32], 32, (add ZERO, ONE, LNEG, MNEG, MAJ_ZERO, MAJ_ONE, MAJ_LNEG, MAJ_MNEG)>;
def ID_REG    : RegisterClass<"DPU", [i32], 32, (add ID, ID2, ID4, ID8, MAJ_ID, MAJ_ID2, MAJ_ID4, MAJ_ID8)>;
def ZERO_REG  : RegisterClass<"DPU", [i32], 32, (add ZERO, MAJ_ZERO)>;
def OP_REG    : RegisterClass<"DPU", [i32], 32, (add GP_REG, CONST_REG, ID_REG)>;
def GPZ_REG   : RegisterClass<"DPU", [i32], 32, (add GP_REG, ZERO_REG)>;

// 64 bits registers are the combinations of 2 consecutive registers.
def GP64_REG  : RegisterClass<"DPU", [i64], 64,
                          (add D0, D2, D4, D6, D8, D10, D12, D14, D16, D18, D20, D22, MAJ_D0, MAJ_D2, MAJ_D4, MAJ_D6, MAJ_D8, MAJ_D10, MAJ_D12, MAJ_D14, MAJ_D16, MAJ_D18, MAJ_D20, MAJ_D22)>;

def S0:   DPUReg<0, "s0">;
def S1:   DPUReg<1, "s1">;
def S2:   DPUReg<2, "s2">;
def S3:   DPUReg<3, "s3">;
def S4:   DPUReg<4, "s4">;
def S5:   DPUReg<5, "s5">;
def S6:   DPUReg<6, "s6">;
def S7:   DPUReg<7, "s7">;
def S8:   DPUReg<8, "s8">;
def S9:   DPUReg<9, "s9">;
def S10:  DPUReg<10, "s10">;
def S11:  DPUReg<11, "s11">;
def S12:  DPUReg<12, "s12">;
def S13:  DPUReg<13, "s13">;
def S14:  DPUReg<14, "s14">;
def S15:  DPUReg<15, "s15">;
def S16:  DPUReg<16, "s16">;
def S17:  DPUReg<17, "s17">;
def S18:  DPUReg<18, "s18">;
def S19:  DPUReg<19, "s19">;
def S20:  DPUReg<20, "s20">;
def S21:  DPUReg<21, "s21">;
def S22:  DPUReg<22, "s22">;
def S23:  DPUReg<23, "s23">;
def CST_REG: RegisterClass<"DPU", [i32], 32, (add CONST_REG, ID_REG)>;
def SAFE_REG   : RegisterClass<"DPU", [i32], 32, (add (sequence "S%u", 0, 23))>;
def SAFE_OP_REG: RegisterClass<"DPU", [i32], 32, (add SAFE_REG, CST_REG)>;
def NZ_OP_REG: RegisterClass<"DPU", [i32], 32, (add GP_REG, ID_REG, ONE, LNEG, MNEG)>;

// todo end JORDI

class DPUAsmRegOperand<RegisterClass regs> : AsmOperandClass {
  let ParserMethod = "parseAnyRegister";
  let RenderMethod = "addRegOperands";
  let Name = regs ## "_AsmReg";
}

class DPURegOperand<RegisterClass regs> : RegisterOperand<regs> {
  let ParserMatchClass = DPUAsmRegOperand<regs>;
}

def SimpleReg             : DPURegOperand<GP_REG>;
def DoubleReg             : DPURegOperand<GP64_REG>;
def SimpleRegOrCst        : DPURegOperand<OP_REG>;
def SafeRegOrCst          : DPURegOperand<SAFE_OP_REG>;
def SafeReg               : DPURegOperand<SAFE_REG>;
def CstReg                : DPURegOperand<CST_REG>;
def SimpleRegOrCstButZero : DPURegOperand<NZ_OP_REG>;
def ZeroRegister          : DPURegOperand<ZERO_REG>;
