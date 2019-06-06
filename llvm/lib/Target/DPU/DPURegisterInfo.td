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
// R17: reserved for internal computation and system operations... not given to the compiler
def R17: DPUReg<17, "r17">, DwarfRegNum<[17]>;
// D18: reserved for internal computation and system operations... not given to the compiler
def R18 : DPUReg<18, "r18">, DwarfRegNum<[18]>;
def R19 : DPUReg<19, "r19">, DwarfRegNum<[19]>;
// RVALHI: reserved as partial part of D20, used as register to return 64 bits values
def RVALHI : DPUReg<20, "r20">, DwarfRegNum<[20]>;
// R21: reserved to put the functions' returned values.
def RVAL : DPUReg<21, "r21">, DwarfRegNum<[21]>;
// R22: reserved as the stack pointer, SP
def STKP : DPUReg<22, "r22">, DwarfRegNum<[22]>;
// R23: reserved as the return address for functions
def RADD : DPUReg<23, "r23">, DwarfRegNum<[23]>;

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
  // Note: D16 is not used, because it encloses system registers.
  def D16: DPUReg<16, "d16", [R17, R16]>;
  // D18: reserved for internal computation and system operations... not given to the compiler
  def D18: DPUReg<18, "d18", [R19, R18]>;
  // To return long values from functions
  def RDVAL: DPUReg<20, "d20", [RVAL, RVALHI]>;
  def RDFUN: DPUReg<22, "d22", [RADD, STKP]>;
}

// Constant registers.
def ZERO: DPUReg<24, "zero">;
def ONE: DPUReg<25, "one">;
def LNEG: DPUReg<26, "lneg">;
def MNEG:  DPUReg<27, "mneg">;
// Thread id registers. Return the thread identification for the
// current thread, times 1, 2, 4, 8.
def ID:    DPUReg<28, "id">;
def ID2:   DPUReg<29, "id2">;
def ID4:   DPUReg<30, "id4">;
def ID8:   DPUReg<31, "id8">;

// Define the register class representing this bank of general
// purpose registers used by ONE thread.
// The general purpose registers (GP_REG) are readable and writable.
// The constant registers (CONST_REG) are read only registers.
// The operand registers (OP_REG) are the collection of any readable register
// that can be used as an instruction operand.
// Hide the reserved registers, so that we are very sure that the compiler will
// not do anything with them.
def GP_REG    : RegisterClass<"DPU", [i32], 32, (add (sequence "R%u", 0, 19), RVAL, RVALHI, RADD, STKP)>;
def GPZ_REG   : RegisterClass<"DPU", [i32], 32, (add GP_REG, ZERO)>;
def CONST_REG : RegisterClass<"DPU", [i32], 32, (add ZERO, ONE, LNEG, MNEG)>;
def ID_REG    : RegisterClass<"DPU", [i32], 32, (add ID, ID2, ID4, ID8)>;
def OP_REG    : RegisterClass<"DPU", [i32], 32, (add GP_REG, CONST_REG, ID_REG)>;
def ZERO_REG  : RegisterClass<"DPU", [i32], 32, (add ZERO)>;

// 64 bits registers are the combinations of 2 consecutive registers.
def GP64_REG  : RegisterClass<"DPU", [i64], 64,
                          (add D0, D2, D4, D6, D8, D10, D12, D14, D16, D18, RDVAL, RDFUN)>;

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
