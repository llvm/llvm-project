//===-- MipsRegisterInfo.td - Mips Register defs -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MIPS register file
//===----------------------------------------------------------------------===//
let Namespace = "Mips" in {
def sub_fpeven : SubRegIndex;
def sub_fpodd  : SubRegIndex;
def sub_32     : SubRegIndex;
def sub_lo     : SubRegIndex;
def sub_hi     : SubRegIndex;
}

// We have banks of 32 registers each.
class MipsReg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "Mips";
}

class MipsRegWithSubRegs<string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  field bits<5> Num;
  let Namespace = "Mips";
}

// Mips CPU Registers
class MipsGPRReg<bits<5> num, string n> : MipsReg<n> {
  let Num = num;
}

// Mips 64-bit CPU Registers
class Mips64GPRReg<bits<5> num, string n, list<Register> subregs>
  : MipsRegWithSubRegs<n, subregs> {
  let Num = num;
  let SubRegIndices = [sub_32];
}

// Mips 32-bit FPU Registers
class FPR<bits<5> num, string n> : MipsReg<n> {
  let Num = num;
}

// Mips 64-bit (aliased) FPU Registers
class AFPR<bits<5> num, string n, list<Register> subregs>
  : MipsRegWithSubRegs<n, subregs> {
  let Num = num;
  let SubRegIndices = [sub_fpeven, sub_fpodd];
  let CoveredBySubRegs = 1;
}

class AFPR64<bits<5> num, string n, list<Register> subregs>
  : MipsRegWithSubRegs<n, subregs> {
  let Num = num;
  let SubRegIndices = [sub_32];
}

// Mips Hardware Registers
class HWR<bits<5> num, string n> : MipsReg<n> {
  let Num = num;
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

let Namespace = "Mips" in {
  // General Purpose Registers
  def ZERO : MipsGPRReg< 0, "zero">, DwarfRegNum<[0]>;
  def AT   : MipsGPRReg< 1, "1">,    DwarfRegNum<[1]>;
  def V0   : MipsGPRReg< 2, "2">,    DwarfRegNum<[2]>;
  def V1   : MipsGPRReg< 3, "3">,    DwarfRegNum<[3]>;
  def A0   : MipsGPRReg< 4, "4">,    DwarfRegNum<[4]>;
  def A1   : MipsGPRReg< 5, "5">,    DwarfRegNum<[5]>;
  def A2   : MipsGPRReg< 6, "6">,    DwarfRegNum<[6]>;
  def A3   : MipsGPRReg< 7, "7">,    DwarfRegNum<[7]>;
  def T0   : MipsGPRReg< 8, "8">,    DwarfRegNum<[8]>;
  def T1   : MipsGPRReg< 9, "9">,    DwarfRegNum<[9]>;
  def T2   : MipsGPRReg< 10, "10">,  DwarfRegNum<[10]>;
  def T3   : MipsGPRReg< 11, "11">,  DwarfRegNum<[11]>;
  def T4   : MipsGPRReg< 12, "12">,  DwarfRegNum<[12]>;
  def T5   : MipsGPRReg< 13, "13">,  DwarfRegNum<[13]>;
  def T6   : MipsGPRReg< 14, "14">,  DwarfRegNum<[14]>;
  def T7   : MipsGPRReg< 15, "15">,  DwarfRegNum<[15]>;
  def S0   : MipsGPRReg< 16, "16">,  DwarfRegNum<[16]>;
  def S1   : MipsGPRReg< 17, "17">,  DwarfRegNum<[17]>;
  def S2   : MipsGPRReg< 18, "18">,  DwarfRegNum<[18]>;
  def S3   : MipsGPRReg< 19, "19">,  DwarfRegNum<[19]>;
  def S4   : MipsGPRReg< 20, "20">,  DwarfRegNum<[20]>;
  def S5   : MipsGPRReg< 21, "21">,  DwarfRegNum<[21]>;
  def S6   : MipsGPRReg< 22, "22">,  DwarfRegNum<[22]>;
  def S7   : MipsGPRReg< 23, "23">,  DwarfRegNum<[23]>;
  def T8   : MipsGPRReg< 24, "24">,  DwarfRegNum<[24]>;
  def T9   : MipsGPRReg< 25, "25">,  DwarfRegNum<[25]>;
  def K0   : MipsGPRReg< 26, "26">,  DwarfRegNum<[26]>;
  def K1   : MipsGPRReg< 27, "27">,  DwarfRegNum<[27]>;
  def GP   : MipsGPRReg< 28, "gp">,  DwarfRegNum<[28]>;
  def SP   : MipsGPRReg< 29, "sp">,  DwarfRegNum<[29]>;
  def FP   : MipsGPRReg< 30, "fp">,  DwarfRegNum<[30]>;
  def RA   : MipsGPRReg< 31, "ra">,  DwarfRegNum<[31]>;

  // General Purpose 64-bit Registers
  def ZERO_64 : Mips64GPRReg< 0, "zero", [ZERO]>, DwarfRegNum<[0]>;
  def AT_64   : Mips64GPRReg< 1, "1",    [AT]>, DwarfRegNum<[1]>;
  def V0_64   : Mips64GPRReg< 2, "2",    [V0]>, DwarfRegNum<[2]>;
  def V1_64   : Mips64GPRReg< 3, "3",    [V1]>, DwarfRegNum<[3]>;
  def A0_64   : Mips64GPRReg< 4, "4",    [A0]>, DwarfRegNum<[4]>;
  def A1_64   : Mips64GPRReg< 5, "5",    [A1]>, DwarfRegNum<[5]>;
  def A2_64   : Mips64GPRReg< 6, "6",    [A2]>, DwarfRegNum<[6]>;
  def A3_64   : Mips64GPRReg< 7, "7",    [A3]>, DwarfRegNum<[7]>;
  def T0_64   : Mips64GPRReg< 8, "8",    [T0]>, DwarfRegNum<[8]>;
  def T1_64   : Mips64GPRReg< 9, "9",    [T1]>, DwarfRegNum<[9]>;
  def T2_64   : Mips64GPRReg< 10, "10",  [T2]>, DwarfRegNum<[10]>;
  def T3_64   : Mips64GPRReg< 11, "11",  [T3]>, DwarfRegNum<[11]>;
  def T4_64   : Mips64GPRReg< 12, "12",  [T4]>, DwarfRegNum<[12]>;
  def T5_64   : Mips64GPRReg< 13, "13",  [T5]>, DwarfRegNum<[13]>;
  def T6_64   : Mips64GPRReg< 14, "14",  [T6]>, DwarfRegNum<[14]>;
  def T7_64   : Mips64GPRReg< 15, "15",  [T7]>, DwarfRegNum<[15]>;
  def S0_64   : Mips64GPRReg< 16, "16",  [S0]>, DwarfRegNum<[16]>;
  def S1_64   : Mips64GPRReg< 17, "17",  [S1]>, DwarfRegNum<[17]>;
  def S2_64   : Mips64GPRReg< 18, "18",  [S2]>, DwarfRegNum<[18]>;
  def S3_64   : Mips64GPRReg< 19, "19",  [S3]>, DwarfRegNum<[19]>;
  def S4_64   : Mips64GPRReg< 20, "20",  [S4]>, DwarfRegNum<[20]>;
  def S5_64   : Mips64GPRReg< 21, "21",  [S5]>, DwarfRegNum<[21]>;
  def S6_64   : Mips64GPRReg< 22, "22",  [S6]>, DwarfRegNum<[22]>;
  def S7_64   : Mips64GPRReg< 23, "23",  [S7]>, DwarfRegNum<[23]>;
  def T8_64   : Mips64GPRReg< 24, "24",  [T8]>, DwarfRegNum<[24]>;
  def T9_64   : Mips64GPRReg< 25, "25",  [T9]>, DwarfRegNum<[25]>;
  def K0_64   : Mips64GPRReg< 26, "26",  [K0]>, DwarfRegNum<[26]>;
  def K1_64   : Mips64GPRReg< 27, "27",  [K1]>, DwarfRegNum<[27]>;
  def GP_64   : Mips64GPRReg< 28, "gp",  [GP]>, DwarfRegNum<[28]>;
  def SP_64   : Mips64GPRReg< 29, "sp",  [SP]>, DwarfRegNum<[29]>;
  def FP_64   : Mips64GPRReg< 30, "fp",  [FP]>, DwarfRegNum<[30]>;
  def RA_64   : Mips64GPRReg< 31, "ra",  [RA]>, DwarfRegNum<[31]>;

  /// Mips Single point precision FPU Registers
  def F0  : FPR< 0,  "f0">, DwarfRegNum<[32]>;
  def F1  : FPR< 1,  "f1">, DwarfRegNum<[33]>;
  def F2  : FPR< 2,  "f2">, DwarfRegNum<[34]>;
  def F3  : FPR< 3,  "f3">, DwarfRegNum<[35]>;
  def F4  : FPR< 4,  "f4">, DwarfRegNum<[36]>;
  def F5  : FPR< 5,  "f5">, DwarfRegNum<[37]>;
  def F6  : FPR< 6,  "f6">, DwarfRegNum<[38]>;
  def F7  : FPR< 7,  "f7">, DwarfRegNum<[39]>;
  def F8  : FPR< 8,  "f8">, DwarfRegNum<[40]>;
  def F9  : FPR< 9,  "f9">, DwarfRegNum<[41]>;
  def F10 : FPR<10, "f10">, DwarfRegNum<[42]>;
  def F11 : FPR<11, "f11">, DwarfRegNum<[43]>;
  def F12 : FPR<12, "f12">, DwarfRegNum<[44]>;
  def F13 : FPR<13, "f13">, DwarfRegNum<[45]>;
  def F14 : FPR<14, "f14">, DwarfRegNum<[46]>;
  def F15 : FPR<15, "f15">, DwarfRegNum<[47]>;
  def F16 : FPR<16, "f16">, DwarfRegNum<[48]>;
  def F17 : FPR<17, "f17">, DwarfRegNum<[49]>;
  def F18 : FPR<18, "f18">, DwarfRegNum<[50]>;
  def F19 : FPR<19, "f19">, DwarfRegNum<[51]>;
  def F20 : FPR<20, "f20">, DwarfRegNum<[52]>;
  def F21 : FPR<21, "f21">, DwarfRegNum<[53]>;
  def F22 : FPR<22, "f22">, DwarfRegNum<[54]>;
  def F23 : FPR<23, "f23">, DwarfRegNum<[55]>;
  def F24 : FPR<24, "f24">, DwarfRegNum<[56]>;
  def F25 : FPR<25, "f25">, DwarfRegNum<[57]>;
  def F26 : FPR<26, "f26">, DwarfRegNum<[58]>;
  def F27 : FPR<27, "f27">, DwarfRegNum<[59]>;
  def F28 : FPR<28, "f28">, DwarfRegNum<[60]>;
  def F29 : FPR<29, "f29">, DwarfRegNum<[61]>;
  def F30 : FPR<30, "f30">, DwarfRegNum<[62]>;
  def F31 : FPR<31, "f31">, DwarfRegNum<[63]>;

  /// Mips Double point precision FPU Registers (aliased
  /// with the single precision to hold 64 bit values)
  def D0  : AFPR< 0,  "f0", [F0,   F1]>;
  def D1  : AFPR< 2,  "f2", [F2,   F3]>;
  def D2  : AFPR< 4,  "f4", [F4,   F5]>;
  def D3  : AFPR< 6,  "f6", [F6,   F7]>;
  def D4  : AFPR< 8,  "f8", [F8,   F9]>;
  def D5  : AFPR<10, "f10", [F10, F11]>;
  def D6  : AFPR<12, "f12", [F12, F13]>;
  def D7  : AFPR<14, "f14", [F14, F15]>;
  def D8  : AFPR<16, "f16", [F16, F17]>;
  def D9  : AFPR<18, "f18", [F18, F19]>;
  def D10 : AFPR<20, "f20", [F20, F21]>;
  def D11 : AFPR<22, "f22", [F22, F23]>;
  def D12 : AFPR<24, "f24", [F24, F25]>;
  def D13 : AFPR<26, "f26", [F26, F27]>;
  def D14 : AFPR<28, "f28", [F28, F29]>;
  def D15 : AFPR<30, "f30", [F30, F31]>;

  /// Mips Double point precision FPU Registers in MFP64 mode.
  def D0_64  : AFPR64<0, "f0", [F0]>, DwarfRegNum<[32]>;
  def D1_64  : AFPR64<1, "f1", [F1]>, DwarfRegNum<[33]>;
  def D2_64  : AFPR64<2, "f2", [F2]>, DwarfRegNum<[34]>;
  def D3_64  : AFPR64<3, "f3", [F3]>, DwarfRegNum<[35]>;
  def D4_64  : AFPR64<4, "f4", [F4]>, DwarfRegNum<[36]>;
  def D5_64  : AFPR64<5, "f5", [F5]>, DwarfRegNum<[37]>;
  def D6_64  : AFPR64<6, "f6", [F6]>, DwarfRegNum<[38]>;
  def D7_64  : AFPR64<7, "f7", [F7]>, DwarfRegNum<[39]>;
  def D8_64  : AFPR64<8, "f8", [F8]>, DwarfRegNum<[40]>;
  def D9_64  : AFPR64<9, "f9", [F9]>, DwarfRegNum<[41]>;
  def D10_64  : AFPR64<10, "f10", [F10]>, DwarfRegNum<[42]>;
  def D11_64  : AFPR64<11, "f11", [F11]>, DwarfRegNum<[43]>;
  def D12_64  : AFPR64<12, "f12", [F12]>, DwarfRegNum<[44]>;
  def D13_64  : AFPR64<13, "f13", [F13]>, DwarfRegNum<[45]>;
  def D14_64  : AFPR64<14, "f14", [F14]>, DwarfRegNum<[46]>;
  def D15_64  : AFPR64<15, "f15", [F15]>, DwarfRegNum<[47]>;
  def D16_64  : AFPR64<16, "f16", [F16]>, DwarfRegNum<[48]>;
  def D17_64  : AFPR64<17, "f17", [F17]>, DwarfRegNum<[49]>;
  def D18_64  : AFPR64<18, "f18", [F18]>, DwarfRegNum<[50]>;
  def D19_64  : AFPR64<19, "f19", [F19]>, DwarfRegNum<[51]>;
  def D20_64  : AFPR64<20, "f20", [F20]>, DwarfRegNum<[52]>;
  def D21_64  : AFPR64<21, "f21", [F21]>, DwarfRegNum<[53]>;
  def D22_64  : AFPR64<22, "f22", [F22]>, DwarfRegNum<[54]>;
  def D23_64  : AFPR64<23, "f23", [F23]>, DwarfRegNum<[55]>;
  def D24_64  : AFPR64<24, "f24", [F24]>, DwarfRegNum<[56]>;
  def D25_64  : AFPR64<25, "f25", [F25]>, DwarfRegNum<[57]>;
  def D26_64  : AFPR64<26, "f26", [F26]>, DwarfRegNum<[58]>;
  def D27_64  : AFPR64<27, "f27", [F27]>, DwarfRegNum<[59]>;
  def D28_64  : AFPR64<28, "f28", [F28]>, DwarfRegNum<[60]>;
  def D29_64  : AFPR64<29, "f29", [F29]>, DwarfRegNum<[61]>;
  def D30_64  : AFPR64<30, "f30", [F30]>, DwarfRegNum<[62]>;
  def D31_64  : AFPR64<31, "f31", [F31]>, DwarfRegNum<[63]>;

  // Hi/Lo registers
  def HI  : Register<"hi">, DwarfRegNum<[64]>;
  def LO  : Register<"lo">, DwarfRegNum<[65]>;

  let SubRegIndices = [sub_32] in {
  def HI64  : RegisterWithSubRegs<"hi", [HI]>;
  def LO64  : RegisterWithSubRegs<"lo", [LO]>;
  }

  // Status flags register
  def FCR31 : Register<"31">;

  // fcc0 register
  def FCC0 : Register<"fcc0">;

  // PC register
  def PC : Register<"pc">;

  // Hardware register $29
  def HWR29 : Register<"29">;
  def HWR29_64 : Register<"29">;

  // Accum registers
  let SubRegIndices = [sub_lo, sub_hi] in
  def AC0 : RegisterWithSubRegs<"ac0", [LO, HI]>;
  def AC1 : Register<"ac1">;
  def AC2 : Register<"ac2">;
  def AC3 : Register<"ac3">;

  def DSPCtrl : Register<"dspctrl">;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

class CPURegsClass<list<ValueType> regTypes> :
  RegisterClass<"Mips", regTypes, 32, (add
  // Reserved
  ZERO, AT,
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3,
  // Not preserved across procedure calls
  T0, T1, T2, T3, T4, T5, T6, T7,
  // Callee save
  S0, S1, S2, S3, S4, S5, S6, S7,
  // Not preserved across procedure calls
  T8, T9,
  // Reserved
  K0, K1, GP, SP, FP, RA)>;

def CPURegs : CPURegsClass<[i32]>;
def DSPRegs : CPURegsClass<[v4i8, v2i16]>;

def CPU64Regs : RegisterClass<"Mips", [i64], 64, (add
// Reserved
  ZERO_64, AT_64,
  // Return Values and Arguments
  V0_64, V1_64, A0_64, A1_64, A2_64, A3_64,
  // Not preserved across procedure calls
  T0_64, T1_64, T2_64, T3_64, T4_64, T5_64, T6_64, T7_64,
  // Callee save
  S0_64, S1_64, S2_64, S3_64, S4_64, S5_64, S6_64, S7_64,
  // Not preserved across procedure calls
  T8_64, T9_64,
  // Reserved
  K0_64, K1_64, GP_64, SP_64, FP_64, RA_64)>;

def CPU16Regs : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3,
  // Callee save
  S0, S1)>;

def CPURAReg : RegisterClass<"Mips", [i32], 32, (add RA)>;

def CPUSPReg : RegisterClass<"Mips", [i32], 32, (add SP)>;

// 64bit fp:
// * FGR64  - 32 64-bit registers
// * AFGR64 - 16 32-bit even registers (32-bit FP Mode)
//
// 32bit fp:
// * FGR32 - 16 32-bit even registers
// * FGR32 - 32 32-bit registers (single float only mode)
def FGR32 : RegisterClass<"Mips", [f32], 32, (sequence "F%u", 0, 31)>;

def AFGR64 : RegisterClass<"Mips", [f64], 64, (add
  // Return Values and Arguments
  D0, D1,
  // Not preserved across procedure calls
  D2, D3, D4, D5,
  // Return Values and Arguments
  D6, D7,
  // Not preserved across procedure calls
  D8, D9,
  // Callee save
  D10, D11, D12, D13, D14, D15)>;

def FGR64 : RegisterClass<"Mips", [f64], 64, (sequence "D%u_64", 0, 31)>;

// Condition Register for floating point operations
def CCR  : RegisterClass<"Mips", [i32], 32, (add FCR31,FCC0)>;

// Hi/Lo Registers
def HILO : RegisterClass<"Mips", [i32], 32, (add HI, LO)>;
def HILO64 : RegisterClass<"Mips", [i64], 64, (add HI64, LO64)>;

// Hardware registers
def HWRegs : RegisterClass<"Mips", [i32], 32, (add HWR29)>;
def HWRegs64 : RegisterClass<"Mips", [i64], 32, (add HWR29_64)>;

// Accumulator Registers
def ACRegs : RegisterClass<"Mips", [i64], 64, (sequence "AC%u", 0, 3)>;
