//===- MipsRegisterInfo.td - Mips Register defs -----------------*- C++ -*-===//
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

// We have banks of 32 registers each.
class MipsReg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "Mips";
}

// Mips CPU Registers
class MipsGPRReg<bits<5> num, string n> : MipsReg<n> {
  let Num = num;
}

// Mips 32-bit FPU Registers
class FPR<bits<5> num, string n> : MipsReg<n> {
  let Num = num;
}

// Mips 64-bit (aliased) FPU Registers
class AFPR<bits<5> num, string n, list<Register> aliases> : MipsReg<n> {
  let Num = num;
  let Aliases = aliases;
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

let Namespace = "Mips" in {

  // General Purpose Registers
  def ZERO : MipsGPRReg< 0, "ZERO">, DwarfRegNum<[0]>;
  def AT   : MipsGPRReg< 1, "AT">,   DwarfRegNum<[1]>;
  def V0   : MipsGPRReg< 2, "2">,    DwarfRegNum<[2]>;
  def V1   : MipsGPRReg< 3, "3">,    DwarfRegNum<[3]>;
  def A0   : MipsGPRReg< 4, "4">,    DwarfRegNum<[5]>;
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
  def GP   : MipsGPRReg< 28, "GP">,  DwarfRegNum<[28]>;
  def SP   : MipsGPRReg< 29, "SP">,  DwarfRegNum<[29]>;
  def FP   : MipsGPRReg< 30, "FP">,  DwarfRegNum<[30]>;
  def RA   : MipsGPRReg< 31, "RA">,  DwarfRegNum<[31]>;
  
  /// Mips Single point precision FPU Registers
  def F0  : FPR< 0,  "F0">, DwarfRegNum<[32]>;
  def F1  : FPR< 1,  "F1">, DwarfRegNum<[33]>;
  def F2  : FPR< 2,  "F2">, DwarfRegNum<[34]>;
  def F3  : FPR< 3,  "F3">, DwarfRegNum<[35]>;
  def F4  : FPR< 4,  "F4">, DwarfRegNum<[36]>;
  def F5  : FPR< 5,  "F5">, DwarfRegNum<[37]>;
  def F6  : FPR< 6,  "F6">, DwarfRegNum<[38]>;
  def F7  : FPR< 7,  "F7">, DwarfRegNum<[39]>;
  def F8  : FPR< 8,  "F8">, DwarfRegNum<[40]>;
  def F9  : FPR< 9,  "F9">, DwarfRegNum<[41]>;
  def F10 : FPR<10, "F10">, DwarfRegNum<[42]>;
  def F11 : FPR<11, "F11">, DwarfRegNum<[43]>;
  def F12 : FPR<12, "F12">, DwarfRegNum<[44]>;
  def F13 : FPR<13, "F13">, DwarfRegNum<[45]>;
  def F14 : FPR<14, "F14">, DwarfRegNum<[46]>;
  def F15 : FPR<15, "F15">, DwarfRegNum<[47]>;
  def F16 : FPR<16, "F16">, DwarfRegNum<[48]>;
  def F17 : FPR<17, "F17">, DwarfRegNum<[49]>;
  def F18 : FPR<18, "F18">, DwarfRegNum<[50]>;
  def F19 : FPR<19, "F19">, DwarfRegNum<[51]>;
  def F20 : FPR<20, "F20">, DwarfRegNum<[52]>;
  def F21 : FPR<21, "F21">, DwarfRegNum<[53]>;
  def F22 : FPR<22, "F22">, DwarfRegNum<[54]>;
  def F23 : FPR<23, "F23">, DwarfRegNum<[55]>;
  def F24 : FPR<24, "F24">, DwarfRegNum<[56]>;
  def F25 : FPR<25, "F25">, DwarfRegNum<[57]>;
  def F26 : FPR<26, "F26">, DwarfRegNum<[58]>;
  def F27 : FPR<27, "F27">, DwarfRegNum<[59]>;
  def F28 : FPR<28, "F28">, DwarfRegNum<[60]>;
  def F29 : FPR<29, "F29">, DwarfRegNum<[61]>;
  def F30 : FPR<30, "F30">, DwarfRegNum<[62]>;
  def F31 : FPR<31, "F31">, DwarfRegNum<[63]>;
  
  /// Mips Double point precision FPU Registers (aliased
  /// with the single precision to hold 64 bit values)
  def D0  : AFPR< 0,  "F0", [F0,   F1]>, DwarfRegNum<[32]>;
  def D1  : AFPR< 2,  "F2", [F2,   F3]>, DwarfRegNum<[34]>;
  def D2  : AFPR< 4,  "F4", [F4,   F5]>, DwarfRegNum<[36]>;
  def D3  : AFPR< 6,  "F6", [F6,   F7]>, DwarfRegNum<[38]>;
  def D4  : AFPR< 8,  "F8", [F8,   F9]>, DwarfRegNum<[40]>;
  def D5  : AFPR<10, "F10", [F10, F11]>, DwarfRegNum<[42]>;
  def D6  : AFPR<12, "F12", [F12, F13]>, DwarfRegNum<[44]>;
  def D7  : AFPR<14, "F14", [F14, F15]>, DwarfRegNum<[46]>;
  def D8  : AFPR<16, "F16", [F16, F17]>, DwarfRegNum<[48]>;
  def D9  : AFPR<18, "F18", [F18, F19]>, DwarfRegNum<[50]>;
  def D10 : AFPR<20, "F20", [F20, F21]>, DwarfRegNum<[52]>;
  def D11 : AFPR<22, "F22", [F22, F23]>, DwarfRegNum<[54]>;
  def D12 : AFPR<24, "F24", [F24, F25]>, DwarfRegNum<[56]>;
  def D13 : AFPR<26, "F26", [F26, F27]>, DwarfRegNum<[58]>;
  def D14 : AFPR<28, "F28", [F28, F29]>, DwarfRegNum<[60]>;
  def D15 : AFPR<30, "F30", [F30, F31]>, DwarfRegNum<[62]>;

  // Hi/Lo registers
  def HI  : Register<"hi">, DwarfRegNum<[64]>;
  def LO  : Register<"lo">, DwarfRegNum<[65]>;

  // Status flags register
  def FCR31 : Register<"FCR31">;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

def CPURegs : RegisterClass<"Mips", [i32], 32, 
  // Return Values and Arguments
  [V0, V1, A0, A1, A2, A3,
  // Not preserved across procedure calls
  T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, 
  // Callee save
  S0, S1, S2, S3, S4, S5, S6, S7,
  // Reserved
  ZERO, AT, K0, K1, GP, SP, FP, RA]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    CPURegsClass::iterator
    CPURegsClass::allocation_order_end(const MachineFunction &MF) const {
      // The last 8 registers on the list above are reserved
      return end()-8;
    }
  }];
}

// * 64bit fp:
//    - FGR64 = 32 64-bit registers (default mode)
//    - AFGR32/AFGR64 = 16 even 32-bit registers (32-bit compatible mode) for
//      single and double access.
// * 32bit fp:
//    - AFGR32/AFGR64 = 16 even 32-bit registers - single and double
//    - FGR32 = 32 32-bit registers (within single-only mode)
def FGR32 : RegisterClass<"Mips", [f32], 32, 
  // Return Values and Arguments
  [F0, F1, F2, F3, F12, F13, F14, F15,
  // Not preserved across procedure calls
  F4, F5, F6, F7, F8, F9, F10, F11, F16, F17, F18, F19, 
  // Callee save
  F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30,
  // Reserved
  F31]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FGR32Class::iterator
    FGR32Class::allocation_order_end(const MachineFunction &MF) const {
      // The last register on the list above is reserved
      return end()-1;
    }
  }];
}

def AFGR32 : RegisterClass<"Mips", [f32], 32, 
  // Return Values and Arguments
  [F0, F2, F12, F14, 
  // Not preserved across procedure calls
  F4, F6, F8, F10, F16, F18, 
  // Callee save
  F20, F22, F24, F26, F28, F30,
  // Reserved
  F31]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    AFGR32Class::iterator
    AFGR32Class::allocation_order_end(const MachineFunction &MF) const {
      // The last register on the list above is reserved
      return end()-1;
    }
  }];
}

def AFGR64 : RegisterClass<"Mips", [f64], 64, 
  // Return Values and Arguments
  [D0, D1, D6, D7,
  // Not preserved across procedure calls
  D2, D3, D4, D5, D8, D9, 
  // Callee save
  D10, D11, D12, D13, D14,
  // Reserved
  D15]>
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    AFGR64Class::iterator
    AFGR64Class::allocation_order_end(const MachineFunction &MF) const {
      // The last register on the list above is reserved
      return end()-1;
    }
  }];
}

def CCR : RegisterClass<"Mips", [i32], 32, [FCR31]> {
  let CopyCost = -1;  // Don't allow copying of status registers.
}

def HILO : RegisterClass<"Mips", [i32], 32, [HI, LO]> {
  //let CopyCost = -1;  // Don't allow copying of hi/lo registers.
}

