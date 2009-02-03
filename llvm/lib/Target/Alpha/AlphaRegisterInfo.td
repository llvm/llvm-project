//===- AlphaRegisterInfo.td - The Alpha Register File ------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the Alpha register set.
//
//===----------------------------------------------------------------------===//

class AlphaReg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "Alpha";
}

// We identify all our registers with a 5-bit ID, for consistency's sake.

// GPR - One of the 32 32-bit general-purpose registers
class GPR<bits<5> num, string n> : AlphaReg<n> {
  let Num = num;
}

// FPR - One of the 32 64-bit floating-point registers
class FPR<bits<5> num, string n> : AlphaReg<n> {
  let Num = num;
}

//#define FP    $15
//#define RA    $26
//#define PV    $27
//#define GP    $29
//#define SP    $30

// General-purpose registers
def R0  : GPR< 0,  "$0">, DwarfRegNum<[0]>;
def R1  : GPR< 1,  "$1">, DwarfRegNum<[1]>;
def R2  : GPR< 2,  "$2">, DwarfRegNum<[2]>;
def R3  : GPR< 3,  "$3">, DwarfRegNum<[3]>;
def R4  : GPR< 4,  "$4">, DwarfRegNum<[4]>;
def R5  : GPR< 5,  "$5">, DwarfRegNum<[5]>;
def R6  : GPR< 6,  "$6">, DwarfRegNum<[6]>;
def R7  : GPR< 7,  "$7">, DwarfRegNum<[7]>;
def R8  : GPR< 8,  "$8">, DwarfRegNum<[8]>;
def R9  : GPR< 9,  "$9">, DwarfRegNum<[9]>;
def R10 : GPR<10, "$10">, DwarfRegNum<[10]>;
def R11 : GPR<11, "$11">, DwarfRegNum<[11]>;
def R12 : GPR<12, "$12">, DwarfRegNum<[12]>;
def R13 : GPR<13, "$13">, DwarfRegNum<[13]>;
def R14 : GPR<14, "$14">, DwarfRegNum<[14]>;
def R15 : GPR<15, "$15">, DwarfRegNum<[15]>;
def R16 : GPR<16, "$16">, DwarfRegNum<[16]>;
def R17 : GPR<17, "$17">, DwarfRegNum<[17]>;
def R18 : GPR<18, "$18">, DwarfRegNum<[18]>;
def R19 : GPR<19, "$19">, DwarfRegNum<[19]>;
def R20 : GPR<20, "$20">, DwarfRegNum<[20]>;
def R21 : GPR<21, "$21">, DwarfRegNum<[21]>;
def R22 : GPR<22, "$22">, DwarfRegNum<[22]>;
def R23 : GPR<23, "$23">, DwarfRegNum<[23]>;
def R24 : GPR<24, "$24">, DwarfRegNum<[24]>;
def R25 : GPR<25, "$25">, DwarfRegNum<[25]>;
def R26 : GPR<26, "$26">, DwarfRegNum<[26]>;
def R27 : GPR<27, "$27">, DwarfRegNum<[27]>;
def R28 : GPR<28, "$28">, DwarfRegNum<[28]>;
def R29 : GPR<29, "$29">, DwarfRegNum<[29]>;
def R30 : GPR<30, "$30">, DwarfRegNum<[30]>;
def R31 : GPR<31, "$31">, DwarfRegNum<[31]>;

// Floating-point registers
def F0  : FPR< 0,  "$f0">, DwarfRegNum<[33]>;
def F1  : FPR< 1,  "$f1">, DwarfRegNum<[34]>;
def F2  : FPR< 2,  "$f2">, DwarfRegNum<[35]>;
def F3  : FPR< 3,  "$f3">, DwarfRegNum<[36]>;
def F4  : FPR< 4,  "$f4">, DwarfRegNum<[37]>;
def F5  : FPR< 5,  "$f5">, DwarfRegNum<[38]>;
def F6  : FPR< 6,  "$f6">, DwarfRegNum<[39]>;
def F7  : FPR< 7,  "$f7">, DwarfRegNum<[40]>;
def F8  : FPR< 8,  "$f8">, DwarfRegNum<[41]>;
def F9  : FPR< 9,  "$f9">, DwarfRegNum<[42]>;
def F10 : FPR<10, "$f10">, DwarfRegNum<[43]>;
def F11 : FPR<11, "$f11">, DwarfRegNum<[44]>;
def F12 : FPR<12, "$f12">, DwarfRegNum<[45]>;
def F13 : FPR<13, "$f13">, DwarfRegNum<[46]>;
def F14 : FPR<14, "$f14">, DwarfRegNum<[47]>;
def F15 : FPR<15, "$f15">, DwarfRegNum<[48]>;
def F16 : FPR<16, "$f16">, DwarfRegNum<[49]>;
def F17 : FPR<17, "$f17">, DwarfRegNum<[50]>;
def F18 : FPR<18, "$f18">, DwarfRegNum<[51]>;
def F19 : FPR<19, "$f19">, DwarfRegNum<[52]>;
def F20 : FPR<20, "$f20">, DwarfRegNum<[53]>;
def F21 : FPR<21, "$f21">, DwarfRegNum<[54]>;
def F22 : FPR<22, "$f22">, DwarfRegNum<[55]>;
def F23 : FPR<23, "$f23">, DwarfRegNum<[56]>;
def F24 : FPR<24, "$f24">, DwarfRegNum<[57]>;
def F25 : FPR<25, "$f25">, DwarfRegNum<[58]>;
def F26 : FPR<26, "$f26">, DwarfRegNum<[59]>;
def F27 : FPR<27, "$f27">, DwarfRegNum<[60]>;
def F28 : FPR<28, "$f28">, DwarfRegNum<[61]>;
def F29 : FPR<29, "$f29">, DwarfRegNum<[62]>;
def F30 : FPR<30, "$f30">, DwarfRegNum<[63]>;
def F31 : FPR<31, "$f31">, DwarfRegNum<[64]>;

  // //#define FP    $15
  // //#define RA    $26
  // //#define PV    $27
  // //#define GP    $29
  // //#define SP    $30
  // $28 is undefined after any and all calls

/// Register classes
def GPRC : RegisterClass<"Alpha", [i64], 64,
     // Volatile
     [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19, R20, R21, R22,
      R23, R24, R25, R28, 
     //Special meaning, but volatile
     R27, //procedure address
     R26, //return address
     R29, //global offset table address
     // Non-volatile
     R9, R10, R11, R12, R13, R14,
// Don't allocate 15, 30, 31
     R15, R30, R31 ]> //zero
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GPRCClass::iterator
    GPRCClass::allocation_order_end(const MachineFunction &MF) const {
        return end()-3;
    }
  }];
}

def F4RC : RegisterClass<"Alpha", [f32], 64, [F0, F1, 
        F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,
        F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30,
        // Saved:
        F2, F3, F4, F5, F6, F7, F8, F9,
        F31 ]> //zero
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    F4RCClass::iterator
    F4RCClass::allocation_order_end(const MachineFunction &MF) const {
        return end()-1;
    }
  }];
}

def F8RC : RegisterClass<"Alpha", [f64], 64, [F0, F1, 
        F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,
        F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30,
        // Saved:
        F2, F3, F4, F5, F6, F7, F8, F9,
        F31 ]> //zero
{
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    F8RCClass::iterator
    F8RCClass::allocation_order_end(const MachineFunction &MF) const {
        return end()-1;
    }
  }];
}
