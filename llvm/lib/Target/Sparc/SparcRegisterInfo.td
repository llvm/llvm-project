//===-- SparcRegisterInfo.td - Sparc Register defs ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Sparc register file 
//===----------------------------------------------------------------------===//

class SparcReg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "SP";
}

class SparcCtrlReg<string n>: Register<n> {
  let Namespace = "SP";
}

let Namespace = "SP" in {
def sub_even : SubRegIndex;
def sub_odd  : SubRegIndex;
}

// Registers are identified with 5-bit ID numbers.
// Ri - 32-bit integer registers
class Ri<bits<5> num, string n> : SparcReg<n> {
  let Num = num;
}
// Rf - 32-bit floating-point registers
class Rf<bits<5> num, string n> : SparcReg<n> {
  let Num = num;
}
// Rd - Slots in the FP register file for 64-bit floating-point values.
class Rd<bits<5> num, string n, list<Register> subregs> : SparcReg<n> {
  let Num = num;
  let SubRegs = subregs;
  let SubRegIndices = [sub_even, sub_odd];
  let CoveredBySubRegs = 1;
}

// Control Registers
def ICC : SparcCtrlReg<"ICC">;
def FCC : SparcCtrlReg<"FCC">;

// Y register
def Y : SparcCtrlReg<"Y">;

// Integer registers
def G0 : Ri< 0, "G0">, DwarfRegNum<[0]>;
def G1 : Ri< 1, "G1">, DwarfRegNum<[1]>;
def G2 : Ri< 2, "G2">, DwarfRegNum<[2]>; 
def G3 : Ri< 3, "G3">, DwarfRegNum<[3]>;
def G4 : Ri< 4, "G4">, DwarfRegNum<[4]>;
def G5 : Ri< 5, "G5">, DwarfRegNum<[5]>; 
def G6 : Ri< 6, "G6">, DwarfRegNum<[6]>;
def G7 : Ri< 7, "G7">, DwarfRegNum<[7]>;
def O0 : Ri< 8, "O0">, DwarfRegNum<[8]>;
def O1 : Ri< 9, "O1">, DwarfRegNum<[9]>;
def O2 : Ri<10, "O2">, DwarfRegNum<[10]>; 
def O3 : Ri<11, "O3">, DwarfRegNum<[11]>;
def O4 : Ri<12, "O4">, DwarfRegNum<[12]>;
def O5 : Ri<13, "O5">, DwarfRegNum<[13]>; 
def O6 : Ri<14, "SP">, DwarfRegNum<[14]>;
def O7 : Ri<15, "O7">, DwarfRegNum<[15]>;
def L0 : Ri<16, "L0">, DwarfRegNum<[16]>;
def L1 : Ri<17, "L1">, DwarfRegNum<[17]>;
def L2 : Ri<18, "L2">, DwarfRegNum<[18]>; 
def L3 : Ri<19, "L3">, DwarfRegNum<[19]>;
def L4 : Ri<20, "L4">, DwarfRegNum<[20]>;
def L5 : Ri<21, "L5">, DwarfRegNum<[21]>; 
def L6 : Ri<22, "L6">, DwarfRegNum<[22]>;
def L7 : Ri<23, "L7">, DwarfRegNum<[23]>;
def I0 : Ri<24, "I0">, DwarfRegNum<[24]>;
def I1 : Ri<25, "I1">, DwarfRegNum<[25]>;
def I2 : Ri<26, "I2">, DwarfRegNum<[26]>; 
def I3 : Ri<27, "I3">, DwarfRegNum<[27]>;
def I4 : Ri<28, "I4">, DwarfRegNum<[28]>;
def I5 : Ri<29, "I5">, DwarfRegNum<[29]>; 
def I6 : Ri<30, "FP">, DwarfRegNum<[30]>;
def I7 : Ri<31, "I7">, DwarfRegNum<[31]>;

// Floating-point registers
def F0  : Rf< 0,  "F0">, DwarfRegNum<[32]>;
def F1  : Rf< 1,  "F1">, DwarfRegNum<[33]>;
def F2  : Rf< 2,  "F2">, DwarfRegNum<[34]>; 
def F3  : Rf< 3,  "F3">, DwarfRegNum<[35]>;
def F4  : Rf< 4,  "F4">, DwarfRegNum<[36]>;
def F5  : Rf< 5,  "F5">, DwarfRegNum<[37]>; 
def F6  : Rf< 6,  "F6">, DwarfRegNum<[38]>;
def F7  : Rf< 7,  "F7">, DwarfRegNum<[39]>;
def F8  : Rf< 8,  "F8">, DwarfRegNum<[40]>; 
def F9  : Rf< 9,  "F9">, DwarfRegNum<[41]>;
def F10 : Rf<10, "F10">, DwarfRegNum<[42]>;
def F11 : Rf<11, "F11">, DwarfRegNum<[43]>; 
def F12 : Rf<12, "F12">, DwarfRegNum<[44]>;
def F13 : Rf<13, "F13">, DwarfRegNum<[45]>;
def F14 : Rf<14, "F14">, DwarfRegNum<[46]>; 
def F15 : Rf<15, "F15">, DwarfRegNum<[47]>;
def F16 : Rf<16, "F16">, DwarfRegNum<[48]>;
def F17 : Rf<17, "F17">, DwarfRegNum<[49]>; 
def F18 : Rf<18, "F18">, DwarfRegNum<[50]>;
def F19 : Rf<19, "F19">, DwarfRegNum<[51]>;
def F20 : Rf<20, "F20">, DwarfRegNum<[52]>; 
def F21 : Rf<21, "F21">, DwarfRegNum<[53]>;
def F22 : Rf<22, "F22">, DwarfRegNum<[54]>;
def F23 : Rf<23, "F23">, DwarfRegNum<[55]>;
def F24 : Rf<24, "F24">, DwarfRegNum<[56]>;
def F25 : Rf<25, "F25">, DwarfRegNum<[57]>;
def F26 : Rf<26, "F26">, DwarfRegNum<[58]>; 
def F27 : Rf<27, "F27">, DwarfRegNum<[59]>;
def F28 : Rf<28, "F28">, DwarfRegNum<[60]>;
def F29 : Rf<29, "F29">, DwarfRegNum<[61]>; 
def F30 : Rf<30, "F30">, DwarfRegNum<[62]>;
def F31 : Rf<31, "F31">, DwarfRegNum<[63]>;

// Aliases of the F* registers used to hold 64-bit fp values (doubles)
def D0  : Rd< 0,  "F0", [F0,   F1]>, DwarfRegNum<[72]>;
def D1  : Rd< 2,  "F2", [F2,   F3]>, DwarfRegNum<[73]>;
def D2  : Rd< 4,  "F4", [F4,   F5]>, DwarfRegNum<[74]>;
def D3  : Rd< 6,  "F6", [F6,   F7]>, DwarfRegNum<[75]>;
def D4  : Rd< 8,  "F8", [F8,   F9]>, DwarfRegNum<[76]>;
def D5  : Rd<10, "F10", [F10, F11]>, DwarfRegNum<[77]>;
def D6  : Rd<12, "F12", [F12, F13]>, DwarfRegNum<[78]>;
def D7  : Rd<14, "F14", [F14, F15]>, DwarfRegNum<[79]>;
def D8  : Rd<16, "F16", [F16, F17]>, DwarfRegNum<[80]>;
def D9  : Rd<18, "F18", [F18, F19]>, DwarfRegNum<[81]>;
def D10 : Rd<20, "F20", [F20, F21]>, DwarfRegNum<[82]>;
def D11 : Rd<22, "F22", [F22, F23]>, DwarfRegNum<[83]>;
def D12 : Rd<24, "F24", [F24, F25]>, DwarfRegNum<[84]>;
def D13 : Rd<26, "F26", [F26, F27]>, DwarfRegNum<[85]>;
def D14 : Rd<28, "F28", [F28, F29]>, DwarfRegNum<[86]>;
def D15 : Rd<30, "F30", [F30, F31]>, DwarfRegNum<[87]>;

// Register classes.
//
// FIXME: the register order should be defined in terms of the preferred
// allocation order...
//
def IntRegs : RegisterClass<"SP", [i32], 32,
                            (add L0, L1, L2, L3, L4, L5, L6,
                                 L7, I0, I1, I2, I3, I4, I5,
                                 O0, O1, O2, O3, O4, O5, O7,
                                 G1,
                                 // Non-allocatable regs:
                                 G2, G3, G4, // FIXME: OK for use only in
                                             // applications, not libraries.
                                 O6, // stack ptr
                                 I6, // frame ptr
                                 I7, // return address
                                 G0, // constant zero
                                 G5, G6, G7 // reserved for kernel
                                 )>;

def FPRegs : RegisterClass<"SP", [f32], 32, (sequence "F%u", 0, 31)>;

def DFPRegs : RegisterClass<"SP", [f64], 64, (sequence "D%u", 0, 15)>;
