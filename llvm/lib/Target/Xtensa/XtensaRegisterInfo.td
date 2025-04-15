//===- XtensaRegisterInfo.td - Xtensa Register defs --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Class definitions.
//===----------------------------------------------------------------------===//

class XtensaReg<string n> : Register<n> {
  let Namespace = "Xtensa";
}

class XtensaRegWithSubRegs<string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let Namespace = "Xtensa";
}

//===----------------------------------------------------------------------===//
// General-purpose registers
//===----------------------------------------------------------------------===//

// Xtensa general purpose regs
class ARReg<bits<4> num, string n, list<string> alt = []> : XtensaReg<n> {
  let HWEncoding{3-0} = num;
  let AltNames = alt;
}

// Return Address
def A0 : ARReg<0, "a0">, DwarfRegNum<[0]>;

// Stack Pointer (callee-saved)
def SP : ARReg<1, "a1", ["sp"]>, DwarfRegNum<[1]>;

// Function Arguments
def A2 : ARReg<2, "a2">, DwarfRegNum<[2]>;
def A3 : ARReg<3, "a3">, DwarfRegNum<[3]>;
def A4 : ARReg<4, "a4">, DwarfRegNum<[4]>;
def A5 : ARReg<5, "a5">, DwarfRegNum<[5]>;
def A6 : ARReg<6, "a6">, DwarfRegNum<[6]>;
def A7 : ARReg<7, "a7">, DwarfRegNum<[7]>;

// Static Chain
def A8 : ARReg<8, "a8">, DwarfRegNum<[8]>;

def A9 : ARReg<9, "a9">, DwarfRegNum<[9]>;
def A10 : ARReg<10, "a10">, DwarfRegNum<[10]>;
def A11 : ARReg<11, "a11">, DwarfRegNum<[11]>;

// Callee-saved
def A12 : ARReg<12, "a12">, DwarfRegNum<[12]>;
def A13 : ARReg<13, "a13">, DwarfRegNum<[13]>;
def A14 : ARReg<14, "a14">, DwarfRegNum<[14]>;

// Stack-Frame Pointer (optional) - Callee-Saved
def A15 : ARReg<15, "a15">, DwarfRegNum<[15]>;

// Register class with allocation order
def AR : RegisterClass<"Xtensa", [i32], 32, (add
  A8, A9, A10, A11, A12, A13, A14, A15,
  A7, A6, A5, A4, A3, A2, A0, SP)>;

//===----------------------------------------------------------------------===//
// Special-purpose registers
//===----------------------------------------------------------------------===//
class SRReg<bits<8> num, string n, list<string> alt = []> : XtensaReg<n> {
  let HWEncoding{7-0} = num;
  let AltNames = alt;
}

// Loop Option Registers
def LBEG : SRReg<0, "lbeg", ["LBEG", "0"]>;
def LEND : SRReg<1, "lend", ["LEND", "1"]>;
def LCOUNT : SRReg<2, "lcount", ["LCOUNT", "2"]>;

// Shift Amount Register
def SAR : SRReg<3, "sar", ["SAR","3"]>;

// Boolean Register
def BREG : SRReg<4, "br", ["BR","4"]>;

// Windowed Register Option registers
def WINDOWBASE : SRReg<72, "windowbase", ["WINDOWBASE", "72"]>;
def WINDOWSTART : SRReg<73, "windowstart", ["WINDOWSTART", "73"]>;

// MAC16 Option registers
def ACCLO : SRReg<16, "acclo", ["ACCLO", "16"]>;
def ACCHI : SRReg<17, "acchi", ["ACCHI", "17"]>;
def M0    : SRReg<32, "m0", ["M0", "32"]>;
def M1    : SRReg<33, "m1", ["M1", "33"]>;
def M2    : SRReg<34, "m2", ["M2", "34"]>;
def M3    : SRReg<35, "m3", ["M3", "35"]>;

def MR01 :  RegisterClass<"Xtensa", [i32], 32, (add M0, M1)>;
def MR23 :  RegisterClass<"Xtensa", [i32], 32, (add M2, M3)>;
def MR   :  RegisterClass<"Xtensa", [i32], 32, (add MR01, MR23)>;

def SR :  RegisterClass<"Xtensa", [i32], 32, (add
  LBEG, LEND, LCOUNT, SAR, BREG, MR, WINDOWBASE, WINDOWSTART)>;

//===----------------------------------------------------------------------===//
// Boolean registers
//===----------------------------------------------------------------------===//
class BReg<bits<4> num, string n> : XtensaReg<n> {
  let HWEncoding{3-0} = num;
}

foreach i = 0-15 in {
  def B#i  : BReg<i, "b"#i>;
}

// Boolean register class
def BR : RegisterClass<"Xtensa", [v1i1], 8, (add B0, B1,
  B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15)> {
  let Size = 8;
}
