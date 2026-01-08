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

// Expected data value for S32C1I operation
def SCOMPARE1 : SRReg<12, "scompare1", ["SCOMPARE1", "12"]>;

// Literal base
def LITBASE : SRReg<5, "litbase", ["LITBASE", "5"]>;

// Windowed Register Option registers
def WINDOWBASE : SRReg<72, "windowbase", ["WINDOWBASE", "72"]>;
def WINDOWSTART : SRReg<73, "windowstart", ["WINDOWSTART", "73"]>;

// Instuction breakpoint enable register
def IBREAKENABLE : SRReg<96, "ibreakenable", ["IBREAKENABLE", "96"]>;

// Memory Control Register
def MEMCTL : SRReg<97, "memctl", ["MEMCTL", "97"]>;

// Atomic Operation Control
def ATOMCTL : SRReg<99, "atomctl", ["ATOMCTL", "99"]>;

def DDR : SRReg<104, "ddr", ["DDR", "104"]>;

// Instuction break address register 0
def IBREAKA0 : SRReg<128, "ibreaka0", ["IBREAKA0", "128"]>;

// Instuction break address register 1
def IBREAKA1 : SRReg<129, "ibreaka1", ["IBREAKA1", "129"]>;

// Data break address register 0
def DBREAKA0 : SRReg<144, "dbreaka0", ["DBREAKA0", "144"]>;

// Data break address register 1
def DBREAKA1 : SRReg<145, "dbreaka1", ["DBREAKA1", "145"]>;

// Data breakpoint control register 0
def DBREAKC0 : SRReg<160, "dbreakc0", ["DBREAKC0", "160"]>;

// Data breakpoint control register 1
def DBREAKC1 : SRReg<161, "dbreakc1", ["DBREAKC1", "161"]>;

def CONFIGID0 : SRReg<176, "configid0", ["CONFIGID0", "176"]>;

// Exception PC1
def EPC1 : SRReg<177, "epc1", ["EPC1", "177"]>;

// Exception PC2
def EPC2 : SRReg<178, "epc2", ["EPC2", "178"]>;

// Exception PC3
def EPC3 : SRReg<179, "epc3", ["EPC3", "179"]>;

// Exception PC4
def EPC4 : SRReg<180, "epc4", ["EPC4", "180"]>;

// Exception PC5
def EPC5 : SRReg<181, "epc5", ["EPC5", "181"]>;

// Exception PC6
def EPC6 : SRReg<182, "epc6", ["EPC6", "182"]>;

// Exception PC7
def EPC7 : SRReg<183, "epc7", ["EPC7", "183"]>;

def DEPC : SRReg<192, "depc", ["DEPC", "192"]>;
def EPS2 : SRReg<194, "eps2", ["EPS2", "194"]>;
def EPS3 : SRReg<195, "eps3", ["EPS3", "195"]>;
def EPS4 : SRReg<196, "eps4", ["EPS4", "196"]>;
def EPS5 : SRReg<197, "eps5", ["EPS5", "197"]>;
def EPS6 : SRReg<198, "eps6", ["EPS6", "198"]>;
def EPS7 : SRReg<199, "eps7", ["EPS7", "199"]>;

def CONFIGID1 : SRReg<208, "configid1", ["CONFIGID1", "208"]>;

def EXCSAVE1 : SRReg<209, "excsave1", ["EXCSAVE1", "209"]>;
def EXCSAVE2 : SRReg<210, "excsave2", ["EXCSAVE2", "210"]>;
def EXCSAVE3 : SRReg<211, "excsave3", ["EXCSAVE3", "211"]>;
def EXCSAVE4 : SRReg<212, "excsave4", ["EXCSAVE4", "212"]>;
def EXCSAVE5 : SRReg<213, "excsave5", ["EXCSAVE5", "213"]>;
def EXCSAVE6 : SRReg<214, "excsave6", ["EXCSAVE6", "214"]>;
def EXCSAVE7 : SRReg<215, "excsave7", ["EXCSAVE7", "215"]>;

def CPENABLE : SRReg<224, "cpenable", ["CPENABLE", "224"]>;

// Interrupt enable mask register
def INTERRUPT : SRReg<226, "interrupt", ["INTERRUPT", "226"]>;

def INTSET : SRReg<226, "intset", ["INTSET"]>;

def INTCLEAR : SRReg<227, "intclear", ["INTCLEAR", "227"]>;

def INTENABLE : SRReg<228, "intenable", ["INTENABLE", "228"]>;

// Processor State
def PS : SRReg<230, "ps", ["PS", "230"]>;

def EXCCAUSE : SRReg<232, "exccause", ["EXCCAUSE", "232"]>;

// Cause of last debug exception register
def DEBUGCAUSE : SRReg<233, "debugcause", ["DEBUGCAUSE", "233"]>;

// Processor Clock Count Register
def CCOUNT : SRReg<234, "ccount", ["CCOUNT", "234"]>;

// Processor ID Register
def PRID : SRReg<235, "prid", ["PRID", "235"]>;

def ICOUNT : SRReg<236, "icount", ["ICOUNT", "236"]>;
def ICOUNTLEVEL : SRReg<237, "icountlevel", ["ICOUNTLEVEL", "237"]>;
def EXCVADDR : SRReg<238, "excvaddr", ["EXCVADDR", "238"]>;

// Cycle number to interrupt register 0
def CCOMPARE0 : SRReg<240, "ccompare0", ["CCOMPARE0", "240"]>;

// Cycle number to interrupt register 1
def CCOMPARE1 : SRReg<241, "ccompare1", ["CCOMPARE1", "241"]>;

// Cycle number to interrupt register 2
def CCOMPARE2 : SRReg<242, "ccompare2", ["CCOMPARE2", "242"]>;

// Vector base register
def VECBASE : SRReg<231, "vecbase", ["VECBASE", "231"]>;

// Xtensa Miscellaneous SR
def MISC0 : SRReg<244, "misc0", ["MISC0", "244"]>;
def MISC1 : SRReg<245, "misc1", ["MISC1", "245"]>;
def MISC2 : SRReg<246, "misc2", ["MISC2", "246"]>;
def MISC3 : SRReg<247, "misc3", ["MISC3", "247"]>;

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
  LBEG, LEND, LCOUNT, SAR, BREG, SCOMPARE1, LITBASE, ACCLO, ACCHI, MR,
  WINDOWBASE, WINDOWSTART, IBREAKENABLE, MEMCTL, ATOMCTL, DDR, IBREAKA0, IBREAKA1,
  DBREAKA0, DBREAKA1, DBREAKC0, DBREAKC1, CONFIGID0, EPC1, EPC2, EPC3, EPC4, EPC5,
  EPC6, EPC7, DEPC, EPS2, EPS3, EPS4, EPS5, EPS6, EPS7, CONFIGID1, EXCSAVE1, EXCSAVE2,
  EXCSAVE3, EXCSAVE4, EXCSAVE5, EXCSAVE6, EXCSAVE7, CPENABLE, INTERRUPT, INTSET, INTCLEAR, INTENABLE,
  PS, VECBASE, EXCCAUSE, DEBUGCAUSE, CCOUNT, PRID, ICOUNT, ICOUNTLEVEL, EXCVADDR, CCOMPARE0,
  CCOMPARE1, CCOMPARE2, MISC0, MISC1, MISC2, MISC3)>;

//===----------------------------------------------------------------------===//
// USER registers
//===----------------------------------------------------------------------===//
class URReg<bits<8> num, string n, list<string> alt = []> : XtensaReg<n> {
  let HWEncoding{7-0} = num;
  let AltNames = alt;
}

// Thread Pointer register
def THREADPTR : URReg<231, "threadptr", ["THREADPTR"]>;

def FCR : URReg<232, "fcr", ["FCR"]>;
def FSR : URReg<233, "fsr", ["FSR"]>;

// DFPAccel registers
def F64R_LO : URReg<234, "f64r_lo", ["F64R_LO"]>;
def F64R_HI : URReg<235, "f64r_hi", ["F64R_HI"]>;
def F64S : URReg<236, "f64s", ["F64S"]>;

def UR :  RegisterClass<"Xtensa", [i32], 32, (add
  THREADPTR, FCR, FSR, F64R_LO, F64R_HI, F64S)>;

//===----------------------------------------------------------------------===//
// Floating-Point registers
//===----------------------------------------------------------------------===//

// Xtensa Floating-Point regs
class FPReg<bits<4> num, string n> : XtensaReg<n> {
  let HWEncoding{3-0} = num;
}

def F0 : FPReg<0, "f0">, DwarfRegNum<[19]>;
def F1 : FPReg<1, "f1">, DwarfRegNum<[20]>;
def F2 : FPReg<2, "f2">, DwarfRegNum<[21]>;
def F3 : FPReg<3, "f3">, DwarfRegNum<[22]>;
def F4 : FPReg<4, "f4">, DwarfRegNum<[23]>;
def F5 : FPReg<5, "f5">, DwarfRegNum<[24]>;
def F6 : FPReg<6, "f6">, DwarfRegNum<[25]>;
def F7 : FPReg<7, "f7">, DwarfRegNum<[26]>;
def F8 : FPReg<8, "f8">, DwarfRegNum<[27]>;
def F9 : FPReg<9, "f9">, DwarfRegNum<[28]>;
def F10 : FPReg<10, "f10">, DwarfRegNum<[29]>;
def F11 : FPReg<11, "f11">, DwarfRegNum<[30]>;
def F12 : FPReg<12, "f12">, DwarfRegNum<[31]>;
def F13 : FPReg<13, "f13">, DwarfRegNum<[32]>;
def F14 : FPReg<14, "f14">, DwarfRegNum<[33]>;
def F15 : FPReg<15, "f15">, DwarfRegNum<[34]>;

// Floating-Point register class with allocation order
def FPR : RegisterClass<"Xtensa", [f32], 32, (add
  F8, F9, F10, F11, F12, F13, F14, F15,
  F7, F6, F5, F4, F3, F2, F1, F0)>;

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
