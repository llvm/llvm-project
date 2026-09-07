//===-- MipsRegisterInfo.td - Mips Register defs -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MIPS register file
//===----------------------------------------------------------------------===//
let Namespace = "Mips" in {
def sub_32     : SubRegIndex<32>;
def sub_64     : SubRegIndex<64>;
def sub_lo     : SubRegIndex<32>;
def sub_hi     : SubRegIndex<32, 32>;
def sub_dsp16_19 : SubRegIndex<4, 16>;
def sub_dsp20    : SubRegIndex<1, 20>;
def sub_dsp21    : SubRegIndex<1, 21>;
def sub_dsp22    : SubRegIndex<1, 22>;
def sub_dsp23    : SubRegIndex<1, 23>;
}

class Unallocatable {
  bit isAllocatable = 0;
}

// We have banks of 32 registers each.
class MipsReg<bits<16> Enc, string n> : Register<n> {
  let HWEncoding = Enc;
  let Namespace = "Mips";
}

class MipsRegWithSubRegs<bits<16> Enc, string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let HWEncoding = Enc;
  let Namespace = "Mips";
}

// Mips CPU Registers.
class MipsGPRReg<bits<16> Enc, string n> : MipsReg<Enc, n>;

// Mips 64-bit CPU Registers
class Mips64GPRReg<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_32];
}

// Mips 32-bit FPU Registers
class FPR<bits<16> Enc, string n> : MipsReg<Enc, n>;

// Mips 64-bit (aliased) FPU Registers
class AFPR<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_lo, sub_hi];
  let CoveredBySubRegs = 1;
}

class AFPR64<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_lo, sub_hi];
  let CoveredBySubRegs = 1;
}

// Mips 128-bit (aliased) MSA Registers
class AFPR128<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_64];
}

// Accumulator Registers
class ACCReg<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_lo, sub_hi];
  let CoveredBySubRegs = 1;
}

// Mips Hardware Registers
class HWR<bits<16> Enc, string n> : MipsReg<Enc, n>;

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

let Namespace = "Mips" in {
  // General Purpose Registers
  let isConstant = true in
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
  let isConstant = true in
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

  // General Purpose nanoMIPS Registers
  let isConstant = true in
  def ZERO_NM : MipsGPRReg< 0, "zero">, DwarfRegNum<[0]>;
  def AT_NM   : MipsGPRReg< 1, "at">,   DwarfRegNum<[1]>;
  def T4_NM   : MipsGPRReg< 2, "t4">,   DwarfRegNum<[2]>;
  def T5_NM   : MipsGPRReg< 3, "t5">,   DwarfRegNum<[3]>;
  def A0_NM   : MipsGPRReg< 4, "a0">,   DwarfRegNum<[4]>;
  def A1_NM   : MipsGPRReg< 5, "a1">,   DwarfRegNum<[5]>;
  def A2_NM   : MipsGPRReg< 6, "a2">,   DwarfRegNum<[6]>;
  def A3_NM   : MipsGPRReg< 7, "a3">,   DwarfRegNum<[7]>;
  def A4_NM   : MipsGPRReg< 8, "a4">,   DwarfRegNum<[8]>;
  def A5_NM   : MipsGPRReg< 9, "a5">,   DwarfRegNum<[9]>;
  def A6_NM   : MipsGPRReg< 10, "a6">,  DwarfRegNum<[10]>;
  def A7_NM   : MipsGPRReg< 11, "a7">,  DwarfRegNum<[11]>;
  def T0_NM   : MipsGPRReg< 12, "t0">,  DwarfRegNum<[12]>;
  def T1_NM   : MipsGPRReg< 13, "t1">,  DwarfRegNum<[13]>;
  def T2_NM   : MipsGPRReg< 14, "t2">,  DwarfRegNum<[14]>;
  def T3_NM   : MipsGPRReg< 15, "t3">,  DwarfRegNum<[15]>;
  def S0_NM   : MipsGPRReg< 16, "s0">,  DwarfRegNum<[16]>;
  def S1_NM   : MipsGPRReg< 17, "s1">,  DwarfRegNum<[17]>;
  def S2_NM   : MipsGPRReg< 18, "s2">,  DwarfRegNum<[18]>;
  def S3_NM   : MipsGPRReg< 19, "s3">,  DwarfRegNum<[19]>;
  def S4_NM   : MipsGPRReg< 20, "s4">,  DwarfRegNum<[20]>;
  def S5_NM   : MipsGPRReg< 21, "s5">,  DwarfRegNum<[21]>;
  def S6_NM   : MipsGPRReg< 22, "s6">,  DwarfRegNum<[22]>;
  def S7_NM   : MipsGPRReg< 23, "s7">,  DwarfRegNum<[23]>;
  def T8_NM   : MipsGPRReg< 24, "t8">,  DwarfRegNum<[24]>;
  def T9_NM   : MipsGPRReg< 25, "t9">,  DwarfRegNum<[25]>;
  def K0_NM   : MipsGPRReg< 26, "k0">,  DwarfRegNum<[26]>;
  def K1_NM   : MipsGPRReg< 27, "k1">,  DwarfRegNum<[27]>;
  def GP_NM   : MipsGPRReg< 28, "gp">,  DwarfRegNum<[28]>;
  def SP_NM   : MipsGPRReg< 29, "sp">,  DwarfRegNum<[29]>;
  def FP_NM   : MipsGPRReg< 30, "fp">,  DwarfRegNum<[30]>;
  def RA_NM   : MipsGPRReg< 31, "ra">,  DwarfRegNum<[31]>;

  /// Mips Single point precision FPU Registers
  foreach I = 0-31 in
  def F#I : FPR<I, "f"#I>, DwarfRegNum<[!add(I, 32)]>;

  // Higher half of 64-bit FP registers.
  foreach I = 0-31 in
  def F_HI#I : FPR<I, "f"#I>, DwarfRegNum<[!add(I, 32)]>;

  /// Mips Double point precision FPU Registers (aliased
  /// with the single precision to hold 64 bit values)
  foreach I = 0-15 in
  def D#I : AFPR<!shl(I, 1), "f"#!shl(I, 1),
                 [!cast<FPR>("F"#!shl(I, 1)),
                  !cast<FPR>("F"#!add(!shl(I, 1), 1))]>;

  /// Mips Double point precision FPU Registers in MFP64 mode.
  foreach I = 0-31 in
  def D#I#_64 : AFPR64<I, "f"#I, [!cast<FPR>("F"#I), !cast<FPR>("F_HI"#I)]>,
                DwarfRegNum<[!add(I, 32)]>;

  /// Mips MSA registers
  /// MSA and FPU cannot both be present unless the FPU has 64-bit registers
  foreach I = 0-31 in
  def W#I : AFPR128<I, "w"#I, [!cast<AFPR64>("D"#I#"_64")]>,
            DwarfRegNum<[!add(I, 32)]>;

  // Hi/Lo registers
  def HI0 : MipsReg<0, "ac0">, DwarfRegNum<[64]>;
  def HI1 : MipsReg<1, "ac1">, DwarfRegNum<[176]>;
  def HI2 : MipsReg<2, "ac2">, DwarfRegNum<[178]>;
  def HI3 : MipsReg<3, "ac3">, DwarfRegNum<[180]>;
  def LO0 : MipsReg<0, "ac0">, DwarfRegNum<[65]>;
  def LO1 : MipsReg<1, "ac1">, DwarfRegNum<[177]>;
  def LO2 : MipsReg<2, "ac2">, DwarfRegNum<[179]>;
  def LO3 : MipsReg<3, "ac3">, DwarfRegNum<[181]>;

  let SubRegIndices = [sub_32] in {
  def HI0_64  : RegisterWithSubRegs<"hi", [HI0]>;
  def LO0_64  : RegisterWithSubRegs<"lo", [LO0]>;
  }

  // FP control registers.
  foreach I = 0-31 in
  def FCR#I : MipsReg<I, ""#I>;

  // FP condition code registers.
  foreach I = 0-7 in
  def FCC#I : MipsReg<I, "fcc"#I>;

  // COP0 registers.
  foreach I = 0-31 in
  def COP0#I : MipsReg<I, ""#I>;

  // COP2 registers.
  foreach I = 0-31 in
  def COP2#I : MipsReg<I, ""#I>;

  // COP3 registers.
  foreach I = 0-31 in
  def COP3#I : MipsReg<I, ""#I>;

  // PC register
  def PC : Register<"pc">;

  // Hardware registers
  def HWR0 : MipsReg<0, "hwr_cpunum">;
  def HWR1 : MipsReg<1, "hwr_synci_step">;
  def HWR2 : MipsReg<2, "hwr_cc">;
  def HWR3 : MipsReg<3, "hwr_ccres">;

  foreach I = 4-31 in
  def HWR#I : MipsReg<I, ""#I>;

  // Accum registers
  foreach I = 0-3 in
  def AC#I : ACCReg<I, "ac"#I,
                    [!cast<Register>("LO"#I), !cast<Register>("HI"#I)]>;

  def AC0_64 : ACCReg<0, "ac0", [LO0_64, HI0_64]>;

  // DSP-ASE control register fields.
  def DSPPos : Register<"">;
  def DSPSCount : Register<"">;
  def DSPCarry : Register<"">;
  def DSPEFI : Register<"">;
  def DSPOutFlag16_19 : Register<"">;
  def DSPOutFlag20 : Register<"">;
  def DSPOutFlag21 : Register<"">;
  def DSPOutFlag22 : Register<"">;
  def DSPOutFlag23 : Register<"">;
  def DSPCCond : Register<"">;

  let SubRegIndices = [sub_dsp16_19, sub_dsp20, sub_dsp21, sub_dsp22,
                       sub_dsp23] in
  def DSPOutFlag : RegisterWithSubRegs<"", [DSPOutFlag16_19, DSPOutFlag20,
                                            DSPOutFlag21, DSPOutFlag22,
                                            DSPOutFlag23]>;

  // MSA-ASE control registers.
  def MSAIR      : MipsReg<0, "0">;
  def MSACSR     : MipsReg<1, "1">;
  def MSAAccess  : MipsReg<2, "2">;
  def MSASave    : MipsReg<3, "3">;
  def MSAModify  : MipsReg<4, "4">;
  def MSARequest : MipsReg<5, "5">;
  def MSAMap     : MipsReg<6, "6">;
  def MSAUnmap   : MipsReg<7, "7">;
  // MSA-ASE fake control registers.
  // These registers do not exist, but instructions like `cfcmsa`
  // and `ctcmsa` allows to specify them.
  foreach I = 8-31 in
  def MSA#I : MipsReg<I, ""#I>;

  // Octeon multiplier and product registers
  def MPL0 : MipsReg<0, "mpl0">;
  def MPL1 : MipsReg<1, "mpl1">;
  def MPL2 : MipsReg<2, "mpl2">;
  def P0 : MipsReg<0, "p0">;
  def P1 : MipsReg<1, "p1">;
  def P2 : MipsReg<2, "p2">;

  // COP0 registers with fixed selects
  def COP0Sel_INDEX           : MipsReg< !or(!shl(0, 5), 0), "index">;
  def COP0Sel_MVPCONTROL      : MipsReg< !or(!shl(0, 5), 1), "mvpcontrol">;
  def COP0Sel_MVPCONF0        : MipsReg< !or(!shl(0, 5), 2), "mvpconf0">;
  def COP0Sel_MVPCONF1        : MipsReg< !or(!shl(0, 5), 3), "mvpconf1">;
  def COP0Sel_VPCONTROL       : MipsReg< !or(!shl(0, 5), 4), "vpcontrol">;
  def COP0Sel_RANDOM          : MipsReg< !or(!shl(1, 5), 0), "random">;
  def COP0Sel_VPECONTROL      : MipsReg< !or(!shl(1, 5), 1), "vpecontrol">;
  def COP0Sel_VPECONF0        : MipsReg< !or(!shl(1, 5), 2), "vpeconf0">;
  def COP0Sel_VPECONF1        : MipsReg< !or(!shl(1, 5), 3), "vpeconf1">;
  def COP0Sel_YQMASK          : MipsReg< !or(!shl(1, 5), 4), "yqmask">;
  def COP0Sel_VPESCHEDULE     : MipsReg< !or(!shl(1, 5), 5), "vpeschedule">;
  def COP0Sel_VPESCHEFBACK    : MipsReg< !or(!shl(1, 5), 6), "vpeschefback">;
  def COP0Sel_VPEOPT          : MipsReg< !or(!shl(1, 5), 7), "vpeopt">;
  def COP0Sel_ENTRYLO0        : MipsReg< !or(!shl(2, 5), 0), "entrylo0">;
  def COP0Sel_TCSTATUS        : MipsReg< !or(!shl(2, 5), 1), "tcstatus">;
  def COP0Sel_TCBIND          : MipsReg< !or(!shl(2, 5), 2), "tcbind">;
  def COP0Sel_TCRESTART       : MipsReg< !or(!shl(2, 5), 3), "tcrestart">;
  def COP0Sel_TCHALT          : MipsReg< !or(!shl(2, 5), 4), "tchalt">;
  def COP0Sel_TCCONTEXT       : MipsReg< !or(!shl(2, 5), 5), "tccontext">;
  def COP0Sel_TCSCHEDULE      : MipsReg< !or(!shl(2, 5), 6), "tcschedule">;
  def COP0Sel_TCSCHEFBACK     : MipsReg< !or(!shl(2, 5), 7), "tcschefback">;
  def COP0Sel_ENTRYLO1        : MipsReg< !or(!shl(3, 5), 0), "entrylo1">;
  def COP0Sel_GLOBALNUMBER    : MipsReg< !or(!shl(3, 5), 1), "globalnumber">;
  def COP0Sel_TCOPT           : MipsReg< !or(!shl(3, 5), 7), "tcopt">;
  def COP0Sel_CONTEXT         : MipsReg< !or(!shl(4, 5), 0), "context">;
  def COP0Sel_CONTEXTCONFIG   : MipsReg< !or(!shl(4, 5), 1), "contextconfig">;
  def COP0Sel_USERLOCAL       : MipsReg< !or(!shl(4, 5), 2), "userlocal">;
  def COP0Sel_XCONTEXTCONFIG  : MipsReg< !or(!shl(4, 5), 3), "xcontextconfig">;
  def COP0Sel_DEBUGCONTEXTID  : MipsReg< !or(!shl(4, 5), 4), "debugcontextid">;
  def COP0Sel_MEMORYMAPID     : MipsReg< !or(!shl(4, 5), 5), "memorymapid">;
  def COP0Sel_PAGEMASK        : MipsReg< !or(!shl(5, 5), 0), "pagemask">;
  def COP0Sel_PAGEGRAIN       : MipsReg< !or(!shl(5, 5), 1), "pagegrain">;
  def COP0Sel_SEGCTL0         : MipsReg< !or(!shl(5, 5), 2), "segctl0">;
  def COP0Sel_SEGCTL1         : MipsReg< !or(!shl(5, 5), 3), "segctl1">;
  def COP0Sel_SEGCTL2         : MipsReg< !or(!shl(5, 5), 4), "segctl2">;
  def COP0Sel_PWBASE          : MipsReg< !or(!shl(5, 5), 5), "pwbase">;
  def COP0Sel_PWFIELD         : MipsReg< !or(!shl(5, 5), 6), "pwfield">;
  def COP0Sel_PWSIZE          : MipsReg< !or(!shl(5, 5), 7), "pwsize">;
  def COP0Sel_WIRED           : MipsReg< !or(!shl(6, 5), 0), "wired">;
  def COP0Sel_SRSCONF0        : MipsReg< !or(!shl(6, 5), 1), "srsconf0">;
  def COP0Sel_SRSCONF1        : MipsReg< !or(!shl(6, 5), 2), "srsconf1">;
  def COP0Sel_SRSCONF2        : MipsReg< !or(!shl(6, 5), 3), "srsconf2">;
  def COP0Sel_SRSCONF3        : MipsReg< !or(!shl(6, 5), 4), "srsconf3">;
  def COP0Sel_SRSCONF4        : MipsReg< !or(!shl(6, 5), 5), "srsconf4">;
  def COP0Sel_PWCTL           : MipsReg< !or(!shl(6, 5), 6), "pwctl">;
  def COP0Sel_HWRENA          : MipsReg< !or(!shl(7, 5), 0), "hwrena">;
  def COP0Sel_BADVADDR        : MipsReg< !or(!shl(8, 5), 0), "badvaddr">;
  def COP0Sel_BADINST         : MipsReg< !or(!shl(8, 5), 1), "badinst">;
  def COP0Sel_BADINSTRP       : MipsReg< !or(!shl(8, 5), 2), "badinstrp">;
  def COP0Sel_BADINSTRX       : MipsReg< !or(!shl(8, 5), 3), "badinstrx">;
  def COP0Sel_COUNT           : MipsReg< !or(!shl(9, 5), 0), "count">;
  def COP0Sel_ENTRYHI         : MipsReg< !or(!shl(10, 5), 0), "entryhi">;
  def COP0Sel_GUESTCTL1       : MipsReg< !or(!shl(10, 5), 4), "guestctl1">;
  def COP0Sel_GUESTCTL2       : MipsReg< !or(!shl(10, 5), 5), "guestctl2">;
  def COP0Sel_GUESTCTL3       : MipsReg< !or(!shl(10, 5), 6), "guestctl3">;
  def COP0Sel_COMPARE         : MipsReg< !or(!shl(11, 5), 0), "compare">;
  def COP0Sel_GUESTCTL0EXT    : MipsReg< !or(!shl(11, 5), 4), "guestctl0ext">;
  def COP0Sel_STATUS          : MipsReg< !or(!shl(12, 5), 0), "status">;
  def COP0Sel_INTCTL          : MipsReg< !or(!shl(12, 5), 1), "intctl">;
  def COP0Sel_SRSCTL          : MipsReg< !or(!shl(12, 5), 2), "srsctl">;
  def COP0Sel_SRSMAP          : MipsReg< !or(!shl(12, 5), 3), "srsmap">;
  def COP0Sel_VIEW_IPL        : MipsReg< !or(!shl(12, 5), 4), "view_ipl">;
  def COP0Sel_SRSMAP2         : MipsReg< !or(!shl(12, 5), 5), "srsmap2">;
  def COP0Sel_GUESTCTL0       : MipsReg< !or(!shl(12, 5), 6), "guestctl0">;
  def COP0Sel_GTOFFSET        : MipsReg< !or(!shl(12, 5), 7), "gtoffset">;
  def COP0Sel_CAUSE           : MipsReg< !or(!shl(13, 5), 0), "cause">;
  def COP0Sel_VIEW_RIPL       : MipsReg< !or(!shl(13, 5), 4), "view_ripl">;
  def COP0Sel_NESTEDEXC       : MipsReg< !or(!shl(13, 5), 5), "nestedexc">;
  def COP0Sel_EPC             : MipsReg< !or(!shl(14, 5), 0), "epc">;
  def COP0Sel_NESTEDEPC       : MipsReg< !or(!shl(14, 5), 2), "nestedepc">;
  def COP0Sel_PRID            : MipsReg< !or(!shl(15, 5), 0), "prid">;
  def COP0Sel_EBASE           : MipsReg< !or(!shl(15, 5), 1), "ebase">;
  def COP0Sel_CDMMBASE        : MipsReg< !or(!shl(15, 5), 2), "cdmmbase">;
  def COP0Sel_CMGCRBASE       : MipsReg< !or(!shl(15, 5), 3), "cmgcrbase">;
  def COP0Sel_BEVVA           : MipsReg< !or(!shl(15, 5), 4), "bevva">;
  def COP0Sel_CONFIG          : MipsReg< !or(!shl(16, 5), 0), "config">;
  def COP0Sel_CONFIG1         : MipsReg< !or(!shl(16, 5), 1), "config1">;
  def COP0Sel_CONFIG2         : MipsReg< !or(!shl(16, 5), 2), "config2">;
  def COP0Sel_CONFIG3         : MipsReg< !or(!shl(16, 5), 3), "config3">;
  def COP0Sel_CONFIG4         : MipsReg< !or(!shl(16, 5), 4), "config4">;
  def COP0Sel_CONFIG5         : MipsReg< !or(!shl(16, 5), 5), "config5">;
  def COP0Sel_LLADDR          : MipsReg< !or(!shl(17, 5), 0), "lladdr">;
  def COP0Sel_MAAR            : MipsReg< !or(!shl(17, 5), 1), "maar">;
  def COP0Sel_MAARI           : MipsReg< !or(!shl(17, 5), 2), "maari">;
  def COP0Sel_WATCHLO0        : MipsReg< !or(!shl(18, 5), 0), "watchlo0">;
  def COP0Sel_WATCHLO1        : MipsReg< !or(!shl(18, 5), 1), "watchlo1">;
  def COP0Sel_WATCHLO2        : MipsReg< !or(!shl(18, 5), 2), "watchlo2">;
  def COP0Sel_WATCHLO3        : MipsReg< !or(!shl(18, 5), 3), "watchlo3">;
  def COP0Sel_WATCHLO4        : MipsReg< !or(!shl(18, 5), 4), "watchlo4">;
  def COP0Sel_WATCHLO5        : MipsReg< !or(!shl(18, 5), 5), "watchlo5">;
  def COP0Sel_WATCHLO6        : MipsReg< !or(!shl(18, 5), 6), "watchlo6">;
  def COP0Sel_WATCHLO7        : MipsReg< !or(!shl(18, 5), 7), "watchlo7">;
  def COP0Sel_WATCHLO8        : MipsReg< !or(!shl(18, 5), 8), "watchlo8">;
  def COP0Sel_WATCHLO9        : MipsReg< !or(!shl(18, 5), 9), "watchlo9">;
  def COP0Sel_WATCHLO10       : MipsReg< !or(!shl(18, 5), 10), "watchlo10">;
  def COP0Sel_WATCHLO11       : MipsReg< !or(!shl(18, 5), 11), "watchlo11">;
  def COP0Sel_WATCHLO12       : MipsReg< !or(!shl(18, 5), 12), "watchlo12">;
  def COP0Sel_WATCHLO13       : MipsReg< !or(!shl(18, 5), 13), "watchlo13">;
  def COP0Sel_WATCHLO14       : MipsReg< !or(!shl(18, 5), 14), "watchlo14">;
  def COP0Sel_WATCHLO15       : MipsReg< !or(!shl(18, 5), 15), "watchlo15">;
  def COP0Sel_WATCHHI0        : MipsReg< !or(!shl(19, 5), 0), "watchhi0">;
  def COP0Sel_WATCHHI1        : MipsReg< !or(!shl(19, 5), 1), "watchhi1">;
  def COP0Sel_WATCHHI2        : MipsReg< !or(!shl(19, 5), 2), "watchhi2">;
  def COP0Sel_WATCHHI3        : MipsReg< !or(!shl(19, 5), 3), "watchhi3">;
  def COP0Sel_WATCHHI4        : MipsReg< !or(!shl(19, 5), 4), "watchhi4">;
  def COP0Sel_WATCHHI5        : MipsReg< !or(!shl(19, 5), 5), "watchhi5">;
  def COP0Sel_WATCHHI6        : MipsReg< !or(!shl(19, 5), 6), "watchhi6">;
  def COP0Sel_WATCHHI7        : MipsReg< !or(!shl(19, 5), 7), "watchhi7">;
  def COP0Sel_WATCHHI8        : MipsReg< !or(!shl(19, 5), 8), "watchhi8">;
  def COP0Sel_WATCHHI9        : MipsReg< !or(!shl(19, 5), 9), "watchhi9">;
  def COP0Sel_WATCHHI10       : MipsReg< !or(!shl(19, 5), 10), "watchhi10">;
  def COP0Sel_WATCHHI11       : MipsReg< !or(!shl(19, 5), 11), "watchhi11">;
  def COP0Sel_WATCHHI12       : MipsReg< !or(!shl(19, 5), 12), "watchhi12">;
  def COP0Sel_WATCHHI13       : MipsReg< !or(!shl(19, 5), 13), "watchhi13">;
  def COP0Sel_WATCHHI14       : MipsReg< !or(!shl(19, 5), 14), "watchhi14">;
  def COP0Sel_WATCHHI15       : MipsReg< !or(!shl(19, 5), 15), "watchhi15">;
  def COP0Sel_XCONTEXT        : MipsReg< !or(!shl(20, 5), 0), "xcontext">;
  def COP0Sel_DEBUG           : MipsReg< !or(!shl(23, 5), 0), "debug">;
  def COP0Sel_TRACECONTROL    : MipsReg< !or(!shl(23, 5), 1), "tracecontrol">;
  def COP0Sel_TRACECONTROL2   : MipsReg< !or(!shl(23, 5), 2), "tracecontrol2">;
  def COP0Sel_USERTRACEDATA1  : MipsReg< !or(!shl(23, 5), 3), "usertracedata1">;
  def COP0Sel_TRACEIBPC       : MipsReg< !or(!shl(23, 5), 4), "traceibpc">;
  def COP0Sel_TRACEDBPC       : MipsReg< !or(!shl(23, 5), 5), "tracedbpc">;
  def COP0Sel_DEBUG2          : MipsReg< !or(!shl(23, 5), 6), "debug2">;
  def COP0Sel_DEPC            : MipsReg< !or(!shl(24, 5), 0), "depc">;
  def COP0Sel_TRACECONTROL3   : MipsReg< !or(!shl(24, 5), 2), "tracecontrol3">;
  def COP0Sel_USERTRACEDATA2  : MipsReg< !or(!shl(24, 5), 3), "usertracedata2">;
  def COP0Sel_PERFCTL0        : MipsReg< !or(!shl(25, 5), 0), "perfctl0">;
  def COP0Sel_PERFCNT0        : MipsReg< !or(!shl(25, 5), 1), "perfcnt0">;
  def COP0Sel_PERFCTL1        : MipsReg< !or(!shl(25, 5), 2), "perfctl1">;
  def COP0Sel_PERFCNT1        : MipsReg< !or(!shl(25, 5), 3), "perfcnt1">;
  def COP0Sel_PERFCTL2        : MipsReg< !or(!shl(25, 5), 4), "perfctl2">;
  def COP0Sel_PERFCNT2        : MipsReg< !or(!shl(25, 5), 5), "perfcnt2">;
  def COP0Sel_PERFCTL3        : MipsReg< !or(!shl(25, 5), 6), "perfctl3">;
  def COP0Sel_PERFCNT3        : MipsReg< !or(!shl(25, 5), 7), "perfcnt3">;
  def COP0Sel_PERFCTL4        : MipsReg< !or(!shl(25, 5), 8), "perfctl4">;
  def COP0Sel_PERFCNT4        : MipsReg< !or(!shl(25, 5), 9), "perfcnt4">;
  def COP0Sel_PERFCTL5        : MipsReg< !or(!shl(25, 5), 10), "perfctl5">;
  def COP0Sel_PERFCNT5        : MipsReg< !or(!shl(25, 5), 11), "perfcnt5">;
  def COP0Sel_PERFCTL6        : MipsReg< !or(!shl(25, 5), 12), "perfctl6">;
  def COP0Sel_PERFCNT6        : MipsReg< !or(!shl(25, 5), 13), "perfcnt6">;
  def COP0Sel_PERFCTL7        : MipsReg< !or(!shl(25, 5), 14), "perfctl7">;
  def COP0Sel_PERFCNT7        : MipsReg< !or(!shl(25, 5), 15), "perfcnt7">;
  def COP0Sel_ERRCTL          : MipsReg< !or(!shl(26, 5), 0), "errctl">;
  def COP0Sel_CACHEERR        : MipsReg< !or(!shl(27, 5), 0), "cacheerr">;
  def COP0Sel_ITAGLO          : MipsReg< !or(!shl(28, 5), 0), "itaglo">;
  def COP0Sel_IDATALO         : MipsReg< !or(!shl(28, 5), 1), "idatalo">;
  def COP0Sel_DTAGLO          : MipsReg< !or(!shl(28, 5), 2), "dtaglo">;
  def COP0Sel_DDATALO         : MipsReg< !or(!shl(28, 5), 3), "ddatalo">;
  def COP0Sel_ITAGHI          : MipsReg< !or(!shl(29, 5), 0), "itaghi">;
  def COP0Sel_IDATAHI         : MipsReg< !or(!shl(29, 5), 1), "idatahi">;
  def COP0Sel_DTAGHI          : MipsReg< !or(!shl(29, 5), 2), "dtaghi">;
  def COP0Sel_DDATAHI         : MipsReg< !or(!shl(29, 5), 3), "ddatahi">;
  def COP0Sel_ERROREPC        : MipsReg< !or(!shl(30, 5), 0), "errorepc">;
  def COP0Sel_DESAVE          : MipsReg< !or(!shl(31, 5), 0), "desave">;
  def COP0Sel_KSCRATCH1       : MipsReg< !or(!shl(31, 5), 2), "kscratch1">;
  def COP0Sel_KSCRATCH2       : MipsReg< !or(!shl(31, 5), 3), "kscratch2">;
  def COP0Sel_KSCRATCH3       : MipsReg< !or(!shl(31, 5), 4), "kscratch3">;
  def COP0Sel_KSCRATCH4       : MipsReg< !or(!shl(31, 5), 5), "kscratch4">;
  def COP0Sel_KSCRATCH5       : MipsReg< !or(!shl(31, 5), 6), "kscratch5">;
  def COP0Sel_KSCRATCH6       : MipsReg< !or(!shl(31, 5), 7), "kscratch6">;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

class GPR32Class<list<ValueType> regTypes> :
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

def GPR32 : GPR32Class<[i32]>;

def GPR32ZERO : RegisterClass<"Mips", [i32], 32, (add
  // Reserved
  ZERO)>;

def GPR32NONZERO : RegisterClass<"Mips", [i32], 32, (add
  // Reserved
  AT,
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

def DSPR  : GPR32Class<[v4i8, v2i16, i32]>;

def GPRMM16 : RegisterClass<"Mips", [i32], 32, (add
  // Callee save
  S0, S1,
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3)>;

def GPRMM16Zero : RegisterClass<"Mips", [i32], 32, (add
  // Reserved
  ZERO,
  // Callee save
  S1,
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3)>;

def GPRMM16MoveP : RegisterClass<"Mips", [i32], 32, (add
  // Reserved
  ZERO,
  // Callee save
  S1,
  // Return Values and Arguments
  V0, V1,
  // Callee save
  S0, S2, S3, S4)>;

def GPRMM16MovePPairFirst : RegisterClass<"Mips", [i32], 32, (add
  // Arguments
  A0, A1, A2)>;

def GPRMM16MovePPairSecond : RegisterClass<"Mips", [i32], 32, (add
  // Arguments
  A1, A2, A3,
  // Callee save
  S5, S6)>;

class GPRNM32Class<list<ValueType> regTypes> :
  RegisterClass<"Mips", regTypes, 32, (add
  // Reserved
  ZERO_NM, AT_NM,
  // Not preserved across procedure calls
  T4_NM, T5_NM,
  // Return Values and Arguments
  A0_NM, A1_NM, A2_NM, A3_NM, A4_NM, A5_NM, A6_NM, A7_NM,
  // Not preserved across procedure calls
  T0_NM, T1_NM, T2_NM, T3_NM,
  // Callee save
  S0_NM, S1_NM, S2_NM, S3_NM, S4_NM, S5_NM, S6_NM, S7_NM,
  // Not preserved across procedure calls
  T8_NM, T9_NM,
  // Reserved
  K0_NM, K1_NM, GP_NM, SP_NM, FP_NM, RA_NM)>;

def GPRNM32 : GPRNM32Class<[i32]>;

class COP0SelRegClassNM<list<ValueType> regTypes> :
  RegisterClass<"Mips", regTypes, 32, (add
  COP0Sel_INDEX, COP0Sel_MVPCONTROL, COP0Sel_MVPCONF0, COP0Sel_MVPCONF1,
  COP0Sel_VPCONTROL, COP0Sel_RANDOM, COP0Sel_VPECONTROL, COP0Sel_VPECONF0,
  COP0Sel_VPECONF1, COP0Sel_YQMASK, COP0Sel_VPESCHEDULE, COP0Sel_VPESCHEFBACK,
  COP0Sel_VPEOPT, COP0Sel_ENTRYLO0, COP0Sel_TCSTATUS, COP0Sel_TCBIND,
  COP0Sel_TCRESTART, COP0Sel_TCHALT, COP0Sel_TCCONTEXT, COP0Sel_TCSCHEDULE,
  COP0Sel_TCSCHEFBACK, COP0Sel_ENTRYLO1, COP0Sel_GLOBALNUMBER, COP0Sel_TCOPT,
  COP0Sel_CONTEXT, COP0Sel_CONTEXTCONFIG, COP0Sel_USERLOCAL,
  COP0Sel_XCONTEXTCONFIG, COP0Sel_DEBUGCONTEXTID, COP0Sel_MEMORYMAPID,
  COP0Sel_PAGEMASK, COP0Sel_PAGEGRAIN, COP0Sel_SEGCTL0, COP0Sel_SEGCTL1,
  COP0Sel_SEGCTL2, COP0Sel_PWBASE, COP0Sel_PWFIELD, COP0Sel_PWSIZE,
  COP0Sel_WIRED, COP0Sel_SRSCONF0, COP0Sel_SRSCONF1, COP0Sel_SRSCONF2,
  COP0Sel_SRSCONF3, COP0Sel_SRSCONF4, COP0Sel_PWCTL, COP0Sel_HWRENA,
  COP0Sel_BADVADDR, COP0Sel_BADINST, COP0Sel_BADINSTRP, COP0Sel_BADINSTRX,
  COP0Sel_COUNT, COP0Sel_ENTRYHI, COP0Sel_GUESTCTL1, COP0Sel_GUESTCTL2,
  COP0Sel_GUESTCTL3, COP0Sel_COMPARE, COP0Sel_GUESTCTL0EXT, COP0Sel_STATUS,
  COP0Sel_INTCTL, COP0Sel_SRSCTL, COP0Sel_SRSMAP, COP0Sel_VIEW_IPL,
  COP0Sel_SRSMAP2, COP0Sel_GUESTCTL0, COP0Sel_GTOFFSET, COP0Sel_CAUSE,
  COP0Sel_VIEW_RIPL, COP0Sel_NESTEDEXC, COP0Sel_EPC, COP0Sel_NESTEDEPC,
  COP0Sel_PRID, COP0Sel_EBASE, COP0Sel_CDMMBASE, COP0Sel_CMGCRBASE,
  COP0Sel_BEVVA, COP0Sel_CONFIG, COP0Sel_CONFIG1, COP0Sel_CONFIG2,
  COP0Sel_CONFIG3, COP0Sel_CONFIG4, COP0Sel_CONFIG5, COP0Sel_LLADDR,
  COP0Sel_MAAR, COP0Sel_MAARI, COP0Sel_WATCHLO0, COP0Sel_WATCHLO1,
  COP0Sel_WATCHLO2, COP0Sel_WATCHLO3, COP0Sel_WATCHLO4, COP0Sel_WATCHLO5,
  COP0Sel_WATCHLO6, COP0Sel_WATCHLO7, COP0Sel_WATCHLO8, COP0Sel_WATCHLO9,
  COP0Sel_WATCHLO10, COP0Sel_WATCHLO11, COP0Sel_WATCHLO12, COP0Sel_WATCHLO13,
  COP0Sel_WATCHLO14, COP0Sel_WATCHLO15, COP0Sel_WATCHHI0, COP0Sel_WATCHHI1,
  COP0Sel_WATCHHI2, COP0Sel_WATCHHI3, COP0Sel_WATCHHI4, COP0Sel_WATCHHI5,
  COP0Sel_WATCHHI6, COP0Sel_WATCHHI7, COP0Sel_WATCHHI8, COP0Sel_WATCHHI9,
  COP0Sel_WATCHHI10, COP0Sel_WATCHHI11, COP0Sel_WATCHHI12, COP0Sel_WATCHHI13,
  COP0Sel_WATCHHI14, COP0Sel_WATCHHI15, COP0Sel_XCONTEXT, COP0Sel_DEBUG,
  COP0Sel_TRACECONTROL, COP0Sel_TRACECONTROL2, COP0Sel_USERTRACEDATA1,
  COP0Sel_TRACEIBPC, COP0Sel_TRACEDBPC, COP0Sel_DEBUG2, COP0Sel_DEPC,
  COP0Sel_TRACECONTROL3, COP0Sel_USERTRACEDATA2, COP0Sel_PERFCTL0,
  COP0Sel_PERFCNT0, COP0Sel_PERFCTL1, COP0Sel_PERFCNT1, COP0Sel_PERFCTL2,
  COP0Sel_PERFCNT2, COP0Sel_PERFCTL3, COP0Sel_PERFCNT3, COP0Sel_PERFCTL4,
  COP0Sel_PERFCNT4, COP0Sel_PERFCTL5, COP0Sel_PERFCNT5, COP0Sel_PERFCTL6,
  COP0Sel_PERFCNT6, COP0Sel_PERFCTL7, COP0Sel_PERFCNT7, COP0Sel_ERRCTL,
  COP0Sel_CACHEERR, COP0Sel_ITAGLO, COP0Sel_IDATALO, COP0Sel_DTAGLO,
  COP0Sel_DDATALO, COP0Sel_ITAGHI, COP0Sel_IDATAHI, COP0Sel_DTAGHI,
  COP0Sel_DDATAHI, COP0Sel_ERROREPC, COP0Sel_DESAVE, COP0Sel_KSCRATCH1,
  COP0Sel_KSCRATCH2, COP0Sel_KSCRATCH3, COP0Sel_KSCRATCH4, COP0Sel_KSCRATCH5,
  COP0Sel_KSCRATCH6)>;

def COP0Sel : COP0SelRegClassNM<[i32]>, Unallocatable;

def GPRNM32_TAIL : RegisterClass<"Mips", [i32], 32, (add
  // Not preserved across procedure calls
  T4_NM, T5_NM,
  // Return Values and Arguments
  A0_NM, A1_NM, A2_NM, A3_NM, A4_NM, A5_NM, A6_NM, A7_NM,
  // Not preserved across procedure calls
  T0_NM, T1_NM, T2_NM, T3_NM,
  // Not preserved across procedure calls
  T8_NM, T9_NM)>;

// nanoMIPS "gpr3" register encoding type.
def GPRNM3 : RegisterClass<"Mips", [i32], 32, (add
  // Callee save
  S0_NM, S1_NM, S2_NM, S3_NM,
  // Return Values and Arguments
  A0_NM, A1_NM, A2_NM, A3_NM)>;

// nanoMIPS "gpr3.zero" register encoding type.
def GPRNM3Z : RegisterClass<"Mips", [i32], 32, (add
  // Reserved
  ZERO_NM,
  // Callee save
  S1_NM, S2_NM, S3_NM,
  // Return Values and Arguments
  A0_NM, A1_NM, A2_NM, A3_NM)>;

// nanoMIPS "gpr4x4" register encoding type.
def GPRNM4 : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  A4_NM, A5_NM, A6_NM, A7_NM, A0_NM, A1_NM, A2_NM, A3_NM,
  // Callee save
  S0_NM, S1_NM, S2_NM, S3_NM, S4_NM, S5_NM, S6_NM, S7_NM)>;

// nanoMIPS "gpr4x4.zero" register encoding type.
def GPRNM4Z : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  A4_NM, A5_NM, A6_NM,
  // Reserved
  ZERO_NM,
  // Return Values and Arguments
  A0_NM, A1_NM, A2_NM, A3_NM,
  // Callee save
  S0_NM, S1_NM, S2_NM, S3_NM, S4_NM, S5_NM, S6_NM, S7_NM)>;

// nanoMIPS non-zero gpr register encoding type.
def GPRNM32NZ : RegisterClass<"Mips", [i32], 32, (add
  // Reserved
  AT_NM,
  // Not preserved across procedure calls
  T4_NM, T5_NM,
  // Return Values and Arguments
  A0_NM, A1_NM, A2_NM, A3_NM, A4_NM, A5_NM, A6_NM, A7_NM,
  // Not preserved across procedure calls
  T0_NM, T1_NM, T2_NM, T3_NM,
  // Callee save
  S0_NM, S1_NM, S2_NM, S3_NM, S4_NM, S5_NM, S6_NM, S7_NM,
  // Not preserved across procedure calls
  T8_NM, T9_NM,
  // Reserved
  K0_NM, K1_NM, GP_NM, SP_NM, FP_NM, RA_NM)>;

// nanoMIPS: Stack pointer, global pointer and return address classes for
// instructions that are limited to a single register
def GPRNMSP : RegisterClass<"Mips", [i32], 32, (add SP_NM)>, Unallocatable;

def GPRNMGP : RegisterClass<"Mips", [i32], 32, (add GP_NM)>, Unallocatable;

def GPRNMRA : RegisterClass<"Mips", [i32], 32, (add RA_NM)>, Unallocatable;

// nanoMIPS "gpr2.reg1" & "gpr2.reg2" register encoding types.
def GPRNM2R1 : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  A0_NM, A1_NM, A2_NM, A3_NM)>;

def GPRNM2R2 : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  A1_NM, A2_NM, A3_NM, A4_NM)>;

def GPRNM1R1 : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  A0_NM, A1_NM)>;

def GPR64 : RegisterClass<"Mips", [i64], 64, (add
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

def CPU16RegsPlusSP : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3,
  // Callee save
  S0, S1,
  SP)>;

def CPURAReg : RegisterClass<"Mips", [i32], 32, (add RA)>, Unallocatable;

def CPUSPReg : RegisterClass<"Mips", [i32], 32, (add SP)>, Unallocatable;

// 64bit fp:
// * FGR64  - 32 64-bit registers
// * AFGR64 - 16 32-bit even registers (32-bit FP Mode)
//
// 32bit fp:
// * FGR32 - 16 32-bit even registers
// * FGR32 - 32 32-bit registers (single float only mode)
def FGR32 : RegisterClass<"Mips", [f32], 32, (sequence "F%u", 0, 31)> {
  // Do not allocate odd registers when given -mattr=+nooddspreg.
  let AltOrders = [(decimate FGR32, 2)];
  let AltOrderSelect = [{
    const auto & S = MF.getSubtarget<MipsSubtarget>();
    return S.isABI_O32() && !S.useOddSPReg();
  }];
}

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

def FGR64 : RegisterClass<"Mips", [f64], 64, (sequence "D%u_64", 0, 31)> {
  // Do not allocate odd registers when given -mattr=+nooddspreg.
  let AltOrders = [(decimate FGR64, 2)];
  let AltOrderSelect = [{
    const auto & S = MF.getSubtarget<MipsSubtarget>();
    return S.isABI_O32() && !S.useOddSPReg();
  }];
}

// FP control registers.
def CCR : RegisterClass<"Mips", [i32], 32, (sequence "FCR%u", 0, 31)>,
          Unallocatable;

// FP condition code registers.
def FCC : RegisterClass<"Mips", [i32], 32, (sequence "FCC%u", 0, 7)>,
          Unallocatable;

// MIPS32r6/MIPS64r6 store FPU condition codes in normal FGR registers.
// This class allows us to represent this in codegen patterns.
def FGR32CC : RegisterClass<"Mips", [i32], 32, (sequence "F%u", 0, 31)>;
def FGR64CC : RegisterClass<"Mips", [i32, f32, f64], 64, (sequence "D%u_64", 0, 31)>;

def MSA128B: RegisterClass<"Mips", [v16i8], 128,
                           (sequence "W%u", 0, 31)>;
def MSA128H: RegisterClass<"Mips", [v8i16, v8f16], 128,
                           (sequence "W%u", 0, 31)>;
def MSA128W: RegisterClass<"Mips", [v4i32, v4f32], 128,
                           (sequence "W%u", 0, 31)>;
def MSA128D: RegisterClass<"Mips", [v2i64, v2f64], 128,
                           (sequence "W%u", 0, 31)>;
def MSA128WEvens: RegisterClass<"Mips", [v4i32, v4f32], 128,
                                (decimate (sequence "W%u", 0, 31), 2)>;

def MSACtrl: RegisterClass<"Mips", [i32], 32, (add
  MSAIR, MSACSR, MSAAccess, MSASave, MSAModify, MSARequest, MSAMap, MSAUnmap,
  (sequence "MSA%u", 8, 31))>, Unallocatable;

// Hi/Lo Registers
def LO32 : RegisterClass<"Mips", [i32], 32, (add LO0)>;
def HI32 : RegisterClass<"Mips", [i32], 32, (add HI0)>;
def LO32DSP : RegisterClass<"Mips", [i32], 32, (sequence "LO%u", 0, 3)>;
def HI32DSP : RegisterClass<"Mips", [i32], 32, (sequence "HI%u", 0, 3)>;
def LO64 : RegisterClass<"Mips", [i64], 64, (add LO0_64)>;
def HI64 : RegisterClass<"Mips", [i64], 64, (add HI0_64)>;

// Hardware registers
def HWRegs : RegisterClass<"Mips", [i32], 32, (sequence "HWR%u", 0, 31)>,
             Unallocatable;

// Accumulator Registers
def ACC64 : RegisterClass<"Mips", [untyped], 64, (add AC0)> {
  let Size = 64;
}

def ACC128 : RegisterClass<"Mips", [untyped], 128, (add AC0_64)> {
  let Size = 128;
}

def ACC64DSP : RegisterClass<"Mips", [untyped], 64, (sequence "AC%u", 0, 3)> {
  let Size = 64;
}

def DSPCC : RegisterClass<"Mips", [v4i8, v2i16], 32, (add DSPCCond)>;

// Coprocessor 0 registers.
def COP0 : RegisterClass<"Mips", [i32], 32, (sequence "COP0%u", 0, 31)>,
           Unallocatable;

// Coprocessor 2 registers.
def COP2 : RegisterClass<"Mips", [i32], 32, (sequence "COP2%u", 0, 31)>,
           Unallocatable;

// Coprocessor 3 registers.
def COP3 : RegisterClass<"Mips", [i32], 32, (sequence "COP3%u", 0, 31)>,
           Unallocatable;

// Stack pointer and global pointer classes for instructions that are limited
// to a single register such as lwgp/lwsp in microMIPS.
def SP32 : RegisterClass<"Mips", [i32], 32, (add SP)>, Unallocatable;
def SP64 : RegisterClass<"Mips", [i64], 64, (add SP_64)>, Unallocatable;
def GP32 : RegisterClass<"Mips", [i32], 32, (add GP)>, Unallocatable;
def GP64 : RegisterClass<"Mips", [i64], 64, (add GP_64)>, Unallocatable;

// Octeon multiplier and product registers
def OCTEON_MPL : RegisterClass<"Mips", [i64], 64, (add MPL0, MPL1, MPL2)>,
                 Unallocatable;
def OCTEON_P : RegisterClass<"Mips", [i64], 64, (add P0, P1, P2)>,
               Unallocatable;

// Register Operands.

class MipsAsmRegOperand : AsmOperandClass {
  let ParserMethod = "parseAnyRegister";
}

def GPR64AsmOperand : MipsAsmRegOperand {
  let Name = "GPR64AsmReg";
  let PredicateMethod = "isGPRAsmReg";
}

def GPR32ZeroAsmOperand : MipsAsmRegOperand {
  let Name = "GPR32ZeroAsmReg";
  let PredicateMethod = "isGPRZeroAsmReg";
}

def GPR32NonZeroAsmOperand : MipsAsmRegOperand {
  let Name = "GPR32NonZeroAsmReg";
  let PredicateMethod = "isGPRNonZeroAsmReg";
}

def GPR32AsmOperand : MipsAsmRegOperand {
  let Name = "GPR32AsmReg";
  let PredicateMethod = "isGPRAsmReg";
}

def GPRMM16AsmOperand : MipsAsmRegOperand {
  let Name = "GPRMM16AsmReg";
  let PredicateMethod = "isMM16AsmReg";
}

def GPRMM16AsmOperandZero : MipsAsmRegOperand {
  let Name = "GPRMM16AsmRegZero";
  let PredicateMethod = "isMM16AsmRegZero";
}

def GPRMM16AsmOperandMoveP : MipsAsmRegOperand {
  let Name = "GPRMM16AsmRegMoveP";
  let PredicateMethod = "isMM16AsmRegMoveP";
}

def GPRMM16AsmOperandMovePPairFirst : MipsAsmRegOperand {
  let Name = "GPRMM16AsmRegMovePPairFirst";
  let PredicateMethod = "isMM16AsmRegMovePPairFirst";
}

def GPRMM16AsmOperandMovePPairSecond : MipsAsmRegOperand {
  let Name = "GPRMM16AsmRegMovePPairSecond";
  let PredicateMethod = "isMM16AsmRegMovePPairSecond";
}

def GPRNM32AsmOperand : MipsAsmRegOperand {
  let Name = "GPRNM32AsmReg";
  let PredicateMethod = "isGPRAsmReg";
}

class GPRNMAsmOperandClass<string Cname, string RegClass,
                           list<AsmOperandClass> Supers = []>
  : MipsAsmRegOperand {
  let Name = "GPRNM" # Cname;
  let PredicateMethod = "isGPRNMAsmReg<Mips::" # RegClass # ">";
  let SuperClasses = Supers;
}

class COP0SelAsmOperandClass<string Cname, list<AsmOperandClass> Supers = []>
  : MipsAsmRegOperand {
  let Name = Cname;
  let PredicateMethod = "isCOP0SelAsmReg";
  let SuperClasses = Supers;
}

def ACC64DSPAsmOperand : MipsAsmRegOperand {
  let Name = "ACC64DSPAsmReg";
  let PredicateMethod = "isACCAsmReg";
}

def HI32DSPAsmOperand : MipsAsmRegOperand {
  let Name = "HI32DSPAsmReg";
  let PredicateMethod = "isACCAsmReg";
}

def LO32DSPAsmOperand : MipsAsmRegOperand {
  let Name = "LO32DSPAsmReg";
  let PredicateMethod = "isACCAsmReg";
}

def CCRAsmOperand : MipsAsmRegOperand {
  let Name = "CCRAsmReg";
}

def AFGR64AsmOperand : MipsAsmRegOperand {
  let Name = "AFGR64AsmReg";
  let PredicateMethod = "isFGRAsmReg";
}

def StrictlyAFGR64AsmOperand : MipsAsmRegOperand {
  let Name = "StrictlyAFGR64AsmReg";
  let PredicateMethod = "isStrictlyFGRAsmReg";
}

def FGR64AsmOperand : MipsAsmRegOperand {
  let Name = "FGR64AsmReg";
  let PredicateMethod = "isFGRAsmReg";
}

def StrictlyFGR64AsmOperand : MipsAsmRegOperand {
  let Name = "StrictlyFGR64AsmReg";
  let PredicateMethod = "isStrictlyFGRAsmReg";
}

def FGR32AsmOperand : MipsAsmRegOperand {
  let Name = "FGR32AsmReg";
  let PredicateMethod = "isFGRAsmReg";
}

def StrictlyFGR32AsmOperand : MipsAsmRegOperand {
  let Name = "StrictlyFGR32AsmReg";
  let PredicateMethod = "isStrictlyFGRAsmReg";
}

def FCCRegsAsmOperand : MipsAsmRegOperand {
  let Name = "FCCAsmReg";
}

def MSA128AsmOperand : MipsAsmRegOperand {
  let Name = "MSA128AsmReg";
}

def MSACtrlAsmOperand : MipsAsmRegOperand {
  let Name = "MSACtrlAsmReg";
}

def GPR32ZeroOpnd : RegisterOperand<GPR32ZERO> {
  let ParserMatchClass = GPR32ZeroAsmOperand;
}

def GPR32NonZeroOpnd : RegisterOperand<GPR32NONZERO> {
  let ParserMatchClass = GPR32NonZeroAsmOperand;
}

def GPR32Opnd : RegisterOperand<GPR32> {
  let ParserMatchClass = GPR32AsmOperand;
}

def GPRMM16Opnd : RegisterOperand<GPRMM16> {
  let ParserMatchClass = GPRMM16AsmOperand;
}

def GPRMM16OpndZero : RegisterOperand<GPRMM16Zero> {
  let ParserMatchClass = GPRMM16AsmOperandZero;
}

def GPRMM16OpndMoveP : RegisterOperand<GPRMM16MoveP> {
  let ParserMatchClass = GPRMM16AsmOperandMoveP;
  let EncoderMethod = "getMovePRegSingleOpValue";
}

def GPRMM16OpndMovePPairFirst : RegisterOperand<GPRMM16MovePPairFirst> {
  let ParserMatchClass = GPRMM16AsmOperandMovePPairFirst;
}

def GPRMM16OpndMovePPairSecond : RegisterOperand<GPRMM16MovePPairSecond> {
  let ParserMatchClass = GPRMM16AsmOperandMovePPairSecond;
}

// In the following nanoMIPS operand definitions, the SuperClasses list passed
// to GPRNMAsmOperandClass indirectly controls mnemonic-matching priority by
// dictating the order in which instructions appear in the MatchTable.
def GPRNM48Opnd : RegisterOperand<GPRNM32> {
  let ParserMatchClass = GPRNMAsmOperandClass<"48Reg5", "GPRNM32RegClassID",
                                              []>;
}

def GPRNM32NZOpnd : RegisterOperand<GPRNM32NZ> {
  let ParserMatchClass = GPRNMAsmOperandClass<"32Reg5NZ", "GPRNM32NZRegClassID",
                                              [GPRNM48Opnd.ParserMatchClass]>;
}

def GPRNM32Opnd : RegisterOperand<GPRNM32> {
  let ParserMatchClass = GPRNMAsmOperandClass<"32Reg5", "GPRNM32RegClassID",
                                              [GPRNM32NZOpnd.ParserMatchClass]>;
}

def COP0SelOpnd : RegisterOperand<COP0Sel> {
  let ParserMatchClass =
    COP0SelAsmOperandClass<"COP0SelReg", [GPRNM32Opnd.ParserMatchClass]>;
}

def GPRNM4Opnd : RegisterOperand<GPRNM4> {
  let ParserMatchClass = GPRNMAsmOperandClass<"16Reg4x4", "GPRNM4RegClassID",
                                              [GPRNM32Opnd.ParserMatchClass]>;
}

def GPRNM4ZOpnd : RegisterOperand<GPRNM4Z> {
  let ParserMatchClass = GPRNMAsmOperandClass<"16Reg4x4Z", "GPRNM4ZRegClassID",
                                              [GPRNM32Opnd.ParserMatchClass]>;
}

def GPRNM16R5Opnd : RegisterOperand<GPRNM32> {
  let ParserMatchClass = GPRNMAsmOperandClass<"16Reg5", "GPRNM32RegClassID",
                                              [GPRNM4Opnd.ParserMatchClass]>;
}

def GPRNM16R5NZOpnd : RegisterOperand<GPRNM32> {
  let ParserMatchClass = GPRNMAsmOperandClass<"16Reg5NZ", "GPRNM32NZRegClassID",
                                              [GPRNM4Opnd.ParserMatchClass]>;
}

def GPRNM16R3Opnd : RegisterOperand<GPRNM3> {
  let ParserMatchClass = GPRNMAsmOperandClass<"16Reg3", "GPRNM3RegClassID",
                                              [GPRNM16R5Opnd.ParserMatchClass]>;
}

def GPRNM16R3ZOpnd : RegisterOperand<GPRNM3Z> {
  let ParserMatchClass = GPRNMAsmOperandClass<"16Reg3Z", "GPRNM3ZRegClassID",
                                              [GPRNM16R5Opnd.ParserMatchClass]>;
}

def GPRNMSPOpnd : RegisterOperand<GPRNMSP> {
  let ParserMatchClass = GPRNMAsmOperandClass<"RegSP", "GPRNMSPRegClassID",
                                              [GPRNM16R3Opnd.ParserMatchClass]>;
}

def GPRNMGPOpnd : RegisterOperand<GPRNMGP> {
  let ParserMatchClass = GPRNMAsmOperandClass<"RegGP", "GPRNMGPRegClassID",
                                              [GPRNM16R3Opnd.ParserMatchClass]>;
}

def GPRNMRAOpnd : RegisterOperand<GPRNMRA> {
  let ParserMatchClass = GPRNMAsmOperandClass<"RegRA", "GPRNMRARegClassID",
                                             [GPRNM16R3Opnd.ParserMatchClass]>;
}

def GPRNM2R1Opnd : RegisterOperand<GPRNM2R1> {
  let ParserMatchClass = GPRNMAsmOperandClass<"2Reg1", "GPRNM2R1RegClassID",
                                              [GPRNM4Opnd.ParserMatchClass]>;
}

def GPRNM2R2Opnd : RegisterOperand<GPRNM2R2> {
  let ParserMatchClass = GPRNMAsmOperandClass<"2Reg2", "GPRNM2R2RegClassID",
                                              [GPRNM4Opnd.ParserMatchClass]>;
}

def GPRNM1R1Opnd : RegisterOperand<GPRNM1R1> {
  let ParserMatchClass = GPRNMAsmOperandClass<"1Reg1", "GPRNM1R1RegClassID",
                                              [GPRNM4Opnd.ParserMatchClass]>;
}

def GPR64Opnd : RegisterOperand<GPR64> {
  let ParserMatchClass = GPR64AsmOperand;
}

def DSPROpnd : RegisterOperand<DSPR> {
  let ParserMatchClass = GPR32AsmOperand;
}

def CCROpnd : RegisterOperand<CCR> {
  let ParserMatchClass = CCRAsmOperand;
}

def HWRegsAsmOperand : MipsAsmRegOperand {
  let Name = "HWRegsAsmReg";
}

def COP0AsmOperand : MipsAsmRegOperand {
  let Name = "COP0AsmReg";
}

def COP2AsmOperand : MipsAsmRegOperand {
  let Name = "COP2AsmReg";
}

def COP3AsmOperand : MipsAsmRegOperand {
  let Name = "COP3AsmReg";
}

def HWRegsOpnd : RegisterOperand<HWRegs> {
  let ParserMatchClass = HWRegsAsmOperand;
}

def AFGR64Opnd : RegisterOperand<AFGR64> {
  let ParserMatchClass = AFGR64AsmOperand;
}

def StrictlyAFGR64Opnd : RegisterOperand<AFGR64> {
  let ParserMatchClass = StrictlyAFGR64AsmOperand;
}

def FGR64Opnd : RegisterOperand<FGR64> {
  let ParserMatchClass = FGR64AsmOperand;
}

def StrictlyFGR64Opnd : RegisterOperand<FGR64> {
  let ParserMatchClass = StrictlyFGR64AsmOperand;
}

def FGR32Opnd : RegisterOperand<FGR32> {
  let ParserMatchClass = FGR32AsmOperand;
}

def StrictlyFGR32Opnd : RegisterOperand<FGR32> {
  let ParserMatchClass = StrictlyFGR32AsmOperand;
}

def FGR64CCOpnd : RegisterOperand<FGR64CC> {
  // The assembler doesn't use register classes so we can re-use
  // FGR64AsmOperand.
  let ParserMatchClass = FGR64AsmOperand;
}

def FGR32CCOpnd : RegisterOperand<FGR32CC> {
  // The assembler doesn't use register classes so we can re-use
  // FGR32AsmOperand.
  let ParserMatchClass = FGR32AsmOperand;
}

def FCCRegsOpnd : RegisterOperand<FCC> {
  let ParserMatchClass = FCCRegsAsmOperand;
}

def LO32DSPOpnd : RegisterOperand<LO32DSP> {
  let ParserMatchClass = LO32DSPAsmOperand;
}

def HI32DSPOpnd : RegisterOperand<HI32DSP> {
  let ParserMatchClass = HI32DSPAsmOperand;
}

def ACC64DSPOpnd : RegisterOperand<ACC64DSP> {
  let ParserMatchClass = ACC64DSPAsmOperand;
}

def COP0Opnd : RegisterOperand<COP0> {
  let ParserMatchClass = COP0AsmOperand;
}

def COP2Opnd : RegisterOperand<COP2> {
  let ParserMatchClass = COP2AsmOperand;
}

def COP3Opnd : RegisterOperand<COP3> {
  let ParserMatchClass = COP3AsmOperand;
}

def MSA128BOpnd : RegisterOperand<MSA128B> {
  let ParserMatchClass = MSA128AsmOperand;
}

def MSA128HOpnd : RegisterOperand<MSA128H> {
  let ParserMatchClass = MSA128AsmOperand;
}

def MSA128WOpnd : RegisterOperand<MSA128W> {
  let ParserMatchClass = MSA128AsmOperand;
}

def MSA128DOpnd : RegisterOperand<MSA128D> {
  let ParserMatchClass = MSA128AsmOperand;
}

def MSA128CROpnd : RegisterOperand<MSACtrl> {
  let ParserMatchClass = MSACtrlAsmOperand;
}

def mips_ptr_rc : RegClassByHwMode<
  [MIPS32, MIPS64],
  [GPR32, GPR64]>;

def ptr_gpr16mm_rc : RegClassByHwMode<
  [MIPS32, MIPS64],
  [GPRMM16, GPRMM16]>; // FIXME: Why even use this if it's just a constant

def ptr_sp_rc : RegClassByHwMode<
  [MIPS32, MIPS64],
  [SP32, SP64]>;

def ptr_gp_rc : RegClassByHwMode<
  [MIPS32, MIPS64],
  [GP32, GP64]>;
