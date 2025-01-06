//=- AArch64RegisterInfo.td - Describe the AArch64 Registers -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//


class AArch64Reg<bits<16> enc, string n, list<Register> subregs = [],
               list<string> altNames = []>
        : Register<n, altNames> {
  let HWEncoding = enc;
  let Namespace = "AArch64";
  let SubRegs = subregs;
}

let Namespace = "AArch64" in {
  // SubRegIndexes for GPR registers
  def sub_32   : SubRegIndex<32>;
  def sube64   : SubRegIndex<64>;
  def subo64   : SubRegIndex<64>;
  def sube32   : SubRegIndex<32>;
  def subo32   : SubRegIndex<32>;

  // SubRegIndexes for FPR/Vector registers
  def bsub    : SubRegIndex<8, 0>;
  def hsub    : SubRegIndex<16, 0>;
  def ssub    : SubRegIndex<32, 0>;
  def dsub    : SubRegIndex<64, 0>;
  def zsub    : SubRegIndex<128, 0>;

  // The _hi SubRegIndexes describe the high bits of a register which are not
  // separately addressable. They need to be described so that partially
  // overlapping registers end up with a different lane mask. This is required
  // to enable subreg liveness tracking.
  //
  // For example: 8-bit B0 is a sub-register of 16-bit H0.
  // * B0 is described with 'bsub'.
  // * H0 is described with 'bsub + bsub_hi' == 'hsub'.
  def bsub_hi   : SubRegIndex<8, 8>;
  def hsub_hi   : SubRegIndex<16, 16>;
  def ssub_hi   : SubRegIndex<32, 32>;
  def dsub_hi   : SubRegIndex<64, 64>;
  def zsub_hi   : SubRegIndex<-1, 128>;
  // sub_32_hi describes the top 32 bits in e.g. X0
  def sub_32_hi : SubRegIndex<32, 32>;
  // Note: Code depends on these having consecutive numbers
  def zsub0 : SubRegIndex<-1>;
  def zsub1 : SubRegIndex<-1>;
  def zsub2 : SubRegIndex<-1>;
  def zsub3 : SubRegIndex<-1>;
  // Note: Code depends on these having consecutive numbers
  def qsub0 : SubRegIndex<128>;
  def qsub1 : ComposedSubRegIndex<zsub1, zsub>;
  def qsub2 : ComposedSubRegIndex<zsub2, zsub>;
  def qsub3 : ComposedSubRegIndex<zsub3, zsub>;
  // Note: Code depends on these having consecutive numbers
  def dsub0 : SubRegIndex<64>;
  def dsub1 : ComposedSubRegIndex<qsub1, dsub>;
  def dsub2 : ComposedSubRegIndex<qsub2, dsub>;
  def dsub3 : ComposedSubRegIndex<qsub3, dsub>;

  // SubRegIndexes for SME Matrix tiles
  def zasubb  : SubRegIndex<2048>; // (16 x 16)/1 bytes  = 2048 bits
  def zasubh0 : SubRegIndex<1024>; // (16 x 16)/2 bytes  = 1024 bits
  def zasubh1 : SubRegIndex<1024>; // (16 x 16)/2 bytes  = 1024 bits
  def zasubs0 : SubRegIndex<512>;  // (16 x 16)/4 bytes  = 512 bits
  def zasubs1 : SubRegIndex<512>;  // (16 x 16)/4 bytes  = 512 bits
  def zasubd0 : SubRegIndex<256>;  // (16 x 16)/8 bytes  = 256 bits
  def zasubd1 : SubRegIndex<256>;  // (16 x 16)/8 bytes  = 256 bits
  def zasubq0 : SubRegIndex<128>;  // (16 x 16)/16 bytes = 128 bits
  def zasubq1 : SubRegIndex<128>;  // (16 x 16)/16 bytes = 128 bits

  // SubRegIndexes for SVE Predicates
  def psub  : SubRegIndex<-1>;
  // Note: Code depends on these having consecutive numbers
  def psub0 : SubRegIndex<-1>;
  def psub1 : SubRegIndex<-1>;
}

let Namespace = "AArch64" in {
  def vreg : RegAltNameIndex;
  def vlist1 : RegAltNameIndex;
}

//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//

foreach i = 0-30 in {
  // Define W0_HI, W1_HI, .. W30_HI
  def W#i#_HI : AArch64Reg<-1,  "w"#i#"_hi"> { let isArtificial = 1; }
}
def WSP_HI : AArch64Reg<-1,  "wsp_hi"> { let isArtificial = 1; }
def WZR_HI : AArch64Reg<-1,  "wzr_hi"> { let isArtificial = 1; }

def W0    : AArch64Reg<0,   "w0" >, DwarfRegNum<[0]>;
def W1    : AArch64Reg<1,   "w1" >, DwarfRegNum<[1]>;
def W2    : AArch64Reg<2,   "w2" >, DwarfRegNum<[2]>;
def W3    : AArch64Reg<3,   "w3" >, DwarfRegNum<[3]>;
def W4    : AArch64Reg<4,   "w4" >, DwarfRegNum<[4]>;
def W5    : AArch64Reg<5,   "w5" >, DwarfRegNum<[5]>;
def W6    : AArch64Reg<6,   "w6" >, DwarfRegNum<[6]>;
def W7    : AArch64Reg<7,   "w7" >, DwarfRegNum<[7]>;
def W8    : AArch64Reg<8,   "w8" >, DwarfRegNum<[8]>;
def W9    : AArch64Reg<9,   "w9" >, DwarfRegNum<[9]>;
def W10   : AArch64Reg<10, "w10">, DwarfRegNum<[10]>;
def W11   : AArch64Reg<11, "w11">, DwarfRegNum<[11]>;
def W12   : AArch64Reg<12, "w12">, DwarfRegNum<[12]>;
def W13   : AArch64Reg<13, "w13">, DwarfRegNum<[13]>;
def W14   : AArch64Reg<14, "w14">, DwarfRegNum<[14]>;
def W15   : AArch64Reg<15, "w15">, DwarfRegNum<[15]>;
def W16   : AArch64Reg<16, "w16">, DwarfRegNum<[16]>;
def W17   : AArch64Reg<17, "w17">, DwarfRegNum<[17]>;
def W18   : AArch64Reg<18, "w18">, DwarfRegNum<[18]>;
def W19   : AArch64Reg<19, "w19">, DwarfRegNum<[19]>;
def W20   : AArch64Reg<20, "w20">, DwarfRegNum<[20]>;
def W21   : AArch64Reg<21, "w21">, DwarfRegNum<[21]>;
def W22   : AArch64Reg<22, "w22">, DwarfRegNum<[22]>;
def W23   : AArch64Reg<23, "w23">, DwarfRegNum<[23]>;
def W24   : AArch64Reg<24, "w24">, DwarfRegNum<[24]>;
def W25   : AArch64Reg<25, "w25">, DwarfRegNum<[25]>;
def W26   : AArch64Reg<26, "w26">, DwarfRegNum<[26]>;
def W27   : AArch64Reg<27, "w27">, DwarfRegNum<[27]>;
def W28   : AArch64Reg<28, "w28">, DwarfRegNum<[28]>;
def W29   : AArch64Reg<29, "w29">, DwarfRegNum<[29]>;
def W30   : AArch64Reg<30, "w30">, DwarfRegNum<[30]>;
def WSP   : AArch64Reg<31, "wsp">, DwarfRegNum<[31]>;
def WZR   : AArch64Reg<31, "wzr">, DwarfRegAlias<WSP> { let isConstant = true; }

let SubRegIndices = [sub_32, sub_32_hi], CoveredBySubRegs = 1 in {
def X0    : AArch64Reg<0,   "x0",  [W0,  W0_HI]>, DwarfRegAlias<W0>;
def X1    : AArch64Reg<1,   "x1",  [W1,  W1_HI]>, DwarfRegAlias<W1>;
def X2    : AArch64Reg<2,   "x2",  [W2,  W2_HI]>, DwarfRegAlias<W2>;
def X3    : AArch64Reg<3,   "x3",  [W3,  W3_HI]>, DwarfRegAlias<W3>;
def X4    : AArch64Reg<4,   "x4",  [W4,  W4_HI]>, DwarfRegAlias<W4>;
def X5    : AArch64Reg<5,   "x5",  [W5,  W5_HI]>, DwarfRegAlias<W5>;
def X6    : AArch64Reg<6,   "x6",  [W6,  W6_HI]>, DwarfRegAlias<W6>;
def X7    : AArch64Reg<7,   "x7",  [W7,  W7_HI]>, DwarfRegAlias<W7>;
def X8    : AArch64Reg<8,   "x8",  [W8,  W8_HI]>, DwarfRegAlias<W8>;
def X9    : AArch64Reg<9,   "x9",  [W9,  W9_HI]>, DwarfRegAlias<W9>;
def X10   : AArch64Reg<10, "x10", [W10, W10_HI]>, DwarfRegAlias<W10>;
def X11   : AArch64Reg<11, "x11", [W11, W11_HI]>, DwarfRegAlias<W11>;
def X12   : AArch64Reg<12, "x12", [W12, W12_HI]>, DwarfRegAlias<W12>;
def X13   : AArch64Reg<13, "x13", [W13, W13_HI]>, DwarfRegAlias<W13>;
def X14   : AArch64Reg<14, "x14", [W14, W14_HI]>, DwarfRegAlias<W14>;
def X15   : AArch64Reg<15, "x15", [W15, W15_HI]>, DwarfRegAlias<W15>;
def X16   : AArch64Reg<16, "x16", [W16, W16_HI]>, DwarfRegAlias<W16>;
def X17   : AArch64Reg<17, "x17", [W17, W17_HI]>, DwarfRegAlias<W17>;
def X18   : AArch64Reg<18, "x18", [W18, W18_HI]>, DwarfRegAlias<W18>;
def X19   : AArch64Reg<19, "x19", [W19, W19_HI]>, DwarfRegAlias<W19>;
def X20   : AArch64Reg<20, "x20", [W20, W20_HI]>, DwarfRegAlias<W20>;
def X21   : AArch64Reg<21, "x21", [W21, W21_HI]>, DwarfRegAlias<W21>;
def X22   : AArch64Reg<22, "x22", [W22, W22_HI]>, DwarfRegAlias<W22>;
def X23   : AArch64Reg<23, "x23", [W23, W23_HI]>, DwarfRegAlias<W23>;
def X24   : AArch64Reg<24, "x24", [W24, W24_HI]>, DwarfRegAlias<W24>;
def X25   : AArch64Reg<25, "x25", [W25, W25_HI]>, DwarfRegAlias<W25>;
def X26   : AArch64Reg<26, "x26", [W26, W26_HI]>, DwarfRegAlias<W26>;
def X27   : AArch64Reg<27, "x27", [W27, W27_HI]>, DwarfRegAlias<W27>;
def X28   : AArch64Reg<28, "x28", [W28, W28_HI]>, DwarfRegAlias<W28>;
def FP    : AArch64Reg<29, "x29", [W29, W29_HI]>, DwarfRegAlias<W29>;
def LR    : AArch64Reg<30, "x30", [W30, W30_HI]>, DwarfRegAlias<W30>;
def SP    : AArch64Reg<31, "sp",  [WSP, WSP_HI]>, DwarfRegAlias<WSP>;
def XZR   : AArch64Reg<31, "xzr", [WZR, WZR_HI]>, DwarfRegAlias<WSP> { let isConstant = true; }
}

// Condition code register.
def NZCV  : AArch64Reg<0, "nzcv">;

// First fault status register
def FFR : AArch64Reg<0, "ffr">, DwarfRegNum<[47]>;

// Purely virtual Vector Granule (VG) Dwarf register
def VG : AArch64Reg<0, "vg">, DwarfRegNum<[46]>;

// Floating-point control register
def FPCR : AArch64Reg<0, "fpcr">;

// Floating-point Mode Register
def FPMR : AArch64Reg<0, "fpmr">;

// Floating-point status register.
def FPSR : AArch64Reg<0, "fpsr">;

// GPR register classes with the intersections of GPR32/GPR32sp and
// GPR64/GPR64sp for use by the coalescer.
def GPR32common : RegisterClass<"AArch64", [i32], 32, (sequence "W%u", 0, 30)> {
  let AltOrders = [(rotl GPR32common, 8)];
  let AltOrderSelect = [{ return 1; }];
}
def GPR64common : RegisterClass<"AArch64", [i64], 64,
                                (add (sequence "X%u", 0, 28), FP, LR)> {
  let AltOrders = [(rotl GPR64common, 8)];
  let AltOrderSelect = [{ return 1; }];
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::GPR64commonRegClassID, 0, 31>";
}
// GPR register classes which exclude SP/WSP.
def GPR32 : RegisterClass<"AArch64", [i32], 32, (add GPR32common, WZR)> {
  let AltOrders = [(rotl GPR32, 8)];
  let AltOrderSelect = [{ return 1; }];
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::GPR32RegClassID, 0, 32>";
}
def GPR64 : RegisterClass<"AArch64", [i64], 64, (add GPR64common, XZR)> {
  let AltOrders = [(rotl GPR64, 8)];
  let AltOrderSelect = [{ return 1; }];
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::GPR64RegClassID, 0, 32>";
}

// GPR register classes which include SP/WSP.
def GPR32sp : RegisterClass<"AArch64", [i32], 32, (add GPR32common, WSP)> {
  let AltOrders = [(rotl GPR32sp, 8)];
  let AltOrderSelect = [{ return 1; }];
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::GPR32spRegClassID, 0, 32>";
}
def GPR64sp : RegisterClass<"AArch64", [i64], 64, (add GPR64common, SP)> {
  let AltOrders = [(rotl GPR64sp, 8)];
  let AltOrderSelect = [{ return 1; }];
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::GPR64spRegClassID,0, 32>";
}

def GPR32sponly : RegisterClass<"AArch64", [i32], 32, (add WSP)>;
def GPR64sponly : RegisterClass<"AArch64", [i64], 64, (add SP)>;

def GPR64spPlus0Operand : AsmOperandClass {
  let Name = "GPR64sp0";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isGPR64<AArch64::GPR64spRegClassID>";
  let ParserMethod = "tryParseGPR64sp0Operand";
}

def GPR64sp0 : RegisterOperand<GPR64sp> {
  let ParserMatchClass = GPR64spPlus0Operand;
}

// GPR32/GPR64 but with zero-register substitution enabled.
// TODO: Roll this out to GPR32/GPR64/GPR32all/GPR64all.
def GPR32z : RegisterOperand<GPR32> {
  let GIZeroRegister = WZR;
}
def GPR64z : RegisterOperand<GPR64> {
  let GIZeroRegister = XZR;
}

// GPR argument registers.
def GPR32arg : RegisterClass<"AArch64", [i32], 32, (sequence "W%u", 0, 7)>;
def GPR64arg : RegisterClass<"AArch64", [i64], 64, (sequence "X%u", 0, 7)>;

// GPR register classes which include WZR/XZR AND SP/WSP. This is not a
// constraint used by any instructions, it is used as a common super-class.
def GPR32all : RegisterClass<"AArch64", [i32], 32, (add GPR32common, WZR, WSP)>;
def GPR64all : RegisterClass<"AArch64", [i64], 64, (add GPR64common, XZR, SP)>;

// For tail calls, we can't use callee-saved registers, as they are restored
// to the saved value before the tail call, which would clobber a call address.
// This is for indirect tail calls to store the address of the destination.
def tcGPR64 : RegisterClass<"AArch64", [i64], 64, (sub GPR64common, X19, X20, X21,
                                                     X22, X23, X24, X25, X26,
                                                     X27, X28, FP, LR)>;

// Restricted sets of tail call registers, for use when branch target
// enforcement or PAuthLR are enabled.
// For BTI, x16 and x17 are the only registers which can be used to indirectly
// branch (not call) to the "BTI c" instruction at the start of a BTI-protected
// function.
// For PAuthLR, x16 must be used in the function epilogue for other purposes,
// so cannot hold the function pointer.
def tcGPRx17 : RegisterClass<"AArch64", [i64], 64, (add X17)>;
def tcGPRx16x17 : RegisterClass<"AArch64", [i64], 64, (add X16, X17)>;
def tcGPRnotx16 : RegisterClass<"AArch64", [i64], 64, (sub tcGPR64, X16)>;
// LR checking code expects either x16 or x17 to be available as a scratch
// register - for that reason restrict one of two register operands of
// AUTH_TCRETURN* pseudos.
def tcGPRnotx16x17 : RegisterClass<"AArch64", [i64], 64, (sub tcGPR64, X16, X17)>;

// Register set that excludes registers that are reserved for procedure calls.
// This is used for pseudo-instructions that are actually implemented using a
// procedure call.
def GPR64noip : RegisterClass<"AArch64", [i64], 64, (sub GPR64, X16, X17, LR)> {
  let AltOrders = [(rotl GPR64noip, 8)];
  let AltOrderSelect = [{ return 1; }];
}

// GPR register classes for post increment amount of vector load/store that
// has alternate printing when Rm=31 and prints a constant immediate value
// equal to the total number of bytes transferred.

// FIXME: TableGen *should* be able to do these itself now. There appears to be
// a bug in counting how many operands a Post-indexed MCInst should have which
// means the aliases don't trigger.
def GPR64pi1  : RegisterOperand<GPR64, "printPostIncOperand<1>">;
def GPR64pi2  : RegisterOperand<GPR64, "printPostIncOperand<2>">;
def GPR64pi3  : RegisterOperand<GPR64, "printPostIncOperand<3>">;
def GPR64pi4  : RegisterOperand<GPR64, "printPostIncOperand<4>">;
def GPR64pi6  : RegisterOperand<GPR64, "printPostIncOperand<6>">;
def GPR64pi8  : RegisterOperand<GPR64, "printPostIncOperand<8>">;
def GPR64pi12 : RegisterOperand<GPR64, "printPostIncOperand<12>">;
def GPR64pi16 : RegisterOperand<GPR64, "printPostIncOperand<16>">;
def GPR64pi24 : RegisterOperand<GPR64, "printPostIncOperand<24>">;
def GPR64pi32 : RegisterOperand<GPR64, "printPostIncOperand<32>">;
def GPR64pi48 : RegisterOperand<GPR64, "printPostIncOperand<48>">;
def GPR64pi64 : RegisterOperand<GPR64, "printPostIncOperand<64>">;

// Condition code regclass.
def CCR : RegisterClass<"AArch64", [i32], 32, (add NZCV)> {
  let CopyCost = -1;  // Don't allow copying of status registers.

  // CCR is not allocatable.
  let isAllocatable = 0;
}

//===----------------------------------------------------------------------===//
// Floating Point Scalar Registers
//===----------------------------------------------------------------------===//

foreach i = 0-31 in {
  def B#i#_HI : AArch64Reg<-1,  "b"#i#"_hi"> { let isArtificial = 1; }
  def H#i#_HI : AArch64Reg<-1,  "h"#i#"_hi"> { let isArtificial = 1; }
  def S#i#_HI : AArch64Reg<-1,  "s"#i#"_hi"> { let isArtificial = 1; }
  def D#i#_HI : AArch64Reg<-1,  "d"#i#"_hi"> { let isArtificial = 1; }
  def Q#i#_HI : AArch64Reg<-1,  "q"#i#"_hi"> { let isArtificial = 1; }
}

def B0    : AArch64Reg<0,   "b0">, DwarfRegNum<[64]>;
def B1    : AArch64Reg<1,   "b1">, DwarfRegNum<[65]>;
def B2    : AArch64Reg<2,   "b2">, DwarfRegNum<[66]>;
def B3    : AArch64Reg<3,   "b3">, DwarfRegNum<[67]>;
def B4    : AArch64Reg<4,   "b4">, DwarfRegNum<[68]>;
def B5    : AArch64Reg<5,   "b5">, DwarfRegNum<[69]>;
def B6    : AArch64Reg<6,   "b6">, DwarfRegNum<[70]>;
def B7    : AArch64Reg<7,   "b7">, DwarfRegNum<[71]>;
def B8    : AArch64Reg<8,   "b8">, DwarfRegNum<[72]>;
def B9    : AArch64Reg<9,   "b9">, DwarfRegNum<[73]>;
def B10   : AArch64Reg<10, "b10">, DwarfRegNum<[74]>;
def B11   : AArch64Reg<11, "b11">, DwarfRegNum<[75]>;
def B12   : AArch64Reg<12, "b12">, DwarfRegNum<[76]>;
def B13   : AArch64Reg<13, "b13">, DwarfRegNum<[77]>;
def B14   : AArch64Reg<14, "b14">, DwarfRegNum<[78]>;
def B15   : AArch64Reg<15, "b15">, DwarfRegNum<[79]>;
def B16   : AArch64Reg<16, "b16">, DwarfRegNum<[80]>;
def B17   : AArch64Reg<17, "b17">, DwarfRegNum<[81]>;
def B18   : AArch64Reg<18, "b18">, DwarfRegNum<[82]>;
def B19   : AArch64Reg<19, "b19">, DwarfRegNum<[83]>;
def B20   : AArch64Reg<20, "b20">, DwarfRegNum<[84]>;
def B21   : AArch64Reg<21, "b21">, DwarfRegNum<[85]>;
def B22   : AArch64Reg<22, "b22">, DwarfRegNum<[86]>;
def B23   : AArch64Reg<23, "b23">, DwarfRegNum<[87]>;
def B24   : AArch64Reg<24, "b24">, DwarfRegNum<[88]>;
def B25   : AArch64Reg<25, "b25">, DwarfRegNum<[89]>;
def B26   : AArch64Reg<26, "b26">, DwarfRegNum<[90]>;
def B27   : AArch64Reg<27, "b27">, DwarfRegNum<[91]>;
def B28   : AArch64Reg<28, "b28">, DwarfRegNum<[92]>;
def B29   : AArch64Reg<29, "b29">, DwarfRegNum<[93]>;
def B30   : AArch64Reg<30, "b30">, DwarfRegNum<[94]>;
def B31   : AArch64Reg<31, "b31">, DwarfRegNum<[95]>;

let SubRegIndices = [bsub, bsub_hi] in {
def H0    : AArch64Reg<0,   "h0", [B0, B0_HI]>, DwarfRegAlias<B0>;
def H1    : AArch64Reg<1,   "h1", [B1, B1_HI]>, DwarfRegAlias<B1>;
def H2    : AArch64Reg<2,   "h2", [B2, B2_HI]>, DwarfRegAlias<B2>;
def H3    : AArch64Reg<3,   "h3", [B3, B3_HI]>, DwarfRegAlias<B3>;
def H4    : AArch64Reg<4,   "h4", [B4, B4_HI]>, DwarfRegAlias<B4>;
def H5    : AArch64Reg<5,   "h5", [B5, B5_HI]>, DwarfRegAlias<B5>;
def H6    : AArch64Reg<6,   "h6", [B6, B6_HI]>, DwarfRegAlias<B6>;
def H7    : AArch64Reg<7,   "h7", [B7, B7_HI]>, DwarfRegAlias<B7>;
def H8    : AArch64Reg<8,   "h8", [B8, B8_HI]>, DwarfRegAlias<B8>;
def H9    : AArch64Reg<9,   "h9", [B9, B9_HI]>, DwarfRegAlias<B9>;
def H10   : AArch64Reg<10, "h10", [B10, B10_HI]>, DwarfRegAlias<B10>;
def H11   : AArch64Reg<11, "h11", [B11, B11_HI]>, DwarfRegAlias<B11>;
def H12   : AArch64Reg<12, "h12", [B12, B12_HI]>, DwarfRegAlias<B12>;
def H13   : AArch64Reg<13, "h13", [B13, B13_HI]>, DwarfRegAlias<B13>;
def H14   : AArch64Reg<14, "h14", [B14, B14_HI]>, DwarfRegAlias<B14>;
def H15   : AArch64Reg<15, "h15", [B15, B15_HI]>, DwarfRegAlias<B15>;
def H16   : AArch64Reg<16, "h16", [B16, B16_HI]>, DwarfRegAlias<B16>;
def H17   : AArch64Reg<17, "h17", [B17, B17_HI]>, DwarfRegAlias<B17>;
def H18   : AArch64Reg<18, "h18", [B18, B18_HI]>, DwarfRegAlias<B18>;
def H19   : AArch64Reg<19, "h19", [B19, B19_HI]>, DwarfRegAlias<B19>;
def H20   : AArch64Reg<20, "h20", [B20, B20_HI]>, DwarfRegAlias<B20>;
def H21   : AArch64Reg<21, "h21", [B21, B21_HI]>, DwarfRegAlias<B21>;
def H22   : AArch64Reg<22, "h22", [B22, B22_HI]>, DwarfRegAlias<B22>;
def H23   : AArch64Reg<23, "h23", [B23, B23_HI]>, DwarfRegAlias<B23>;
def H24   : AArch64Reg<24, "h24", [B24, B24_HI]>, DwarfRegAlias<B24>;
def H25   : AArch64Reg<25, "h25", [B25, B25_HI]>, DwarfRegAlias<B25>;
def H26   : AArch64Reg<26, "h26", [B26, B26_HI]>, DwarfRegAlias<B26>;
def H27   : AArch64Reg<27, "h27", [B27, B27_HI]>, DwarfRegAlias<B27>;
def H28   : AArch64Reg<28, "h28", [B28, B28_HI]>, DwarfRegAlias<B28>;
def H29   : AArch64Reg<29, "h29", [B29, B29_HI]>, DwarfRegAlias<B29>;
def H30   : AArch64Reg<30, "h30", [B30, B30_HI]>, DwarfRegAlias<B30>;
def H31   : AArch64Reg<31, "h31", [B31, B31_HI]>, DwarfRegAlias<B31>;
}

let SubRegIndices = [hsub, hsub_hi] in {
def S0    : AArch64Reg<0,   "s0", [H0, H0_HI]>, DwarfRegAlias<B0>;
def S1    : AArch64Reg<1,   "s1", [H1, H1_HI]>, DwarfRegAlias<B1>;
def S2    : AArch64Reg<2,   "s2", [H2, H2_HI]>, DwarfRegAlias<B2>;
def S3    : AArch64Reg<3,   "s3", [H3, H3_HI]>, DwarfRegAlias<B3>;
def S4    : AArch64Reg<4,   "s4", [H4, H4_HI]>, DwarfRegAlias<B4>;
def S5    : AArch64Reg<5,   "s5", [H5, H5_HI]>, DwarfRegAlias<B5>;
def S6    : AArch64Reg<6,   "s6", [H6, H6_HI]>, DwarfRegAlias<B6>;
def S7    : AArch64Reg<7,   "s7", [H7, H7_HI]>, DwarfRegAlias<B7>;
def S8    : AArch64Reg<8,   "s8", [H8, H8_HI]>, DwarfRegAlias<B8>;
def S9    : AArch64Reg<9,   "s9", [H9, H9_HI]>, DwarfRegAlias<B9>;
def S10   : AArch64Reg<10, "s10", [H10, H10_HI]>, DwarfRegAlias<B10>;
def S11   : AArch64Reg<11, "s11", [H11, H11_HI]>, DwarfRegAlias<B11>;
def S12   : AArch64Reg<12, "s12", [H12, H12_HI]>, DwarfRegAlias<B12>;
def S13   : AArch64Reg<13, "s13", [H13, H13_HI]>, DwarfRegAlias<B13>;
def S14   : AArch64Reg<14, "s14", [H14, H14_HI]>, DwarfRegAlias<B14>;
def S15   : AArch64Reg<15, "s15", [H15, H15_HI]>, DwarfRegAlias<B15>;
def S16   : AArch64Reg<16, "s16", [H16, H16_HI]>, DwarfRegAlias<B16>;
def S17   : AArch64Reg<17, "s17", [H17, H17_HI]>, DwarfRegAlias<B17>;
def S18   : AArch64Reg<18, "s18", [H18, H18_HI]>, DwarfRegAlias<B18>;
def S19   : AArch64Reg<19, "s19", [H19, H19_HI]>, DwarfRegAlias<B19>;
def S20   : AArch64Reg<20, "s20", [H20, H20_HI]>, DwarfRegAlias<B20>;
def S21   : AArch64Reg<21, "s21", [H21, H21_HI]>, DwarfRegAlias<B21>;
def S22   : AArch64Reg<22, "s22", [H22, H22_HI]>, DwarfRegAlias<B22>;
def S23   : AArch64Reg<23, "s23", [H23, H23_HI]>, DwarfRegAlias<B23>;
def S24   : AArch64Reg<24, "s24", [H24, H24_HI]>, DwarfRegAlias<B24>;
def S25   : AArch64Reg<25, "s25", [H25, H25_HI]>, DwarfRegAlias<B25>;
def S26   : AArch64Reg<26, "s26", [H26, H26_HI]>, DwarfRegAlias<B26>;
def S27   : AArch64Reg<27, "s27", [H27, H27_HI]>, DwarfRegAlias<B27>;
def S28   : AArch64Reg<28, "s28", [H28, H28_HI]>, DwarfRegAlias<B28>;
def S29   : AArch64Reg<29, "s29", [H29, H29_HI]>, DwarfRegAlias<B29>;
def S30   : AArch64Reg<30, "s30", [H30, H30_HI]>, DwarfRegAlias<B30>;
def S31   : AArch64Reg<31, "s31", [H31, H31_HI]>, DwarfRegAlias<B31>;
}

let SubRegIndices = [ssub, ssub_hi], RegAltNameIndices = [vreg, vlist1] in {
def D0    : AArch64Reg<0,   "d0", [S0, S0_HI], ["v0", ""]>, DwarfRegAlias<B0>;
def D1    : AArch64Reg<1,   "d1", [S1, S1_HI], ["v1", ""]>, DwarfRegAlias<B1>;
def D2    : AArch64Reg<2,   "d2", [S2, S2_HI], ["v2", ""]>, DwarfRegAlias<B2>;
def D3    : AArch64Reg<3,   "d3", [S3, S3_HI], ["v3", ""]>, DwarfRegAlias<B3>;
def D4    : AArch64Reg<4,   "d4", [S4, S4_HI], ["v4", ""]>, DwarfRegAlias<B4>;
def D5    : AArch64Reg<5,   "d5", [S5, S5_HI], ["v5", ""]>, DwarfRegAlias<B5>;
def D6    : AArch64Reg<6,   "d6", [S6, S6_HI], ["v6", ""]>, DwarfRegAlias<B6>;
def D7    : AArch64Reg<7,   "d7", [S7, S7_HI], ["v7", ""]>, DwarfRegAlias<B7>;
def D8    : AArch64Reg<8,   "d8", [S8, S8_HI], ["v8", ""]>, DwarfRegAlias<B8>;
def D9    : AArch64Reg<9,   "d9", [S9, S9_HI], ["v9", ""]>, DwarfRegAlias<B9>;
def D10   : AArch64Reg<10, "d10", [S10, S10_HI], ["v10", ""]>, DwarfRegAlias<B10>;
def D11   : AArch64Reg<11, "d11", [S11, S11_HI], ["v11", ""]>, DwarfRegAlias<B11>;
def D12   : AArch64Reg<12, "d12", [S12, S12_HI], ["v12", ""]>, DwarfRegAlias<B12>;
def D13   : AArch64Reg<13, "d13", [S13, S13_HI], ["v13", ""]>, DwarfRegAlias<B13>;
def D14   : AArch64Reg<14, "d14", [S14, S14_HI], ["v14", ""]>, DwarfRegAlias<B14>;
def D15   : AArch64Reg<15, "d15", [S15, S15_HI], ["v15", ""]>, DwarfRegAlias<B15>;
def D16   : AArch64Reg<16, "d16", [S16, S16_HI], ["v16", ""]>, DwarfRegAlias<B16>;
def D17   : AArch64Reg<17, "d17", [S17, S17_HI], ["v17", ""]>, DwarfRegAlias<B17>;
def D18   : AArch64Reg<18, "d18", [S18, S18_HI], ["v18", ""]>, DwarfRegAlias<B18>;
def D19   : AArch64Reg<19, "d19", [S19, S19_HI], ["v19", ""]>, DwarfRegAlias<B19>;
def D20   : AArch64Reg<20, "d20", [S20, S20_HI], ["v20", ""]>, DwarfRegAlias<B20>;
def D21   : AArch64Reg<21, "d21", [S21, S21_HI], ["v21", ""]>, DwarfRegAlias<B21>;
def D22   : AArch64Reg<22, "d22", [S22, S22_HI], ["v22", ""]>, DwarfRegAlias<B22>;
def D23   : AArch64Reg<23, "d23", [S23, S23_HI], ["v23", ""]>, DwarfRegAlias<B23>;
def D24   : AArch64Reg<24, "d24", [S24, S24_HI], ["v24", ""]>, DwarfRegAlias<B24>;
def D25   : AArch64Reg<25, "d25", [S25, S25_HI], ["v25", ""]>, DwarfRegAlias<B25>;
def D26   : AArch64Reg<26, "d26", [S26, S26_HI], ["v26", ""]>, DwarfRegAlias<B26>;
def D27   : AArch64Reg<27, "d27", [S27, S27_HI], ["v27", ""]>, DwarfRegAlias<B27>;
def D28   : AArch64Reg<28, "d28", [S28, S28_HI], ["v28", ""]>, DwarfRegAlias<B28>;
def D29   : AArch64Reg<29, "d29", [S29, S29_HI], ["v29", ""]>, DwarfRegAlias<B29>;
def D30   : AArch64Reg<30, "d30", [S30, S30_HI], ["v30", ""]>, DwarfRegAlias<B30>;
def D31   : AArch64Reg<31, "d31", [S31, S31_HI], ["v31", ""]>, DwarfRegAlias<B31>;
}

let SubRegIndices = [dsub, dsub_hi], RegAltNameIndices = [vreg, vlist1] in {
def Q0    : AArch64Reg<0,   "q0", [D0, D0_HI], ["v0", ""]>, DwarfRegAlias<B0>;
def Q1    : AArch64Reg<1,   "q1", [D1, D1_HI], ["v1", ""]>, DwarfRegAlias<B1>;
def Q2    : AArch64Reg<2,   "q2", [D2, D2_HI], ["v2", ""]>, DwarfRegAlias<B2>;
def Q3    : AArch64Reg<3,   "q3", [D3, D3_HI], ["v3", ""]>, DwarfRegAlias<B3>;
def Q4    : AArch64Reg<4,   "q4", [D4, D4_HI], ["v4", ""]>, DwarfRegAlias<B4>;
def Q5    : AArch64Reg<5,   "q5", [D5, D5_HI], ["v5", ""]>, DwarfRegAlias<B5>;
def Q6    : AArch64Reg<6,   "q6", [D6, D6_HI], ["v6", ""]>, DwarfRegAlias<B6>;
def Q7    : AArch64Reg<7,   "q7", [D7, D7_HI], ["v7", ""]>, DwarfRegAlias<B7>;
def Q8    : AArch64Reg<8,   "q8", [D8, D8_HI], ["v8", ""]>, DwarfRegAlias<B8>;
def Q9    : AArch64Reg<9,   "q9", [D9, D9_HI], ["v9", ""]>, DwarfRegAlias<B9>;
def Q10   : AArch64Reg<10, "q10", [D10, D10_HI], ["v10", ""]>, DwarfRegAlias<B10>;
def Q11   : AArch64Reg<11, "q11", [D11, D11_HI], ["v11", ""]>, DwarfRegAlias<B11>;
def Q12   : AArch64Reg<12, "q12", [D12, D12_HI], ["v12", ""]>, DwarfRegAlias<B12>;
def Q13   : AArch64Reg<13, "q13", [D13, D13_HI], ["v13", ""]>, DwarfRegAlias<B13>;
def Q14   : AArch64Reg<14, "q14", [D14, D14_HI], ["v14", ""]>, DwarfRegAlias<B14>;
def Q15   : AArch64Reg<15, "q15", [D15, D15_HI], ["v15", ""]>, DwarfRegAlias<B15>;
def Q16   : AArch64Reg<16, "q16", [D16, D16_HI], ["v16", ""]>, DwarfRegAlias<B16>;
def Q17   : AArch64Reg<17, "q17", [D17, D17_HI], ["v17", ""]>, DwarfRegAlias<B17>;
def Q18   : AArch64Reg<18, "q18", [D18, D18_HI], ["v18", ""]>, DwarfRegAlias<B18>;
def Q19   : AArch64Reg<19, "q19", [D19, D19_HI], ["v19", ""]>, DwarfRegAlias<B19>;
def Q20   : AArch64Reg<20, "q20", [D20, D20_HI], ["v20", ""]>, DwarfRegAlias<B20>;
def Q21   : AArch64Reg<21, "q21", [D21, D21_HI], ["v21", ""]>, DwarfRegAlias<B21>;
def Q22   : AArch64Reg<22, "q22", [D22, D22_HI], ["v22", ""]>, DwarfRegAlias<B22>;
def Q23   : AArch64Reg<23, "q23", [D23, D23_HI], ["v23", ""]>, DwarfRegAlias<B23>;
def Q24   : AArch64Reg<24, "q24", [D24, D24_HI], ["v24", ""]>, DwarfRegAlias<B24>;
def Q25   : AArch64Reg<25, "q25", [D25, D25_HI], ["v25", ""]>, DwarfRegAlias<B25>;
def Q26   : AArch64Reg<26, "q26", [D26, D26_HI], ["v26", ""]>, DwarfRegAlias<B26>;
def Q27   : AArch64Reg<27, "q27", [D27, D27_HI], ["v27", ""]>, DwarfRegAlias<B27>;
def Q28   : AArch64Reg<28, "q28", [D28, D28_HI], ["v28", ""]>, DwarfRegAlias<B28>;
def Q29   : AArch64Reg<29, "q29", [D29, D29_HI], ["v29", ""]>, DwarfRegAlias<B29>;
def Q30   : AArch64Reg<30, "q30", [D30, D30_HI], ["v30", ""]>, DwarfRegAlias<B30>;
def Q31   : AArch64Reg<31, "q31", [D31, D31_HI], ["v31", ""]>, DwarfRegAlias<B31>;
}

def FPR8  : RegisterClass<"AArch64", [i8], 8, (sequence "B%u", 0, 31)> {
  let Size = 8;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::FPR8RegClassID, 0, 32>";
}
def FPR16 : RegisterClass<"AArch64", [f16, bf16, i16], 16, (sequence "H%u", 0, 31)> {
  let Size = 16;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::FPR16RegClassID, 0, 32>";
}

def FPR16_lo : RegisterClass<"AArch64", [f16], 16, (trunc FPR16, 16)> {
  let Size = 16;
}
def FPR32 : RegisterClass<"AArch64", [f32, i32], 32,(sequence "S%u", 0, 31)> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::FPR32RegClassID, 0, 32>";
}
def FPR64 : RegisterClass<"AArch64", [f64, i64, v2f32, v1f64, v8i8, v4i16, v2i32,
                                      v1i64, v4f16, v4bf16],
                                     64, (sequence "D%u", 0, 31)> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::FPR64RegClassID, 0, 32>";
}
def FPR64_lo : RegisterClass<"AArch64",
                             [v8i8, v4i16, v2i32, v1i64, v4f16, v4bf16, v2f32,
                              v1f64],
                             64, (trunc FPR64, 16)>;

// We don't (yet) have an f128 legal type, so don't use that here. We
// normalize 128-bit vectors to v2f64 for arg passing and such, so use
// that here.
def FPR128 : RegisterClass<"AArch64",
                           [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64, f128,
                            v8f16, v8bf16],
                           128, (sequence "Q%u", 0, 31)> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::FPR128RegClassID, 0, 32>";
}

// The lower 16 vector registers.  Some instructions can only take registers
// in this range.
def FPR128_lo : RegisterClass<"AArch64",
                              [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64, v8f16,
                               v8bf16],
                              128, (trunc FPR128, 16)> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::FPR128RegClassID, 0, 16>";
}

// The lower 8 vector registers.  Some instructions can only take registers
// in this range.
def FPR128_0to7 : RegisterClass<"AArch64",
                                [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64, v8f16,
                                 v8bf16],
                                128, (trunc FPR128, 8)> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::FPR128RegClassID, 0, 8>";
}

// Pairs, triples, and quads of 64-bit vector registers.
def DSeqPairs : RegisterTuples<[dsub0, dsub1], [(rotl FPR64, 0), (rotl FPR64, 1)]>;
def DSeqTriples : RegisterTuples<[dsub0, dsub1, dsub2],
                                 [(rotl FPR64, 0), (rotl FPR64, 1),
                                  (rotl FPR64, 2)]>;
def DSeqQuads : RegisterTuples<[dsub0, dsub1, dsub2, dsub3],
                               [(rotl FPR64, 0), (rotl FPR64, 1),
                                (rotl FPR64, 2), (rotl FPR64, 3)]>;
def DD   : RegisterClass<"AArch64", [untyped], 64, (add DSeqPairs)> {
  let Size = 128;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::DDRegClassID, 0, 32>";
}
def DDD  : RegisterClass<"AArch64", [untyped], 64, (add DSeqTriples)> {
  let Size = 192;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::DDDRegClassID, 0, 32>";
}
def DDDD : RegisterClass<"AArch64", [untyped], 64, (add DSeqQuads)> {
  let Size = 256;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::DDDDRegClassID, 0, 32>";
}

// Pairs, triples, and quads of 128-bit vector registers.
def QSeqPairs : RegisterTuples<[qsub0, qsub1], [(rotl FPR128, 0), (rotl FPR128, 1)]>;
def QSeqTriples : RegisterTuples<[qsub0, qsub1, qsub2],
                                 [(rotl FPR128, 0), (rotl FPR128, 1),
                                  (rotl FPR128, 2)]>;
def QSeqQuads : RegisterTuples<[qsub0, qsub1, qsub2, qsub3],
                               [(rotl FPR128, 0), (rotl FPR128, 1),
                                (rotl FPR128, 2), (rotl FPR128, 3)]>;
def QQ   : RegisterClass<"AArch64", [untyped], 128, (add QSeqPairs)> {
  let Size = 256;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::QQRegClassID, 0, 32>";
}
def QQQ  : RegisterClass<"AArch64", [untyped], 128, (add QSeqTriples)> {
  let Size = 384;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::QQQRegClassID, 0, 32>";
}
def QQQQ : RegisterClass<"AArch64", [untyped], 128, (add QSeqQuads)> {
  let Size = 512;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::QQQQRegClassID, 0, 32>";
}


// Vector operand versions of the FP registers. Alternate name printing and
// assembler matching.
def VectorReg64AsmOperand : AsmOperandClass {
  let Name = "VectorReg64";
  let PredicateMethod = "isNeonVectorReg";
}
def VectorReg128AsmOperand : AsmOperandClass {
  let Name = "VectorReg128";
  let PredicateMethod = "isNeonVectorReg";
}

def V64  : RegisterOperand<FPR64, "printVRegOperand"> {
  let ParserMatchClass = VectorReg64AsmOperand;
}

def V128 : RegisterOperand<FPR128, "printVRegOperand"> {
  let ParserMatchClass = VectorReg128AsmOperand;
}

def VectorRegLoAsmOperand : AsmOperandClass {
  let Name = "VectorRegLo";
  let PredicateMethod = "isNeonVectorRegLo";
}
def V64_lo : RegisterOperand<FPR64_lo, "printVRegOperand"> {
  let ParserMatchClass = VectorRegLoAsmOperand;
}
def V128_lo : RegisterOperand<FPR128_lo, "printVRegOperand"> {
  let ParserMatchClass = VectorRegLoAsmOperand;
}

def VectorReg0to7AsmOperand : AsmOperandClass {
  let Name = "VectorReg0to7";
  let PredicateMethod = "isNeonVectorReg0to7";
}

def V128_0to7 : RegisterOperand<FPR128_0to7, "printVRegOperand"> {
  let ParserMatchClass = VectorReg0to7AsmOperand;
}

class TypedVecListAsmOperand<int count, string vecty, int lanes, int eltsize>
    : AsmOperandClass {
  let Name = "TypedVectorList" # count # "_" # lanes # eltsize;

  let PredicateMethod
      = "isTypedVectorList<RegKind::NeonVector, " # count # ", " # lanes # ", " # eltsize # ">";
  let RenderMethod = "addVectorListOperands<" # vecty  # ", "  # count # ">";
}

class TypedVecListRegOperand<RegisterClass Reg, int lanes, string eltsize>
    : RegisterOperand<Reg, "printTypedVectorList<" # lanes # ", '"
                                                   # eltsize # "'>">;

multiclass VectorList<int count, RegisterClass Reg64, RegisterClass Reg128> {
  // With implicit types (probably on instruction instead). E.g. { v0, v1 } or {v0, v2, v4}.
  def _64AsmOperand : AsmOperandClass {
    let Name = NAME # "64";
    let PredicateMethod = "isImplicitlyTypedVectorList<RegKind::NeonVector, " # count # ">";
    let RenderMethod = "addVectorListOperands<AArch64Operand::VecListIdx_DReg, " # count # ">";
  }

  def "64" : RegisterOperand<Reg64, "printImplicitlyTypedVectorList"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_64AsmOperand");
  }

  def _128AsmOperand : AsmOperandClass {
    let Name = NAME # "128";
    let PredicateMethod = "isImplicitlyTypedVectorList<RegKind::NeonVector, " # count # ">";
    let RenderMethod = "addVectorListOperands<AArch64Operand::VecListIdx_QReg, " # count # ">";
  }

  def "128" : RegisterOperand<Reg128, "printImplicitlyTypedVectorList"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_128AsmOperand");
  }

  // With implicit types (probably on instruction instead), consecutive registers. E.g. { v0, v1, v2 }
  def _Consecutive128AsmOperand : AsmOperandClass {
    let Name = NAME # "Consecutive128";
    let PredicateMethod = "isImplicitlyTypedVectorList<RegKind::NeonVector, " # count # ", true>";
    let RenderMethod = "addVectorListOperands<AArch64Operand::VecListIdx_QReg, " # count # ", true>";
  }

  def "Consecutive128" : RegisterOperand<Reg128, "printImplicitlyTypedVectorList"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_Consecutive128AsmOperand");
  }

  // 64-bit register lists with explicit type.

  // { v0.8b, v1.8b }
  def _8bAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_DReg", 8, 8>;
  def "8b" : TypedVecListRegOperand<Reg64, 8, "b"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_8bAsmOperand");
  }

  // { v0.4h, v1.4h }
  def _4hAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_DReg", 4, 16>;
  def "4h" : TypedVecListRegOperand<Reg64, 4, "h"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_4hAsmOperand");
  }

  // { v0.2s, v1.2s }
  def _2sAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_DReg", 2, 32>;
  def "2s" : TypedVecListRegOperand<Reg64, 2, "s"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_2sAsmOperand");
  }

  // { v0.1d, v1.1d }
  def _1dAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_DReg", 1, 64>;
  def "1d" : TypedVecListRegOperand<Reg64, 1, "d"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_1dAsmOperand");
  }

  // 128-bit register lists with explicit type

  // { v0.16b, v1.16b }
  def _16bAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 16, 8>;
  def "16b" : TypedVecListRegOperand<Reg128, 16, "b"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_16bAsmOperand");
  }

  // { v0.8h, v1.8h }
  def _8hAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 8, 16>;
  def "8h" : TypedVecListRegOperand<Reg128, 8, "h"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_8hAsmOperand");
  }

  // { v0.4s, v1.4s }
  def _4sAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 4, 32>;
  def "4s" : TypedVecListRegOperand<Reg128, 4, "s"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_4sAsmOperand");
  }

  // { v0.2d, v1.2d }
  def _2dAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 2, 64>;
  def "2d" : TypedVecListRegOperand<Reg128, 2, "d"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_2dAsmOperand");
  }

  // { v0.b, v1.b }
  def _bAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 0, 8>;
  def "b" : TypedVecListRegOperand<Reg128, 0, "b"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_bAsmOperand");
  }

  // { v0.h, v1.h }
  def _hAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 0, 16>;
  def "h" : TypedVecListRegOperand<Reg128, 0, "h"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_hAsmOperand");
  }

  // { v0.s, v1.s }
  def _sAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 0, 32>;
  def "s" : TypedVecListRegOperand<Reg128, 0, "s"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_sAsmOperand");
  }

  // { v0.d, v1.d }
  def _dAsmOperand : TypedVecListAsmOperand<count, "AArch64Operand::VecListIdx_QReg", 0, 64>;
  def "d" : TypedVecListRegOperand<Reg128, 0, "d"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_dAsmOperand");
  }


}

defm VecListOne   : VectorList<1, FPR64, FPR128>;
defm VecListTwo   : VectorList<2, DD,    QQ>;
defm VecListThree : VectorList<3, DDD,   QQQ>;
defm VecListFour  : VectorList<4, DDDD,  QQQQ>;

class FPRAsmOperand<string RC> : AsmOperandClass {
  let Name = "FPRAsmOperand" # RC;
  let PredicateMethod = "isGPR64<AArch64::" # RC # "RegClassID>";
  let RenderMethod = "addRegOperands";
}

// Register operand versions of the scalar FP registers.
def FPR8Op  : RegisterOperand<FPR8, "printOperand"> {
  let ParserMatchClass = FPRAsmOperand<"FPR8">;
}

def FPR16Op  : RegisterOperand<FPR16, "printOperand"> {
  let ParserMatchClass = FPRAsmOperand<"FPR16">;
}

def FPR16Op_lo  : RegisterOperand<FPR16_lo, "printOperand"> {
  let ParserMatchClass = FPRAsmOperand<"FPR16_lo">;
}

def FPR32Op  : RegisterOperand<FPR32, "printOperand"> {
  let ParserMatchClass = FPRAsmOperand<"FPR32">;
}

def FPR64Op  : RegisterOperand<FPR64, "printOperand"> {
  let ParserMatchClass = FPRAsmOperand<"FPR64">;
}

def FPR128Op : RegisterOperand<FPR128, "printOperand"> {
  let ParserMatchClass = FPRAsmOperand<"FPR128">;
}

//===----------------------------------------------------------------------===//
// ARMv8.1a atomic CASP register operands


def WSeqPairs : RegisterTuples<[sube32, subo32],
                               [(decimate (rotl GPR32, 0), 2),
                                (decimate (rotl GPR32, 1), 2)]>;
def XSeqPairs : RegisterTuples<[sube64, subo64],
                               [(decimate (rotl GPR64, 0), 2),
                                (decimate (rotl GPR64, 1), 2)]>;

def WSeqPairsClass   : RegisterClass<"AArch64", [untyped], 32,
                                     (add WSeqPairs)>{
  let Size = 64;
}
def XSeqPairsClass   : RegisterClass<"AArch64", [untyped], 64,
                                     (add XSeqPairs)>{
  let Size = 128;
}


let RenderMethod = "addRegOperands", ParserMethod="tryParseGPRSeqPair" in {
  def WSeqPairsAsmOperandClass : AsmOperandClass { let Name = "WSeqPair"; }
  def XSeqPairsAsmOperandClass : AsmOperandClass { let Name = "XSeqPair"; }
}

def WSeqPairClassOperand :
    RegisterOperand<WSeqPairsClass, "printGPRSeqPairsClassOperand<32>"> {
  let ParserMatchClass = WSeqPairsAsmOperandClass;
}
def XSeqPairClassOperand :
    RegisterOperand<XSeqPairsClass, "printGPRSeqPairsClassOperand<64>"> {
  let ParserMatchClass = XSeqPairsAsmOperandClass;
}
// Reuse the parsing and register numbers from XSeqPairs, but encoding is different.
def MrrsMssrPairClassOperand :
    RegisterOperand<XSeqPairsClass, "printGPRSeqPairsClassOperand<64>"> {
  let ParserMatchClass = XSeqPairsAsmOperandClass;
}
def SyspXzrPairOperandMatcherClass : AsmOperandClass {
  let Name = "SyspXzrPair";
  let RenderMethod = "addSyspXzrPairOperand";
  let ParserMethod = "tryParseSyspXzrPair";
}
def SyspXzrPairOperand :
    RegisterOperand<GPR64, "printSyspXzrPair"> { // needed to allow alias with XZR operand
  let ParserMatchClass = SyspXzrPairOperandMatcherClass;
}



//===----- END: v8.1a atomic CASP register operands -----------------------===//

//===----------------------------------------------------------------------===//
// Armv8.7a accelerator extension register operands: 8 consecutive GPRs
// starting with an even one

let Namespace = "AArch64" in {
  foreach i = 0-7 in
    def "x8sub_"#i : SubRegIndex<64, !mul(64, i)>;
}

def Tuples8X : RegisterTuples<
  !foreach(i, [0,1,2,3,4,5,6,7], !cast<SubRegIndex>("x8sub_"#i)),
  !foreach(i, [0,1,2,3,4,5,6,7], (trunc (decimate (rotl GPR64, i), 2), 12))>;

def GPR64x8Class : RegisterClass<"AArch64", [i64x8], 512, (trunc Tuples8X, 12)> {
  let Size = 512;
}
def GPR64x8AsmOp : AsmOperandClass {
  let Name = "GPR64x8";
  let ParserMethod = "tryParseGPR64x8";
  let RenderMethod = "addRegOperands";
}
def GPR64x8 : RegisterOperand<GPR64x8Class, "printGPR64x8"> {
  let ParserMatchClass = GPR64x8AsmOp;
  let PrintMethod = "printGPR64x8";
}

//===----- END: v8.7a accelerator extension register operands -------------===//

// SVE predicate-as-counter registers
  def PN0    : AArch64Reg<0,   "pn0">, DwarfRegNum<[48]>;
  def PN1    : AArch64Reg<1,   "pn1">, DwarfRegNum<[49]>;
  def PN2    : AArch64Reg<2,   "pn2">, DwarfRegNum<[50]>;
  def PN3    : AArch64Reg<3,   "pn3">, DwarfRegNum<[51]>;
  def PN4    : AArch64Reg<4,   "pn4">, DwarfRegNum<[52]>;
  def PN5    : AArch64Reg<5,   "pn5">, DwarfRegNum<[53]>;
  def PN6    : AArch64Reg<6,   "pn6">, DwarfRegNum<[54]>;
  def PN7    : AArch64Reg<7,   "pn7">, DwarfRegNum<[55]>;
  def PN8    : AArch64Reg<8,   "pn8">, DwarfRegNum<[56]>;
  def PN9    : AArch64Reg<9,   "pn9">, DwarfRegNum<[57]>;
  def PN10   : AArch64Reg<10, "pn10">, DwarfRegNum<[58]>;
  def PN11   : AArch64Reg<11, "pn11">, DwarfRegNum<[59]>;
  def PN12   : AArch64Reg<12, "pn12">, DwarfRegNum<[60]>;
  def PN13   : AArch64Reg<13, "pn13">, DwarfRegNum<[61]>;
  def PN14   : AArch64Reg<14, "pn14">, DwarfRegNum<[62]>;
  def PN15   : AArch64Reg<15, "pn15">, DwarfRegNum<[63]>;

// SVE predicate registers
let SubRegIndices = [psub] in {
  def P0    : AArch64Reg<0,   "p0", [PN0]>, DwarfRegAlias<PN0>;
  def P1    : AArch64Reg<1,   "p1", [PN1]>, DwarfRegAlias<PN1>;
  def P2    : AArch64Reg<2,   "p2", [PN2]>, DwarfRegAlias<PN2>;
  def P3    : AArch64Reg<3,   "p3", [PN3]>, DwarfRegAlias<PN3>;
  def P4    : AArch64Reg<4,   "p4", [PN4]>, DwarfRegAlias<PN4>;
  def P5    : AArch64Reg<5,   "p5", [PN5]>, DwarfRegAlias<PN5>;
  def P6    : AArch64Reg<6,   "p6", [PN6]>, DwarfRegAlias<PN6>;
  def P7    : AArch64Reg<7,   "p7", [PN7]>, DwarfRegAlias<PN7>;
  def P8    : AArch64Reg<8,   "p8", [PN8]>, DwarfRegAlias<PN8>;
  def P9    : AArch64Reg<9,   "p9", [PN9]>, DwarfRegAlias<PN9>;
  def P10   : AArch64Reg<10, "p10", [PN10]>, DwarfRegAlias<PN10>;
  def P11   : AArch64Reg<11, "p11", [PN11]>, DwarfRegAlias<PN11>;
  def P12   : AArch64Reg<12, "p12", [PN12]>, DwarfRegAlias<PN12>;
  def P13   : AArch64Reg<13, "p13", [PN13]>, DwarfRegAlias<PN13>;
  def P14   : AArch64Reg<14, "p14", [PN14]>, DwarfRegAlias<PN14>;
  def P15   : AArch64Reg<15, "p15", [PN15]>, DwarfRegAlias<PN15>;
}

// SVE variable-size vector registers
let SubRegIndices = [zsub, zsub_hi] in {
def Z0    : AArch64Reg<0,   "z0",  [Q0, Q0_HI]>, DwarfRegNum<[96]>;
def Z1    : AArch64Reg<1,   "z1",  [Q1, Q1_HI]>, DwarfRegNum<[97]>;
def Z2    : AArch64Reg<2,   "z2",  [Q2, Q2_HI]>, DwarfRegNum<[98]>;
def Z3    : AArch64Reg<3,   "z3",  [Q3, Q3_HI]>, DwarfRegNum<[99]>;
def Z4    : AArch64Reg<4,   "z4",  [Q4, Q4_HI]>, DwarfRegNum<[100]>;
def Z5    : AArch64Reg<5,   "z5",  [Q5, Q5_HI]>, DwarfRegNum<[101]>;
def Z6    : AArch64Reg<6,   "z6",  [Q6, Q6_HI]>, DwarfRegNum<[102]>;
def Z7    : AArch64Reg<7,   "z7",  [Q7, Q7_HI]>, DwarfRegNum<[103]>;
def Z8    : AArch64Reg<8,   "z8",  [Q8, Q8_HI]>, DwarfRegNum<[104]>;
def Z9    : AArch64Reg<9,   "z9",  [Q9, Q9_HI]>, DwarfRegNum<[105]>;
def Z10   : AArch64Reg<10, "z10", [Q10, Q10_HI]>, DwarfRegNum<[106]>;
def Z11   : AArch64Reg<11, "z11", [Q11, Q11_HI]>, DwarfRegNum<[107]>;
def Z12   : AArch64Reg<12, "z12", [Q12, Q12_HI]>, DwarfRegNum<[108]>;
def Z13   : AArch64Reg<13, "z13", [Q13, Q13_HI]>, DwarfRegNum<[109]>;
def Z14   : AArch64Reg<14, "z14", [Q14, Q14_HI]>, DwarfRegNum<[110]>;
def Z15   : AArch64Reg<15, "z15", [Q15, Q15_HI]>, DwarfRegNum<[111]>;
def Z16   : AArch64Reg<16, "z16", [Q16, Q16_HI]>, DwarfRegNum<[112]>;
def Z17   : AArch64Reg<17, "z17", [Q17, Q17_HI]>, DwarfRegNum<[113]>;
def Z18   : AArch64Reg<18, "z18", [Q18, Q18_HI]>, DwarfRegNum<[114]>;
def Z19   : AArch64Reg<19, "z19", [Q19, Q19_HI]>, DwarfRegNum<[115]>;
def Z20   : AArch64Reg<20, "z20", [Q20, Q20_HI]>, DwarfRegNum<[116]>;
def Z21   : AArch64Reg<21, "z21", [Q21, Q21_HI]>, DwarfRegNum<[117]>;
def Z22   : AArch64Reg<22, "z22", [Q22, Q22_HI]>, DwarfRegNum<[118]>;
def Z23   : AArch64Reg<23, "z23", [Q23, Q23_HI]>, DwarfRegNum<[119]>;
def Z24   : AArch64Reg<24, "z24", [Q24, Q24_HI]>, DwarfRegNum<[120]>;
def Z25   : AArch64Reg<25, "z25", [Q25, Q25_HI]>, DwarfRegNum<[121]>;
def Z26   : AArch64Reg<26, "z26", [Q26, Q26_HI]>, DwarfRegNum<[122]>;
def Z27   : AArch64Reg<27, "z27", [Q27, Q27_HI]>, DwarfRegNum<[123]>;
def Z28   : AArch64Reg<28, "z28", [Q28, Q28_HI]>, DwarfRegNum<[124]>;
def Z29   : AArch64Reg<29, "z29", [Q29, Q29_HI]>, DwarfRegNum<[125]>;
def Z30   : AArch64Reg<30, "z30", [Q30, Q30_HI]>, DwarfRegNum<[126]>;
def Z31   : AArch64Reg<31, "z31", [Q31, Q31_HI]>, DwarfRegNum<[127]>;
}

// Enum describing the element size for destructive
// operations.
class ElementSizeEnum<bits<3> val> {
  bits<3> Value = val;
}

def ElementSizeNone : ElementSizeEnum<0>;
def ElementSizeB    : ElementSizeEnum<1>;
def ElementSizeH    : ElementSizeEnum<2>;
def ElementSizeS    : ElementSizeEnum<3>;
def ElementSizeD    : ElementSizeEnum<4>;
def ElementSizeQ    : ElementSizeEnum<5>;  // Unused

class SVERegOp <string Suffix, AsmOperandClass C,
                ElementSizeEnum Size,
                RegisterClass RC> : RegisterOperand<RC> {
  ElementSizeEnum ElementSize;

  let ElementSize = Size;
  let PrintMethod = !if(!eq(Suffix, ""),
                        "printSVERegOp<>",
                        "printSVERegOp<'" # Suffix # "'>");
  let ParserMatchClass = C;
}

class ZPRRegOp <string Suffix, AsmOperandClass C, ElementSizeEnum Size,
                RegisterClass RC> : SVERegOp<Suffix, C, Size, RC> {}

//******************************************************************************

// SVE predicate register classes.
class PPRClass<int firstreg, int lastreg, int step = 1> : RegisterClass<"AArch64",
                                  [ nxv16i1, nxv8i1, nxv4i1, nxv2i1, nxv1i1 ], 16,
                                  (sequence "P%u", firstreg, lastreg, step)> {
  let Size = 16;
}

def PPR    : PPRClass<0, 15> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::PPRRegClassID, 0, 16>";
}
def PPR_3b : PPRClass<0, 7> { // Restricted 3 bit SVE predicate register class.
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::PPRRegClassID, 0, 8>";
}
def PPR_p8to15 : PPRClass<8, 15> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::PNRRegClassID, 8, 8>";
}

def PPRMul2 : PPRClass<0, 14, 2>;

class PPRAsmOperand <string name, string RegClass, int Width>: AsmOperandClass {
  let Name = "SVE" # name # "Reg";
  let PredicateMethod = "isSVEPredicateVectorRegOfWidth<"
                            # Width # ", " # "AArch64::" # RegClass # "RegClassID>";
  let DiagnosticType = "InvalidSVE" # name # "Reg";
  let RenderMethod = "addRegOperands";
  let ParserMethod = "tryParseSVEPredicateVector<RegKind::SVEPredicateVector>";
}

def PPRAsmOpAny   : PPRAsmOperand<"PredicateAny",   "PPR",     0>;
def PPRAsmOp8     : PPRAsmOperand<"PredicateB",     "PPR",     8>;
def PPRAsmOp16    : PPRAsmOperand<"PredicateH",     "PPR",    16>;
def PPRAsmOp32    : PPRAsmOperand<"PredicateS",     "PPR",    32>;
def PPRAsmOp64    : PPRAsmOperand<"PredicateD",     "PPR",    64>;
def PPRAsmOp3bAny : PPRAsmOperand<"Predicate3bAny", "PPR_3b",  0>;

class PPRRegOp <string Suffix, AsmOperandClass C, ElementSizeEnum Size,
                RegisterClass RC> : SVERegOp<Suffix, C, Size, RC> {}

def PPRAny   : PPRRegOp<"",  PPRAsmOpAny,   ElementSizeNone, PPR>;
def PPR8     : PPRRegOp<"b", PPRAsmOp8,     ElementSizeB,    PPR>;
def PPR16    : PPRRegOp<"h", PPRAsmOp16,    ElementSizeH,    PPR>;
def PPR32    : PPRRegOp<"s", PPRAsmOp32,    ElementSizeS,    PPR>;
def PPR64    : PPRRegOp<"d", PPRAsmOp64,    ElementSizeD,    PPR>;
def PPR3bAny : PPRRegOp<"",  PPRAsmOp3bAny, ElementSizeNone, PPR_3b>;

class PNRClass<int firstreg, int lastreg> : RegisterClass<
                                  "AArch64",
                                  [ aarch64svcount ], 16,
                                  (sequence "PN%u", firstreg, lastreg)> {
  let Size = 16;
}

def PNR        : PNRClass<0, 15> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::PNRRegClassID, 0, 16>";
}
def PNR_3b     : PNRClass<0, 7>;
def PNR_p8to15 : PNRClass<8, 15>;

// SVE predicate-as-counter operand
class PNRAsmOperand<string name, string RegClass, int Width>: AsmOperandClass {
  let Name = "SVE" # name # "Reg";
  let PredicateMethod = "isSVEPredicateAsCounterRegOfWidth<"
                            # Width # ", " # "AArch64::"
                            # RegClass # "RegClassID>";
  let DiagnosticType = "InvalidSVE" # name # "Reg";
  let RenderMethod = "addRegOperands";
  let ParserMethod   = "tryParseSVEPredicateVector<RegKind::SVEPredicateAsCounter>";
}

def PNRAsmOpAny: PNRAsmOperand<"PNPredicateAny", "PNR", 0>;
def PNRAsmOp8  : PNRAsmOperand<"PNPredicateB",   "PNR", 8>;
def PNRAsmOp16 : PNRAsmOperand<"PNPredicateH",   "PNR", 16>;
def PNRAsmOp32 : PNRAsmOperand<"PNPredicateS",   "PNR", 32>;
def PNRAsmOp64 : PNRAsmOperand<"PNPredicateD",   "PNR", 64>;

class PNRRegOp<string Suffix, AsmOperandClass C, int Size, RegisterClass RC>
    : SVERegOp<Suffix, C, ElementSizeNone, RC> {
  let PrintMethod = "printPredicateAsCounter<" # Size # ">";
}
def PNRAny : PNRRegOp<"",  PNRAsmOpAny, 0,  PNR>;
def PNR8   : PNRRegOp<"b", PNRAsmOp8,   8,  PNR>;
def PNR16  : PNRRegOp<"h", PNRAsmOp16,  16, PNR>;
def PNR32  : PNRRegOp<"s", PNRAsmOp32,  32, PNR>;
def PNR64  : PNRRegOp<"d", PNRAsmOp64,  64, PNR>;

def PNRAsmAny_p8to15  : PNRAsmOperand<"PNPredicateAny_p8to15", "PNR_p8to15", 0>;
def PNRAsmOp8_p8to15  : PNRAsmOperand<"PNPredicateB_p8to15",   "PNR_p8to15", 8>;
def PNRAsmOp16_p8to15 : PNRAsmOperand<"PNPredicateH_p8to15",   "PNR_p8to15", 16>;
def PNRAsmOp32_p8to15 : PNRAsmOperand<"PNPredicateS_p8to15",   "PNR_p8to15", 32>;
def PNRAsmOp64_p8to15 : PNRAsmOperand<"PNPredicateD_p8to15",   "PNR_p8to15", 64>;

class PNRP8to15RegOp<string Suffix, AsmOperandClass C, int Width, RegisterClass RC>
    : SVERegOp<Suffix, C, ElementSizeNone, RC> {
  let PrintMethod   = "printPredicateAsCounter<" # Width # ">";
  let EncoderMethod = "EncodePNR_p8to15";
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::PNRRegClassID, 8, 8>";
}

def PNRAny_p8to15 : PNRP8to15RegOp<"",  PNRAsmAny_p8to15,  0,  PNR_p8to15>;
def PNR8_p8to15   : PNRP8to15RegOp<"b", PNRAsmOp8_p8to15,  8,  PNR_p8to15>;
def PNR16_p8to15  : PNRP8to15RegOp<"h", PNRAsmOp16_p8to15, 16, PNR_p8to15>;
def PNR32_p8to15  : PNRP8to15RegOp<"s", PNRAsmOp32_p8to15, 32, PNR_p8to15>;
def PNR64_p8to15  : PNRP8to15RegOp<"d", PNRAsmOp64_p8to15, 64, PNR_p8to15>;

class PPRorPNRClass : RegisterClass<
                                  "AArch64",
                                  [ nxv16i1, nxv8i1, nxv4i1, nxv2i1, nxv1i1, aarch64svcount ], 16,
                                  (add PPR, PNR)> {
  let Size = 16;
}

class PPRorPNRAsmOperand<string name, string RegClass, int Width>: AsmOperandClass {
  let Name = "SVE" # name # "Reg";
  let PredicateMethod = "isSVEPredicateOrPredicateAsCounterRegOfWidth<"
                            # Width # ", " # "AArch64::"
                            # RegClass # "RegClassID>";
  let DiagnosticType = "InvalidSVE" # name # "Reg";
  let RenderMethod = "addPPRorPNRRegOperands";
  let ParserMethod = "tryParseSVEPredicateOrPredicateAsCounterVector";
}

def PPRorPNR         : PPRorPNRClass {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::PPRorPNRRegClassID, 0, 16>";
}
def PPRorPNRAsmOp8   : PPRorPNRAsmOperand<"PPRorPNRB", "PPRorPNR", 8>;
def PPRorPNRAsmOpAny : PPRorPNRAsmOperand<"PPRorPNRAny", "PPRorPNR", 0>;
def PPRorPNRAny      : PPRRegOp<"", PPRorPNRAsmOpAny, ElementSizeNone, PPRorPNR>;
def PPRorPNR8        : PPRRegOp<"b", PPRorPNRAsmOp8, ElementSizeB, PPRorPNR>;

// Pairs of SVE predicate vector registers.
def PSeqPairs : RegisterTuples<[psub0, psub1], [(rotl PPR, 0), (rotl PPR, 1)]>;

def PPR2 : RegisterClass<"AArch64", [untyped], 16, (add PSeqPairs)> {
  let Size = 32;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::PPR2RegClassID, 0, 16>";
}

class PPRVectorList<int ElementWidth, int NumRegs> : AsmOperandClass {
  let Name = "SVEPredicateList" # NumRegs # "x" # ElementWidth;
  let ParserMethod = "tryParseVectorList<RegKind::SVEPredicateVector>";
  let PredicateMethod = "isTypedVectorList<RegKind::SVEPredicateVector, "
                            # NumRegs #", 0, "#ElementWidth #">";
  let RenderMethod = "addVectorListOperands<AArch64Operand::VecListIdx_PReg, "
                         # NumRegs #">";
}

def PP_b : RegisterOperand<PPR2, "printTypedVectorList<0,'b'>"> {
  let ParserMatchClass = PPRVectorList<8, 2>;
}

def PP_h : RegisterOperand<PPR2, "printTypedVectorList<0,'h'>"> {
  let ParserMatchClass = PPRVectorList<16, 2>;
}

def PP_s : RegisterOperand<PPR2, "printTypedVectorList<0,'s'>"> {
  let ParserMatchClass = PPRVectorList<32, 2>;
}

def PP_d : RegisterOperand<PPR2, "printTypedVectorList<0,'d'>"> {
  let ParserMatchClass = PPRVectorList<64, 2>;
}

// SVE2 multiple-of-2 multi-predicate-vector operands
def PPR2Mul2 : RegisterClass<"AArch64", [untyped], 16, (add (decimate PSeqPairs, 2))> {
  let Size = 32;
}

class PPRVectorListMul<int ElementWidth, int NumRegs> : PPRVectorList<ElementWidth, NumRegs> {
  let Name = "SVEPredicateListMul" # NumRegs # "x" # ElementWidth;
  let DiagnosticType = "Invalid" # Name;
  let PredicateMethod =
      "isTypedVectorListMultiple<RegKind::SVEPredicateVector, " # NumRegs # ", 0, "
                                                                # ElementWidth #
                                                                ", AArch64::PPRMul2RegClassID>";
}

let EncoderMethod = "EncodeRegMul_MinMax<2, 0, 14>",
    DecoderMethod = "DecodePPR2Mul2RegisterClass" in {
  def PP_b_mul_r : RegisterOperand<PPR2Mul2, "printTypedVectorList<0,'b'>"> {
    let ParserMatchClass = PPRVectorListMul<8, 2>;
  }

  def PP_h_mul_r : RegisterOperand<PPR2Mul2, "printTypedVectorList<0,'h'>"> {
    let ParserMatchClass = PPRVectorListMul<16, 2>;
  }

  def PP_s_mul_r : RegisterOperand<PPR2Mul2, "printTypedVectorList<0,'s'>"> {
    let ParserMatchClass = PPRVectorListMul<32, 2>;
  }

  def PP_d_mul_r : RegisterOperand<PPR2Mul2, "printTypedVectorList<0,'d'>"> {
    let ParserMatchClass = PPRVectorListMul<64, 2>;
  }
}  // end let EncoderMethod/DecoderMethod


//===----------------------------------------------------------------------===//
// SVE vector register classes
class ZPRClass<int firstreg, int lastreg, int step = 1> : RegisterClass<"AArch64",
                                            [nxv16i8, nxv8i16, nxv4i32, nxv2i64,
                                             nxv2f16, nxv4f16, nxv8f16,
                                             nxv2bf16, nxv4bf16, nxv8bf16,
                                             nxv2f32, nxv4f32,
                                             nxv2f64],
                                            128, (sequence "Z%u", firstreg, lastreg, step)> {
  let Size = 128;
}

def ZPRMul2    : ZPRClass<0,  30, 2>;
def ZPRMul4    : ZPRClass<0,  28, 4>;
def ZPRMul2_Lo : ZPRClass<0,  14, 2>;
def ZPRMul2_Hi : ZPRClass<16, 30, 2>;

def ZPR    : ZPRClass<0, 31> {
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPRRegClassID, 0, 32>";
}
def ZPR_4b : ZPRClass<0, 15> { // Restricted 4 bit SVE vector register class.
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPRRegClassID, 0, 16>";
}
def ZPR_3b : ZPRClass<0, 7> {  // Restricted 3 bit SVE vector register class.
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPRRegClassID, 0, 8>";
}

class ZPRAsmOperand<string name, int Width, string RegClassSuffix = "">
    : AsmOperandClass {
  let Name = "SVE" # name # "Reg";
  let PredicateMethod = "isSVEDataVectorRegOfWidth<"
                            # Width # ", AArch64::ZPR"
                            # RegClassSuffix # "RegClassID>";
  let RenderMethod = "addRegOperands";
  let DiagnosticType = "InvalidZPR" # RegClassSuffix # Width;
  let ParserMethod = "tryParseSVEDataVector<false, "
                               # !if(!eq(Width, 0), "false", "true") # ">";
}

def ZPRAsmOpAny : ZPRAsmOperand<"VectorAny", 0>;
def ZPRAsmOp8   : ZPRAsmOperand<"VectorB",   8>;
def ZPRAsmOp16  : ZPRAsmOperand<"VectorH",   16>;
def ZPRAsmOp32  : ZPRAsmOperand<"VectorS",   32>;
def ZPRAsmOp64  : ZPRAsmOperand<"VectorD",   64>;
def ZPRAsmOp128 : ZPRAsmOperand<"VectorQ",   128>;

def ZPRAny  : ZPRRegOp<"",  ZPRAsmOpAny, ElementSizeNone, ZPR>;
def ZPR8    : ZPRRegOp<"b", ZPRAsmOp8,   ElementSizeB, ZPR>;
def ZPR16   : ZPRRegOp<"h", ZPRAsmOp16,  ElementSizeH, ZPR>;
def ZPR32   : ZPRRegOp<"s", ZPRAsmOp32,  ElementSizeS, ZPR>;
def ZPR64   : ZPRRegOp<"d", ZPRAsmOp64,  ElementSizeD, ZPR>;
def ZPR128  : ZPRRegOp<"q", ZPRAsmOp128, ElementSizeQ, ZPR>;

def ZPRAsmOp3b8   : ZPRAsmOperand<"Vector3bB", 8, "_3b">;
def ZPRAsmOp3b16  : ZPRAsmOperand<"Vector3bH", 16, "_3b">;
def ZPRAsmOp3b32  : ZPRAsmOperand<"Vector3bS", 32, "_3b">;

def ZPR3b8  : ZPRRegOp<"b", ZPRAsmOp3b8,  ElementSizeB, ZPR_3b>;
def ZPR3b16 : ZPRRegOp<"h", ZPRAsmOp3b16, ElementSizeH, ZPR_3b>;
def ZPR3b32 : ZPRRegOp<"s", ZPRAsmOp3b32, ElementSizeS, ZPR_3b>;

def ZPRAsmOp4b8   : ZPRAsmOperand<"Vector4bB", 8,  "_4b">;
def ZPRAsmOp4b16  : ZPRAsmOperand<"Vector4bH", 16, "_4b">;
def ZPRAsmOp4b32  : ZPRAsmOperand<"Vector4bS", 32, "_4b">;
def ZPRAsmOp4b64  : ZPRAsmOperand<"Vector4bD", 64, "_4b">;

def ZPR4b8  : ZPRRegOp<"b", ZPRAsmOp4b8,  ElementSizeB, ZPR_4b>;
def ZPR4b16 : ZPRRegOp<"h", ZPRAsmOp4b16, ElementSizeH, ZPR_4b>;
def ZPR4b32 : ZPRRegOp<"s", ZPRAsmOp4b32, ElementSizeS, ZPR_4b>;
def ZPR4b64 : ZPRRegOp<"d", ZPRAsmOp4b64, ElementSizeD, ZPR_4b>;

class ZPRMul2_MinToMaxRegOp<string Suffix, AsmOperandClass C, int  Min, int Max, ElementSizeEnum  Width, RegisterClass RC>
    : ZPRRegOp<Suffix, C, Width, RC> {
  let EncoderMethod = "EncodeRegMul_MinMax<2," # Min # ", " # Max # ">";
  let DecoderMethod = "DecodeZPRMul2_MinMax<" # Min # ", " # Max # ">";
}

def ZPRMul2AsmOp8_Lo  : ZPRAsmOperand<"VectorB_Lo",  8, "Mul2_Lo">;
def ZPRMul2AsmOp8_Hi  : ZPRAsmOperand<"VectorB_Hi",  8, "Mul2_Hi">;
def ZPRMul2AsmOp16_Lo : ZPRAsmOperand<"VectorH_Lo", 16, "Mul2_Lo">;
def ZPRMul2AsmOp16_Hi : ZPRAsmOperand<"VectorH_Hi", 16, "Mul2_Hi">;
def ZPRMul2AsmOp32_Lo : ZPRAsmOperand<"VectorS_Lo", 32, "Mul2_Lo">;
def ZPRMul2AsmOp32_Hi : ZPRAsmOperand<"VectorS_Hi", 32, "Mul2_Hi">;
def ZPRMul2AsmOp64_Lo : ZPRAsmOperand<"VectorD_Lo", 64, "Mul2_Lo">;
def ZPRMul2AsmOp64_Hi : ZPRAsmOperand<"VectorD_Hi", 64, "Mul2_Hi">;

def ZPR_K : RegisterClass<"AArch64", [untyped], 128,
                          (add Z20, Z21, Z22, Z23, Z28, Z29, Z30, Z31)>;

def ZK : RegisterOperand<ZPR_K, "printSVERegOp<>">{
  let EncoderMethod = "EncodeZK";
  let DecoderMethod = "DecodeZK";
  let ParserMatchClass = ZPRAsmOperand<"Vector_20to23or28to31", 0, "_K">;
}

def ZPR8Mul2_Lo  : ZPRMul2_MinToMaxRegOp<"b", ZPRMul2AsmOp8_Lo,   0, 14, ElementSizeB, ZPRMul2_Lo>;
def ZPR8Mul2_Hi  : ZPRMul2_MinToMaxRegOp<"b", ZPRMul2AsmOp8_Hi,  16, 30, ElementSizeB, ZPRMul2_Hi>;
def ZPR16Mul2_Lo : ZPRMul2_MinToMaxRegOp<"h", ZPRMul2AsmOp16_Lo,  0, 14, ElementSizeH, ZPRMul2_Lo>;
def ZPR16Mul2_Hi : ZPRMul2_MinToMaxRegOp<"h", ZPRMul2AsmOp16_Hi, 16, 30, ElementSizeH, ZPRMul2_Hi>;
def ZPR32Mul2_Lo : ZPRMul2_MinToMaxRegOp<"s", ZPRMul2AsmOp32_Lo,  0, 14, ElementSizeS, ZPRMul2_Lo>;
def ZPR32Mul2_Hi : ZPRMul2_MinToMaxRegOp<"s", ZPRMul2AsmOp32_Hi, 16, 30, ElementSizeS, ZPRMul2_Hi>;
def ZPR64Mul2_Lo : ZPRMul2_MinToMaxRegOp<"d", ZPRMul2AsmOp64_Lo,  0, 14, ElementSizeD, ZPRMul2_Lo>;
def ZPR64Mul2_Hi : ZPRMul2_MinToMaxRegOp<"d", ZPRMul2AsmOp64_Hi, 16, 30, ElementSizeD, ZPRMul2_Hi>;

class FPRasZPR<int Width> : AsmOperandClass{
  let Name = "FPR" # Width # "asZPR";
  let PredicateMethod = "isFPRasZPR<AArch64::FPR" # Width # "RegClassID>";
  let RenderMethod = "addFPRasZPRRegOperands<" # Width # ">";
}

class FPRasZPROperand<int Width> : RegisterOperand<ZPR> {
  let ParserMatchClass = FPRasZPR<Width>;
  let PrintMethod = "printZPRasFPR<" # Width # ">";
}

def FPR8asZPR   : FPRasZPROperand<8>;
def FPR16asZPR  : FPRasZPROperand<16>;
def FPR32asZPR  : FPRasZPROperand<32>;
def FPR64asZPR  : FPRasZPROperand<64>;
def FPR128asZPR : FPRasZPROperand<128>;

// Pairs, triples, and quads of SVE vector registers.
def ZSeqPairs   : RegisterTuples<[zsub0, zsub1], [(rotl ZPR, 0), (rotl ZPR, 1)]>;
def ZSeqTriples : RegisterTuples<[zsub0, zsub1, zsub2], [(rotl ZPR, 0), (rotl ZPR, 1), (rotl ZPR, 2)]>;
def ZSeqQuads   : RegisterTuples<[zsub0, zsub1, zsub2, zsub3], [(rotl ZPR, 0), (rotl ZPR, 1), (rotl ZPR, 2), (rotl ZPR, 3)]>;

def ZPR2   : RegisterClass<"AArch64", [untyped], 128, (add ZSeqPairs)>  {
  let Size = 256;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPR2RegClassID, 0, 32>";
}
def ZPR3  : RegisterClass<"AArch64", [untyped], 128, (add ZSeqTriples)> {
  let Size = 384;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPR3RegClassID, 0, 32>";
}
def ZPR4 : RegisterClass<"AArch64", [untyped], 128, (add ZSeqQuads)> {
  let Size = 512;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPR4RegClassID, 0, 32>";
}

class ZPRVectorList<int ElementWidth, int NumRegs> : AsmOperandClass {
  let Name = "SVEVectorList" # NumRegs # ElementWidth;
  let ParserMethod = "tryParseVectorList<RegKind::SVEDataVector>";
  let PredicateMethod =
      "isTypedVectorList<RegKind::SVEDataVector, " #NumRegs #", 0, " #ElementWidth #">";
  let RenderMethod = "addVectorListOperands<AArch64Operand::VecListIdx_ZReg, " # NumRegs # ">";
}

def Z_b  : RegisterOperand<ZPR,  "printTypedVectorList<0,'b'>"> {
  let ParserMatchClass = ZPRVectorList<8, 1>;
}

def Z_h  : RegisterOperand<ZPR,  "printTypedVectorList<0,'h'>"> {
  let ParserMatchClass = ZPRVectorList<16, 1>;
}

def Z_s  : RegisterOperand<ZPR,  "printTypedVectorList<0,'s'>"> {
  let ParserMatchClass = ZPRVectorList<32, 1>;
}

def Z_d  : RegisterOperand<ZPR,  "printTypedVectorList<0,'d'>"> {
  let ParserMatchClass = ZPRVectorList<64, 1>;
}

def Z_q  : RegisterOperand<ZPR,  "printTypedVectorList<0,'q'>"> {
  let ParserMatchClass = ZPRVectorList<128, 1>;
}

def ZZ_b  : RegisterOperand<ZPR2, "printTypedVectorList<0,'b'>"> {
  let ParserMatchClass = ZPRVectorList<8, 2>;
}

def ZZ_h  : RegisterOperand<ZPR2, "printTypedVectorList<0,'h'>"> {
  let ParserMatchClass = ZPRVectorList<16, 2>;
}

def ZZ_s  : RegisterOperand<ZPR2, "printTypedVectorList<0,'s'>"> {
  let ParserMatchClass = ZPRVectorList<32, 2>;
}

def ZZ_d  : RegisterOperand<ZPR2, "printTypedVectorList<0,'d'>"> {
  let ParserMatchClass = ZPRVectorList<64, 2>;
}

def ZZ_q  : RegisterOperand<ZPR2, "printTypedVectorList<0,'q'>"> {
  let ParserMatchClass = ZPRVectorList<128, 2>;
}

def ZZZ_b  : RegisterOperand<ZPR3, "printTypedVectorList<0,'b'>"> {
  let ParserMatchClass = ZPRVectorList<8, 3>;
}

def ZZZ_h  : RegisterOperand<ZPR3, "printTypedVectorList<0,'h'>"> {
  let ParserMatchClass = ZPRVectorList<16, 3>;
}

def ZZZ_s  : RegisterOperand<ZPR3, "printTypedVectorList<0,'s'>"> {
  let ParserMatchClass = ZPRVectorList<32, 3>;
}

def ZZZ_d  : RegisterOperand<ZPR3, "printTypedVectorList<0,'d'>"> {
  let ParserMatchClass = ZPRVectorList<64, 3>;
}

def ZZZ_q  : RegisterOperand<ZPR3, "printTypedVectorList<0,'q'>"> {
  let ParserMatchClass = ZPRVectorList<128, 3>;
}

def ZZZZ_b : RegisterOperand<ZPR4, "printTypedVectorList<0,'b'>"> {
  let ParserMatchClass = ZPRVectorList<8, 4>;
}

def ZZZZ_h : RegisterOperand<ZPR4, "printTypedVectorList<0,'h'>"> {
  let ParserMatchClass = ZPRVectorList<16, 4>;
}

def ZZZZ_s : RegisterOperand<ZPR4, "printTypedVectorList<0,'s'>"> {
  let ParserMatchClass = ZPRVectorList<32, 4>;
}

def ZZZZ_d : RegisterOperand<ZPR4, "printTypedVectorList<0,'d'>"> {
  let ParserMatchClass = ZPRVectorList<64, 4>;
}

def ZZZZ_q : RegisterOperand<ZPR4, "printTypedVectorList<0,'q'>"> {
  let ParserMatchClass = ZPRVectorList<128, 4>;
}

// SME2 multiple-of-2 or 4 multi-vector operands
def ZPR2Mul2 : RegisterClass<"AArch64", [untyped], 128, (add (decimate ZSeqPairs, 2))> {
  let Size = 256;
}

def ZPR4Mul4 : RegisterClass<"AArch64", [untyped], 128, (add (decimate ZSeqQuads, 4))> {
  let Size = 512;
}

class ZPRVectorListMul<int ElementWidth, int NumRegs, string RegClassSuffix = "">
  : ZPRVectorList<ElementWidth, NumRegs> {
  let Name = "SVEVectorList" # NumRegs # "x" # ElementWidth # RegClassSuffix;
  let DiagnosticType = "Invalid" # Name;
  let PredicateMethod =
      "isTypedVectorListMultiple<RegKind::SVEDataVector, "
                                 # NumRegs # ", 0, "
                                 # ElementWidth #  ", "
                                 # "AArch64::ZPR" # RegClassSuffix # "RegClassID" # ">";
}

let EncoderMethod = "EncodeRegMul_MinMax<2, 0, 30>",
    DecoderMethod = "DecodeZPR2Mul2RegisterClass<0, 30>" in {
  def ZZ_mul_r : RegisterOperand<ZPR2Mul2, "printTypedVectorList<0,0>"> {
    let ParserMatchClass = ZPRVectorListMul<0, 2, "Mul2">;
  }

  def ZZ_b_mul_r : RegisterOperand<ZPR2Mul2, "printTypedVectorList<0,'b'>"> {
    let ParserMatchClass = ZPRVectorListMul<8, 2, "Mul2">;
  }

  def ZZ_h_mul_r : RegisterOperand<ZPR2Mul2, "printTypedVectorList<0,'h'>"> {
    let ParserMatchClass = ZPRVectorListMul<16, 2, "Mul2">;
  }

  def ZZ_s_mul_r : RegisterOperand<ZPR2Mul2, "printTypedVectorList<0,'s'>"> {
    let ParserMatchClass = ZPRVectorListMul<32, 2, "Mul2">;
  }

  def ZZ_d_mul_r : RegisterOperand<ZPR2Mul2, "printTypedVectorList<0,'d'>"> {
    let ParserMatchClass = ZPRVectorListMul<64, 2, "Mul2">;
  }

  def ZZ_q_mul_r : RegisterOperand<ZPR2Mul2, "printTypedVectorList<0,'q'>"> {
    let ParserMatchClass = ZPRVectorListMul<128, 2, "Mul2">;
  }
} // end let EncoderMethod/DecoderMethod

let EncoderMethod = "EncodeRegMul_MinMax<4, 0, 28>",
    DecoderMethod = "DecodeZPR4Mul4RegisterClass" in {
  def ZZZZ_b_mul_r : RegisterOperand<ZPR4Mul4, "printTypedVectorList<0,'b'>"> {
    let ParserMatchClass = ZPRVectorListMul<8, 4, "Mul4">;
  }

  def ZZZZ_h_mul_r : RegisterOperand<ZPR4Mul4, "printTypedVectorList<0,'h'>"> {
    let ParserMatchClass = ZPRVectorListMul<16, 4, "Mul4">;
  }

  def ZZZZ_s_mul_r : RegisterOperand<ZPR4Mul4, "printTypedVectorList<0,'s'>"> {
    let ParserMatchClass = ZPRVectorListMul<32, 4, "Mul4">;
  }

  def ZZZZ_d_mul_r : RegisterOperand<ZPR4Mul4, "printTypedVectorList<0,'d'>"> {
    let ParserMatchClass = ZPRVectorListMul<64, 4, "Mul4">;
  }

  def ZZZZ_q_mul_r : RegisterOperand<ZPR4Mul4, "printTypedVectorList<0,'q'>"> {
    let ParserMatchClass = ZPRVectorListMul<128, 4, "Mul4">;
  }
} // end let EncoderMethod/DecoderMethod

// Pairs of consecutive ZPR, starting with an even register, split into
// Lo=0-14 and Hi=16-30.
def ZPR2Mul2_Lo : RegisterClass<"AArch64", [untyped], 128,
                                (trunc (decimate ZSeqPairs, 2), 8)>  {
  let Size = 256;
}

def ZPR2Mul2_Hi : RegisterClass<"AArch64", [untyped], 128,
                                (trunc (rotr (decimate ZSeqPairs, 2), 8), 8)>  {
  let Size = 256;
}

let EncoderMethod = "EncodeRegMul_MinMax<2, 0, 14>",
    DecoderMethod = "DecodeZPR2Mul2RegisterClass<0, 16>" in {
  def ZZ_b_mul_r_Lo : RegisterOperand<ZPR2Mul2_Lo, "printTypedVectorList<0,'b'>"> {
    let ParserMatchClass = ZPRVectorListMul<8, 2, "Mul2_Lo">;
  }

  def ZZ_h_mul_r_Lo : RegisterOperand<ZPR2Mul2_Lo, "printTypedVectorList<0,'h'>"> {
    let ParserMatchClass = ZPRVectorListMul<16, 2, "Mul2_Lo">;
  }

  def ZZ_s_mul_r_Lo : RegisterOperand<ZPR2Mul2_Lo, "printTypedVectorList<0,'s'>"> {
    let ParserMatchClass = ZPRVectorListMul<32, 2, "Mul2_Lo">;
  }

  def ZZ_d_mul_r_Lo : RegisterOperand<ZPR2Mul2_Lo, "printTypedVectorList<0,'d'>"> {
    let ParserMatchClass = ZPRVectorListMul<64, 2, "Mul2_Lo">;
  }
}

let EncoderMethod = "EncodeRegMul_MinMax<2, 16, 30>",
    DecoderMethod = "DecodeZPR2Mul2RegisterClass<16, 31>" in {
  def ZZ_b_mul_r_Hi : RegisterOperand<ZPR2Mul2_Hi, "printTypedVectorList<0,'b'>"> {
    let ParserMatchClass = ZPRVectorListMul<8, 2, "Mul2_Hi">;
  }

  def ZZ_h_mul_r_Hi : RegisterOperand<ZPR2Mul2_Hi, "printTypedVectorList<0,'h'>"> {
    let ParserMatchClass = ZPRVectorListMul<16, 2, "Mul2_Hi">;
  }

  def ZZ_s_mul_r_Hi : RegisterOperand<ZPR2Mul2_Hi, "printTypedVectorList<0,'s'>"> {
    let ParserMatchClass = ZPRVectorListMul<32, 2, "Mul2_Hi">;
  }

  def ZZ_d_mul_r_Hi : RegisterOperand<ZPR2Mul2_Hi, "printTypedVectorList<0,'d'>"> {
    let ParserMatchClass = ZPRVectorListMul<64, 2, "Mul2_Hi">;
  }
 } // end let EncoderMethod/DecoderMethod

// SME2 strided multi-vector operands

// ZStridedPairs
//
// A group of two Z vectors with strided numbering consisting of:
//   Zn+0.T and Zn+8.T
// where n is in the range 0 to 7 and 16 to 23 inclusive, and T is one of B, H,
// S, or D.

// Z0_Z8, Z1_Z9, Z2_Z10, Z3_Z11, Z4_Z12, Z5_Z13, Z6_Z14, Z7_Z15
def ZStridedPairsLo : RegisterTuples<[zsub0, zsub1], [
  (trunc (rotl ZPR, 0), 8), (trunc (rotl ZPR, 8), 8)
]>;

// Z16_Z24, Z17_Z25, Z18_Z26, Z19_Z27, Z20_Z28, Z21_Z29, Z22_Z30, Z23_Z31
def ZStridedPairsHi : RegisterTuples<[zsub0, zsub1], [
  (trunc (rotl ZPR, 16), 8), (trunc (rotl ZPR, 24), 8)
]>;

// ZStridedQuads
//
// A group of four Z vectors with strided numbering consisting of:
//   Zn+0.T, Zn+4.T, Zn+8.T and Zn+12.T
// where n is in the range 0 to 3 and 16 to 19 inclusive, and T is one of B, H,
// S, or D.

// Z0_Z4_Z8_Z12, Z1_Z5_Z9_Z13, Z2_Z6_Z10_Z14, Z3_Z7_Z11_Z15
def ZStridedQuadsLo : RegisterTuples<[zsub0, zsub1, zsub2, zsub3], [
  (trunc (rotl ZPR, 0), 4), (trunc (rotl ZPR, 4), 4),
  (trunc (rotl ZPR, 8), 4), (trunc (rotl ZPR, 12), 4)
]>;
// Z16_Z20_Z24_Z28, Z17_Z21_Z25_Z29, Z18_Z22_Z26_Z30, Z19_Z23_Z27_Z31
def ZStridedQuadsHi : RegisterTuples<[zsub0, zsub1, zsub2, zsub3], [
  (trunc (rotl ZPR, 16), 4), (trunc (rotl ZPR, 20), 4),
  (trunc (rotl ZPR, 24), 4), (trunc (rotl ZPR, 28), 4)
]>;

def ZPR2Strided : RegisterClass<"AArch64", [untyped], 128,
                                (add ZStridedPairsLo, ZStridedPairsHi)>  {
  let Size = 256;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPR2StridedRegClassID, 0, 16>";
}
def ZPR4Strided : RegisterClass<"AArch64", [untyped], 128,
                                (add ZStridedQuadsLo, ZStridedQuadsHi)>  {
  let Size = 512;
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPR4StridedRegClassID, 0, 8>";
}

def ZPR2StridedOrContiguous : RegisterClass<"AArch64", [untyped], 128,
                                (add ZStridedPairsLo, ZStridedPairsHi,
                                (decimate ZSeqPairs, 2))> {
  let Size = 256;
}

class ZPRVectorListStrided<int ElementWidth, int NumRegs, int Stride>
    : ZPRVectorList<ElementWidth, NumRegs> {
  let Name = "SVEVectorListStrided" # NumRegs # "x" # ElementWidth;
  let DiagnosticType = "Invalid" # Name;
  let PredicateMethod = "isTypedVectorListStrided<RegKind::SVEDataVector, "
                        # NumRegs # "," # Stride # "," # ElementWidth # ">";
  let RenderMethod = "addStridedVectorListOperands<" # NumRegs # ">";
}

let EncoderMethod = "EncodeZPR2StridedRegisterClass",
    DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPR2StridedRegClassID, 0, 16>" in {
  def ZZ_b_strided
      : RegisterOperand<ZPR2Strided, "printTypedVectorList<0, 'b'>"> {
    let ParserMatchClass = ZPRVectorListStrided<8, 2, 8>;
  }

  def ZZ_h_strided
      : RegisterOperand<ZPR2Strided, "printTypedVectorList<0, 'h'>"> {
    let ParserMatchClass = ZPRVectorListStrided<16, 2, 8>;
  }

  def ZZ_s_strided
      : RegisterOperand<ZPR2Strided, "printTypedVectorList<0,'s'>"> {
    let ParserMatchClass = ZPRVectorListStrided<32, 2, 8>;
  }

  def ZZ_d_strided
      : RegisterOperand<ZPR2Strided, "printTypedVectorList<0,'d'>"> {
    let ParserMatchClass = ZPRVectorListStrided<64, 2, 8>;
  }

  def ZZ_b_strided_and_contiguous
      : RegisterOperand<ZPR2StridedOrContiguous, "printTypedVectorList<0,'b'>">;
  def ZZ_h_strided_and_contiguous
      : RegisterOperand<ZPR2StridedOrContiguous, "printTypedVectorList<0,'h'>">;
  def ZZ_s_strided_and_contiguous
      : RegisterOperand<ZPR2StridedOrContiguous, "printTypedVectorList<0,'s'>">;
  def ZZ_d_strided_and_contiguous
      : RegisterOperand<ZPR2StridedOrContiguous, "printTypedVectorList<0,'d'>">;
}

def ZPR4StridedOrContiguous : RegisterClass<"AArch64", [untyped], 128,
                                (add ZStridedQuadsLo, ZStridedQuadsHi,
                                (decimate ZSeqQuads, 4))>  {
  let Size = 512;
}

let EncoderMethod = "EncodeZPR4StridedRegisterClass",
    DecoderMethod = "DecodeSimpleRegisterClass<AArch64::ZPR4StridedRegClassID, 0, 16>" in {
  def ZZZZ_b_strided
      : RegisterOperand<ZPR4Strided, "printTypedVectorList<0,'b'>"> {
    let ParserMatchClass = ZPRVectorListStrided<8, 4, 4>;
  }

  def ZZZZ_h_strided
      : RegisterOperand<ZPR4Strided, "printTypedVectorList<0,'h'>"> {
    let ParserMatchClass = ZPRVectorListStrided<16, 4, 4>;
  }

  def ZZZZ_s_strided
      : RegisterOperand<ZPR4Strided, "printTypedVectorList<0,'s'>"> {
    let ParserMatchClass = ZPRVectorListStrided<32, 4, 4>;
  }

  def ZZZZ_d_strided
      : RegisterOperand<ZPR4Strided, "printTypedVectorList<0,'d'>"> {
    let ParserMatchClass = ZPRVectorListStrided<64, 4, 4>;
  }

  def ZZZZ_b_strided_and_contiguous
      : RegisterOperand<ZPR4StridedOrContiguous, "printTypedVectorList<0,'b'>">;
  def ZZZZ_h_strided_and_contiguous
      : RegisterOperand<ZPR4StridedOrContiguous, "printTypedVectorList<0,'h'>">;
  def ZZZZ_s_strided_and_contiguous
      : RegisterOperand<ZPR4StridedOrContiguous, "printTypedVectorList<0,'s'>">;
  def ZZZZ_d_strided_and_contiguous
      : RegisterOperand<ZPR4StridedOrContiguous, "printTypedVectorList<0,'d'>">;
}

class ZPRExtendAsmOperand<string ShiftExtend, int RegWidth, int Scale,
                          bit ScaleAlwaysSame = 0b0> : AsmOperandClass {
  let Name = "ZPRExtend" # ShiftExtend # RegWidth # Scale
                         # !if(ScaleAlwaysSame, "Only", "");

  let PredicateMethod = "isSVEDataVectorRegWithShiftExtend<"
                          # RegWidth # ", AArch64::ZPRRegClassID, "
                          # "AArch64_AM::" # ShiftExtend # ", "
                          # Scale # ", "
                          # !if(ScaleAlwaysSame, "true", "false")
                          # ">";
  let DiagnosticType = "InvalidZPR" # RegWidth # ShiftExtend # Scale;
  let RenderMethod = "addRegOperands";
  let ParserMethod = "tryParseSVEDataVector<true, true>";
}

class ZPRExtendRegisterOperand<bit SignExtend, bit IsLSL, string Repr,
                               int RegWidth, int Scale, string Suffix = "">
    : RegisterOperand<ZPR> {
  let ParserMatchClass =
    !cast<AsmOperandClass>("ZPR" # RegWidth # "AsmOpndExt" # Repr # Scale # Suffix);
  let PrintMethod = "printRegWithShiftExtend<"
                          # !if(SignExtend, "true", "false") # ", "
                          # Scale # ", "
                          # !if(IsLSL, "'x'", "'w'") # ", "
                          # !if(!eq(RegWidth, 32), "'s'", "'d'") # ">";
}

foreach RegWidth = [32, 64] in {
  // UXTW(8|16|32|64)
  def ZPR#RegWidth#AsmOpndExtUXTW8Only : ZPRExtendAsmOperand<"UXTW", RegWidth, 8, 0b1>;
  def ZPR#RegWidth#AsmOpndExtUXTW8     : ZPRExtendAsmOperand<"UXTW", RegWidth, 8>;
  def ZPR#RegWidth#AsmOpndExtUXTW16    : ZPRExtendAsmOperand<"UXTW", RegWidth, 16>;
  def ZPR#RegWidth#AsmOpndExtUXTW32    : ZPRExtendAsmOperand<"UXTW", RegWidth, 32>;
  def ZPR#RegWidth#AsmOpndExtUXTW64    : ZPRExtendAsmOperand<"UXTW", RegWidth, 64>;

  def ZPR#RegWidth#ExtUXTW8Only        : ZPRExtendRegisterOperand<0b0, 0b0, "UXTW", RegWidth, 8, "Only">;
  def ZPR#RegWidth#ExtUXTW8            : ZPRExtendRegisterOperand<0b0, 0b0, "UXTW", RegWidth, 8>;
  def ZPR#RegWidth#ExtUXTW16           : ZPRExtendRegisterOperand<0b0, 0b0, "UXTW", RegWidth, 16>;
  def ZPR#RegWidth#ExtUXTW32           : ZPRExtendRegisterOperand<0b0, 0b0, "UXTW", RegWidth, 32>;
  def ZPR#RegWidth#ExtUXTW64           : ZPRExtendRegisterOperand<0b0, 0b0, "UXTW", RegWidth, 64>;

  // SXTW(8|16|32|64)
  def ZPR#RegWidth#AsmOpndExtSXTW8Only : ZPRExtendAsmOperand<"SXTW", RegWidth, 8, 0b1>;
  def ZPR#RegWidth#AsmOpndExtSXTW8     : ZPRExtendAsmOperand<"SXTW", RegWidth, 8>;
  def ZPR#RegWidth#AsmOpndExtSXTW16    : ZPRExtendAsmOperand<"SXTW", RegWidth, 16>;
  def ZPR#RegWidth#AsmOpndExtSXTW32    : ZPRExtendAsmOperand<"SXTW", RegWidth, 32>;
  def ZPR#RegWidth#AsmOpndExtSXTW64    : ZPRExtendAsmOperand<"SXTW", RegWidth, 64>;

  def ZPR#RegWidth#ExtSXTW8Only        : ZPRExtendRegisterOperand<0b1, 0b0, "SXTW", RegWidth, 8, "Only">;
  def ZPR#RegWidth#ExtSXTW8            : ZPRExtendRegisterOperand<0b1, 0b0, "SXTW", RegWidth, 8>;
  def ZPR#RegWidth#ExtSXTW16           : ZPRExtendRegisterOperand<0b1, 0b0, "SXTW", RegWidth, 16>;
  def ZPR#RegWidth#ExtSXTW32           : ZPRExtendRegisterOperand<0b1, 0b0, "SXTW", RegWidth, 32>;
  def ZPR#RegWidth#ExtSXTW64           : ZPRExtendRegisterOperand<0b1, 0b0, "SXTW", RegWidth, 64>;

  // LSL(8|16|32|64)
  def ZPR#RegWidth#AsmOpndExtLSL8      : ZPRExtendAsmOperand<"LSL", RegWidth, 8>;
  def ZPR#RegWidth#AsmOpndExtLSL16     : ZPRExtendAsmOperand<"LSL", RegWidth, 16>;
  def ZPR#RegWidth#AsmOpndExtLSL32     : ZPRExtendAsmOperand<"LSL", RegWidth, 32>;
  def ZPR#RegWidth#AsmOpndExtLSL64     : ZPRExtendAsmOperand<"LSL", RegWidth, 64>;
  def ZPR#RegWidth#ExtLSL8             : ZPRExtendRegisterOperand<0b0, 0b1, "LSL", RegWidth, 8>;
  def ZPR#RegWidth#ExtLSL16            : ZPRExtendRegisterOperand<0b0, 0b1, "LSL", RegWidth, 16>;
  def ZPR#RegWidth#ExtLSL32            : ZPRExtendRegisterOperand<0b0, 0b1, "LSL", RegWidth, 32>;
  def ZPR#RegWidth#ExtLSL64            : ZPRExtendRegisterOperand<0b0, 0b1, "LSL", RegWidth, 64>;
}

class GPR64ShiftExtendAsmOperand <string AsmOperandName, int Scale, string RegClass> : AsmOperandClass {
  let Name = AsmOperandName # Scale;
  let PredicateMethod = "isGPR64WithShiftExtend<AArch64::"#RegClass#"RegClassID, " # Scale # ">";
  let DiagnosticType = "Invalid" # AsmOperandName # Scale;
  let RenderMethod = "addRegOperands";
  let ParserMethod = "tryParseGPROperand<true>";
}

class GPR64ExtendRegisterOperand<string Name, int Scale, RegisterClass RegClass> : RegisterOperand<RegClass>{
  let ParserMatchClass = !cast<AsmOperandClass>(Name);
  let PrintMethod = "printRegWithShiftExtend<false, " # Scale # ", 'x', 0>";
}

foreach Scale = [8, 16, 32, 64, 128] in {
  def GPR64shiftedAsmOpnd # Scale : GPR64ShiftExtendAsmOperand<"GPR64shifted", Scale, "GPR64">;
  def GPR64shifted # Scale : GPR64ExtendRegisterOperand<"GPR64shiftedAsmOpnd" # Scale, Scale, GPR64>;

  def GPR64NoXZRshiftedAsmOpnd # Scale : GPR64ShiftExtendAsmOperand<"GPR64NoXZRshifted", Scale, "GPR64common">;
  def GPR64NoXZRshifted # Scale : GPR64ExtendRegisterOperand<"GPR64NoXZRshiftedAsmOpnd" # Scale, Scale, GPR64common>;
}

// Accumulator array tiles.
def ZAQ0  : AArch64Reg<0,  "za0.q">;
def ZAQ1  : AArch64Reg<1,  "za1.q">;
def ZAQ2  : AArch64Reg<2,  "za2.q">;
def ZAQ3  : AArch64Reg<3,  "za3.q">;
def ZAQ4  : AArch64Reg<4,  "za4.q">;
def ZAQ5  : AArch64Reg<5,  "za5.q">;
def ZAQ6  : AArch64Reg<6,  "za6.q">;
def ZAQ7  : AArch64Reg<7,  "za7.q">;
def ZAQ8  : AArch64Reg<8,  "za8.q">;
def ZAQ9  : AArch64Reg<9,  "za9.q">;
def ZAQ10 : AArch64Reg<10, "za10.q">;
def ZAQ11 : AArch64Reg<11, "za11.q">;
def ZAQ12 : AArch64Reg<12, "za12.q">;
def ZAQ13 : AArch64Reg<13, "za13.q">;
def ZAQ14 : AArch64Reg<14, "za14.q">;
def ZAQ15 : AArch64Reg<15, "za15.q">;

let SubRegIndices = [zasubq0, zasubq1] in {
  def ZAD0 : AArch64Reg<0, "za0.d", [ZAQ0, ZAQ8]>;
  def ZAD1 : AArch64Reg<1, "za1.d", [ZAQ1, ZAQ9]>;
  def ZAD2 : AArch64Reg<2, "za2.d", [ZAQ2, ZAQ10]>;
  def ZAD3 : AArch64Reg<3, "za3.d", [ZAQ3, ZAQ11]>;
  def ZAD4 : AArch64Reg<4, "za4.d", [ZAQ4, ZAQ12]>;
  def ZAD5 : AArch64Reg<5, "za5.d", [ZAQ5, ZAQ13]>;
  def ZAD6 : AArch64Reg<6, "za6.d", [ZAQ6, ZAQ14]>;
  def ZAD7 : AArch64Reg<7, "za7.d", [ZAQ7, ZAQ15]>;
}

let SubRegIndices = [zasubd0, zasubd1] in {
  def ZAS0 : AArch64Reg<0, "za0.s", [ZAD0, ZAD4]>;
  def ZAS1 : AArch64Reg<1, "za1.s", [ZAD1, ZAD5]>;
  def ZAS2 : AArch64Reg<2, "za2.s", [ZAD2, ZAD6]>;
  def ZAS3 : AArch64Reg<3, "za3.s", [ZAD3, ZAD7]>;
}

let SubRegIndices = [zasubs0, zasubs1] in {
  def ZAH0 : AArch64Reg<0, "za0.h", [ZAS0, ZAS2]>;
  def ZAH1 : AArch64Reg<1, "za1.h", [ZAS1, ZAS3]>;
}

let SubRegIndices = [zasubh0, zasubh1] in {
  def ZAB0 : AArch64Reg<0, "za0.b", [ZAH0, ZAH1]>;
}

let SubRegIndices = [zasubb] in {
  def ZA : AArch64Reg<0, "za", [ZAB0]>;
}

def ZT0 : AArch64Reg<0, "zt0">;

// SME Register Classes

let isAllocatable = 0 in {
  // Accumulator array
  def MPR : RegisterClass<"AArch64", [untyped], 2048, (add ZA)> {
    let Size = 2048;
  }

  // Accumulator array as single tiles
  def MPR8    : RegisterClass<"AArch64", [untyped], 2048, (add (sequence "ZAB%u", 0, 0))> {
    let Size = 2048;
  }
  def MPR16   : RegisterClass<"AArch64", [untyped], 1024, (add (sequence "ZAH%u", 0, 1))> {
    let Size = 1024;
  }
  def MPR32   : RegisterClass<"AArch64", [untyped],  512, (add (sequence "ZAS%u", 0, 3))> {
    let Size = 512;
  }
  def MPR64   : RegisterClass<"AArch64", [untyped],  256, (add (sequence "ZAD%u", 0, 7))> {
    let Size = 256;
  }
  def MPR128  : RegisterClass<"AArch64", [untyped],  128, (add (sequence "ZAQ%u", 0, 15))> {
    let Size = 128;
  }
}

def ZTR : RegisterClass<"AArch64", [untyped], 512, (add ZT0)> {
  let Size = 512;
  let DiagnosticType = "InvalidLookupTable";
}
// SME Register Operands
// There are three types of SME matrix register operands:
// * Tiles:
//
//   These tiles make up the larger accumulator matrix. The tile representation
//   has an element type suffix, e.g. za0.b or za15.q and can be any of the
//   registers:
//          ZAQ0..ZAQ15
//          ZAD0..ZAD7
//          ZAS0..ZAS3
//          ZAH0..ZAH1
//       or ZAB0
//
// * Tile vectors:
//
//   Their representation is similar to regular tiles, but they have an extra
//   'h' or 'v' to tell how the vector at [reg+offset] is layed out in the tile,
//   horizontally or vertically.
//
//   e.g. za1h.h or za15v.q, which corresponds to vectors in registers ZAH1 and
//   ZAQ15, respectively. The horizontal/vertical is more a property of the
//   instruction, than a property of the asm-operand itself, or its register.
//   The distinction is required for the parsing/printing of the operand,
//   as from a compiler's perspective, the whole tile is read/written.
//
// * Accumulator matrix:
//
//   This is the entire matrix accumulator register ZA (<=> ZAB0), printed as
//   'za'.

//
// Tiles
//

class MatrixTileAsmOperand<string RC, int EltSize> : AsmOperandClass {
  let Name = "MatrixTile" # EltSize;
  let DiagnosticType = "Invalid" # Name;
  let ParserMethod = "tryParseMatrixRegister";
  let RenderMethod = "addMatrixOperands";
  let PredicateMethod = "isMatrixRegOperand<"
                          # "MatrixKind::Tile" # ", "
                          # EltSize # ", AArch64::" # RC # "RegClassID>";
}

class MatrixTileOperand<int EltSize, int NumBitsForTile, RegisterClass RC>
    : RegisterOperand<RC> {
  let ParserMatchClass = MatrixTileAsmOperand<!cast<string>(RC), EltSize>;
  let DecoderMethod = "DecodeMatrixTile<" # NumBitsForTile # ">";
  let PrintMethod = "printMatrixTile";
}

def TileOp16  : MatrixTileOperand<16, 1, MPR16>;
def TileOp32  : MatrixTileOperand<32, 2, MPR32>;
def TileOp64  : MatrixTileOperand<64, 3, MPR64>;

//
// Tile vectors (horizontal and vertical)
//

class MatrixTileVectorAsmOperand<string RC, int EltSize, int IsVertical>
    : AsmOperandClass {
  let Name = "MatrixTileVector" # !if(IsVertical, "V", "H") # EltSize;
  let DiagnosticType = "Invalid" # Name;
  let ParserMethod = "tryParseMatrixRegister";
  let RenderMethod = "addMatrixOperands";
  let PredicateMethod = "isMatrixRegOperand<"
                          # "MatrixKind::"
                          # !if(IsVertical, "Col", "Row") # ", "
                          # EltSize # ", AArch64::" # RC # "RegClassID>";
}

class MatrixTileVectorOperand<int EltSize, int NumBitsForTile,
                              RegisterClass RC, int IsVertical>
    : RegisterOperand<RC> {
  let ParserMatchClass = MatrixTileVectorAsmOperand<!cast<string>(RC), EltSize,
                                                    IsVertical>;
  let DecoderMethod = "DecodeMatrixTile<" # NumBitsForTile # ">";
  let PrintMethod = "printMatrixTileVector<" # IsVertical # ">";
}

def TileVectorOpH8   : MatrixTileVectorOperand<  8, 0, MPR8,   0>;
def TileVectorOpH16  : MatrixTileVectorOperand< 16, 1, MPR16,  0>;
def TileVectorOpH32  : MatrixTileVectorOperand< 32, 2, MPR32,  0>;
def TileVectorOpH64  : MatrixTileVectorOperand< 64, 3, MPR64,  0>;
def TileVectorOpH128 : MatrixTileVectorOperand<128, 4, MPR128, 0>;

def TileVectorOpV8   : MatrixTileVectorOperand<  8, 0, MPR8,   1>;
def TileVectorOpV16  : MatrixTileVectorOperand< 16, 1, MPR16,  1>;
def TileVectorOpV32  : MatrixTileVectorOperand< 32, 2, MPR32,  1>;
def TileVectorOpV64  : MatrixTileVectorOperand< 64, 3, MPR64,  1>;
def TileVectorOpV128 : MatrixTileVectorOperand<128, 4, MPR128, 1>;

//
// Accumulator matrix
//

class MatrixAsmOperand<string RC, int EltSize> : AsmOperandClass {
  let Name = "Matrix" # !if(EltSize, !cast<string>(EltSize), "");
  let DiagnosticType = "Invalid" # Name;
  let ParserMethod = "tryParseMatrixRegister";
  let RenderMethod = "addMatrixOperands";
  let PredicateMethod = "isMatrixRegOperand<"
                          # "MatrixKind::Array" # ", "
                          # EltSize # ", AArch64::" # RC # "RegClassID>";
}

class MatrixOperand<RegisterClass RC, int EltSize> : RegisterOperand<RC> {
  let ParserMatchClass = MatrixAsmOperand<!cast<string>(RC), EltSize>;
  let PrintMethod = "printMatrix<" # EltSize # ">";
}

def MatrixOp : MatrixOperand<MPR, 0>;
// SME2 register operands and classes
def MatrixOp8 : MatrixOperand<MPR, 8>;
def MatrixOp16 : MatrixOperand<MPR, 16>;
def MatrixOp32 : MatrixOperand<MPR, 32>;
def MatrixOp64 : MatrixOperand<MPR, 64>;

class MatrixTileListAsmOperand : AsmOperandClass {
  let Name = "MatrixTileList";
  let ParserMethod = "tryParseMatrixTileList";
  let RenderMethod = "addMatrixTileListOperands";
  let PredicateMethod = "isMatrixTileList";
}

class MatrixTileListOperand : Operand<i8> {
  let ParserMatchClass = MatrixTileListAsmOperand<>;
  let DecoderMethod = "DecodeMatrixTileListRegisterClass";
  let EncoderMethod = "EncodeMatrixTileListRegisterClass";
  let PrintMethod = "printMatrixTileList";
}

def MatrixTileList : MatrixTileListOperand<>;

def MatrixIndexGPR32_8_11 : RegisterClass<"AArch64", [i32], 32, (sequence "W%u", 8, 11)> {
   let DiagnosticType = "InvalidMatrixIndexGPR32_8_11";
   let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::MatrixIndexGPR32_8_11RegClassID, 0, 4>";
}
def MatrixIndexGPR32_12_15 : RegisterClass<"AArch64", [i32], 32, (sequence "W%u", 12, 15)> {
  let DiagnosticType = "InvalidMatrixIndexGPR32_12_15";
  let DecoderMethod = "DecodeSimpleRegisterClass<AArch64::MatrixIndexGPR32_12_15RegClassID, 0, 4>";
}
def MatrixIndexGPR32Op8_11 : RegisterOperand<MatrixIndexGPR32_8_11> {
  let EncoderMethod = "encodeMatrixIndexGPR32<AArch64::W8>";
}
def MatrixIndexGPR32Op12_15 : RegisterOperand<MatrixIndexGPR32_12_15> {
  let EncoderMethod = "encodeMatrixIndexGPR32<AArch64::W12>";
}

def SVCROperand : AsmOperandClass {
  let Name = "SVCR";
  let ParserMethod = "tryParseSVCR";
  let DiagnosticType = "Invalid" # Name;
}

def svcr_op : Operand<i32>, TImmLeaf<i32, [{
    return AArch64SVCR::lookupSVCRByEncoding(Imm) != nullptr;
  }]> {
  let ParserMatchClass = SVCROperand;
  let PrintMethod = "printSVCROp";
  let DecoderMethod = "DecodeSVCROp";
  let MCOperandPredicate = [{
    if (!MCOp.isImm())
      return false;
    return AArch64SVCR::lookupSVCRByEncoding(MCOp.getImm()) != nullptr;
  }];
}

let isAllocatable = 0, GeneratePressureSet = 0 in {
  def W_HI_DummyRC : RegisterClass<"AArch64", [untyped], 0, (add (sequence "W%u_HI", 0, 30), WZR_HI, WSP_HI)>;
  def B_HI_DummyRC : RegisterClass<"AArch64", [untyped], 0, (sequence "B%u_HI", 0, 31)>;
  def H_HI_DummyRC : RegisterClass<"AArch64", [untyped], 0, (sequence "H%u_HI", 0, 31)>;
  def S_HI_DummyRC : RegisterClass<"AArch64", [untyped], 0, (sequence "S%u_HI", 0, 31)>;
  def D_HI_DummyRC : RegisterClass<"AArch64", [untyped], 0, (sequence "D%u_HI", 0, 31)>;
  def Q_HI_DummyRC : RegisterClass<"AArch64", [untyped], 0, (sequence "Q%u_HI", 0, 31)>;
}

//===----------------------------------------------------------------------===//
// Register categories.
//

def GeneralPurposeRegisters : RegisterCategory<[GPR64, GPR32]>;

def FIXED_REGS : RegisterClass<"AArch64", [i64], 64, (add FP, SP, VG, FFR)>;
def FixedRegisters : RegisterCategory<[CCR, FIXED_REGS]>;
