//=- AArch64RegisterInfo.td - Describe the AArch64 Regisers --*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  def sub_32 : SubRegIndex<32>;

  def bsub : SubRegIndex<8>;
  def hsub : SubRegIndex<16>;
  def ssub : SubRegIndex<32>;
  def dsub : SubRegIndex<32>;
  def sube32 : SubRegIndex<32>;
  def subo32 : SubRegIndex<32>;
  def qhisub : SubRegIndex<64>;
  def qsub : SubRegIndex<64>;
  def sube64 : SubRegIndex<64>;
  def subo64 : SubRegIndex<64>;
  // Note: Code depends on these having consecutive numbers
  def dsub0 : SubRegIndex<64>;
  def dsub1 : SubRegIndex<64>;
  def dsub2 : SubRegIndex<64>;
  def dsub3 : SubRegIndex<64>;
  // Note: Code depends on these having consecutive numbers
  def qsub0 : SubRegIndex<128>;
  def qsub1 : SubRegIndex<128>;
  def qsub2 : SubRegIndex<128>;
  def qsub3 : SubRegIndex<128>;
}

let Namespace = "AArch64" in {
  def vreg : RegAltNameIndex;
  def vlist1 : RegAltNameIndex;
}

//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//
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
def WZR   : AArch64Reg<31, "wzr">, DwarfRegAlias<WSP>;

let SubRegIndices = [sub_32] in {
def X0    : AArch64Reg<0,   "x0",  [W0]>, DwarfRegAlias<W0>;
def X1    : AArch64Reg<1,   "x1",  [W1]>, DwarfRegAlias<W1>;
def X2    : AArch64Reg<2,   "x2",  [W2]>, DwarfRegAlias<W2>;
def X3    : AArch64Reg<3,   "x3",  [W3]>, DwarfRegAlias<W3>;
def X4    : AArch64Reg<4,   "x4",  [W4]>, DwarfRegAlias<W4>;
def X5    : AArch64Reg<5,   "x5",  [W5]>, DwarfRegAlias<W5>;
def X6    : AArch64Reg<6,   "x6",  [W6]>, DwarfRegAlias<W6>;
def X7    : AArch64Reg<7,   "x7",  [W7]>, DwarfRegAlias<W7>;
def X8    : AArch64Reg<8,   "x8",  [W8]>, DwarfRegAlias<W8>;
def X9    : AArch64Reg<9,   "x9",  [W9]>, DwarfRegAlias<W9>;
def X10   : AArch64Reg<10, "x10", [W10]>, DwarfRegAlias<W10>;
def X11   : AArch64Reg<11, "x11", [W11]>, DwarfRegAlias<W11>;
def X12   : AArch64Reg<12, "x12", [W12]>, DwarfRegAlias<W12>;
def X13   : AArch64Reg<13, "x13", [W13]>, DwarfRegAlias<W13>;
def X14   : AArch64Reg<14, "x14", [W14]>, DwarfRegAlias<W14>;
def X15   : AArch64Reg<15, "x15", [W15]>, DwarfRegAlias<W15>;
def X16   : AArch64Reg<16, "x16", [W16]>, DwarfRegAlias<W16>;
def X17   : AArch64Reg<17, "x17", [W17]>, DwarfRegAlias<W17>;
def X18   : AArch64Reg<18, "x18", [W18]>, DwarfRegAlias<W18>;
def X19   : AArch64Reg<19, "x19", [W19]>, DwarfRegAlias<W19>;
def X20   : AArch64Reg<20, "x20", [W20]>, DwarfRegAlias<W20>;
def X21   : AArch64Reg<21, "x21", [W21]>, DwarfRegAlias<W21>;
def X22   : AArch64Reg<22, "x22", [W22]>, DwarfRegAlias<W22>;
def X23   : AArch64Reg<23, "x23", [W23]>, DwarfRegAlias<W23>;
def X24   : AArch64Reg<24, "x24", [W24]>, DwarfRegAlias<W24>;
def X25   : AArch64Reg<25, "x25", [W25]>, DwarfRegAlias<W25>;
def X26   : AArch64Reg<26, "x26", [W26]>, DwarfRegAlias<W26>;
def X27   : AArch64Reg<27, "x27", [W27]>, DwarfRegAlias<W27>;
def X28   : AArch64Reg<28, "x28", [W28]>, DwarfRegAlias<W28>;
def FP    : AArch64Reg<29, "x29", [W29]>, DwarfRegAlias<W29>;
def LR    : AArch64Reg<30, "x30", [W30]>, DwarfRegAlias<W30>;
def SP    : AArch64Reg<31, "sp",  [WSP]>, DwarfRegAlias<WSP>;
def XZR   : AArch64Reg<31, "xzr", [WZR]>, DwarfRegAlias<WSP>;
}

// Condition code register.
def NZCV  : AArch64Reg<0, "nzcv">;

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
}
// GPR register classes which exclude SP/WSP.
def GPR32 : RegisterClass<"AArch64", [i32], 32, (add GPR32common, WZR)> {
  let AltOrders = [(rotl GPR32, 8)];
  let AltOrderSelect = [{ return 1; }];
}
def GPR64 : RegisterClass<"AArch64", [i64], 64, (add GPR64common, XZR)> {
  let AltOrders = [(rotl GPR64, 8)];
  let AltOrderSelect = [{ return 1; }];
}

// GPR register classes which include SP/WSP.
def GPR32sp : RegisterClass<"AArch64", [i32], 32, (add GPR32common, WSP)> {
  let AltOrders = [(rotl GPR32sp, 8)];
  let AltOrderSelect = [{ return 1; }];
}
def GPR64sp : RegisterClass<"AArch64", [i64], 64, (add GPR64common, SP)> {
  let AltOrders = [(rotl GPR64sp, 8)];
  let AltOrderSelect = [{ return 1; }];
}

def GPR32sponly : RegisterClass<"AArch64", [i32], 32, (add WSP)>;
def GPR64sponly : RegisterClass<"AArch64", [i64], 64, (add SP)>;

def GPR64spPlus0Operand : AsmOperandClass {
  let Name = "GPR64sp0";
  let RenderMethod = "addRegOperands";
  let ParserMethod = "tryParseGPR64sp0Operand";
}

def GPR64sp0 : RegisterOperand<GPR64sp> {
  let ParserMatchClass = GPR64spPlus0Operand;
}

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

let SubRegIndices = [bsub] in {
def H0    : AArch64Reg<0,   "h0", [B0]>, DwarfRegAlias<B0>;
def H1    : AArch64Reg<1,   "h1", [B1]>, DwarfRegAlias<B1>;
def H2    : AArch64Reg<2,   "h2", [B2]>, DwarfRegAlias<B2>;
def H3    : AArch64Reg<3,   "h3", [B3]>, DwarfRegAlias<B3>;
def H4    : AArch64Reg<4,   "h4", [B4]>, DwarfRegAlias<B4>;
def H5    : AArch64Reg<5,   "h5", [B5]>, DwarfRegAlias<B5>;
def H6    : AArch64Reg<6,   "h6", [B6]>, DwarfRegAlias<B6>;
def H7    : AArch64Reg<7,   "h7", [B7]>, DwarfRegAlias<B7>;
def H8    : AArch64Reg<8,   "h8", [B8]>, DwarfRegAlias<B8>;
def H9    : AArch64Reg<9,   "h9", [B9]>, DwarfRegAlias<B9>;
def H10   : AArch64Reg<10, "h10", [B10]>, DwarfRegAlias<B10>;
def H11   : AArch64Reg<11, "h11", [B11]>, DwarfRegAlias<B11>;
def H12   : AArch64Reg<12, "h12", [B12]>, DwarfRegAlias<B12>;
def H13   : AArch64Reg<13, "h13", [B13]>, DwarfRegAlias<B13>;
def H14   : AArch64Reg<14, "h14", [B14]>, DwarfRegAlias<B14>;
def H15   : AArch64Reg<15, "h15", [B15]>, DwarfRegAlias<B15>;
def H16   : AArch64Reg<16, "h16", [B16]>, DwarfRegAlias<B16>;
def H17   : AArch64Reg<17, "h17", [B17]>, DwarfRegAlias<B17>;
def H18   : AArch64Reg<18, "h18", [B18]>, DwarfRegAlias<B18>;
def H19   : AArch64Reg<19, "h19", [B19]>, DwarfRegAlias<B19>;
def H20   : AArch64Reg<20, "h20", [B20]>, DwarfRegAlias<B20>;
def H21   : AArch64Reg<21, "h21", [B21]>, DwarfRegAlias<B21>;
def H22   : AArch64Reg<22, "h22", [B22]>, DwarfRegAlias<B22>;
def H23   : AArch64Reg<23, "h23", [B23]>, DwarfRegAlias<B23>;
def H24   : AArch64Reg<24, "h24", [B24]>, DwarfRegAlias<B24>;
def H25   : AArch64Reg<25, "h25", [B25]>, DwarfRegAlias<B25>;
def H26   : AArch64Reg<26, "h26", [B26]>, DwarfRegAlias<B26>;
def H27   : AArch64Reg<27, "h27", [B27]>, DwarfRegAlias<B27>;
def H28   : AArch64Reg<28, "h28", [B28]>, DwarfRegAlias<B28>;
def H29   : AArch64Reg<29, "h29", [B29]>, DwarfRegAlias<B29>;
def H30   : AArch64Reg<30, "h30", [B30]>, DwarfRegAlias<B30>;
def H31   : AArch64Reg<31, "h31", [B31]>, DwarfRegAlias<B31>;
}

let SubRegIndices = [hsub] in {
def S0    : AArch64Reg<0,   "s0", [H0]>, DwarfRegAlias<B0>;
def S1    : AArch64Reg<1,   "s1", [H1]>, DwarfRegAlias<B1>;
def S2    : AArch64Reg<2,   "s2", [H2]>, DwarfRegAlias<B2>;
def S3    : AArch64Reg<3,   "s3", [H3]>, DwarfRegAlias<B3>;
def S4    : AArch64Reg<4,   "s4", [H4]>, DwarfRegAlias<B4>;
def S5    : AArch64Reg<5,   "s5", [H5]>, DwarfRegAlias<B5>;
def S6    : AArch64Reg<6,   "s6", [H6]>, DwarfRegAlias<B6>;
def S7    : AArch64Reg<7,   "s7", [H7]>, DwarfRegAlias<B7>;
def S8    : AArch64Reg<8,   "s8", [H8]>, DwarfRegAlias<B8>;
def S9    : AArch64Reg<9,   "s9", [H9]>, DwarfRegAlias<B9>;
def S10   : AArch64Reg<10, "s10", [H10]>, DwarfRegAlias<B10>;
def S11   : AArch64Reg<11, "s11", [H11]>, DwarfRegAlias<B11>;
def S12   : AArch64Reg<12, "s12", [H12]>, DwarfRegAlias<B12>;
def S13   : AArch64Reg<13, "s13", [H13]>, DwarfRegAlias<B13>;
def S14   : AArch64Reg<14, "s14", [H14]>, DwarfRegAlias<B14>;
def S15   : AArch64Reg<15, "s15", [H15]>, DwarfRegAlias<B15>;
def S16   : AArch64Reg<16, "s16", [H16]>, DwarfRegAlias<B16>;
def S17   : AArch64Reg<17, "s17", [H17]>, DwarfRegAlias<B17>;
def S18   : AArch64Reg<18, "s18", [H18]>, DwarfRegAlias<B18>;
def S19   : AArch64Reg<19, "s19", [H19]>, DwarfRegAlias<B19>;
def S20   : AArch64Reg<20, "s20", [H20]>, DwarfRegAlias<B20>;
def S21   : AArch64Reg<21, "s21", [H21]>, DwarfRegAlias<B21>;
def S22   : AArch64Reg<22, "s22", [H22]>, DwarfRegAlias<B22>;
def S23   : AArch64Reg<23, "s23", [H23]>, DwarfRegAlias<B23>;
def S24   : AArch64Reg<24, "s24", [H24]>, DwarfRegAlias<B24>;
def S25   : AArch64Reg<25, "s25", [H25]>, DwarfRegAlias<B25>;
def S26   : AArch64Reg<26, "s26", [H26]>, DwarfRegAlias<B26>;
def S27   : AArch64Reg<27, "s27", [H27]>, DwarfRegAlias<B27>;
def S28   : AArch64Reg<28, "s28", [H28]>, DwarfRegAlias<B28>;
def S29   : AArch64Reg<29, "s29", [H29]>, DwarfRegAlias<B29>;
def S30   : AArch64Reg<30, "s30", [H30]>, DwarfRegAlias<B30>;
def S31   : AArch64Reg<31, "s31", [H31]>, DwarfRegAlias<B31>;
}

let SubRegIndices = [ssub], RegAltNameIndices = [vreg, vlist1] in {
def D0    : AArch64Reg<0,   "d0", [S0], ["v0", ""]>, DwarfRegAlias<B0>;
def D1    : AArch64Reg<1,   "d1", [S1], ["v1", ""]>, DwarfRegAlias<B1>;
def D2    : AArch64Reg<2,   "d2", [S2], ["v2", ""]>, DwarfRegAlias<B2>;
def D3    : AArch64Reg<3,   "d3", [S3], ["v3", ""]>, DwarfRegAlias<B3>;
def D4    : AArch64Reg<4,   "d4", [S4], ["v4", ""]>, DwarfRegAlias<B4>;
def D5    : AArch64Reg<5,   "d5", [S5], ["v5", ""]>, DwarfRegAlias<B5>;
def D6    : AArch64Reg<6,   "d6", [S6], ["v6", ""]>, DwarfRegAlias<B6>;
def D7    : AArch64Reg<7,   "d7", [S7], ["v7", ""]>, DwarfRegAlias<B7>;
def D8    : AArch64Reg<8,   "d8", [S8], ["v8", ""]>, DwarfRegAlias<B8>;
def D9    : AArch64Reg<9,   "d9", [S9], ["v9", ""]>, DwarfRegAlias<B9>;
def D10   : AArch64Reg<10, "d10", [S10], ["v10", ""]>, DwarfRegAlias<B10>;
def D11   : AArch64Reg<11, "d11", [S11], ["v11", ""]>, DwarfRegAlias<B11>;
def D12   : AArch64Reg<12, "d12", [S12], ["v12", ""]>, DwarfRegAlias<B12>;
def D13   : AArch64Reg<13, "d13", [S13], ["v13", ""]>, DwarfRegAlias<B13>;
def D14   : AArch64Reg<14, "d14", [S14], ["v14", ""]>, DwarfRegAlias<B14>;
def D15   : AArch64Reg<15, "d15", [S15], ["v15", ""]>, DwarfRegAlias<B15>;
def D16   : AArch64Reg<16, "d16", [S16], ["v16", ""]>, DwarfRegAlias<B16>;
def D17   : AArch64Reg<17, "d17", [S17], ["v17", ""]>, DwarfRegAlias<B17>;
def D18   : AArch64Reg<18, "d18", [S18], ["v18", ""]>, DwarfRegAlias<B18>;
def D19   : AArch64Reg<19, "d19", [S19], ["v19", ""]>, DwarfRegAlias<B19>;
def D20   : AArch64Reg<20, "d20", [S20], ["v20", ""]>, DwarfRegAlias<B20>;
def D21   : AArch64Reg<21, "d21", [S21], ["v21", ""]>, DwarfRegAlias<B21>;
def D22   : AArch64Reg<22, "d22", [S22], ["v22", ""]>, DwarfRegAlias<B22>;
def D23   : AArch64Reg<23, "d23", [S23], ["v23", ""]>, DwarfRegAlias<B23>;
def D24   : AArch64Reg<24, "d24", [S24], ["v24", ""]>, DwarfRegAlias<B24>;
def D25   : AArch64Reg<25, "d25", [S25], ["v25", ""]>, DwarfRegAlias<B25>;
def D26   : AArch64Reg<26, "d26", [S26], ["v26", ""]>, DwarfRegAlias<B26>;
def D27   : AArch64Reg<27, "d27", [S27], ["v27", ""]>, DwarfRegAlias<B27>;
def D28   : AArch64Reg<28, "d28", [S28], ["v28", ""]>, DwarfRegAlias<B28>;
def D29   : AArch64Reg<29, "d29", [S29], ["v29", ""]>, DwarfRegAlias<B29>;
def D30   : AArch64Reg<30, "d30", [S30], ["v30", ""]>, DwarfRegAlias<B30>;
def D31   : AArch64Reg<31, "d31", [S31], ["v31", ""]>, DwarfRegAlias<B31>;
}

let SubRegIndices = [dsub], RegAltNameIndices = [vreg, vlist1] in {
def Q0    : AArch64Reg<0,   "q0", [D0], ["v0", ""]>, DwarfRegAlias<B0>;
def Q1    : AArch64Reg<1,   "q1", [D1], ["v1", ""]>, DwarfRegAlias<B1>;
def Q2    : AArch64Reg<2,   "q2", [D2], ["v2", ""]>, DwarfRegAlias<B2>;
def Q3    : AArch64Reg<3,   "q3", [D3], ["v3", ""]>, DwarfRegAlias<B3>;
def Q4    : AArch64Reg<4,   "q4", [D4], ["v4", ""]>, DwarfRegAlias<B4>;
def Q5    : AArch64Reg<5,   "q5", [D5], ["v5", ""]>, DwarfRegAlias<B5>;
def Q6    : AArch64Reg<6,   "q6", [D6], ["v6", ""]>, DwarfRegAlias<B6>;
def Q7    : AArch64Reg<7,   "q7", [D7], ["v7", ""]>, DwarfRegAlias<B7>;
def Q8    : AArch64Reg<8,   "q8", [D8], ["v8", ""]>, DwarfRegAlias<B8>;
def Q9    : AArch64Reg<9,   "q9", [D9], ["v9", ""]>, DwarfRegAlias<B9>;
def Q10   : AArch64Reg<10, "q10", [D10], ["v10", ""]>, DwarfRegAlias<B10>;
def Q11   : AArch64Reg<11, "q11", [D11], ["v11", ""]>, DwarfRegAlias<B11>;
def Q12   : AArch64Reg<12, "q12", [D12], ["v12", ""]>, DwarfRegAlias<B12>;
def Q13   : AArch64Reg<13, "q13", [D13], ["v13", ""]>, DwarfRegAlias<B13>;
def Q14   : AArch64Reg<14, "q14", [D14], ["v14", ""]>, DwarfRegAlias<B14>;
def Q15   : AArch64Reg<15, "q15", [D15], ["v15", ""]>, DwarfRegAlias<B15>;
def Q16   : AArch64Reg<16, "q16", [D16], ["v16", ""]>, DwarfRegAlias<B16>;
def Q17   : AArch64Reg<17, "q17", [D17], ["v17", ""]>, DwarfRegAlias<B17>;
def Q18   : AArch64Reg<18, "q18", [D18], ["v18", ""]>, DwarfRegAlias<B18>;
def Q19   : AArch64Reg<19, "q19", [D19], ["v19", ""]>, DwarfRegAlias<B19>;
def Q20   : AArch64Reg<20, "q20", [D20], ["v20", ""]>, DwarfRegAlias<B20>;
def Q21   : AArch64Reg<21, "q21", [D21], ["v21", ""]>, DwarfRegAlias<B21>;
def Q22   : AArch64Reg<22, "q22", [D22], ["v22", ""]>, DwarfRegAlias<B22>;
def Q23   : AArch64Reg<23, "q23", [D23], ["v23", ""]>, DwarfRegAlias<B23>;
def Q24   : AArch64Reg<24, "q24", [D24], ["v24", ""]>, DwarfRegAlias<B24>;
def Q25   : AArch64Reg<25, "q25", [D25], ["v25", ""]>, DwarfRegAlias<B25>;
def Q26   : AArch64Reg<26, "q26", [D26], ["v26", ""]>, DwarfRegAlias<B26>;
def Q27   : AArch64Reg<27, "q27", [D27], ["v27", ""]>, DwarfRegAlias<B27>;
def Q28   : AArch64Reg<28, "q28", [D28], ["v28", ""]>, DwarfRegAlias<B28>;
def Q29   : AArch64Reg<29, "q29", [D29], ["v29", ""]>, DwarfRegAlias<B29>;
def Q30   : AArch64Reg<30, "q30", [D30], ["v30", ""]>, DwarfRegAlias<B30>;
def Q31   : AArch64Reg<31, "q31", [D31], ["v31", ""]>, DwarfRegAlias<B31>;
}

def FPR8  : RegisterClass<"AArch64", [untyped], 8, (sequence "B%u", 0, 31)> {
  let Size = 8;
}
def FPR16 : RegisterClass<"AArch64", [f16], 16, (sequence "H%u", 0, 31)> {
  let Size = 16;
}
def FPR32 : RegisterClass<"AArch64", [f32, i32], 32,(sequence "S%u", 0, 31)>;
def FPR64 : RegisterClass<"AArch64", [f64, i64, v2f32, v1f64, v8i8, v4i16, v2i32,
                                    v1i64, v4f16],
                                    64, (sequence "D%u", 0, 31)>;
// We don't (yet) have an f128 legal type, so don't use that here. We
// normalize 128-bit vectors to v2f64 for arg passing and such, so use
// that here.
def FPR128 : RegisterClass<"AArch64",
                           [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64, f128,
                            v8f16],
                           128, (sequence "Q%u", 0, 31)>;

// The lower 16 vector registers.  Some instructions can only take registers
// in this range.
def FPR128_lo : RegisterClass<"AArch64",
                              [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                              128, (trunc FPR128, 16)>;

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
}
def DDD  : RegisterClass<"AArch64", [untyped], 64, (add DSeqTriples)> {
  let Size = 196;
}
def DDDD : RegisterClass<"AArch64", [untyped], 64, (add DSeqQuads)> {
  let Size = 256;
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
}
def QQQ  : RegisterClass<"AArch64", [untyped], 128, (add QSeqTriples)> {
  let Size = 384;
}
def QQQQ : RegisterClass<"AArch64", [untyped], 128, (add QSeqQuads)> {
  let Size = 512;
}


// Vector operand versions of the FP registers. Alternate name printing and
// assmebler matching.
def VectorReg64AsmOperand : AsmOperandClass {
  let Name = "VectorReg64";
  let PredicateMethod = "isVectorReg";
}
def VectorReg128AsmOperand : AsmOperandClass {
  let Name = "VectorReg128";
  let PredicateMethod = "isVectorReg";
}

def V64  : RegisterOperand<FPR64, "printVRegOperand"> {
  let ParserMatchClass = VectorReg64AsmOperand;
}

def V128 : RegisterOperand<FPR128, "printVRegOperand"> {
  let ParserMatchClass = VectorReg128AsmOperand;
}

def VectorRegLoAsmOperand : AsmOperandClass { let Name = "VectorRegLo"; }
def V128_lo : RegisterOperand<FPR128_lo, "printVRegOperand"> {
  let ParserMatchClass = VectorRegLoAsmOperand;
}

class TypedVecListAsmOperand<int count, int regsize, int lanes, string kind>
    : AsmOperandClass {
  let Name = "TypedVectorList" # count # "_" # lanes # kind;

  let PredicateMethod
      = "isTypedVectorList<" # count # ", " # lanes # ", '" # kind # "'>";
  let RenderMethod = "addVectorList" # regsize # "Operands<" # count # ">";
}

class TypedVecListRegOperand<RegisterClass Reg, int lanes, string kind>
    : RegisterOperand<Reg, "printTypedVectorList<" # lanes # ", '"
                                                   # kind # "'>">;

multiclass VectorList<int count, RegisterClass Reg64, RegisterClass Reg128> {
  // With implicit types (probably on instruction instead). E.g. { v0, v1 }
  def _64AsmOperand : AsmOperandClass {
    let Name = NAME # "64";
    let PredicateMethod = "isImplicitlyTypedVectorList<" # count # ">";
    let RenderMethod = "addVectorList64Operands<" # count # ">";
  }

  def "64" : RegisterOperand<Reg64, "printImplicitlyTypedVectorList"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_64AsmOperand");
  }

  def _128AsmOperand : AsmOperandClass {
    let Name = NAME # "128";
    let PredicateMethod = "isImplicitlyTypedVectorList<" # count # ">";
    let RenderMethod = "addVectorList128Operands<" # count # ">";
  }

  def "128" : RegisterOperand<Reg128, "printImplicitlyTypedVectorList"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_128AsmOperand");
  }

  // 64-bit register lists with explicit type.

  // { v0.8b, v1.8b }
  def _8bAsmOperand : TypedVecListAsmOperand<count, 64, 8, "b">;
  def "8b" : TypedVecListRegOperand<Reg64, 8, "b"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_8bAsmOperand");
  }

  // { v0.4h, v1.4h }
  def _4hAsmOperand : TypedVecListAsmOperand<count, 64, 4, "h">;
  def "4h" : TypedVecListRegOperand<Reg64, 4, "h"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_4hAsmOperand");
  }

  // { v0.2s, v1.2s }
  def _2sAsmOperand : TypedVecListAsmOperand<count, 64, 2, "s">;
  def "2s" : TypedVecListRegOperand<Reg64, 2, "s"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_2sAsmOperand");
  }

  // { v0.1d, v1.1d }
  def _1dAsmOperand : TypedVecListAsmOperand<count, 64, 1, "d">;
  def "1d" : TypedVecListRegOperand<Reg64, 1, "d"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_1dAsmOperand");
  }

  // 128-bit register lists with explicit type

  // { v0.16b, v1.16b }
  def _16bAsmOperand : TypedVecListAsmOperand<count, 128, 16, "b">;
  def "16b" : TypedVecListRegOperand<Reg128, 16, "b"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_16bAsmOperand");
  }

  // { v0.8h, v1.8h }
  def _8hAsmOperand : TypedVecListAsmOperand<count, 128, 8, "h">;
  def "8h" : TypedVecListRegOperand<Reg128, 8, "h"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_8hAsmOperand");
  }

  // { v0.4s, v1.4s }
  def _4sAsmOperand : TypedVecListAsmOperand<count, 128, 4, "s">;
  def "4s" : TypedVecListRegOperand<Reg128, 4, "s"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_4sAsmOperand");
  }

  // { v0.2d, v1.2d }
  def _2dAsmOperand : TypedVecListAsmOperand<count, 128, 2, "d">;
  def "2d" : TypedVecListRegOperand<Reg128, 2, "d"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_2dAsmOperand");
  }

  // { v0.b, v1.b }
  def _bAsmOperand : TypedVecListAsmOperand<count, 128, 0, "b">;
  def "b" : TypedVecListRegOperand<Reg128, 0, "b"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_bAsmOperand");
  }

  // { v0.h, v1.h }
  def _hAsmOperand : TypedVecListAsmOperand<count, 128, 0, "h">;
  def "h" : TypedVecListRegOperand<Reg128, 0, "h"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_hAsmOperand");
  }

  // { v0.s, v1.s }
  def _sAsmOperand : TypedVecListAsmOperand<count, 128, 0, "s">;
  def "s" : TypedVecListRegOperand<Reg128, 0, "s"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_sAsmOperand");
  }

  // { v0.d, v1.d }
  def _dAsmOperand : TypedVecListAsmOperand<count, 128, 0, "d">;
  def "d" : TypedVecListRegOperand<Reg128, 0, "d"> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME # "_dAsmOperand");
  }


}

defm VecListOne   : VectorList<1, FPR64, FPR128>;
defm VecListTwo   : VectorList<2, DD,    QQ>;
defm VecListThree : VectorList<3, DDD,   QQQ>;
defm VecListFour  : VectorList<4, DDDD,  QQQQ>;


// Register operand versions of the scalar FP registers.
def FPR16Op : RegisterOperand<FPR16, "printOperand">;
def FPR32Op : RegisterOperand<FPR32, "printOperand">;
def FPR64Op : RegisterOperand<FPR64, "printOperand">;
def FPR128Op : RegisterOperand<FPR128, "printOperand">;


//===----------------------------------------------------------------------===//
// ARMv8.1a atomic CASP register operands


def WSeqPairs : RegisterTuples<[sube32, subo32], 
                               [(rotl GPR32, 0), (rotl GPR32, 1)]>;
def XSeqPairs : RegisterTuples<[sube64, subo64], 
                               [(rotl GPR64, 0), (rotl GPR64, 1)]>;

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


//===----- END: v8.1a atomic CASP register operands -----------------------===//
