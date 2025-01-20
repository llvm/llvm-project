//===-- RISCVRegisterInfo.td - RISC-V Register defs --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the RISC-V register files
//===----------------------------------------------------------------------===//

let Namespace = "RISCV" in {
class RISCVReg<bits<5> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}

class RISCVRegWithSubRegs<bits<5> Enc, string n, list<Register> subregs,
                          list<string> alt = []>
      : RegisterWithSubRegs<n, subregs> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}

class RISCVReg16<bits<5> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}

def sub_16 : SubRegIndex<16>;
class RISCVReg32<RISCVReg16 subreg>
  : RISCVRegWithSubRegs<subreg.HWEncoding{4-0}, subreg.AsmName, [subreg],
                        subreg.AltNames> {
  let SubRegIndices = [sub_16];
}

// Because RISCVReg64 register have AsmName and AltNames that alias with their
// 16/32-bit sub-register, RISCVAsmParser will need to coerce a register number
// from a RISCVReg16/RISCVReg32 to the equivalent RISCVReg64 when appropriate.
def sub_32 : SubRegIndex<32>;
class RISCVReg64<RISCVReg32 subreg>
  : RISCVRegWithSubRegs<subreg.HWEncoding{4-0}, subreg.AsmName, [subreg],
                        subreg.AltNames> {
  let SubRegIndices = [sub_32];
}

let FallbackRegAltNameIndex = NoRegAltName in
def ABIRegAltName : RegAltNameIndex;

def sub_vrm4_0 : SubRegIndex<256>;
def sub_vrm4_1 : SubRegIndex<256, 256>;
def sub_vrm2_0 : SubRegIndex<128>;
def sub_vrm2_1 : SubRegIndex<128, 128>;
def sub_vrm2_2 : ComposedSubRegIndex<sub_vrm4_1, sub_vrm2_0>;
def sub_vrm2_3 : ComposedSubRegIndex<sub_vrm4_1, sub_vrm2_1>;
def sub_vrm1_0 : SubRegIndex<64>;
def sub_vrm1_1 : SubRegIndex<64, 64>;
def sub_vrm1_2 : ComposedSubRegIndex<sub_vrm2_1, sub_vrm1_0>;
def sub_vrm1_3 : ComposedSubRegIndex<sub_vrm2_1, sub_vrm1_1>;
def sub_vrm1_4 : ComposedSubRegIndex<sub_vrm2_2, sub_vrm1_0>;
def sub_vrm1_5 : ComposedSubRegIndex<sub_vrm2_2, sub_vrm1_1>;
def sub_vrm1_6 : ComposedSubRegIndex<sub_vrm2_3, sub_vrm1_0>;
def sub_vrm1_7 : ComposedSubRegIndex<sub_vrm2_3, sub_vrm1_1>;

// GPR sizes change with HwMode.
def sub_gpr_even : SubRegIndex<32> {
  let SubRegRanges = SubRegRangeByHwMode<[RV32, RV64],
                                         [SubRegRange<32>, SubRegRange<64>]>;
}
def sub_gpr_odd  : SubRegIndex<32, 32> {
  let SubRegRanges = SubRegRangeByHwMode<[RV32, RV64],
                                         [SubRegRange<32, 32>, SubRegRange<64, 64>]>;
}
} // Namespace = "RISCV"

//===----------------------------------------------------------------------===//
// General Purpose Registers (aka Integer Registers)
//===----------------------------------------------------------------------===//

// CostPerUse is set higher for registers that may not be compressible as they
// are not part of GPRC, the most restrictive register class used by the
// compressed instruction set. This will influence the greedy register
// allocator to reduce the use of registers that can't be encoded in 16 bit
// instructions.

let RegAltNameIndices = [ABIRegAltName] in {
  // 16-bit sub-registers for use by Zhinx. Having a 16-bit sub-register reduces
  // the spill size for these operations.
  let isConstant = true in
  def X0_H  : RISCVReg<0, "x0", ["zero"]>;
  let CostPerUse = [0, 1] in {
  def X1_H  : RISCVReg<1, "x1", ["ra"]>;
  def X2_H  : RISCVReg<2, "x2", ["sp"]>;
  def X3_H  : RISCVReg<3, "x3", ["gp"]>;
  def X4_H  : RISCVReg<4, "x4", ["tp"]>;
  def X5_H  : RISCVReg<5, "x5", ["t0"]>;
  def X6_H  : RISCVReg<6, "x6", ["t1"]>;
  def X7_H  : RISCVReg<7, "x7", ["t2"]>;
  }
  def X8_H  : RISCVReg<8, "x8", ["s0", "fp"]>;
  def X9_H  : RISCVReg<9, "x9", ["s1"]>;
  def X10_H : RISCVReg<10,"x10", ["a0"]>;
  def X11_H : RISCVReg<11,"x11", ["a1"]>;
  def X12_H : RISCVReg<12,"x12", ["a2"]>;
  def X13_H : RISCVReg<13,"x13", ["a3"]>;
  def X14_H : RISCVReg<14,"x14", ["a4"]>;
  def X15_H : RISCVReg<15,"x15", ["a5"]>;
  let CostPerUse = [0, 1] in {
  def X16_H : RISCVReg<16,"x16", ["a6"]>;
  def X17_H : RISCVReg<17,"x17", ["a7"]>;
  def X18_H : RISCVReg<18,"x18", ["s2"]>;
  def X19_H : RISCVReg<19,"x19", ["s3"]>;
  def X20_H : RISCVReg<20,"x20", ["s4"]>;
  def X21_H : RISCVReg<21,"x21", ["s5"]>;
  def X22_H : RISCVReg<22,"x22", ["s6"]>;
  def X23_H : RISCVReg<23,"x23", ["s7"]>;
  def X24_H : RISCVReg<24,"x24", ["s8"]>;
  def X25_H : RISCVReg<25,"x25", ["s9"]>;
  def X26_H : RISCVReg<26,"x26", ["s10"]>;
  def X27_H : RISCVReg<27,"x27", ["s11"]>;
  def X28_H : RISCVReg<28,"x28", ["t3"]>;
  def X29_H : RISCVReg<29,"x29", ["t4"]>;
  def X30_H : RISCVReg<30,"x30", ["t5"]>;
  def X31_H : RISCVReg<31,"x31", ["t6"]>;
  }

  let SubRegIndices = [sub_16] in {
  let isConstant = true in
  def X0_W  : RISCVRegWithSubRegs<0, "x0", [X0_H], ["zero"]>;
  let CostPerUse = [0, 1] in {
  def X1_W  : RISCVRegWithSubRegs<1, "x1", [X1_H], ["ra"]>;
  def X2_W  : RISCVRegWithSubRegs<2, "x2", [X2_H], ["sp"]>;
  def X3_W  : RISCVRegWithSubRegs<3, "x3", [X3_H], ["gp"]>;
  def X4_W  : RISCVRegWithSubRegs<4, "x4", [X4_H], ["tp"]>;
  def X5_W  : RISCVRegWithSubRegs<5, "x5", [X5_H], ["t0"]>;
  def X6_W  : RISCVRegWithSubRegs<6, "x6", [X6_H], ["t1"]>;
  def X7_W  : RISCVRegWithSubRegs<7, "x7", [X7_H], ["t2"]>;
  }
  def X8_W  : RISCVRegWithSubRegs<8, "x8", [X8_H], ["s0", "fp"]>;
  def X9_W  : RISCVRegWithSubRegs<9, "x9", [X9_H], ["s1"]>;
  def X10_W : RISCVRegWithSubRegs<10,"x10", [X10_H], ["a0"]>;
  def X11_W : RISCVRegWithSubRegs<11,"x11", [X11_H], ["a1"]>;
  def X12_W : RISCVRegWithSubRegs<12,"x12", [X12_H], ["a2"]>;
  def X13_W : RISCVRegWithSubRegs<13,"x13", [X13_H], ["a3"]>;
  def X14_W : RISCVRegWithSubRegs<14,"x14", [X14_H], ["a4"]>;
  def X15_W : RISCVRegWithSubRegs<15,"x15", [X15_H], ["a5"]>;
  let CostPerUse = [0, 1] in {
  def X16_W : RISCVRegWithSubRegs<16,"x16", [X16_H], ["a6"]>;
  def X17_W : RISCVRegWithSubRegs<17,"x17", [X17_H], ["a7"]>;
  def X18_W : RISCVRegWithSubRegs<18,"x18", [X18_H], ["s2"]>;
  def X19_W : RISCVRegWithSubRegs<19,"x19", [X19_H], ["s3"]>;
  def X20_W : RISCVRegWithSubRegs<20,"x20", [X20_H], ["s4"]>;
  def X21_W : RISCVRegWithSubRegs<21,"x21", [X21_H], ["s5"]>;
  def X22_W : RISCVRegWithSubRegs<22,"x22", [X22_H], ["s6"]>;
  def X23_W : RISCVRegWithSubRegs<23,"x23", [X23_H], ["s7"]>;
  def X24_W : RISCVRegWithSubRegs<24,"x24", [X24_H], ["s8"]>;
  def X25_W : RISCVRegWithSubRegs<25,"x25", [X25_H], ["s9"]>;
  def X26_W : RISCVRegWithSubRegs<26,"x26", [X26_H], ["s10"]>;
  def X27_W : RISCVRegWithSubRegs<27,"x27", [X27_H], ["s11"]>;
  def X28_W : RISCVRegWithSubRegs<28,"x28", [X28_H], ["t3"]>;
  def X29_W : RISCVRegWithSubRegs<29,"x29", [X29_H], ["t4"]>;
  def X30_W : RISCVRegWithSubRegs<30,"x30", [X30_H], ["t5"]>;
  def X31_W : RISCVRegWithSubRegs<31,"x31", [X31_H], ["t6"]>;
  }
  }

  let SubRegIndices = [sub_32] in {
  let isConstant = true in
  def X0  : RISCVRegWithSubRegs<0, "x0", [X0_W], ["zero"]>, DwarfRegNum<[0]>;
  let CostPerUse = [0, 1] in {
  def X1  : RISCVRegWithSubRegs<1, "x1", [X1_W], ["ra"]>, DwarfRegNum<[1]>;
  def X2  : RISCVRegWithSubRegs<2, "x2", [X2_W], ["sp"]>, DwarfRegNum<[2]>;
  def X3  : RISCVRegWithSubRegs<3, "x3", [X3_W], ["gp"]>, DwarfRegNum<[3]>;
  def X4  : RISCVRegWithSubRegs<4, "x4", [X4_W], ["tp"]>, DwarfRegNum<[4]>;
  def X5  : RISCVRegWithSubRegs<5, "x5", [X5_W], ["t0"]>, DwarfRegNum<[5]>;
  def X6  : RISCVRegWithSubRegs<6, "x6", [X6_W], ["t1"]>, DwarfRegNum<[6]>;
  def X7  : RISCVRegWithSubRegs<7, "x7", [X7_W], ["t2"]>, DwarfRegNum<[7]>;
  }
  def X8  : RISCVRegWithSubRegs<8, "x8", [X8_W], ["s0", "fp"]>, DwarfRegNum<[8]>;
  def X9  : RISCVRegWithSubRegs<9, "x9", [X9_W], ["s1"]>, DwarfRegNum<[9]>;
  def X10 : RISCVRegWithSubRegs<10,"x10", [X10_W], ["a0"]>, DwarfRegNum<[10]>;
  def X11 : RISCVRegWithSubRegs<11,"x11", [X11_W], ["a1"]>, DwarfRegNum<[11]>;
  def X12 : RISCVRegWithSubRegs<12,"x12", [X12_W], ["a2"]>, DwarfRegNum<[12]>;
  def X13 : RISCVRegWithSubRegs<13,"x13", [X13_W], ["a3"]>, DwarfRegNum<[13]>;
  def X14 : RISCVRegWithSubRegs<14,"x14", [X14_W], ["a4"]>, DwarfRegNum<[14]>;
  def X15 : RISCVRegWithSubRegs<15,"x15", [X15_W], ["a5"]>, DwarfRegNum<[15]>;
  let CostPerUse = [0, 1] in {
  def X16 : RISCVRegWithSubRegs<16,"x16", [X16_W], ["a6"]>, DwarfRegNum<[16]>;
  def X17 : RISCVRegWithSubRegs<17,"x17", [X17_W], ["a7"]>, DwarfRegNum<[17]>;
  def X18 : RISCVRegWithSubRegs<18,"x18", [X18_W], ["s2"]>, DwarfRegNum<[18]>;
  def X19 : RISCVRegWithSubRegs<19,"x19", [X19_W], ["s3"]>, DwarfRegNum<[19]>;
  def X20 : RISCVRegWithSubRegs<20,"x20", [X20_W], ["s4"]>, DwarfRegNum<[20]>;
  def X21 : RISCVRegWithSubRegs<21,"x21", [X21_W], ["s5"]>, DwarfRegNum<[21]>;
  def X22 : RISCVRegWithSubRegs<22,"x22", [X22_W], ["s6"]>, DwarfRegNum<[22]>;
  def X23 : RISCVRegWithSubRegs<23,"x23", [X23_W], ["s7"]>, DwarfRegNum<[23]>;
  def X24 : RISCVRegWithSubRegs<24,"x24", [X24_W], ["s8"]>, DwarfRegNum<[24]>;
  def X25 : RISCVRegWithSubRegs<25,"x25", [X25_W], ["s9"]>, DwarfRegNum<[25]>;
  def X26 : RISCVRegWithSubRegs<26,"x26", [X26_W], ["s10"]>, DwarfRegNum<[26]>;
  def X27 : RISCVRegWithSubRegs<27,"x27", [X27_W], ["s11"]>, DwarfRegNum<[27]>;
  def X28 : RISCVRegWithSubRegs<28,"x28", [X28_W], ["t3"]>, DwarfRegNum<[28]>;
  def X29 : RISCVRegWithSubRegs<29,"x29", [X29_W], ["t4"]>, DwarfRegNum<[29]>;
  def X30 : RISCVRegWithSubRegs<30,"x30", [X30_W], ["t5"]>, DwarfRegNum<[30]>;
  def X31 : RISCVRegWithSubRegs<31,"x31", [X31_W], ["t6"]>, DwarfRegNum<[31]>;
  }
  }
}

def XLenVT : ValueTypeByHwMode<[RV32, RV64],
                               [i32,  i64]>;
defvar XLenPairVT = untyped;

// Allow f64 in GPR for ZDINX on RV64.
def XLenFVT : ValueTypeByHwMode<[RV64],
                                [f64]>;
def XLenPairFVT : ValueTypeByHwMode<[RV32],
                                    [f64]>;
def XLenRI : RegInfoByHwMode<
      [RV32,              RV64],
      [RegInfo<32,32,32>, RegInfo<64,64,64>]>;

class RISCVRegisterClass<list<ValueType> regTypes, int align, dag regList>
    : RegisterClass<"RISCV", regTypes, align, regList> {
  bit IsVRegClass = 0;
  int VLMul = 1;
  int NF = 1;

  let Size = !if(IsVRegClass, !mul(VLMul, NF, 64), 0);

  let TSFlags{0} = IsVRegClass;
  let TSFlags{3-1} = !logtwo(VLMul);
  let TSFlags{6-4} = !sub(NF, 1);
}

class GPRRegisterClass<dag regList>
    : RISCVRegisterClass<[XLenVT, XLenFVT, i32, i16], 32, regList> {
  let RegInfos = XLenRI;
}

// The order of registers represents the preferred allocation sequence.
// Registers are listed in the order caller-save, callee-save, specials.
def GPR : GPRRegisterClass<(add (sequence "X%u", 10, 17),
                                (sequence "X%u", 5, 7),
                                (sequence "X%u", 28, 31),
                                (sequence "X%u", 8, 9),
                                (sequence "X%u", 18, 27),
                                (sequence "X%u", 0, 4))>;

def GPRX0 : GPRRegisterClass<(add X0)>;
def GPRX1 : GPRRegisterClass<(add X1)>;
def GPRX5 : GPRRegisterClass<(add X5)>;

def GPRNoX0 : GPRRegisterClass<(sub GPR, X0)>;

def GPRNoX0X2 : GPRRegisterClass<(sub GPR, X0, X2)>;

def GPRX7 : GPRRegisterClass<(add X7)>;

// Don't use X1 or X5 for JALR since that is a hint to pop the return address
// stack on some microarchitectures. Also remove the reserved registers X0, X2,
// X3, and X4 as it reduces the number of register classes that get synthesized
// by tablegen.
def GPRJALR : GPRRegisterClass<(sub GPR, (sequence "X%u", 0, 5))>;

def GPRJALRNonX7 : GPRRegisterClass<(sub GPRJALR, X7)>;

def GPRC : GPRRegisterClass<(add (sequence "X%u", 10, 15),
                                 (sequence "X%u", 8, 9))>;

// For indirect tail calls, we can't use callee-saved registers, as they are
// restored to the saved value before the tail call, which would clobber a call
// address. We shouldn't use x5 since that is a hint for to pop the return
// address stack on some microarchitectures.
def GPRTC : GPRRegisterClass<(add (sequence "X%u", 6, 7),
                                  (sequence "X%u", 10, 17),
                                  (sequence "X%u", 28, 31))>;
def GPRTCNonX7 : GPRRegisterClass<(sub GPRTC, X7)>;

def SP : GPRRegisterClass<(add X2)>;

// Saved Registers from s0 to s7, for C.MVA01S07 instruction in Zcmp extension
def SR07 : GPRRegisterClass<(add (sequence "X%u", 8, 9),
                                 (sequence "X%u", 18, 23))>;

def GPRX1X5 :  GPRRegisterClass<(add X1, X5)>;

//===----------------------------------------------------------------------===//
// Even-Odd GPR Pairs
//===----------------------------------------------------------------------===//

def XLenPairRI : RegInfoByHwMode<
      [RV32,                RV64],
      [RegInfo<64, 64, 32>, RegInfo<128, 128, 64>]>;

// Dummy zero register for use in the register pair containing X0 (as X1 is
// not read to or written when the X0 register pair is used).
def DUMMY_REG_PAIR_WITH_X0 : RISCVReg<0, "0">;

// Must add DUMMY_REG_PAIR_WITH_X0 to a separate register class to prevent the
// register's existence from changing codegen (due to the regPressureSetLimit
// for the GPR register class being altered).
def GPRAll : GPRRegisterClass<(add GPR, DUMMY_REG_PAIR_WITH_X0)>;

let RegAltNameIndices = [ABIRegAltName] in {
  def X0_Pair : RISCVRegWithSubRegs<0, X0.AsmName,
                                    [X0, DUMMY_REG_PAIR_WITH_X0],
                                    X0.AltNames> {
    let SubRegIndices = [sub_gpr_even, sub_gpr_odd];
    let CoveredBySubRegs = 1;
  }
  foreach I = 1-15 in {
    defvar Index = !shl(I, 1);
    defvar IndexP1 = !add(Index, 1);
    defvar Reg = !cast<Register>("X"#Index);
    defvar RegP1 = !cast<Register>("X"#IndexP1);
    def "X" # Index #"_X" # IndexP1 : RISCVRegWithSubRegs<Index,
                                                          Reg.AsmName,
                                                          [Reg, RegP1],
                                                          Reg.AltNames> {
      let SubRegIndices = [sub_gpr_even, sub_gpr_odd];
      let CoveredBySubRegs = 1;
    }
  }
}

let RegInfos = XLenPairRI,
    DecoderMethod = "DecodeGPRPairRegisterClass" in {
def GPRPair : RISCVRegisterClass<[XLenPairVT, XLenPairFVT], 64, (add
    X10_X11, X12_X13, X14_X15, X16_X17,
    X6_X7,
    X28_X29, X30_X31,
    X8_X9,
    X18_X19, X20_X21, X22_X23, X24_X25, X26_X27,
    X0_Pair, X2_X3, X4_X5
)>;

def GPRPairNoX0 : RISCVRegisterClass<[XLenPairVT, XLenPairFVT], 64, (sub GPRPair, X0_Pair)>;
} // let RegInfos = XLenPairRI, DecoderMethod = "DecodeGPRPairRegisterClass"

let RegInfos = XLenPairRI in
def GPRPairC : RISCVRegisterClass<[XLenPairVT, XLenPairFVT], 64, (add
  X10_X11, X12_X13, X14_X15, X8_X9
)>;

//===----------------------------------------------------------------------===//
// Floating Point registers
//===----------------------------------------------------------------------===//

let RegAltNameIndices = [ABIRegAltName] in {
  def F0_H  : RISCVReg16<0, "f0", ["ft0"]>, DwarfRegNum<[32]>;
  def F1_H  : RISCVReg16<1, "f1", ["ft1"]>, DwarfRegNum<[33]>;
  def F2_H  : RISCVReg16<2, "f2", ["ft2"]>, DwarfRegNum<[34]>;
  def F3_H  : RISCVReg16<3, "f3", ["ft3"]>, DwarfRegNum<[35]>;
  def F4_H  : RISCVReg16<4, "f4", ["ft4"]>, DwarfRegNum<[36]>;
  def F5_H  : RISCVReg16<5, "f5", ["ft5"]>, DwarfRegNum<[37]>;
  def F6_H  : RISCVReg16<6, "f6", ["ft6"]>, DwarfRegNum<[38]>;
  def F7_H  : RISCVReg16<7, "f7", ["ft7"]>, DwarfRegNum<[39]>;
  def F8_H  : RISCVReg16<8, "f8", ["fs0"]>, DwarfRegNum<[40]>;
  def F9_H  : RISCVReg16<9, "f9", ["fs1"]>, DwarfRegNum<[41]>;
  def F10_H : RISCVReg16<10,"f10", ["fa0"]>, DwarfRegNum<[42]>;
  def F11_H : RISCVReg16<11,"f11", ["fa1"]>, DwarfRegNum<[43]>;
  def F12_H : RISCVReg16<12,"f12", ["fa2"]>, DwarfRegNum<[44]>;
  def F13_H : RISCVReg16<13,"f13", ["fa3"]>, DwarfRegNum<[45]>;
  def F14_H : RISCVReg16<14,"f14", ["fa4"]>, DwarfRegNum<[46]>;
  def F15_H : RISCVReg16<15,"f15", ["fa5"]>, DwarfRegNum<[47]>;
  def F16_H : RISCVReg16<16,"f16", ["fa6"]>, DwarfRegNum<[48]>;
  def F17_H : RISCVReg16<17,"f17", ["fa7"]>, DwarfRegNum<[49]>;
  def F18_H : RISCVReg16<18,"f18", ["fs2"]>, DwarfRegNum<[50]>;
  def F19_H : RISCVReg16<19,"f19", ["fs3"]>, DwarfRegNum<[51]>;
  def F20_H : RISCVReg16<20,"f20", ["fs4"]>, DwarfRegNum<[52]>;
  def F21_H : RISCVReg16<21,"f21", ["fs5"]>, DwarfRegNum<[53]>;
  def F22_H : RISCVReg16<22,"f22", ["fs6"]>, DwarfRegNum<[54]>;
  def F23_H : RISCVReg16<23,"f23", ["fs7"]>, DwarfRegNum<[55]>;
  def F24_H : RISCVReg16<24,"f24", ["fs8"]>, DwarfRegNum<[56]>;
  def F25_H : RISCVReg16<25,"f25", ["fs9"]>, DwarfRegNum<[57]>;
  def F26_H : RISCVReg16<26,"f26", ["fs10"]>, DwarfRegNum<[58]>;
  def F27_H : RISCVReg16<27,"f27", ["fs11"]>, DwarfRegNum<[59]>;
  def F28_H : RISCVReg16<28,"f28", ["ft8"]>, DwarfRegNum<[60]>;
  def F29_H : RISCVReg16<29,"f29", ["ft9"]>, DwarfRegNum<[61]>;
  def F30_H : RISCVReg16<30,"f30", ["ft10"]>, DwarfRegNum<[62]>;
  def F31_H : RISCVReg16<31,"f31", ["ft11"]>, DwarfRegNum<[63]>;

  foreach Index = 0-31 in {
    def F#Index#_F : RISCVReg32<!cast<RISCVReg16>("F"#Index#"_H")>,
      DwarfRegAlias<!cast<Register>("F"#Index#"_H")>;
  }

  foreach Index = 0-31 in {
    def F#Index#_D : RISCVReg64<!cast<RISCVReg32>("F"#Index#"_F")>,
      DwarfRegAlias<!cast<Register>("F"#Index#"_H")>;
  }
}

// The order of registers represents the preferred allocation sequence,
// meaning caller-save regs are listed before callee-save.
// We start by allocating argument registers in reverse order since they are
// compressible.
def FPR16 : RISCVRegisterClass<[f16, bf16], 16, (add
    (sequence "F%u_H", 15, 10), // fa5-fa0
    (sequence "F%u_H", 0, 7),   // ft0-f7
    (sequence "F%u_H", 16, 17), // fa6-fa7
    (sequence "F%u_H", 28, 31), // ft8-ft11
    (sequence "F%u_H", 8, 9),   // fs0-fs1
    (sequence "F%u_H", 18, 27)  // fs2-fs11
)>;

def FPR16C : RISCVRegisterClass<[f16, bf16], 16, (add
    (sequence "F%u_H", 15, 10),
    (sequence "F%u_H", 8, 9)
)>;

def FPR32 : RISCVRegisterClass<[f32], 32, (add
    (sequence "F%u_F", 15, 10),
    (sequence "F%u_F", 0, 7),
    (sequence "F%u_F", 16, 17),
    (sequence "F%u_F", 28, 31),
    (sequence "F%u_F", 8, 9),
    (sequence "F%u_F", 18, 27)
)>;

def FPR32C : RISCVRegisterClass<[f32], 32, (add
  (sequence "F%u_F", 15, 10),
  (sequence "F%u_F", 8, 9)
)>;

// The order of registers represents the preferred allocation sequence,
// meaning caller-save regs are listed before callee-save.
def FPR64 : RISCVRegisterClass<[f64], 64, (add
    (sequence "F%u_D", 15, 10),
    (sequence "F%u_D", 0, 7),
    (sequence "F%u_D", 16, 17),
    (sequence "F%u_D", 28, 31),
    (sequence "F%u_D", 8, 9),
    (sequence "F%u_D", 18, 27)
)>;

def FPR64C : RISCVRegisterClass<[f64], 64, (add
  (sequence "F%u_D", 15, 10),
  (sequence "F%u_D", 8, 9)
)>;

//===----------------------------------------------------------------------===//
// GPR Classes for "H/F/D in X"
//===----------------------------------------------------------------------===//

// 16-bit GPR sub-register class used by Zhinx instructions.
def GPRF16 : RISCVRegisterClass<[f16], 16, (add (sequence "X%u_H", 10, 17),
                                                (sequence "X%u_H", 5, 7),
                                                (sequence "X%u_H", 28, 31),
                                                (sequence "X%u_H", 8, 9),
                                                (sequence "X%u_H", 18, 27),
                                                (sequence "X%u_H", 0, 4))>;
def GPRF16C : RISCVRegisterClass<[f16], 16, (add (sequence "X%u_H", 10, 15),
                                                 (sequence "X%u_H", 8, 9))>;
def GPRF16NoX0 : RISCVRegisterClass<[f16], 16, (sub GPRF16, X0_H)>;

def GPRF32 : RISCVRegisterClass<[f32], 32, (add (sequence "X%u_W", 10, 17),
                                                (sequence "X%u_W", 5, 7),
                                                (sequence "X%u_W", 28, 31),
                                                (sequence "X%u_W", 8, 9),
                                                (sequence "X%u_W", 18, 27),
                                                (sequence "X%u_W", 0, 4))>;
def GPRF32C : RISCVRegisterClass<[f32], 32, (add (sequence "X%u_W", 10, 15),
                                                 (sequence "X%u_W", 8, 9))>;
def GPRF32NoX0 : RISCVRegisterClass<[f32], 32, (sub GPRF32, X0_W)>;

//===----------------------------------------------------------------------===//
// Vector type mapping to LLVM types.
//===----------------------------------------------------------------------===//

// The V vector extension requires that VLEN >= 128 and <= 65536.
// Additionally, the only supported ELEN values are 32 and 64,
// thus `vscale` can be defined as VLEN/64,
// allowing the same types with either ELEN value.
//
//         MF8    MF4     MF2     M1      M2      M4       M8
// i64*    N/A    N/A     N/A     nxv1i64 nxv2i64 nxv4i64  nxv8i64
// i32     N/A    N/A     nxv1i32 nxv2i32 nxv4i32 nxv8i32  nxv16i32
// i16     N/A    nxv1i16 nxv2i16 nxv4i16 nxv8i16 nxv16i16 nxv32i16
// i8      nxv1i8 nxv2i8  nxv4i8  nxv8i8  nxv16i8 nxv32i8  nxv64i8
// double* N/A    N/A     N/A     nxv1f64 nxv2f64 nxv4f64  nxv8f64
// float   N/A    N/A     nxv1f32 nxv2f32 nxv4f32 nxv8f32  nxv16f32
// half    N/A    nxv1f16 nxv2f16 nxv4f16 nxv8f16 nxv16f16 nxv32f16
// * ELEN=64

defvar vint8mf8_t = nxv1i8;
defvar vint8mf4_t = nxv2i8;
defvar vint8mf2_t = nxv4i8;
defvar vint8m1_t = nxv8i8;
defvar vint8m2_t = nxv16i8;
defvar vint8m4_t = nxv32i8;
defvar vint8m8_t = nxv64i8;

defvar vint16mf4_t = nxv1i16;
defvar vint16mf2_t = nxv2i16;
defvar vint16m1_t  = nxv4i16;
defvar vint16m2_t  = nxv8i16;
defvar vint16m4_t  = nxv16i16;
defvar vint16m8_t  = nxv32i16;

defvar vint32mf2_t = nxv1i32;
defvar vint32m1_t  = nxv2i32;
defvar vint32m2_t  = nxv4i32;
defvar vint32m4_t  = nxv8i32;
defvar vint32m8_t  = nxv16i32;

defvar vint64m1_t = nxv1i64;
defvar vint64m2_t = nxv2i64;
defvar vint64m4_t = nxv4i64;
defvar vint64m8_t = nxv8i64;

defvar vfloat16mf4_t = nxv1f16;
defvar vfloat16mf2_t = nxv2f16;
defvar vfloat16m1_t  = nxv4f16;
defvar vfloat16m2_t  = nxv8f16;
defvar vfloat16m4_t  = nxv16f16;
defvar vfloat16m8_t  = nxv32f16;

defvar vbfloat16mf4_t = nxv1bf16;
defvar vbfloat16mf2_t = nxv2bf16;
defvar vbfloat16m1_t  = nxv4bf16;
defvar vbfloat16m2_t  = nxv8bf16;
defvar vbfloat16m4_t  = nxv16bf16;
defvar vbfloat16m8_t  = nxv32bf16;

defvar vfloat32mf2_t = nxv1f32;
defvar vfloat32m1_t  = nxv2f32;
defvar vfloat32m2_t  = nxv4f32;
defvar vfloat32m4_t  = nxv8f32;
defvar vfloat32m8_t  = nxv16f32;

defvar vfloat64m1_t = nxv1f64;
defvar vfloat64m2_t = nxv2f64;
defvar vfloat64m4_t = nxv4f64;
defvar vfloat64m8_t = nxv8f64;

defvar vbool1_t  = nxv64i1;
defvar vbool2_t  = nxv32i1;
defvar vbool4_t  = nxv16i1;
defvar vbool8_t  = nxv8i1;
defvar vbool16_t = nxv4i1;
defvar vbool32_t = nxv2i1;
defvar vbool64_t = nxv1i1;

// There is no need to define register classes for fractional LMUL.
defvar LMULList = [1, 2, 4, 8];

//===----------------------------------------------------------------------===//
// Utility classes for segment load/store.
//===----------------------------------------------------------------------===//
// The set of legal NF for LMUL = lmul.
// LMUL <= 1, NF = 2, 3, 4, 5, 6, 7, 8
// LMUL == 2, NF = 2, 3, 4
// LMUL == 4, NF = 2
// LMUL == 8, no legal NF
class NFList<int lmul> {
  list<int> L = !cond(!eq(lmul, 8): [],
                      !eq(lmul, 4): [2],
                      !eq(lmul, 2): [2, 3, 4],
                      true: [2, 3, 4, 5, 6, 7, 8]);
}

// Generate [start, end) SubRegIndex list.
class SubRegSet<int nf, int lmul> {
  list<SubRegIndex> L = !foldl([]<SubRegIndex>,
                               !range(0, 8),
                               AccList, i,
                               !listconcat(AccList,
                                 !if(!lt(i, nf),
                                   [!cast<SubRegIndex>("sub_vrm" # lmul # "_" # i)],
                                   [])));
}

// Collect the valid indexes into 'R' under NF and LMUL values from TUPLE_INDEX.
// When NF = 2, the valid TUPLE_INDEX is 0 and 1.
// For example, when LMUL = 4, the potential valid indexes is
// [8, 12, 16, 20, 24, 28, 4]. However, not all these indexes are valid under
// NF = 2. For example, 28 is not valid under LMUL = 4, NF = 2 and TUPLE_INDEX = 0.
// The filter is
//   (tuple_index + i) x lmul <= (tuple_index x lmul) + 32 - (nf x lmul)
//
// Use START = 0, LMUL = 4 and NF = 2 as the example,
//   i x 4 <= 24
// The class will return [8, 12, 16, 20, 24, 4].
// Use START = 1, LMUL = 4 and NF = 2 as the example,
//   (1 + i) x 4 <= 28
// The class will return [12, 16, 20, 24, 28, 8].
//
class IndexSet<int tuple_index, int nf, int lmul, bit isV0 = false> {
  list<int> R =
    !foldl([]<int>,
              !if(isV0, [0],
                !cond(
                  !eq(lmul, 1): !listconcat(!range(8, 32), !range(1, 8)),
                  !eq(lmul, 2): !listconcat(!range(4, 16), !range(1, 4)),
                  !eq(lmul, 4): !listconcat(!range(2, 8), !range(1, 2)))),
              L, i,
              !listconcat(L,
                          !if(!le(!mul(!add(i, tuple_index), lmul),
                                  !sub(!add(32, !mul(tuple_index, lmul)), !mul(nf, lmul))),
                              [!mul(!add(i, tuple_index), lmul)], [])));
}

// This class returns a list of vector register collections.
// For example, for NF = 2 and LMUL = 4,
// L would be:
//   ([ V8M4, V12M4, V16M4, V20M4, V24M4, V4M4],
//    [V12M4, V16M4, V20M4, V24M4, V28M4, V8M4])
// Names are the starting register of each register list,
// in this example:
//   ["v8", "v12", "v16", "v20", "v24", "v4"]
class VRegList<list<dag> LIn, int start, int nf, int lmul, bit isV0> {
  list<dag> L =
    !if(!ge(start, nf),
        LIn,
        !listconcat(
          [!dag(add,
                !foreach(i, IndexSet<start, nf, lmul, isV0>.R,
                  !cast<Register>("V" # i # !cond(!eq(lmul, 2): "M2",
                                                  !eq(lmul, 4): "M4",
                                                  true: ""))),
                !listsplat("",
                  !size(IndexSet<start, nf, lmul, isV0>.R)))],
          VRegList<LIn, !add(start, 1), nf, lmul, isV0>.L));
  list<string> Names =
    !if(!ge(start, nf), [],
        !foreach(i, IndexSet<start, nf, lmul, isV0>.R, "v" # i));
}

//===----------------------------------------------------------------------===//
// Vector registers
//===----------------------------------------------------------------------===//

foreach Index = !range(0, 32, 1) in {
  def V#Index : RISCVReg<Index, "v"#Index>, DwarfRegNum<[!add(Index, 96)]>;
}

foreach Index = !range(0, 32, 2) in {
  def V#Index#M2 : RISCVRegWithSubRegs<Index, "v"#Index,
                     [!cast<Register>("V"#Index),
                      !cast<Register>("V"#!add(Index, 1))]>,
                   DwarfRegAlias<!cast<Register>("V"#Index)> {
    let SubRegIndices = [sub_vrm1_0, sub_vrm1_1];
  }
}

foreach Index = !range(0, 32, 4) in {
  def V#Index#M4 : RISCVRegWithSubRegs<Index, "v"#Index,
                     [!cast<Register>("V"#Index#"M2"),
                      !cast<Register>("V"#!add(Index, 2)#"M2")]>,
                   DwarfRegAlias<!cast<Register>("V"#Index)> {
    let SubRegIndices = [sub_vrm2_0, sub_vrm2_1];
  }
}

foreach Index = !range(0, 32, 8) in {
  def V#Index#M8 : RISCVRegWithSubRegs<Index, "v"#Index,
                     [!cast<Register>("V"#Index#"M4"),
                      !cast<Register>("V"#!add(Index, 4)#"M4")]>,
                   DwarfRegAlias<!cast<Register>("V"#Index)> {
    let SubRegIndices = [sub_vrm4_0, sub_vrm4_1];
  }
}

def VTYPE  : RISCVReg<0, "vtype">;
def VL     : RISCVReg<0, "vl">;
def VXSAT  : RISCVReg<0, "vxsat">;
def VXRM   : RISCVReg<0, "vxrm">;
let isConstant = true in
def VLENB  : RISCVReg<0, "vlenb">,
             DwarfRegNum<[!add(4096, SysRegVLENB.Encoding)]>;

def VCSR : RISCVRegisterClass<[XLenVT], 32,
                              (add VTYPE, VL, VLENB)> {
  let RegInfos = XLenRI;
  let isAllocatable = 0;
}


foreach m = [1, 2, 4] in {
  foreach n = NFList<m>.L in {
    defvar RegListWOV0 = VRegList<[], 0, n, m, false>;
    defvar RegListWV0 = VRegList<[], 0, n, m, true>;
    def "VN" # n # "M" # m # "NoV0": RegisterTuples<
                                       SubRegSet<n, m>.L,
                                       RegListWOV0.L,
                                       RegListWOV0.Names>;
    def "VN" # n # "M" # m # "V0" : RegisterTuples<
                                       SubRegSet<n, m>.L,
                                       RegListWV0.L,
                                       RegListWV0.Names>;
  }
}

class VReg<list<ValueType> regTypes, dag regList, int Vlmul>
    : RISCVRegisterClass<regTypes,
                         64, // The maximum supported ELEN is 64.
                         regList> {
  let IsVRegClass = 1;
  let VLMul = Vlmul;
}

defvar VMaskVTs = [vbool1_t, vbool2_t, vbool4_t, vbool8_t, vbool16_t,
                   vbool32_t, vbool64_t];

defvar VM1VTs = [vint8m1_t, vint16m1_t, vint32m1_t, vint64m1_t,
                 vbfloat16m1_t, vfloat16m1_t, vfloat32m1_t,
                 vfloat64m1_t, vint8mf2_t, vint8mf4_t, vint8mf8_t,
                 vint16mf2_t, vint16mf4_t, vint32mf2_t,
                 vfloat16mf4_t, vfloat16mf2_t, vbfloat16mf4_t,
                 vbfloat16mf2_t, vfloat32mf2_t];

defvar VM2VTs = [vint8m2_t, vint16m2_t, vint32m2_t, vint64m2_t,
                 vfloat16m2_t, vbfloat16m2_t,
                 vfloat32m2_t, vfloat64m2_t];

defvar VM4VTs = [vint8m4_t, vint16m4_t, vint32m4_t, vint64m4_t,
                 vfloat16m4_t, vbfloat16m4_t,
                 vfloat32m4_t, vfloat64m4_t];

defvar VM8VTs = [vint8m8_t, vint16m8_t, vint32m8_t, vint64m8_t,
                 vfloat16m8_t, vbfloat16m8_t,
                 vfloat32m8_t, vfloat64m8_t];

// We reverse the order of last 8 registers so that we don't needlessly prevent
// allocation of higher lmul register groups while still putting v0 last in the
// allocation order.

def VR : VReg<!listconcat(VM1VTs, VMaskVTs),
              (add (sequence "V%u", 8, 31),
                   (sequence "V%u", 7, 0)), 1>;

def VRNoV0 : VReg<!listconcat(VM1VTs, VMaskVTs), (sub VR, V0), 1>;

def VRM2 : VReg<VM2VTs, (add (sequence "V%uM2", 8, 31, 2),
                             (sequence "V%uM2", 6, 0, 2)), 2>;

def VRM2NoV0 : VReg<VM2VTs, (sub VRM2, V0M2), 2>;

def VRM4 : VReg<VM4VTs, (add V8M4, V12M4, V16M4, V20M4,
                             V24M4, V28M4, V4M4, V0M4), 4>;

def VRM4NoV0 : VReg<VM4VTs, (sub VRM4, V0M4), 4>;

def VRM8 : VReg<VM8VTs, (add V8M8, V16M8, V24M8, V0M8), 8>;

def VRM8NoV0 : VReg<VM8VTs, (sub VRM8, V0M8), 8>;

def VMV0 : VReg<VMaskVTs, (add V0), 1>;

// The register class is added for inline assembly for vector mask types.
def VM : VReg<VMaskVTs, (add VR), 1>;

defvar VTupM1N2VTs = [riscv_nxv8i8x2, riscv_nxv4i8x2, riscv_nxv2i8x2, riscv_nxv1i8x2];
defvar VTupM1N3VTs = [riscv_nxv8i8x3, riscv_nxv4i8x3, riscv_nxv2i8x3, riscv_nxv1i8x3];
defvar VTupM1N4VTs = [riscv_nxv8i8x4, riscv_nxv4i8x4, riscv_nxv2i8x4, riscv_nxv1i8x4];
defvar VTupM1N5VTs = [riscv_nxv8i8x5, riscv_nxv4i8x5, riscv_nxv2i8x5, riscv_nxv1i8x5];
defvar VTupM1N6VTs = [riscv_nxv8i8x6, riscv_nxv4i8x6, riscv_nxv2i8x6, riscv_nxv1i8x6];
defvar VTupM1N7VTs = [riscv_nxv8i8x7, riscv_nxv4i8x7, riscv_nxv2i8x7, riscv_nxv1i8x7];
defvar VTupM1N8VTs = [riscv_nxv8i8x8, riscv_nxv4i8x8, riscv_nxv2i8x8, riscv_nxv1i8x8];
defvar VTupM2N2VTs = [riscv_nxv16i8x2];
defvar VTupM2N3VTs = [riscv_nxv16i8x3];
defvar VTupM2N4VTs = [riscv_nxv16i8x4];
defvar VTupM4N2VTs = [riscv_nxv32i8x2];
class VTupRegList<int LMUL, int NF> {
  list<ValueType> L = !cond(!and(!eq(LMUL, 1), !eq(NF, 2)): VTupM1N2VTs,
                            !and(!eq(LMUL, 1), !eq(NF, 3)): VTupM1N3VTs,
                            !and(!eq(LMUL, 1), !eq(NF, 4)): VTupM1N4VTs,
                            !and(!eq(LMUL, 1), !eq(NF, 5)): VTupM1N5VTs,
                            !and(!eq(LMUL, 1), !eq(NF, 6)): VTupM1N6VTs,
                            !and(!eq(LMUL, 1), !eq(NF, 7)): VTupM1N7VTs,
                            !and(!eq(LMUL, 1), !eq(NF, 8)): VTupM1N8VTs,
                            !and(!eq(LMUL, 2), !eq(NF, 2)): VTupM2N2VTs,
                            !and(!eq(LMUL, 2), !eq(NF, 3)): VTupM2N3VTs,
                            !and(!eq(LMUL, 2), !eq(NF, 4)): VTupM2N4VTs,
                            !and(!eq(LMUL, 4), !eq(NF, 2)): VTupM4N2VTs);
}

foreach m = LMULList in {
  foreach nf = NFList<m>.L in {
    let NF = nf in {
      def "VRN" # nf # "M" # m # "NoV0"
        : VReg<VTupRegList<m, nf>.L,
               (add !cast<RegisterTuples>("VN" # nf # "M" # m # "NoV0")),
               m>;
      def "VRN" # nf # "M" # m
        : VReg<VTupRegList<m, nf>.L,
               (add !cast<RegisterTuples>("VN" # nf # "M" # m # "NoV0"),
                    !cast<RegisterTuples>("VN" # nf # "M" # m # "V0")),
               m>;
    }
  }
}

//===----------------------------------------------------------------------===//
// Special registers
//===----------------------------------------------------------------------===//

def FFLAGS : RISCVReg<0, "fflags">;
def FRM    : RISCVReg<0, "frm">;

// Shadow Stack register
def SSP    : RISCVReg<0, "ssp">;

// Dummy SiFive VCIX state register
def SF_VCIX_STATE : RISCVReg<0, "sf.vcix_state">;
def : RISCVRegisterClass<[XLenVT], 32, (add SF_VCIX_STATE)> {
  let RegInfos = XLenRI;
  let isAllocatable = 0;
}
