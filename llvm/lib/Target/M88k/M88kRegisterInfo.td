//===-- M88kRegisterInfo.td - M88k Register definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class M88kReg<bits<5> Enc, string n> : Register<n> {
  let HWEncoding{15-5} = 0;
  let HWEncoding{4-0} = Enc;
  let Namespace = "M88k";
}

class M88kCRReg<bits<6> Enc, string n> : Register<n> {
  let HWEncoding{15-6} = 0;
  let HWEncoding{5-0} = Enc;
  let Namespace = "M88k";
}

class M88kRegWithSubregs<string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let Namespace = "M88k";
}

// Multiclass to define RegisterOperand, RegisterClass and AsmOperand together.
multiclass M88kRegisterClass<list<ValueType> types, int size, int alignment,
                             dag regList, bit allocatable = 1> {
  def AsmOperand : AsmOperandClass {
    let Name = NAME;
    let ParserMethod = "parse"#NAME#;
  }
  let isAllocatable = allocatable in
    def RC : RegisterClass<"M88k", types, alignment, regList> {
      let Size = size;
    }
  def "" : RegisterOperand<!cast<RegisterClass>(NAME#"RC")> {
    let ParserMatchClass = !cast<AsmOperandClass>(NAME#"AsmOperand");
    let DecoderMethod = "decode"#NAME#"RegisterClass";
  }
}

let Namespace = "M88k" in {
  def sub_hi : SubRegIndex<32, 0>;
  def sub_lo : SubRegIndex<32, 32>;
}

// General-purpose registers
foreach I = 0-31 in {
  def R#I : M88kReg<I, "r"#I>;
}

// Register pairs.
// TODO Check let isArtificial = 1
let AsmName = "", SubRegIndices = [sub_hi, sub_lo], CoveredBySubRegs = true in {
  foreach I = 0-15 in {
    defvar Rx = "R" # !mul(2,I);
    defvar Ry = "R" # !add(!mul(2,I),1);
    def Rx#_#Ry : M88kRegWithSubregs<Rx#_#Ry,
                                    [!cast<Register>(Rx), !cast<Register>(Ry)]>;
  }
}

defm GPR : M88kRegisterClass<[i32], 32, 32,
                             (add (sequence "R%u", 0, 31))>;
defm GPR64 : M88kRegisterClass<[i64, f64], 64, 32,
                               (add R0_R1, R2_R3, R4_R5, R6_R7, R8_R9,
                                    R10_R11, R12_R13, R14_R15, R16_R17, R18_R19,
                                    R20_R21, R22_R23, R24_R25, R26_R27, R28_R29,
                                    R30_R31)>;

// Experimental: GPR with floating point types.
defm FPR32 : M88kRegisterClass<[f32], 32, 32,
                               (add GPRRC)>;

// Extended (floating point) registers.
foreach I = 0-31 in {
  def X#I : M88kReg<I, "x"#I>;
}

defm XR : M88kRegisterClass<[f32, f64, f80], 128, 32,
                            (add (sequence "X%u", 0, 31))>;

//def XROpnd : RegisterOperand<XR>;

// Control registers.
foreach I = 0-63 in {
  def CR#I : M88kCRReg<I, "cr"#I>;
}

foreach I = 0-63 in {
  def FCR#I : M88kCRReg<I, "fcr"#I>;
}

defm CR : M88kRegisterClass<[i32], 32, 32, (add (sequence "CR%u", 0, 63)), 0>;
defm FCR : M88kRegisterClass<[i32], 32, 32, (add (sequence "FCR%u", 0, 63)), 0>;

/*
Ideas:

  bits<1> HWRegFile = regFile; // HW register file: 0 = GPR, 1 = XPR

ValueType -> Register
class getVregSrcForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 128), VReg_128,
                        !if(!eq(VT.Size, 96), VReg_96,
                          !if(!eq(VT.Size, 64), VReg_64,
                            !if(!eq(VT.Size, 48), VReg_64,
                              VGPR_32))));
}

class isFloatType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, f16.Value),
                !eq(SrcVT.Value, f32.Value),
                !eq(SrcVT.Value, f64.Value),
                !eq(SrcVT.Value, v2f16.Value),
                !eq(SrcVT.Value, v4f16.Value),
                !eq(SrcVT.Value, v2f32.Value),
                !eq(SrcVT.Value, v2f64.Value),
                !eq(SrcVT.Value, v4f64.Value));
}

*/