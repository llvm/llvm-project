//===-- SIRegisterInfo.td - SI Register defs ---------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the SI registers
//===----------------------------------------------------------------------===//

class SIReg <string n, bits<16> encoding = 0> : Register<n> {
  let Namespace = "AMDGPU";
  let HWEncoding = encoding;
}

// Special Registers
def VCC_LO : SIReg<"vcc_lo", 106>;
def VCC_HI : SIReg<"vcc_hi", 107>;

// VCC for 64-bit instructions
def VCC : RegisterWithSubRegs<"VCC", [VCC_LO, VCC_HI]> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 106;
}

def EXEC_LO : SIReg<"exec_lo", 126>;
def EXEC_HI : SIReg<"exec_hi", 127>;

def EXEC : RegisterWithSubRegs<"EXEC", [EXEC_LO, EXEC_HI]> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 126;
}

def SCC : SIReg<"SCC", 253>;
def M0 : SIReg <"M0", 124>;

def FLAT_SCR_LO : SIReg<"flat_scr_lo", 104>; // Offset in units of 256-bytes.
def FLAT_SCR_HI : SIReg<"flat_scr_hi", 105>; // Size is the per-thread scratch size, in bytes.

// Pair to indicate location of scratch space for flat accesses.
def FLAT_SCR : RegisterWithSubRegs <"FLAT_SCR", [FLAT_SCR_LO, FLAT_SCR_HI]> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 104;
}

// SGPR registers
foreach Index = 0-101 in {
  def SGPR#Index : SIReg <"SGPR"#Index, Index>;
}

// VGPR registers
foreach Index = 0-255 in {
  def VGPR#Index : SIReg <"VGPR"#Index, Index> {
    let HWEncoding{8} = 1;
  }
}

//===----------------------------------------------------------------------===//
//  Groupings using register classes and tuples
//===----------------------------------------------------------------------===//

// SGPR 32-bit registers
def SGPR_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
                            (add (sequence "SGPR%u", 0, 101))>;

// SGPR 64-bit registers
def SGPR_64Regs : RegisterTuples<[sub0, sub1],
                             [(add (decimate (trunc SGPR_32, 101), 2)),
                              (add (decimate (shl SGPR_32, 1), 2))]>;

// SGPR 128-bit registers
def SGPR_128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                              [(add (decimate (trunc SGPR_32, 99), 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4))]>;

// SGPR 256-bit registers
def SGPR_256 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7],
                              [(add (decimate (trunc SGPR_32, 95), 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4)),
                               (add (decimate (shl SGPR_32, 4), 4)),
                               (add (decimate (shl SGPR_32, 5), 4)),
                               (add (decimate (shl SGPR_32, 6), 4)),
                               (add (decimate (shl SGPR_32, 7), 4))]>;

// SGPR 512-bit registers
def SGPR_512 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7,
                               sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15],
                              [(add (decimate (trunc SGPR_32, 87), 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4)),
                               (add (decimate (shl SGPR_32, 4), 4)),
                               (add (decimate (shl SGPR_32, 5), 4)),
                               (add (decimate (shl SGPR_32, 6), 4)),
                               (add (decimate (shl SGPR_32, 7), 4)),
                               (add (decimate (shl SGPR_32, 8), 4)),
                               (add (decimate (shl SGPR_32, 9), 4)),
                               (add (decimate (shl SGPR_32, 10), 4)),
                               (add (decimate (shl SGPR_32, 11), 4)),
                               (add (decimate (shl SGPR_32, 12), 4)),
                               (add (decimate (shl SGPR_32, 13), 4)),
                               (add (decimate (shl SGPR_32, 14), 4)),
                               (add (decimate (shl SGPR_32, 15), 4))]>;

// VGPR 32-bit registers
def VGPR_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
                            (add (sequence "VGPR%u", 0, 255))>;

// VGPR 64-bit registers
def VGPR_64 : RegisterTuples<[sub0, sub1],
                             [(add (trunc VGPR_32, 255)),
                              (add (shl VGPR_32, 1))]>;

// VGPR 96-bit registers
def VGPR_96 : RegisterTuples<[sub0, sub1, sub2],
                             [(add (trunc VGPR_32, 254)),
                              (add (shl VGPR_32, 1)),
                              (add (shl VGPR_32, 2))]>;

// VGPR 128-bit registers
def VGPR_128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                              [(add (trunc VGPR_32, 253)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3))]>;

// VGPR 256-bit registers
def VGPR_256 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7],
                              [(add (trunc VGPR_32, 249)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3)),
                               (add (shl VGPR_32, 4)),
                               (add (shl VGPR_32, 5)),
                               (add (shl VGPR_32, 6)),
                               (add (shl VGPR_32, 7))]>;

// VGPR 512-bit registers
def VGPR_512 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7,
                               sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15],
                              [(add (trunc VGPR_32, 241)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3)),
                               (add (shl VGPR_32, 4)),
                               (add (shl VGPR_32, 5)),
                               (add (shl VGPR_32, 6)),
                               (add (shl VGPR_32, 7)),
                               (add (shl VGPR_32, 8)),
                               (add (shl VGPR_32, 9)),
                               (add (shl VGPR_32, 10)),
                               (add (shl VGPR_32, 11)),
                               (add (shl VGPR_32, 12)),
                               (add (shl VGPR_32, 13)),
                               (add (shl VGPR_32, 14)),
                               (add (shl VGPR_32, 15))]>;

//===----------------------------------------------------------------------===//
//  Register classes used as source and destination
//===----------------------------------------------------------------------===//

// Special register classes for predicates and the M0 register
def SCCReg : RegisterClass<"AMDGPU", [i32, i1], 32, (add SCC)> {
  let CopyCost = -1; // Theoretically it is possible to read from SCC,
                     // but it should never be necessary.
}

def VCCReg : RegisterClass<"AMDGPU", [i64, i1], 64, (add VCC)>;
def EXECReg : RegisterClass<"AMDGPU", [i64, i1], 64, (add EXEC)>;
def M0Reg : RegisterClass<"AMDGPU", [i32], 32, (add M0)>;

// Register class for all scalar registers (SGPRs + Special Registers)
def SReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
  (add SGPR_32, M0Reg, VCC_LO, VCC_HI, EXEC_LO, EXEC_HI, FLAT_SCR_LO, FLAT_SCR_HI)
>;

def SGPR_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64], 64, (add SGPR_64Regs)>;

def SReg_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64, i1], 64,
  (add SGPR_64, VCCReg, EXECReg, FLAT_SCR)
>;

def SReg_128 : RegisterClass<"AMDGPU", [v4i32, v16i8], 128, (add SGPR_128)>;

def SReg_256 : RegisterClass<"AMDGPU", [v32i8, v8i32, v8f32], 256, (add SGPR_256)>;

def SReg_512 : RegisterClass<"AMDGPU", [v64i8, v16i32], 512, (add SGPR_512)>;

// Register class for all vector registers (VGPRs + Interploation Registers)
def VReg_64 : RegisterClass<"AMDGPU", [i64, f64, v2i32, v2f32], 64, (add VGPR_64)>;

def VReg_96 : RegisterClass<"AMDGPU", [untyped], 96, (add VGPR_96)> {
  let Size = 96;
}

def VReg_128 : RegisterClass<"AMDGPU", [v4i32, v4f32], 128, (add VGPR_128)>;

def VReg_256 : RegisterClass<"AMDGPU", [v32i8, v8i32, v8f32], 256, (add VGPR_256)>;

def VReg_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 512, (add VGPR_512)>;

def VReg_1 : RegisterClass<"AMDGPU", [i1], 32, (add VGPR_32)> {
  let Size = 32;
}

class RegImmOperand <RegisterClass rc> : RegisterOperand<rc> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_IMM32";
}

class RegInlineOperand <RegisterClass rc> : RegisterOperand<rc> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_INLINE_C";
}

//===----------------------------------------------------------------------===//
//  SSrc_* Operands with an SGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

def SSrc_32 : RegImmOperand<SReg_32>;

def SSrc_64 : RegImmOperand<SReg_64>;

//===----------------------------------------------------------------------===//
//  SCSrc_* Operands with an SGPR or a inline constant
//===----------------------------------------------------------------------===//

def SCSrc_32 : RegInlineOperand<SReg_32>;

//===----------------------------------------------------------------------===//
//  VSrc_* Operands with an SGPR, VGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

def VS_32 : RegisterClass<"AMDGPU", [i32, f32], 32, (add VGPR_32, SReg_32)>;

def VS_64 : RegisterClass<"AMDGPU", [i64, f64], 64, (add VReg_64, SReg_64)>;

def VSrc_32 : RegImmOperand<VS_32>;

def VSrc_64 : RegImmOperand<VS_64>;

//===----------------------------------------------------------------------===//
//  VCSrc_* Operands with an SGPR, VGPR or an inline constant
//===----------------------------------------------------------------------===//

def VCSrc_32 : RegInlineOperand<VS_32>;

def VCSrc_64 : RegInlineOperand<VS_64>;

//===----------------------------------------------------------------------===//
// SGPR and VGPR register classes
//===----------------------------------------------------------------------===//

def VSrc_128 : RegisterClass<"AMDGPU", [v4i32, v4f32], 128,
                             (add VReg_128, SReg_128)>;
