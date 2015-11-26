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
class SIReg <string n, bits<16> regIdx = 0> : Register<n>,
  DwarfRegNum<[!cast<int>(HWEncoding)]> {
  let Namespace = "AMDGPU";

  // This is the not yet the complete register encoding. An additional
  // bit is set for VGPRs.
  let HWEncoding = regIdx;
}

// Special Registers
def VCC_LO : SIReg<"vcc_lo", 106>;
def VCC_HI : SIReg<"vcc_hi", 107>;

// VCC for 64-bit instructions
def VCC : RegisterWithSubRegs<"vcc", [VCC_LO, VCC_HI]>,
          DwarfRegAlias<VCC_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 106;
}

def EXEC_LO : SIReg<"exec_lo", 126>;
def EXEC_HI : SIReg<"exec_hi", 127>;

def EXEC : RegisterWithSubRegs<"EXEC", [EXEC_LO, EXEC_HI]>,
           DwarfRegAlias<EXEC_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 126;
}

def SCC : SIReg<"scc", 253>;
def M0 : SIReg <"m0", 124>;

def FLAT_SCR_LO : SIReg<"flat_scr_lo", 104>; // Offset in units of 256-bytes.
def FLAT_SCR_HI : SIReg<"flat_scr_hi", 105>; // Size is the per-thread scratch size, in bytes.

// Pair to indicate location of scratch space for flat accesses.
def FLAT_SCR : RegisterWithSubRegs <"flat_scr", [FLAT_SCR_LO, FLAT_SCR_HI]>,
               DwarfRegAlias<FLAT_SCR_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 104;
}

// SGPR registers
foreach Index = 0-103 in {
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

// TODO: Do we need to set DwarfRegAlias on register tuples?

// SGPR 32-bit registers
def SGPR_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
                            (add (sequence "SGPR%u", 0, 103))>;

// SGPR 64-bit registers
def SGPR_64Regs : RegisterTuples<[sub0, sub1],
                             [(add (decimate SGPR_32, 2)),
                              (add (decimate (shl SGPR_32, 1), 2))]>;

// SGPR 128-bit registers
def SGPR_128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                              [(add (decimate SGPR_32, 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4))]>;

// SGPR 256-bit registers
def SGPR_256 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7],
                              [(add (decimate SGPR_32, 4)),
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
                              [(add (decimate SGPR_32, 4)),
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
def VGPR_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
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

class RegImmMatcher<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addRegOrImmOperands";
}

// Register class for all scalar registers (SGPRs + Special Registers)
def SReg_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
  (add SGPR_32, M0, VCC_LO, VCC_HI, EXEC_LO, EXEC_HI, FLAT_SCR_LO, FLAT_SCR_HI)
>;

def SGPR_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64], 32, (add SGPR_64Regs)>;

def SReg_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64, i1], 32,
  (add SGPR_64, VCC, EXEC, FLAT_SCR)
>;

def SReg_128 : RegisterClass<"AMDGPU", [v4i32, v16i8, v2i64], 32, (add SGPR_128)> {
  // Requires 2 s_mov_b64 to copy
  let CopyCost = 2;
}

def SReg_256 : RegisterClass<"AMDGPU", [v32i8, v8i32, v8f32], 32, (add SGPR_256)> {
  // Requires 4 s_mov_b64 to copy
  let CopyCost = 4;
}

def SReg_512 : RegisterClass<"AMDGPU", [v64i8, v16i32], 32, (add SGPR_512)> {
  // Requires 8 s_mov_b64 to copy
  let CopyCost = 8;
}

// Register class for all vector registers (VGPRs + Interploation Registers)
def VReg_64 : RegisterClass<"AMDGPU", [i64, f64, v2i32, v2f32], 32, (add VGPR_64)> {
  // Requires 2 v_mov_b32 to copy
  let CopyCost = 2;
}

def VReg_96 : RegisterClass<"AMDGPU", [untyped], 32, (add VGPR_96)> {
  let Size = 96;

  // Requires 3 v_mov_b32 to copy
  let CopyCost = 3;
}

def VReg_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64, v2f64], 32, (add VGPR_128)> {
  // Requires 4 v_mov_b32 to copy
  let CopyCost = 4;
}

def VReg_256 : RegisterClass<"AMDGPU", [v32i8, v8i32, v8f32], 32, (add VGPR_256)> {
  let CopyCost = 8;
}

def VReg_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32, (add VGPR_512)> {
  let CopyCost = 16;
}

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

def SSrc_32 : RegImmOperand<SReg_32> {
  let ParserMatchClass = RegImmMatcher<"SSrc32">;
}

def SSrc_64 : RegImmOperand<SReg_64> {
  let ParserMatchClass = RegImmMatcher<"SSrc64">;
}

//===----------------------------------------------------------------------===//
//  SCSrc_* Operands with an SGPR or a inline constant
//===----------------------------------------------------------------------===//

def SCSrc_32 : RegInlineOperand<SReg_32> {
  let ParserMatchClass = RegImmMatcher<"SCSrc32">;
}

//===----------------------------------------------------------------------===//
//  VSrc_* Operands with an SGPR, VGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

def VS_32 : RegisterClass<"AMDGPU", [i32, f32], 32, (add VGPR_32, SReg_32)>;

def VS_64 : RegisterClass<"AMDGPU", [i64, f64], 32, (add VReg_64, SReg_64)> {
  let CopyCost = 2;
}

def VSrc_32 : RegisterOperand<VS_32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_IMM32";
  let ParserMatchClass = RegImmMatcher<"VSrc32">;
}

def VSrc_64 : RegisterOperand<VS_64> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_IMM32";
  let ParserMatchClass = RegImmMatcher<"VSrc64">;
}

//===----------------------------------------------------------------------===//
//  VCSrc_* Operands with an SGPR, VGPR or an inline constant
//===----------------------------------------------------------------------===//

def VCSrc_32 : RegisterOperand<VS_32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_INLINE_C";
  let ParserMatchClass = RegImmMatcher<"VCSrc32">;
}

def VCSrc_64 : RegisterOperand<VS_64> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_INLINE_C";
  let ParserMatchClass = RegImmMatcher<"VCSrc64">;
}

//===----------------------------------------------------------------------===//
//  SCSrc_* Operands with an SGPR or an inline constant
//===----------------------------------------------------------------------===//

def SCSrc_64 : RegisterOperand<SReg_64> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_INLINE_C";
  let ParserMatchClass = RegImmMatcher<"SCSrc64">;
}
