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

// Trap handler registers
def TBA_LO : SIReg<"tba_lo", 108>;
def TBA_HI : SIReg<"tba_hi", 109>;

def TBA : RegisterWithSubRegs<"tba", [TBA_LO, TBA_HI]>,
          DwarfRegAlias<TBA_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 108;
}

def TMA_LO : SIReg<"tma_lo", 110>;
def TMA_HI : SIReg<"tma_hi", 111>;

def TMA : RegisterWithSubRegs<"tma", [TMA_LO, TMA_HI]>,
          DwarfRegAlias<TMA_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 110;
}

def TTMP0 : SIReg <"ttmp0", 112>;
def TTMP1 : SIReg <"ttmp1", 113>;
def TTMP2 : SIReg <"ttmp2", 114>;
def TTMP3 : SIReg <"ttmp3", 115>;
def TTMP4 : SIReg <"ttmp4", 116>;
def TTMP5 : SIReg <"ttmp5", 117>;
def TTMP6 : SIReg <"ttmp6", 118>;
def TTMP7 : SIReg <"ttmp7", 119>;
def TTMP8 : SIReg <"ttmp8", 120>;
def TTMP9 : SIReg <"ttmp9", 121>;
def TTMP10 : SIReg <"ttmp10", 122>;
def TTMP11 : SIReg <"ttmp11", 123>;

multiclass FLAT_SCR_LOHI_m <string n, bits<16> ci_e, bits<16> vi_e> {
  def _ci : SIReg<n, ci_e>;
  def _vi : SIReg<n, vi_e>;
  def "" : SIReg<"", 0>;
}

class FlatReg <Register lo, Register hi, bits<16> encoding> :
    RegisterWithSubRegs<"flat_scratch", [lo, hi]>,
    DwarfRegAlias<lo> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = encoding;
}

defm FLAT_SCR_LO : FLAT_SCR_LOHI_m<"flat_scratch_lo", 104, 102>; // Offset in units of 256-bytes.
defm FLAT_SCR_HI : FLAT_SCR_LOHI_m<"flat_scratch_hi", 105, 103>; // Size is the per-thread scratch size, in bytes.

def FLAT_SCR_ci : FlatReg<FLAT_SCR_LO_ci, FLAT_SCR_HI_ci, 104>;
def FLAT_SCR_vi : FlatReg<FLAT_SCR_LO_vi, FLAT_SCR_HI_vi, 102>;
def FLAT_SCR : FlatReg<FLAT_SCR_LO, FLAT_SCR_HI, 0>;

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

def SCC_CLASS : RegisterClass<"AMDGPU", [i1], 1, (add SCC)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

// TODO: Do we need to set DwarfRegAlias on register tuples?

// SGPR 32-bit registers
def SGPR_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
                            (add (sequence "SGPR%u", 0, 103))> {
  let AllocationPriority = 1;
}

// SGPR 64-bit registers
def SGPR_64Regs : RegisterTuples<[sub0, sub1],
                             [(add (decimate SGPR_32, 2)),
                              (add (decimate (shl SGPR_32, 1), 2))]>;

// SGPR 128-bit registers
def SGPR_128Regs : RegisterTuples<[sub0, sub1, sub2, sub3],
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

// Trap handler TMP 32-bit registers
def TTMP_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
                            (add (sequence "TTMP%u", 0, 11))> {
  let isAllocatable = 0;
}

// Trap handler TMP 64-bit registers
def TTMP_64Regs : RegisterTuples<[sub0, sub1],
                             [(add (decimate TTMP_32, 2)),
                              (add (decimate (shl TTMP_32, 1), 2))]>;

// Trap handler TMP 128-bit registers
def TTMP_128Regs : RegisterTuples<[sub0, sub1, sub2, sub3],
                              [(add (decimate TTMP_32, 4)),
                               (add (decimate (shl TTMP_32, 1), 4)),
                               (add (decimate (shl TTMP_32, 2), 4)),
                               (add (decimate (shl TTMP_32, 3), 4))]>;

// VGPR 32-bit registers
def VGPR_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
                            (add (sequence "VGPR%u", 0, 255))> {
  let AllocationPriority = 1;
  let Size = 32;
}

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

// Subset of SReg_32 without M0 for SMRD instructions and alike.
// See comments in SIInstructions.td for more info.
def SReg_32_XM0 : RegisterClass<"AMDGPU", [i32, f32], 32,
  (add SGPR_32, VCC_LO, VCC_HI, EXEC_LO, EXEC_HI, FLAT_SCR_LO, FLAT_SCR_HI,
   TTMP_32, TMA_LO, TMA_HI, TBA_LO, TBA_HI)> {
  let AllocationPriority = 1;
}

// Register class for all scalar registers (SGPRs + Special Registers)
def SReg_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
  (add SReg_32_XM0, M0)> {
  let AllocationPriority = 1;
}

def SGPR_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64], 32, (add SGPR_64Regs)> {
  let AllocationPriority = 2;
}

def TTMP_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64], 32, (add TTMP_64Regs)> {
  let isAllocatable = 0;
}

def SReg_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64, i1], 32,
  (add SGPR_64, VCC, EXEC, FLAT_SCR, TTMP_64, TBA, TMA)> {
  let AllocationPriority = 2;
}

// Requires 2 s_mov_b64 to copy
let CopyCost = 2 in {

def SGPR_128 : RegisterClass<"AMDGPU", [v4i32, v16i8, v2i64], 32, (add SGPR_128Regs)> {
  let AllocationPriority = 4;
}

def TTMP_128 : RegisterClass<"AMDGPU", [v4i32, v16i8, v2i64], 32, (add TTMP_128Regs)> {
  let isAllocatable = 0;
}

def SReg_128 : RegisterClass<"AMDGPU", [v4i32, v16i8, v2i64], 32, (add SGPR_128, TTMP_128)> {
  let AllocationPriority = 4;
}

} // End CopyCost = 2

def SReg_256 : RegisterClass<"AMDGPU", [v8i32, v8f32], 32, (add SGPR_256)> {
  // Requires 4 s_mov_b64 to copy
  let CopyCost = 4;
  let AllocationPriority = 5;
}

def SReg_512 : RegisterClass<"AMDGPU", [v64i8, v16i32], 32, (add SGPR_512)> {
  // Requires 8 s_mov_b64 to copy
  let CopyCost = 8;
  let AllocationPriority = 6;
}

// Register class for all vector registers (VGPRs + Interploation Registers)
def VReg_64 : RegisterClass<"AMDGPU", [i64, f64, v2i32, v2f32], 32, (add VGPR_64)> {
  let Size = 64;

  // Requires 2 v_mov_b32 to copy
  let CopyCost = 2;
  let AllocationPriority = 2;
}

def VReg_96 : RegisterClass<"AMDGPU", [untyped], 32, (add VGPR_96)> {
  let Size = 96;

  // Requires 3 v_mov_b32 to copy
  let CopyCost = 3;
  let AllocationPriority = 3;
}

def VReg_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64, v2f64], 32, (add VGPR_128)> {
  let Size = 128;

  // Requires 4 v_mov_b32 to copy
  let CopyCost = 4;
  let AllocationPriority = 4;
}

def VReg_256 : RegisterClass<"AMDGPU", [v8i32, v8f32], 32, (add VGPR_256)> {
  let Size = 256;
  let CopyCost = 8;
  let AllocationPriority = 5;
}

def VReg_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32, (add VGPR_512)> {
  let Size = 512;
  let CopyCost = 16;
  let AllocationPriority = 6;
}

def VReg_1 : RegisterClass<"AMDGPU", [i1], 32, (add VGPR_32)> {
  let Size = 32;
}

def VS_32 : RegisterClass<"AMDGPU", [i32, f32], 32, (add VGPR_32, SReg_32)> {
  let isAllocatable = 0;
}

def VS_64 : RegisterClass<"AMDGPU", [i64, f64], 32, (add VReg_64, SReg_64)> {
  let isAllocatable = 0;
}

//===----------------------------------------------------------------------===//
//  Register operands
//===----------------------------------------------------------------------===//

class RegImmOperand <RegisterClass rc> : RegisterOperand<rc> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_IMM32";
}

class RegInlineOperand <RegisterClass rc> : RegisterOperand<rc> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_INLINE_C";
}

class RegImmMatcher<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addRegOrImmOperands";
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

def SCSrc_64 : RegInlineOperand<SReg_64> {
  let ParserMatchClass = RegImmMatcher<"SCSrc64">;
}

//===----------------------------------------------------------------------===//
//  VSrc_* Operands with an SGPR, VGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

def VSrc_32 : RegImmOperand<VS_32> {
  let ParserMatchClass = RegImmMatcher<"VSrc32">;
}

def VSrc_64 : RegImmOperand<VS_64> {
  let ParserMatchClass = RegImmMatcher<"VSrc64">;
}

//===----------------------------------------------------------------------===//
//  VSrc_* Operands with an VGPR
//===----------------------------------------------------------------------===//

// This is for operands with the enum(9), VSrc encoding restriction,
// but only allows VGPRs.
def VRegSrc_32 : RegisterOperand<VGPR_32> {
  //let ParserMatchClass = RegImmMatcher<"VRegSrc32">;
  let DecoderMethod = "DecodeVS_32RegisterClass";
}

//===----------------------------------------------------------------------===//
//  VCSrc_* Operands with an SGPR, VGPR or an inline constant
//===----------------------------------------------------------------------===//

def VCSrc_32 : RegInlineOperand<VS_32> {
  let ParserMatchClass = RegImmMatcher<"VCSrc32">;
}

def VCSrc_64 : RegInlineOperand<VS_64> {
  let ParserMatchClass = RegImmMatcher<"VCSrc64">;
}
