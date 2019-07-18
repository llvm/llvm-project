//===-- SIRegisterInfo.td - SI Register defs ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Helpers
//===----------------------------------------------------------------------===//

class getSubRegs<int size> {
  list<SubRegIndex> ret2 = [sub0, sub1];
  list<SubRegIndex> ret3 = [sub0, sub1, sub2];
  list<SubRegIndex> ret4 = [sub0, sub1, sub2, sub3];
  list<SubRegIndex> ret5 = [sub0, sub1, sub2, sub3, sub4];
  list<SubRegIndex> ret8 = [sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7];
  list<SubRegIndex> ret16 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11,
                             sub12, sub13, sub14, sub15];
  list<SubRegIndex> ret32 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11,
                             sub12, sub13, sub14, sub15,
                             sub16, sub17, sub18, sub19,
                             sub20, sub21, sub22, sub23,
                             sub24, sub25, sub26, sub27,
                             sub28, sub29, sub30, sub31];

  list<SubRegIndex> ret = !if(!eq(size, 2), ret2,
                              !if(!eq(size, 3), ret3,
                                  !if(!eq(size, 4), ret4,
                                      !if(!eq(size, 5), ret5,
                                          !if(!eq(size, 8), ret8,
                                              !if(!eq(size, 16), ret16, ret32))))));
}

let Namespace = "AMDGPU" in {
defset list<RegAltNameIndex> AllRegAltNameIndices = {
  def Reg32   : RegAltNameIndex;
  def Reg64   : RegAltNameIndex;
  def Reg96   : RegAltNameIndex;
  def Reg128  : RegAltNameIndex;
  def Reg160  : RegAltNameIndex;
  def Reg256  : RegAltNameIndex;
  def Reg512  : RegAltNameIndex;
  def Reg1024 : RegAltNameIndex;
}
}

//===----------------------------------------------------------------------===//
//  Declarations that describe the SI registers
//===----------------------------------------------------------------------===//
class SIReg <string n, bits<16> regIdx = 0, string prefix = "",
             int regNo = !cast<int>(regIdx)> :
  Register<n, !if(!eq(prefix, ""),
                [ n, n, n, n, n, n, n, n ],
                [ prefix # regNo,
                  prefix # "[" # regNo # ":" # !and(!add(regNo, 1), 255) # "]",
                  prefix # "[" # regNo # ":" # !and(!add(regNo, 2), 255) # "]",
                  prefix # "[" # regNo # ":" # !and(!add(regNo, 3), 255) # "]",
                  prefix # "[" # regNo # ":" # !and(!add(regNo, 4), 255) # "]",
                  prefix # "[" # regNo # ":" # !and(!add(regNo, 7), 255) # "]",
                  prefix # "[" # regNo # ":" # !and(!add(regNo, 15), 255) # "]",
                  prefix # "[" # regNo # ":" # !and(!add(regNo, 31), 255) # "]",
                ])>,
  DwarfRegNum<[!cast<int>(HWEncoding)]> {
  let Namespace = "AMDGPU";
  let RegAltNameIndices = AllRegAltNameIndices;

  // This is the not yet the complete register encoding. An additional
  // bit is set for VGPRs.
  let HWEncoding = regIdx;
}

class SIRegisterWithSubRegs<string n, list<Register> subregs> :
  RegisterWithSubRegs<n, subregs> {
  let RegAltNameIndices = AllRegAltNameIndices;
  let AltNames = [ n, n, n, n, n, n, n, n ];
}

// Special Registers
def VCC_LO : SIReg<"vcc_lo", 106>;
def VCC_HI : SIReg<"vcc_hi", 107>;

// Pseudo-registers: Used as placeholders during isel and immediately
// replaced, never seeing the verifier.
def PRIVATE_RSRC_REG : SIReg<"private_rsrc", 0>;
def FP_REG : SIReg<"fp", 0>;
def SP_REG : SIReg<"sp", 0>;
def SCRATCH_WAVE_OFFSET_REG : SIReg<"scratch_wave_offset", 0>;

// VCC for 64-bit instructions
def VCC : SIRegisterWithSubRegs<"vcc", [VCC_LO, VCC_HI]>,
          DwarfRegAlias<VCC_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 106;
}

def EXEC_LO : SIReg<"exec_lo", 126>;
def EXEC_HI : SIReg<"exec_hi", 127>;

def EXEC : SIRegisterWithSubRegs<"exec", [EXEC_LO, EXEC_HI]>,
           DwarfRegAlias<EXEC_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 126;
}

// 32-bit real registers, for MC only.
// May be used with both 32-bit and 64-bit operands.
def SRC_VCCZ : SIReg<"src_vccz", 251>;
def SRC_EXECZ : SIReg<"src_execz", 252>;
def SRC_SCC : SIReg<"src_scc", 253>;

// 1-bit pseudo register, for codegen only.
// Should never be emitted.
def SCC : SIReg<"scc">;

def M0 : SIReg <"m0", 124>;
def SGPR_NULL : SIReg<"null", 125>;

def SRC_SHARED_BASE : SIReg<"src_shared_base", 235>;
def SRC_SHARED_LIMIT : SIReg<"src_shared_limit", 236>;
def SRC_PRIVATE_BASE : SIReg<"src_private_base", 237>;
def SRC_PRIVATE_LIMIT : SIReg<"src_private_limit", 238>;
def SRC_POPS_EXITING_WAVE_ID : SIReg<"src_pops_exiting_wave_id", 239>;

def LDS_DIRECT : SIReg <"src_lds_direct", 254>;

def XNACK_MASK_LO : SIReg<"xnack_mask_lo", 104>;
def XNACK_MASK_HI : SIReg<"xnack_mask_hi", 105>;

def XNACK_MASK : SIRegisterWithSubRegs<"xnack_mask", [XNACK_MASK_LO, XNACK_MASK_HI]>,
                 DwarfRegAlias<XNACK_MASK_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 104;
}

// Trap handler registers
def TBA_LO : SIReg<"tba_lo", 108>;
def TBA_HI : SIReg<"tba_hi", 109>;

def TBA : SIRegisterWithSubRegs<"tba", [TBA_LO, TBA_HI]>,
          DwarfRegAlias<TBA_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 108;
}

def TMA_LO : SIReg<"tma_lo", 110>;
def TMA_HI : SIReg<"tma_hi", 111>;

def TMA : SIRegisterWithSubRegs<"tma", [TMA_LO, TMA_HI]>,
          DwarfRegAlias<TMA_LO> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = 110;
}

foreach Index = 0-15 in {
  def TTMP#Index#_vi         : SIReg<"ttmp"#Index, !add(112, Index)>;
  def TTMP#Index#_gfx9_gfx10 : SIReg<"ttmp"#Index, !add(108, Index)>;
  def TTMP#Index             : SIReg<"ttmp"#Index, 0>;
}

multiclass FLAT_SCR_LOHI_m <string n, bits<16> ci_e, bits<16> vi_e> {
  def _ci : SIReg<n, ci_e>;
  def _vi : SIReg<n, vi_e>;
  def "" : SIReg<n, 0>;
}

class FlatReg <Register lo, Register hi, bits<16> encoding> :
    SIRegisterWithSubRegs<"flat_scratch", [lo, hi]>,
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
foreach Index = 0-105 in {
  def SGPR#Index : SIReg <"SGPR"#Index, Index, "s">;
}

// VGPR registers
foreach Index = 0-255 in {
  def VGPR#Index : SIReg <"VGPR"#Index, Index, "v"> {
    let HWEncoding{8} = 1;
  }
}

// AccVGPR registers
foreach Index = 0-255 in {
  def AGPR#Index : SIReg <"AGPR"#Index, Index, "a"> {
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

def M0_CLASS : RegisterClass<"AMDGPU", [i32], 32, (add M0)> {
  let CopyCost = 1;
  let isAllocatable = 0;
}

// TODO: Do we need to set DwarfRegAlias on register tuples?

// SGPR 32-bit registers
def SGPR_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                            (add (sequence "SGPR%u", 0, 105)), Reg32> {
  // Give all SGPR classes higher priority than VGPR classes, because
  // we want to spill SGPRs to VGPRs.
  let AllocationPriority = 9;
}

// SGPR 64-bit registers
def SGPR_64Regs : RegisterTuples<getSubRegs<2>.ret,
                             [(add (decimate SGPR_32, 2)),
                              (add (decimate (shl SGPR_32, 1), 2))]>;

// SGPR 96-bit registers. No operations use these, but for symmetry with 96-bit VGPRs.
def SGPR_96Regs : RegisterTuples<getSubRegs<3>.ret,
                            [(add (decimate SGPR_32, 3)),
                             (add (decimate (shl SGPR_32, 1), 3)),
                             (add (decimate (shl SGPR_32, 2), 3))]>;

// SGPR 128-bit registers
def SGPR_128Regs : RegisterTuples<getSubRegs<4>.ret,
                              [(add (decimate SGPR_32, 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4))]>;

// SGPR 160-bit registers. No operations use these, but for symmetry with 160-bit VGPRs.
def SGPR_160Regs : RegisterTuples<getSubRegs<5>.ret,
                            [(add (decimate SGPR_32, 4)),
                             (add (decimate (shl SGPR_32, 1), 4)),
                             (add (decimate (shl SGPR_32, 2), 4)),
                             (add (decimate (shl SGPR_32, 3), 4)),
                             (add (decimate (shl SGPR_32, 4), 4))]>;

// SGPR 256-bit registers
def SGPR_256Regs : RegisterTuples<getSubRegs<8>.ret,
                              [(add (decimate SGPR_32, 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4)),
                               (add (decimate (shl SGPR_32, 4), 4)),
                               (add (decimate (shl SGPR_32, 5), 4)),
                               (add (decimate (shl SGPR_32, 6), 4)),
                               (add (decimate (shl SGPR_32, 7), 4))]>;

// SGPR 512-bit registers
def SGPR_512Regs : RegisterTuples<getSubRegs<16>.ret,
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

// SGPR 1024-bit registers
def SGPR_1024Regs : RegisterTuples<getSubRegs<32>.ret,
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
                               (add (decimate (shl SGPR_32, 15), 4)),
                               (add (decimate (shl SGPR_32, 16), 4)),
                               (add (decimate (shl SGPR_32, 17), 4)),
                               (add (decimate (shl SGPR_32, 18), 4)),
                               (add (decimate (shl SGPR_32, 19), 4)),
                               (add (decimate (shl SGPR_32, 20), 4)),
                               (add (decimate (shl SGPR_32, 21), 4)),
                               (add (decimate (shl SGPR_32, 22), 4)),
                               (add (decimate (shl SGPR_32, 23), 4)),
                               (add (decimate (shl SGPR_32, 24), 4)),
                               (add (decimate (shl SGPR_32, 25), 4)),
                               (add (decimate (shl SGPR_32, 26), 4)),
                               (add (decimate (shl SGPR_32, 27), 4)),
                               (add (decimate (shl SGPR_32, 28), 4)),
                               (add (decimate (shl SGPR_32, 29), 4)),
                               (add (decimate (shl SGPR_32, 30), 4)),
                               (add (decimate (shl SGPR_32, 31), 4))]>;

// Trap handler TMP 32-bit registers
def TTMP_32 : RegisterClass<"AMDGPU", [i32, f32, v2i16, v2f16], 32,
                            (add (sequence "TTMP%u", 0, 15))> {
  let isAllocatable = 0;
}

// Trap handler TMP 64-bit registers
def TTMP_64Regs : RegisterTuples<getSubRegs<2>.ret,
                             [(add (decimate TTMP_32, 2)),
                              (add (decimate (shl TTMP_32, 1), 2))]>;

// Trap handler TMP 128-bit registers
def TTMP_128Regs : RegisterTuples<getSubRegs<4>.ret,
                              [(add (decimate TTMP_32, 4)),
                               (add (decimate (shl TTMP_32, 1), 4)),
                               (add (decimate (shl TTMP_32, 2), 4)),
                               (add (decimate (shl TTMP_32, 3), 4))]>;

def TTMP_256Regs : RegisterTuples<getSubRegs<8>.ret,
                              [(add (decimate TTMP_32, 4)),
                               (add (decimate (shl TTMP_32, 1), 4)),
                               (add (decimate (shl TTMP_32, 2), 4)),
                               (add (decimate (shl TTMP_32, 3), 4)),
                               (add (decimate (shl TTMP_32, 4), 4)),
                               (add (decimate (shl TTMP_32, 5), 4)),
                               (add (decimate (shl TTMP_32, 6), 4)),
                               (add (decimate (shl TTMP_32, 7), 4))]>;

def TTMP_512Regs : RegisterTuples<getSubRegs<16>.ret,
                              [(add (decimate TTMP_32, 4)),
                               (add (decimate (shl TTMP_32, 1), 4)),
                               (add (decimate (shl TTMP_32, 2), 4)),
                               (add (decimate (shl TTMP_32, 3), 4)),
                               (add (decimate (shl TTMP_32, 4), 4)),
                               (add (decimate (shl TTMP_32, 5), 4)),
                               (add (decimate (shl TTMP_32, 6), 4)),
                               (add (decimate (shl TTMP_32, 7), 4)),
                               (add (decimate (shl TTMP_32, 8), 4)),
                               (add (decimate (shl TTMP_32, 9), 4)),
                               (add (decimate (shl TTMP_32, 10), 4)),
                               (add (decimate (shl TTMP_32, 11), 4)),
                               (add (decimate (shl TTMP_32, 12), 4)),
                               (add (decimate (shl TTMP_32, 13), 4)),
                               (add (decimate (shl TTMP_32, 14), 4)),
                               (add (decimate (shl TTMP_32, 15), 4))]>;

class TmpRegTuplesBase<int index, int size,
                       list<Register> subRegs,
                       list<SubRegIndex> indices = getSubRegs<size>.ret,
                       int index1 = !add(index, !add(size, -1)),
                       string name = "ttmp["#index#":"#index1#"]"> :
  SIRegisterWithSubRegs<name, subRegs> {
  let HWEncoding = subRegs[0].HWEncoding;
  let SubRegIndices = indices;
}

class TmpRegTuples<string tgt,
                   int size,
                   int index0,
                   int index1 = !add(index0, 1),
                   int index2 = !add(index0, !if(!eq(size, 2), 1, 2)),
                   int index3 = !add(index0, !if(!eq(size, 2), 1, 3)),
                   int index4 = !add(index0, !if(!eq(size, 8), 4, 1)),
                   int index5 = !add(index0, !if(!eq(size, 8), 5, 1)),
                   int index6 = !add(index0, !if(!eq(size, 8), 6, 1)),
                   int index7 = !add(index0, !if(!eq(size, 8), 7, 1)),
                   Register r0 = !cast<Register>("TTMP"#index0#tgt),
                   Register r1 = !cast<Register>("TTMP"#index1#tgt),
                   Register r2 = !cast<Register>("TTMP"#index2#tgt),
                   Register r3 = !cast<Register>("TTMP"#index3#tgt),
                   Register r4 = !cast<Register>("TTMP"#index4#tgt),
                   Register r5 = !cast<Register>("TTMP"#index5#tgt),
                   Register r6 = !cast<Register>("TTMP"#index6#tgt),
                   Register r7 = !cast<Register>("TTMP"#index7#tgt)> :
  TmpRegTuplesBase<index0, size,
                   !if(!eq(size, 2), [r0, r1],
                       !if(!eq(size, 4), [r0, r1, r2, r3],
                                         [r0, r1, r2, r3, r4, r5, r6, r7])),
                   getSubRegs<size>.ret>;

foreach Index = {0, 2, 4, 6, 8, 10, 12, 14} in {
  def TTMP#Index#_TTMP#!add(Index,1)#_vi         : TmpRegTuples<"_vi",   2, Index>;
  def TTMP#Index#_TTMP#!add(Index,1)#_gfx9_gfx10 : TmpRegTuples<"_gfx9_gfx10", 2, Index>;
}

foreach Index = {0, 4, 8, 12} in {
  def TTMP#Index#_TTMP#!add(Index,1)#
                 _TTMP#!add(Index,2)#
                 _TTMP#!add(Index,3)#_vi : TmpRegTuples<"_vi",   4, Index>;
  def TTMP#Index#_TTMP#!add(Index,1)#
                 _TTMP#!add(Index,2)#
                 _TTMP#!add(Index,3)#_gfx9_gfx10 : TmpRegTuples<"_gfx9_gfx10", 4, Index>;
}

foreach Index = {0, 4, 8} in {
  def TTMP#Index#_TTMP#!add(Index,1)#
                 _TTMP#!add(Index,2)#
                 _TTMP#!add(Index,3)#
                 _TTMP#!add(Index,4)#
                 _TTMP#!add(Index,5)#
                 _TTMP#!add(Index,6)#
                 _TTMP#!add(Index,7)#_vi : TmpRegTuples<"_vi",   8, Index>;
  def TTMP#Index#_TTMP#!add(Index,1)#
                 _TTMP#!add(Index,2)#
                 _TTMP#!add(Index,3)#
                 _TTMP#!add(Index,4)#
                 _TTMP#!add(Index,5)#
                 _TTMP#!add(Index,6)#
                 _TTMP#!add(Index,7)#_gfx9_gfx10 : TmpRegTuples<"_gfx9_gfx10", 8, Index>;
}

def TTMP0_TTMP1_TTMP2_TTMP3_TTMP4_TTMP5_TTMP6_TTMP7_TTMP8_TTMP9_TTMP10_TTMP11_TTMP12_TTMP13_TTMP14_TTMP15_vi :
  TmpRegTuplesBase<0, 16,
                   [TTMP0_vi, TTMP1_vi, TTMP2_vi, TTMP3_vi,
                    TTMP4_vi, TTMP5_vi, TTMP6_vi, TTMP7_vi,
                    TTMP8_vi, TTMP9_vi, TTMP10_vi, TTMP11_vi,
                    TTMP12_vi, TTMP13_vi, TTMP14_vi, TTMP15_vi]>;

def TTMP0_TTMP1_TTMP2_TTMP3_TTMP4_TTMP5_TTMP6_TTMP7_TTMP8_TTMP9_TTMP10_TTMP11_TTMP12_TTMP13_TTMP14_TTMP15_gfx9_gfx10 :
  TmpRegTuplesBase<0, 16,
                   [TTMP0_gfx9_gfx10, TTMP1_gfx9_gfx10, TTMP2_gfx9_gfx10, TTMP3_gfx9_gfx10,
                    TTMP4_gfx9_gfx10, TTMP5_gfx9_gfx10, TTMP6_gfx9_gfx10, TTMP7_gfx9_gfx10,
                    TTMP8_gfx9_gfx10, TTMP9_gfx9_gfx10, TTMP10_gfx9_gfx10, TTMP11_gfx9_gfx10,
                    TTMP12_gfx9_gfx10, TTMP13_gfx9_gfx10, TTMP14_gfx9_gfx10, TTMP15_gfx9_gfx10]>;

// VGPR 32-bit registers
// i16/f16 only on VI+
def VGPR_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                            (add (sequence "VGPR%u", 0, 255)), Reg32> {
  let AllocationPriority = 1;
  let Size = 32;
}

// VGPR 64-bit registers
def VGPR_64 : RegisterTuples<getSubRegs<2>.ret,
                             [(add (trunc VGPR_32, 255)),
                              (add (shl VGPR_32, 1))]>;

// VGPR 96-bit registers
def VGPR_96 : RegisterTuples<getSubRegs<3>.ret,
                             [(add (trunc VGPR_32, 254)),
                              (add (shl VGPR_32, 1)),
                              (add (shl VGPR_32, 2))]>;

// VGPR 128-bit registers
def VGPR_128 : RegisterTuples<getSubRegs<4>.ret,
                              [(add (trunc VGPR_32, 253)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3))]>;

// VGPR 160-bit registers
def VGPR_160 : RegisterTuples<getSubRegs<5>.ret,
                             [(add (trunc VGPR_32, 252)),
                              (add (shl VGPR_32, 1)),
                              (add (shl VGPR_32, 2)),
                              (add (shl VGPR_32, 3)),
                              (add (shl VGPR_32, 4))]>;

// VGPR 256-bit registers
def VGPR_256 : RegisterTuples<getSubRegs<8>.ret,
                              [(add (trunc VGPR_32, 249)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3)),
                               (add (shl VGPR_32, 4)),
                               (add (shl VGPR_32, 5)),
                               (add (shl VGPR_32, 6)),
                               (add (shl VGPR_32, 7))]>;

// VGPR 512-bit registers
def VGPR_512 : RegisterTuples<getSubRegs<16>.ret,
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

// VGPR 1024-bit registers
def VGPR_1024 : RegisterTuples<getSubRegs<32>.ret,
                              [(add (trunc VGPR_32, 225)),
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
                               (add (shl VGPR_32, 15)),
                               (add (shl VGPR_32, 16)),
                               (add (shl VGPR_32, 17)),
                               (add (shl VGPR_32, 18)),
                               (add (shl VGPR_32, 19)),
                               (add (shl VGPR_32, 20)),
                               (add (shl VGPR_32, 21)),
                               (add (shl VGPR_32, 22)),
                               (add (shl VGPR_32, 23)),
                               (add (shl VGPR_32, 24)),
                               (add (shl VGPR_32, 25)),
                               (add (shl VGPR_32, 26)),
                               (add (shl VGPR_32, 27)),
                               (add (shl VGPR_32, 28)),
                               (add (shl VGPR_32, 29)),
                               (add (shl VGPR_32, 30)),
                               (add (shl VGPR_32, 31))]>;

// AccVGPR 32-bit registers
def AGPR_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                            (add (sequence "AGPR%u", 0, 255)), Reg32> {
  let AllocationPriority = 1;
  let Size = 32;
}

// AGPR 64-bit registers
def AGPR_64 : RegisterTuples<getSubRegs<2>.ret,
                             [(add (trunc AGPR_32, 255)),
                              (add (shl AGPR_32, 1))]>;

// AGPR 128-bit registers
def AGPR_128 : RegisterTuples<getSubRegs<4>.ret,
                              [(add (trunc AGPR_32, 253)),
                               (add (shl AGPR_32, 1)),
                               (add (shl AGPR_32, 2)),
                               (add (shl AGPR_32, 3))]>;

// AGPR 512-bit registers
def AGPR_512 : RegisterTuples<getSubRegs<16>.ret,
                              [(add (trunc AGPR_32, 241)),
                               (add (shl AGPR_32, 1)),
                               (add (shl AGPR_32, 2)),
                               (add (shl AGPR_32, 3)),
                               (add (shl AGPR_32, 4)),
                               (add (shl AGPR_32, 5)),
                               (add (shl AGPR_32, 6)),
                               (add (shl AGPR_32, 7)),
                               (add (shl AGPR_32, 8)),
                               (add (shl AGPR_32, 9)),
                               (add (shl AGPR_32, 10)),
                               (add (shl AGPR_32, 11)),
                               (add (shl AGPR_32, 12)),
                               (add (shl AGPR_32, 13)),
                               (add (shl AGPR_32, 14)),
                               (add (shl AGPR_32, 15))]>;

// AGPR 1024-bit registers
def AGPR_1024 : RegisterTuples<getSubRegs<32>.ret,
                              [(add (trunc AGPR_32, 225)),
                               (add (shl AGPR_32, 1)),
                               (add (shl AGPR_32, 2)),
                               (add (shl AGPR_32, 3)),
                               (add (shl AGPR_32, 4)),
                               (add (shl AGPR_32, 5)),
                               (add (shl AGPR_32, 6)),
                               (add (shl AGPR_32, 7)),
                               (add (shl AGPR_32, 8)),
                               (add (shl AGPR_32, 9)),
                               (add (shl AGPR_32, 10)),
                               (add (shl AGPR_32, 11)),
                               (add (shl AGPR_32, 12)),
                               (add (shl AGPR_32, 13)),
                               (add (shl AGPR_32, 14)),
                               (add (shl AGPR_32, 15)),
                               (add (shl AGPR_32, 16)),
                               (add (shl AGPR_32, 17)),
                               (add (shl AGPR_32, 18)),
                               (add (shl AGPR_32, 19)),
                               (add (shl AGPR_32, 20)),
                               (add (shl AGPR_32, 21)),
                               (add (shl AGPR_32, 22)),
                               (add (shl AGPR_32, 23)),
                               (add (shl AGPR_32, 24)),
                               (add (shl AGPR_32, 25)),
                               (add (shl AGPR_32, 26)),
                               (add (shl AGPR_32, 27)),
                               (add (shl AGPR_32, 28)),
                               (add (shl AGPR_32, 29)),
                               (add (shl AGPR_32, 30)),
                               (add (shl AGPR_32, 31))]>;

//===----------------------------------------------------------------------===//
//  Register classes used as source and destination
//===----------------------------------------------------------------------===//

def Pseudo_SReg_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
  (add FP_REG, SP_REG, SCRATCH_WAVE_OFFSET_REG), Reg32> {
  let isAllocatable = 0;
  let CopyCost = -1;
}

def Pseudo_SReg_128 : RegisterClass<"AMDGPU", [v4i32, v2i64, v2f64], 32,
  (add PRIVATE_RSRC_REG), Reg128> {
  let isAllocatable = 0;
  let CopyCost = -1;
}

def LDS_DIRECT_CLASS : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
  (add LDS_DIRECT), Reg32> {
  let isAllocatable = 0;
  let CopyCost = -1;
}

// Subset of SReg_32 without M0 for SMRD instructions and alike.
// See comments in SIInstructions.td for more info.
def SReg_32_XM0_XEXEC : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SGPR_32, VCC_LO, VCC_HI, FLAT_SCR_LO, FLAT_SCR_HI, XNACK_MASK_LO, XNACK_MASK_HI,
   SGPR_NULL, TTMP_32, TMA_LO, TMA_HI, TBA_LO, TBA_HI, SRC_SHARED_BASE, SRC_SHARED_LIMIT,
   SRC_PRIVATE_BASE, SRC_PRIVATE_LIMIT, SRC_POPS_EXITING_WAVE_ID,
   SRC_VCCZ, SRC_EXECZ, SRC_SCC), Reg32> {
  let AllocationPriority = 10;
}

def SReg_32_XEXEC_HI : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SReg_32_XM0_XEXEC, EXEC_LO, M0_CLASS), Reg32> {
  let AllocationPriority = 10;
}

def SReg_32_XM0 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SReg_32_XM0_XEXEC, EXEC_LO, EXEC_HI), Reg32> {
  let AllocationPriority = 10;
}

// Register class for all scalar registers (SGPRs + Special Registers)
def SReg_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SReg_32_XM0, M0_CLASS, EXEC_LO, EXEC_HI, SReg_32_XEXEC_HI), Reg32> {
  let AllocationPriority = 10;
}

def SRegOrLds_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SReg_32_XM0, M0_CLASS, EXEC_LO, EXEC_HI, SReg_32_XEXEC_HI, LDS_DIRECT_CLASS),
  Reg32> {
  let isAllocatable = 0;
}

def SGPR_64 : RegisterClass<"AMDGPU", [v2i32, i64, v2f32, f64, v4i16, v4f16], 32,
                            (add SGPR_64Regs), Reg64> {
  let CopyCost = 1;
  let AllocationPriority = 11;
}

// CCR (call clobbered registers) SGPR 64-bit registers
def CCR_SGPR_64 : RegisterClass<"AMDGPU", SGPR_64.RegTypes, 32,
                                (add (trunc SGPR_64, 16)), Reg64> {
  let CopyCost = SGPR_64.CopyCost;
  let AllocationPriority = SGPR_64.AllocationPriority;
}

def TTMP_64 : RegisterClass<"AMDGPU", [v2i32, i64, f64, v4i16, v4f16], 32,
                            (add TTMP_64Regs)> {
  let isAllocatable = 0;
}

def SReg_64_XEXEC : RegisterClass<"AMDGPU", [v2i32, i64, v2f32, f64, i1, v4i16, v4f16], 32,
  (add SGPR_64, VCC, FLAT_SCR, XNACK_MASK, TTMP_64, TBA, TMA), Reg64> {
  let CopyCost = 1;
  let AllocationPriority = 13;
}

def SReg_64 : RegisterClass<"AMDGPU", [v2i32, i64, v2f32, f64, i1, v4i16, v4f16], 32,
  (add SReg_64_XEXEC, EXEC), Reg64> {
  let CopyCost = 1;
  let AllocationPriority = 13;
}

def SReg_1_XEXEC : RegisterClass<"AMDGPU", [i1], 32,
  (add SReg_64_XEXEC, SReg_32_XM0_XEXEC)> {
  let CopyCost = 1;
  let isAllocatable = 0;
}

def SReg_1 : RegisterClass<"AMDGPU", [i1], 32,
  (add SReg_1_XEXEC, EXEC, EXEC_LO)> {
  let CopyCost = 1;
  let isAllocatable = 0;
}

// Requires 2 s_mov_b64 to copy
let CopyCost = 2 in {

// There are no 3-component scalar instructions, but this is needed
// for symmetry with VGPRs.
def SGPR_96 : RegisterClass<"AMDGPU", [v3i32, v3f32], 32,
  (add SGPR_96Regs), Reg96> {
  let AllocationPriority = 14;
}

def SReg_96 : RegisterClass<"AMDGPU", [v3i32, v3f32], 32,
  (add SGPR_96), Reg96> {
  let AllocationPriority = 14;
}

def SGPR_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64], 32,
                             (add SGPR_128Regs), Reg128> {
  let AllocationPriority = 15;
}

def TTMP_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64], 32,
                             (add TTMP_128Regs)> {
  let isAllocatable = 0;
}

def SReg_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64, v2f64], 32,
                             (add SGPR_128, TTMP_128), Reg128> {
  let AllocationPriority = 15;
}

} // End CopyCost = 2

// There are no 5-component scalar instructions, but this is needed
// for symmetry with VGPRs.
def SGPR_160 : RegisterClass<"AMDGPU", [v5i32, v5f32], 32,
                             (add SGPR_160Regs), Reg160> {
  let AllocationPriority = 16;
}

def SReg_160 : RegisterClass<"AMDGPU", [v5i32, v5f32], 32,
                             (add SGPR_160), Reg160> {
  let AllocationPriority = 16;
}

def SGPR_256 : RegisterClass<"AMDGPU", [v8i32, v8f32], 32, (add SGPR_256Regs),
                             Reg256> {
  let AllocationPriority = 17;
}

def TTMP_256 : RegisterClass<"AMDGPU", [v8i32, v8f32], 32, (add TTMP_256Regs)> {
  let isAllocatable = 0;
}

def SReg_256 : RegisterClass<"AMDGPU", [v8i32, v8f32], 32,
                             (add SGPR_256, TTMP_256), Reg256> {
  // Requires 4 s_mov_b64 to copy
  let CopyCost = 4;
  let AllocationPriority = 17;
}

def SGPR_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32,
                             (add SGPR_512Regs), Reg512> {
  let AllocationPriority = 18;
}

def TTMP_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32,
                             (add TTMP_512Regs)> {
  let isAllocatable = 0;
}

def SReg_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32,
                             (add SGPR_512, TTMP_512), Reg512> {
  // Requires 8 s_mov_b64 to copy
  let CopyCost = 8;
  let AllocationPriority = 18;
}

def VRegOrLds_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                                 (add VGPR_32, LDS_DIRECT_CLASS), Reg32> {
  let isAllocatable = 0;
}

def SGPR_1024 : RegisterClass<"AMDGPU", [v32i32, v32f32], 32,
                              (add SGPR_1024Regs), Reg1024> {
  let AllocationPriority = 19;
}

def SReg_1024 : RegisterClass<"AMDGPU", [v32i32, v32f32], 32,
                              (add SGPR_1024), Reg1024> {
  let CopyCost = 16;
  let AllocationPriority = 19;
}

// Register class for all vector registers (VGPRs + Interploation Registers)
def VReg_64 : RegisterClass<"AMDGPU", [i64, f64, v2i32, v2f32, v4f16, v4i16], 32,
                            (add VGPR_64), Reg64> {
  let Size = 64;

  // Requires 2 v_mov_b32 to copy
  let CopyCost = 2;
  let AllocationPriority = 2;
}

def VReg_96 : RegisterClass<"AMDGPU", [v3i32, v3f32], 32, (add VGPR_96), Reg96> {
  let Size = 96;

  // Requires 3 v_mov_b32 to copy
  let CopyCost = 3;
  let AllocationPriority = 3;
}

def VReg_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64, v2f64], 32,
                             (add VGPR_128), Reg128> {
  let Size = 128;

  // Requires 4 v_mov_b32 to copy
  let CopyCost = 4;
  let AllocationPriority = 4;
}

def VReg_160 : RegisterClass<"AMDGPU", [v5i32, v5f32], 32,
                             (add VGPR_160), Reg160> {
  let Size = 160;

  // Requires 5 v_mov_b32 to copy
  let CopyCost = 5;
  let AllocationPriority = 5;
}

def VReg_256 : RegisterClass<"AMDGPU", [v8i32, v8f32], 32,
                             (add VGPR_256), Reg256> {
  let Size = 256;
  let CopyCost = 8;
  let AllocationPriority = 6;
}

def VReg_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32,
                             (add VGPR_512), Reg512> {
  let Size = 512;
  let CopyCost = 16;
  let AllocationPriority = 7;
}

def VReg_1024 : RegisterClass<"AMDGPU", [v32i32, v32f32], 32,
                              (add VGPR_1024), Reg1024> {
  let Size = 1024;
  let CopyCost = 32;
  let AllocationPriority = 8;
}

def AReg_64 : RegisterClass<"AMDGPU", [i64, f64, v2i32, v2f32, v4f16, v4i16], 32,
                            (add AGPR_64), Reg64> {
  let Size = 64;

  let CopyCost = 5;
  let AllocationPriority = 2;
}

def AReg_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64, v2f64], 32,
                             (add AGPR_128), Reg128> {
  let Size = 128;

  // Requires 4 v_accvgpr_write and 4 v_accvgpr_read to copy + burn 1 vgpr
  let CopyCost = 9;
  let AllocationPriority = 4;
}

def AReg_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32,
                             (add AGPR_512), Reg512> {
  let Size = 512;
  let CopyCost = 33;
  let AllocationPriority = 7;
}

def AReg_1024 : RegisterClass<"AMDGPU", [v32i32, v32f32], 32,
                              (add AGPR_1024), Reg1024> {
  let Size = 1024;
  let CopyCost = 65;
  let AllocationPriority = 8;
}

def VReg_1 : RegisterClass<"AMDGPU", [i1], 32, (add VGPR_32), Reg32> {
  let Size = 32;
}

def VS_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                          (add VGPR_32, SReg_32, LDS_DIRECT_CLASS), Reg32> {
  let isAllocatable = 0;
}

def VS_64 : RegisterClass<"AMDGPU", [i64, f64], 32, (add VReg_64, SReg_64),
                          Reg64> {
  let isAllocatable = 0;
}

def AV_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                          (add AGPR_32, VGPR_32), Reg32> {
  let isAllocatable = 0;
}

def AV_64 : RegisterClass<"AMDGPU", [i64, f64, v4f16], 32,
                          (add AReg_64, VReg_64), Reg64> {
  let isAllocatable = 0;
}

//===----------------------------------------------------------------------===//
//  Register operands
//===----------------------------------------------------------------------===//

class RegImmMatcher<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addRegOrImmOperands";
}

multiclass SIRegOperand32 <string rc, string MatchName, string opType,
                           string rc_suffix = "_32"> {
  let OperandNamespace = "AMDGPU" in {
    def _b16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_INT16";
      let ParserMatchClass = RegImmMatcher<MatchName#"B16">;
      let DecoderMethod = "decodeOperand_VSrc16";
    }

    def _f16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_FP16";
      let ParserMatchClass = RegImmMatcher<MatchName#"F16">;
      let DecoderMethod = "decodeOperand_" # rc # "_16";
    }

    def _b32 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_INT32";
      let ParserMatchClass = RegImmMatcher<MatchName#"B32">;
      let DecoderMethod = "decodeOperand_" # rc # rc_suffix;
    }

    def _f32 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_FP32";
      let ParserMatchClass = RegImmMatcher<MatchName#"F32">;
      let DecoderMethod = "decodeOperand_" # rc # rc_suffix;
    }

    def _v2b16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_V2INT16";
      let ParserMatchClass = RegImmMatcher<MatchName#"V2B16">;
      let DecoderMethod = "decodeOperand_VSrcV216";
    }

    def _v2f16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_V2FP16";
      let ParserMatchClass = RegImmMatcher<MatchName#"V2F16">;
      let DecoderMethod = "decodeOperand_VSrcV216";
    }
  }
}

multiclass SIRegOperand <string rc, string MatchName, string opType> :
  SIRegOperand32<rc, MatchName, opType> {
  let OperandNamespace = "AMDGPU" in {
    def _b64 : RegisterOperand<!cast<RegisterClass>(rc#"_64")> {
      let OperandType = opType#"_INT64";
      let ParserMatchClass = RegImmMatcher<MatchName#"B64">;
    }

    def _f64 : RegisterOperand<!cast<RegisterClass>(rc#"_64")> {
      let OperandType = opType#"_FP64";
      let ParserMatchClass = RegImmMatcher<MatchName#"F64">;
    }
  }
}

// FIXME: 64-bit sources can sometimes use 32-bit constants.
multiclass RegImmOperand <string rc, string MatchName>
  : SIRegOperand<rc, MatchName, "OPERAND_REG_IMM">;

multiclass RegInlineOperand <string rc, string MatchName>
  : SIRegOperand<rc, MatchName, "OPERAND_REG_INLINE_C">;

multiclass RegInlineOperand32 <string rc, string MatchName,
                               string rc_suffix = "_32">
  : SIRegOperand32<rc, MatchName, "OPERAND_REG_INLINE_C", rc_suffix>;

multiclass RegInlineOperandAC <string rc, string MatchName,
                               string rc_suffix = "_32">
  : SIRegOperand32<rc, MatchName, "OPERAND_REG_INLINE_AC", rc_suffix>;

//===----------------------------------------------------------------------===//
//  SSrc_* Operands with an SGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

defm SSrc : RegImmOperand<"SReg", "SSrc">;

def SSrcOrLds_b32 : RegisterOperand<SRegOrLds_32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_IMM_INT32";
  let ParserMatchClass = RegImmMatcher<"SSrcOrLdsB32">;
}

//===----------------------------------------------------------------------===//
//  SCSrc_* Operands with an SGPR or a inline constant
//===----------------------------------------------------------------------===//

defm SCSrc : RegInlineOperand<"SReg", "SCSrc"> ;

//===----------------------------------------------------------------------===//
//  VSrc_* Operands with an SGPR, VGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

defm VSrc : RegImmOperand<"VS", "VSrc">;

def VSrc_128 : RegisterOperand<VReg_128> {
  let DecoderMethod = "DecodeVS_128RegisterClass";
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
//  ASrc_* Operands with an AccVGPR
//===----------------------------------------------------------------------===//

def ARegSrc_32 : RegisterOperand<AGPR_32> {
  let DecoderMethod = "DecodeAGPR_32RegisterClass";
  let EncoderMethod = "getAVOperandEncoding";
}

//===----------------------------------------------------------------------===//
//  VCSrc_* Operands with an SGPR, VGPR or an inline constant
//===----------------------------------------------------------------------===//

defm VCSrc : RegInlineOperand<"VS", "VCSrc">;

//===----------------------------------------------------------------------===//
//  VISrc_* Operands with a VGPR or an inline constant
//===----------------------------------------------------------------------===//

defm VISrc : RegInlineOperand32<"VGPR", "VISrc">;

//===----------------------------------------------------------------------===//
//  AVSrc_* Operands with an AGPR or VGPR
//===----------------------------------------------------------------------===//

def AVSrc_32 : RegisterOperand<AV_32> {
  let DecoderMethod = "DecodeAV_32RegisterClass";
  let EncoderMethod = "getAVOperandEncoding";
}

def AVSrc_64 : RegisterOperand<AV_64> {
  let DecoderMethod = "DecodeAV_64RegisterClass";
  let EncoderMethod = "getAVOperandEncoding";
}

//===----------------------------------------------------------------------===//
//  ACSrc_* Operands with an AGPR or an inline constant
//===----------------------------------------------------------------------===//

defm AISrc      : RegInlineOperandAC<"AGPR", "AISrc">;
defm AISrc_128  : RegInlineOperandAC<"AReg", "AISrc_128",  "_128">;
defm AISrc_512  : RegInlineOperandAC<"AReg", "AISrc_512",  "_512">;
defm AISrc_1024 : RegInlineOperandAC<"AReg", "AISrc_1024", "_1024">;
