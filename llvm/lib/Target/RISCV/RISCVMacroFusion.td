//==----- RISCVMacroFusion.td - Macro Fusion Definitions -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the macro fusion predicators.

// Fuse LUI followed by ADDI or ADDIW:
//   rd = imm[31:0] which decomposes to
//   lui rd, imm[31:12]
//   addi(w) rd, rd, imm[11:0]
def TuneLUIADDIFusion
  : SimpleFusion<"lui-addi-fusion", "HasLUIADDIFusion",
                 "Enable LUI+ADDI macro fusion",
                 CheckOpcode<[LUI]>,
                 CheckOpcode<[ADDI, ADDIW]>>;

// Fuse AUIPC followed by ADDI:
//   auipc rd, imm20
//   addi rd, rd, imm12
def TuneAUIPCADDIFusion
  : SimpleFusion<"auipc-addi-fusion", "HasAUIPCADDIFusion",
                 "Enable AUIPC+ADDI macrofusion",
                 CheckOpcode<[AUIPC]>,
                 CheckOpcode<[ADDI]>>;

// Fuse zero extension of halfword:
//   slli rd, rs1, 48
//   srli rd, rd, 48
def TuneZExtHFusion
  : SimpleFusion<"zexth-fusion", "HasZExtHFusion",
                 "Enable SLLI+SRLI to be fused to zero extension of halfword",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 48>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 48>
                 ]>>;

// Fuse zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, 32
def TuneZExtWFusion
  : SimpleFusion<"zextw-fusion", "HasZExtWFusion",
                 "Enable SLLI+SRLI to be fused to zero extension of word",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>>;

// Fuse shifted zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, x
//   where 0 <= x < 32
def TuneShiftedZExtWFusion
  : SimpleFusion<"shifted-zextw-fusion", "HasShiftedZExtWFusion",
                 "Enable SLLI+SRLI to be fused when computing (shifted) word zero extension",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperandRange<2, 0, 31>
                 ]>>;

// Fuse load with add:
//   add rd, rs1, rs2
//   ld rd, 0(rd)
def TuneLDADDFusion
  : SimpleFusion<"ld-add-fusion", "HasLDADDFusion", "Enable LD+ADD macrofusion",
                 CheckOpcode<[ADD]>,
                 CheckAll<[
                   CheckOpcode<[LD]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 0>
                 ]>>;

// Fuse add with lw:
//   add rd, rs1, rs2
//   lw rd, 0(rd)
def TuneADDLWFusion
  : SimpleFusion<"add-lw-fusion", "HasADDLWFusion", "Enable ADD+LW macrofusion",
                 CheckOpcode<[ADD]>,
                 CheckAll<[
                   CheckOpcode<[LW]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 0>
                 ]>>;

// Fuse AUIPC followed by LD:
//   auipc rd, imm20
//   ld rd, imm12(rd)
def TuneAUIPCLDFusion
  : SimpleFusion<"auipc-ld-fusion", "HasAUIPCLDFusion",
                 "Enable AUIPC+LD macrofusion",
                 CheckOpcode<[AUIPC]>,
                 CheckOpcode<[LD]>>;

// Fuse LUI followed by LD:
//   lui rd, imm[31:12]
//   ld rd, imm12(rd)
def TuneLUILDFusion
  : SimpleFusion<"lui-ld-fusion", "HasLUILDFusion",
                 "Enable LUI+LD macrofusion",
                 CheckOpcode<[LUI]>,
                 CheckOpcode<[LD]>>;

// Bitfield extract fusion: similar to TuneShiftedZExtWFusion
// but without the immediate restriction
//   slli rd, rs1, imm12
//   srli rd, rd, imm12
def TuneBFExtFusion
  : SimpleFusion<"bfext-fusion", "HasBFExtFusion",
                 "Enable SLLI+SRLI (bitfield extract) macrofusion",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                 ]>>;

// Fuse ADDI followed by LD
//   addi rd, rs1, imm12
//   ld rd, imm12(rd)
def TuneADDILDFusion
  : SimpleFusion<"addi-ld-fusion", "HasADDILDFusion",
                 "Enable ADDI+LD macrofusion",
                 CheckOpcode<[ADDI]>,
                 CheckOpcode<[LD]>>;

// Fuse ADDI followed by LW
//   addi rd, rs1, imm12
//   lw rd, imm12(rd)
def TuneADDILWFusion
  : SimpleFusion<"addi-lw-fusion", "HasADDILWFusion",
                 "Enable ADDI+LW macrofusion",
                 CheckOpcode<[ADDI]>,
                 CheckOpcode<[LW]>>;

// Fuse ADDUW followed by LW
//   adduw rd, rs1, rs2
//   lw rd, imm12(rd)
def TuneADDUWLWFusion
  : SimpleFusion<"adduw-lw-fusion", "HasADDUWLWFusion",
                 "Enable ADD_UW+LW macrofusion",
                 CheckOpcode<[ADD_UW]>,
                 CheckOpcode<[LW]>>;

// Fuse SHXADD followed by a load (lb, lh, lw, ld)
//   shXadd rd, rs1, rs2
//   load rd, imm12(rd)
def TuneSHXADDLoadFusion
  : SimpleFusion<"shxadd-load-fusion", "HasSHXADDLoadFusion",
                 "Enable SH(1|2|3)ADD + load macrofusion",
                 CheckOpcode<[SH1ADD, SH2ADD, SH3ADD]>,
                 CheckOpcode<[LB, LH, LW, LD]>>;

// Fuse SHXADD.UW followed by a load (lb, lh, lw, ld)
//   shXadd.uw rd, rs1, rs2
//   load rd, imm12(rd)
def TuneSHXADDUWLoadFusion
  : SimpleFusion<"shxadduw-load-fusion", "HasSHXADDUWLoadFusion",
                 "Enable SH(1|2|3)ADDUW + load macrofusion",
                 CheckOpcode<[SH1ADD_UW, SH2ADD_UW, SH3ADD_UW]>,
                 CheckOpcode<[LB, LH, LW, LD]>>;
