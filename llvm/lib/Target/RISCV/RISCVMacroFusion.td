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

defvar Load = [LB, LH, LW, LD, LBU, LHU, LWU];

// Fuse add(.uw) followed by a load (lb, lh, lw, ld, lbu, lhu, lwu):
//   add(.uw) rd, rs1, rs2
//   load rd, imm12(rd)
def TuneADDLoadFusion
  : SimpleFusion<"add-load-fusion", "HasADDLoadFusion", "Enable ADD(.UW) + load macrofusion",
                 CheckOpcode<[ADD, ADD_UW]>,
                 CheckOpcode<Load>>;

// Fuse AUIPC followed by by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   auipc rd, imm20
//   load rd, imm12(rd)
def TuneAUIPCLoadFusion
  : SimpleFusion<"auipc-load-fusion", "HasAUIPCLoadFusion",
                 "Enable AUIPC + load macrofusion",
                 CheckOpcode<[AUIPC]>,
                 CheckOpcode<Load>>;

// Fuse LUI followed by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   lui rd, imm[31:12]
//   load rd, imm12(rd)
def TuneLUILoadFusion
  : SimpleFusion<"lui-load-fusion", "HasLUILoadFusion",
                 "Enable LUI + load macrofusion",
                 CheckOpcode<[LUI]>,
                 CheckOpcode<Load>>;

// Bitfield extract fusion: similar to TuneShiftedZExtWFusion
// but without the immediate restriction
//   slli rd, rs1, imm12
//   srli rd, rd, imm12
def TuneBFExtFusion
  : SimpleFusion<"bfext-fusion", "HasBFExtFusion",
                 "Enable SLLI+SRLI (bitfield extract) macrofusion",
                 CheckOpcode<[SLLI]>,
                 CheckOpcode<[SRLI]>>;

// Fuse ADDI followed by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   addi rd, rs1, imm12
//   load rd, imm12(rd)
def TuneADDILoadFusion
  : SimpleFusion<"addi-load-fusion", "HasADDILoadFusion",
                 "Enable ADDI + load macrofusion",
                 CheckOpcode<[ADDI]>,
                 CheckOpcode<Load>>;

// Fuse shXadd(.uw) followed by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   shXadd(.uw) rd, rs1, rs2
//   load rd, imm12(rd)
def TuneSHXADDLoadFusion
  : SimpleFusion<"shxadd-load-fusion", "HasSHXADDLoadFusion",
                 "Enable SH(1|2|3)ADD(.UW) + load macrofusion",
                 CheckOpcode<[SH1ADD, SH2ADD, SH3ADD, SH1ADD_UW, SH2ADD_UW, SH3ADD_UW]>,
                 CheckOpcode<Load>>;
