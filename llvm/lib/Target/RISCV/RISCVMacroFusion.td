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
def : SimpleFusionFor<TuneLUIADDIFusion,
                      CheckOpcode<[LUI]>,
                      CheckOpcode<[ADDI, ADDIW]>>;

// Fuse AUIPC followed by ADDI:
//   auipc rd, imm20
//   addi rd, rd, imm12
def : SimpleFusionFor<TuneAUIPCADDIFusion,
                      CheckOpcode<[AUIPC]>,
                      CheckOpcode<[ADDI]>>;

// Fuse zero extension of halfword:
//   slli rd, rs1, 48
//   srli rd, rd, 48
def : SimpleFusionFor<TuneZExtHFusion,
                      CheckAll<[
                        CheckOpcode<[SLLI]>,
                        CheckImmOperand<2, 48>
                      ]>,
                      CheckAll<[
                        CheckOpcode<[SRLI]>,
                        CheckImmOperand<2, 48>
                      ]>>;

// Fuse zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, 32
def : SimpleFusionFor<TuneZExtWFusion,
                      CheckAll<[
                        CheckOpcode<[SLLI]>,
                        CheckImmOperand<2, 32>
                      ]>,
                      CheckAll<[
                        CheckOpcode<[SRLI]>,
                        CheckImmOperand<2, 32>
                      ]>>;

// Fuse shifted zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, x
//   where 0 <= x < 32
def : SimpleFusionFor<TuneShiftedZExtWFusion,
                      CheckAll<[
                        CheckOpcode<[SLLI]>,
                        CheckImmOperand<2, 32>
                      ]>,
                      CheckAll<[
                        CheckOpcode<[SRLI]>,
                        CheckImmOperandRange<2, 0, 31>
                      ]>>;

// Fuse load with add:
//   add rd, rs1, rs2
//   ld rd, 0(rd)
def : SimpleFusionFor<TuneLDADDFusion,
                      CheckOpcode<[ADD]>,
                      CheckAll<[
                        CheckOpcode<[LD]>,
                        CheckImmOperand<2, 0>
                      ]>>;

defvar Load = [LB, LH, LW, LD, LBU, LHU, LWU];
defvar LoadStore = [LB, LBU, LH, LHU, LW, LWU, LD,
                    FLH, FLW, FLD,
                    SB, SH, SW, SD,
                    FSH, FSW, FSD];
defvar LogicOp = [AND, OR, XOR];
defvar LogicImmOp = [ANDI, ORI, XORI];
defvar ShiftLeft = [SLLI, SLLIW];
defvar ShiftRight = [SRLI, SRLIW, SRAI, SRAIW];

// Fuse add(.uw) followed by a load (lb, lh, lw, ld, lbu, lhu, lwu):
//   add(.uw) rd, rs1, rs2
//   load rd, imm12(rd)
def : SimpleFusionFor<TuneADDLoadFusion,
                      CheckOpcode<[ADD, ADD_UW]>,
                      CheckOpcode<Load>>;

// Fuse AUIPC followed by by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   auipc rd, imm20
//   load rd, imm12(rd)
def : SimpleFusionFor<TuneAUIPCLoadFusion,
                      CheckOpcode<[AUIPC]>,
                      CheckOpcode<Load>>;

// Fuse LUI followed by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   lui rd, imm[31:12]
//   load rd, imm12(rd)
def : SimpleFusionFor<TuneLUILoadFusion,
                      CheckOpcode<[LUI]>,
                      CheckOpcode<Load>>;

defvar RV32GPRLoadStore = [LB, LBU, LH, LHU, LW, SB, SH, SW];
def : FusionFor<TuneQCELILoadStoreFusion,
                [
                  SecondFusionPredicateWithMCInstPredicate<CheckOpcode<RV32GPRLoadStore>>,
                  WildcardTrue,
                  FirstFusionPredicateWithMCInstPredicate<CheckOpcode<[QC_E_LI]>>,
                  OneUse,
                  TieReg<0, 1>,
                ]>;

// Bitfield extract fusion: similar to TuneShiftedZExtWFusion
// but without the immediate restriction
//   slli rd, rs1, imm12
//   srli rd, rd, imm12
def : SimpleFusionFor<TuneBFExtFusion,
                      CheckOpcode<[SLLI]>,
                      CheckOpcode<[SRLI]>>;

// Fuse ADDI followed by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   addi rd, rs1, imm12
//   load rd, imm12(rd)
def : SimpleFusionFor<TuneADDILoadFusion,
                      CheckOpcode<[ADDI]>,
                      CheckOpcode<Load>>;

// Fuse shXadd(.uw) followed by a load (lb, lh, lw, ld, lbu, lhu, lwu)
//   shXadd(.uw) rd, rs1, rs2
//   load rd, imm12(rd)
def : SimpleFusionFor<TuneSHXADDLoadFusion,
                      CheckOpcode<[SH1ADD, SH2ADD, SH3ADD, SH1ADD_UW, SH2ADD_UW, SH3ADD_UW]>,
                      CheckOpcode<Load>>;

// Fuse logic operation followed by another logic operation:
//   and/or/xor rd, rs1, rs2
//   and/or/xor rd, rd, rs3
let IsCommutable = 1 in
def : SimpleFusionFor<TuneFusionLogicRegReg,
                      CheckOpcode<LogicOp>,
                      CheckOpcode<LogicOp>>;

// Fuse logic operation followed by logic immediate operation:
//   and/or/xor rd, rs1, rs2
//   andi/ori/xori rd, rd, imm
def : SimpleFusionFor<TuneFusionLogicRegImm,
                      CheckOpcode<LogicOp>,
                      CheckOpcode<LogicImmOp>>;

// Fuse logic immediate operation followed by logic operation:
//   andi/ori/xori rd, rs1, imm
//   and/or/xor rd, rd, rs2
let IsCommutable = 1 in
def : SimpleFusionFor<TuneFusionLogicImmReg,
                      CheckOpcode<LogicImmOp>,
                      CheckOpcode<LogicOp>>;

// Fuse multiply followed by add:
//   mul(w) rd, rs1, rs2
//   add(w) rd, rd, rs3
def : SimpleFusionFor<TuneFusionMulAdd,
                      CheckOpcode<[MUL, MULW]>,
                      CheckOpcode<[ADD, ADDW]>,
                      epilog = [
                        // Check that both instructions have the same width
                        FusionPredicateWithCode<[{
                          if ((FirstMI->getDesc().TSFlags & RISCVII::IsSignExtendingOpWMask) !=
                              (SecondMI.getDesc().TSFlags & RISCVII::IsSignExtendingOpWMask))
                            return false;
                        }]>,
                        SecondFusionPredicateWithMCInstPredicate<
                          CheckNot<CheckSameRegOperand<0, 2>>
                        >
                      ]>;

// Fuse add with load or store with zero offset:
//   add rd, rs1, rs2
//   load/store rt, 0(rd)
def : FusionFor<TuneFusionAddMem,
                [
                  SecondFusionPredicateWithMCInstPredicate<
                    CheckAll<[
                      CheckOpcode<LoadStore>,
                      CheckImmOperand<2, 0>
                    ]>
                  >,
                  WildcardTrue,
                  FirstFusionPredicateWithMCInstPredicate<
                    CheckAll<[
                      CheckOpcode<[ADD]>,
                      CheckNot<CheckSameRegOperand<0, 1>>,
                      CheckNot<CheckSameRegOperand<0, 2>>
                    ]>
                  >,
                  TieReg<0, 1>
                ]>;

// Fuse shift left followed by shift right for bit extraction:
//   slli(w) rd, rs1, imm1
//   srli(w)/srai(w) rd, rd, imm2
//   where imm1 <= imm2
def : SimpleFusionFor<TuneFusionShiftBitExtract,
                      CheckOpcode<ShiftLeft>,
                      CheckOpcode<ShiftRight>,
                      epilog = [
                        // Check that both instructions have the same width
                        FusionPredicateWithCode<[{
                          if ((FirstMI->getDesc().TSFlags & RISCVII::IsSignExtendingOpWMask) !=
                              (SecondMI.getDesc().TSFlags & RISCVII::IsSignExtendingOpWMask))
                            return false;
                        }]>,
                        // Check that FirstMI's immediate <= SecondMI's immediate
                        FusionPredicateWithCode<[{
                          if (!(FirstMI->getOperand(2).isImm() &&
                                SecondMI.getOperand(2).isImm() &&
                                FirstMI->getOperand(2).getImm() <=
                                SecondMI.getOperand(2).getImm()))
                            return false;
                        }]>,
                      ]>;


// Xqci macro fusions.
include "RISCVMacroFusionXQCI.td"
