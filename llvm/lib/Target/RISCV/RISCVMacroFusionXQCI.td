//==---- RISCVMacroFusionXQCI.td - Macro Fusions for XQCI -----*- tablegen-*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe macro fusion predicators applicable to
// Xqci instructions.

// Fuse two instructions where the destination register of the first
// instruction is the same as the destination register of the second
// instruction.
class FirstRdEqSecondRdFusion<string name, string fieldName, string desc,
                              MCInstPredicate firstPred,
                              MCInstPredicate secondPred,
                              int secondOpIdx=0, int firstOpIdx=0>
  : Fusion<name, fieldName, desc,
           [
             SecondFusionPredicateWithMCInstPredicate<secondPred>,
             WildcardTrue,
             FirstFusionPredicateWithMCInstPredicate<firstPred>,
             TieReg<firstOpIdx, secondOpIdx>
           ]>;

defvar LoadImmOp = [LUI, QC_E_LI, QC_LI];
// 48-bit load store
defvar LongLoadStoreXqciOp = [QC_E_LB, QC_E_LBU, QC_E_LH, QC_E_LHU, QC_E_LW,
                              QC_E_SB, QC_E_SH, QC_E_SW];
// 48-bit logical operations
defvar LongLogicalXqciOp = [QC_E_XORAI, QC_E_ORAI, QC_E_ANDAI, QC_E_ADDAI];

// Fuse load-immediate/move-immediate followed by arithmetic/logic.
def TuneMOVIMMALUXqciFusion
  : SimpleFusion<"xqci-movimm-alu", "HasMOVIMMALUXqciFusion",
                 "Enable MOVIMM + ALU operations macrofusion",
                 CheckAny<[
                   CheckAll<[
                     CheckOpcode<[ADDI]>,
                     CheckRegOperand<1, X0>
                   ]>,
                   CheckOpcode<LoadImmOp>,
                   CheckOpcode<[AUIPC]>,
                 ]>,
                 CheckOpcode<[
                   ADD, ADDI, SUB, MUL,
                   AND, ANDI, ANDN, OR, ORI, ORN, XOR, XORI, XNOR,
                   QC_E_ANDI, QC_E_ORI, QC_E_XORI,
                   SLLI, SRLI, SRAI, QC_LI, QC_E_LI,
                   MAX, MAXU, MIN, MINU
                 ]>>;

// Fuse load-immediate/move-immediate followed by multiply.
def TuneMOVIMMMULXqciFusion
  : SimpleFusion<"xqci-movimm-mul", "HasMOVIMMMULXqciFusion",
                 "Enable MOVIMM + multiply macrofusion",
                 CheckAny<[
                   CheckAll<[
                     CheckOpcode<[ADDI]>,
                     CheckRegOperand<1, X0>
                   ]>,
                   CheckOpcode<LoadImmOp>,
                 ]>,
                 CheckOpcode<[MUL]>>;

// Fuse load-immediate/move-immediate followed by long load/store.
def TuneMOVIMMLoadStoreXqciFusion
  : SimpleFusion<"xqci-movimm-ldst", "HasMOVIMMLoadStoreXqciFusion",
                 "Enable MOVIMM + long Load/Store macrofusion",
                 CheckAny<[
                   CheckAll<[
                     CheckOpcode<[ADDI]>,
                     CheckRegOperand<1, X0>
                   ]>,
                   CheckOpcode<LoadImmOp>,
                 ]>,
                 CheckOpcode<LongLoadStoreXqciOp>>;

// Fuse load-immediate/move-immediate followed by a jump.
def TuneMOVIMMJumpXqciFusion
  : SimpleFusion<"xqci-movimm-jump", "HasMOVIMMJumpXqciFusion",
                 "Enable MOVIMM + Jump macrofusion",
                 CheckAny<[
                   CheckAll<[
                     CheckOpcode<[ADDI]>,
                     CheckRegOperand<1, X0>
                   ]>,
                   CheckOpcode<LoadImmOp>,
                 ]>,
                 CheckOpcode<[JALR]>>;

// Fuse load-immediate/move-immediate followed by long logical operation.
// The destination register of the first operation is the same as the
// destination register of the second operation.
def TuneMOVIMMLongLogicalXciFusion
  : FirstRdEqSecondRdFusion<"xqci-movimm-longlogical",
                 "HasMOVIMMLongLogicalXqciFusion",
                 "Enable MOVIMM + long logical operation macrofusion",
                 CheckAny<[
                   CheckAll<[
                     CheckOpcode<[ADDI]>,
                     CheckRegOperand<1, X0>
                   ]>,
                   CheckOpcode<LoadImmOp>,
                 ]>,
                 CheckOpcode<LongLogicalXqciOp>>;

// Fuse move followed by long logical operation.
def TuneMovALUXqciFusion
  : FirstRdEqSecondRdFusion<"xqci-mov-longlogical", "HasMovALUXqciFusion",
                            "Enable MOV + long logical operation macrofusion",
                            CheckAll<[
                              CheckOpcode<[ADDI]>,
                              CheckIsImmOperand<2>,
                              CheckImmOperand<2, 0>
                            ]>,
                            CheckOpcode<LongLogicalXqciOp>
                            >;
