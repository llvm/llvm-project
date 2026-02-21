//===-- RISCVInstrInfoSFB.td - Pseudos for SFB -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the pseudos for SFB (Short Forward Branch).
//
//===----------------------------------------------------------------------===//

// Represents the Opcode for a branch instruction, used by SFB pseudos
def bcc_opcode : RISCVOp<i32> {
  let OperandType = "OPERAND_BCC_OPCODE";
}

def sfb_rhs : RegisterOperand<GPR> {
  let OperandNamespace = "RISCVOp";
  let OperandType = "OPERAND_SFB_RHS";
}

// cond -> bcc_opcode
def CCtoRISCVBCC : SDNodeXForm<cond, [{
  ISD::CondCode CC = N->get();
  RISCVCC::CondCode RvCC = getRISCVCCForIntCC(CC);
  RvCC = RISCVCC::getInverseBranchCondition(RvCC);
  unsigned BccOpcode = RISCVCC::getBrCond(RvCC);
  return CurDAG->getTargetConstant(BccOpcode, SDLoc(N), MVT::i32);
}]>;

// For each of the short forward branch pseudos, corresponding code for
// getting correct size of the pseduo is needed in getInstSizeInBytes.
let Predicates = [HasShortForwardBranchIALU], isSelect = 1,
    Constraints = "$dst = $falsev", isCommutable = 1 in {
// This instruction moves $truev to $dst when the condition is true. It will
// be expanded to control flow in RISCVExpandPseudoInsts.
def PseudoCCMOVGPR : Pseudo<(outs GPR:$dst),
                            (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$bcc,
                             GPR:$falsev, GPR:$truev),
                            []>,
                     Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp,
                            ReadSFBALU, ReadSFBALU]>;
}

let Predicates = [HasShortForwardBranchIALU] in
def : Pat<(riscv_selectcc (XLenVT GPR:$lhs), GPR:$rhs, cond:$cc, (XLenVT GPR:$truev), GPR:$falsev),
           (PseudoCCMOVGPR GPR:$lhs, GPR:$rhs, (CCtoRISCVBCC $cc), GPR:$falsev, GPR:$truev)>;

// This should always expand to a branch+c.mv so the size is 6 or 4 if the
// branch is compressible.
let Predicates = [HasConditionalMoveFusion, NoShortForwardBranch],
    Constraints = "$dst = $falsev", isCommutable = 1 in {
// This instruction moves $truev to $dst when the condition is true. It will
// be expanded to control flow in RISCVExpandPseudoInsts.
// We use GPRNoX0 because c.mv cannot encode X0.
def PseudoCCMOVGPRNoX0 : Pseudo<(outs GPRNoX0:$dst),
                                (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$bcc,
                                 GPRNoX0:$falsev, GPRNoX0:$truev),
                                []>,
                         Sched<[]>;
}

let Predicates = [HasConditionalMoveFusion, NoShortForwardBranch] in
def :Pat<(riscv_selectcc (XLenVT GPR:$lhs), GPR:$rhs, cond:$cc, (XLenVT GPRNoX0:$truev), GPRNoX0:$falsev),
         (PseudoCCMOVGPRNoX0 GPR:$lhs, GPR:$rhs, (CCtoRISCVBCC $cc), GPRNoX0:$falsev, GPRNoX0:$truev)>;

class SFBALU_rr
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$cc, GPR:$falsev, GPR:$rs1,
                  GPR:$rs2), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU,
             ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Constraints = "$dst = $falsev";
}

class SFBALU_ri
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$cc, GPR:$falsev, GPR:$rs1,
                  simm12_lo:$imm), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Constraints = "$dst = $falsev";
}

class SFBLUI
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$cc, GPR:$falsev,
                  uimm20_lui:$imm), []> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Constraints = "$dst = $falsev";
}

class SFBLoad
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$cc, GPR:$falsev, GPR:$rs1,
                  simm12_lo:$imm), []> {
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;
  let Constraints = "$dst = $falsev";
}

class SFBShift_ri
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$cc, GPR:$falsev, GPR:$rs1,
                  uimmlog2xlen:$imm), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Constraints = "$dst = $falsev";
}

class SFBShiftW_ri
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, sfb_rhs:$rhs, bcc_opcode:$cc, GPR:$falsev, GPR:$rs1,
                  uimm5:$imm), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Constraints = "$dst = $falsev";
}

// Conditional binops, that updates update $dst to (op rs1, rs2) when condition
// is true. Returns $falsev otherwise. Selected by optimizeSelect.
// TODO: Can we use DefaultOperands on the regular binop to accomplish this more
// like how ARM does predication?
let Predicates = [HasShortForwardBranchIALU] in {
def PseudoCCADD : SFBALU_rr;
def PseudoCCSUB : SFBALU_rr;
def PseudoCCSLL : SFBALU_rr;
def PseudoCCSRL : SFBALU_rr;
def PseudoCCSRA : SFBALU_rr;
def PseudoCCAND : SFBALU_rr;
def PseudoCCOR  : SFBALU_rr;
def PseudoCCXOR : SFBALU_rr;

def PseudoCCADDI : SFBALU_ri;
def PseudoCCANDI : SFBALU_ri;
def PseudoCCORI  : SFBALU_ri;
def PseudoCCXORI : SFBALU_ri;

def PseudoCCLUI : SFBLUI;

def PseudoCCSLLI : SFBShift_ri;
def PseudoCCSRLI : SFBShift_ri;
def PseudoCCSRAI : SFBShift_ri;

// RV64I instructions
def PseudoCCADDW : SFBALU_rr;
def PseudoCCSUBW : SFBALU_rr;
def PseudoCCSLLW : SFBALU_rr;
def PseudoCCSRLW : SFBALU_rr;
def PseudoCCSRAW : SFBALU_rr;

def PseudoCCADDIW : SFBALU_ri;

def PseudoCCSLLIW : SFBShiftW_ri;
def PseudoCCSRLIW : SFBShiftW_ri;
def PseudoCCSRAIW : SFBShiftW_ri;

// Zbb/Zbkb instructions
def PseudoCCANDN : SFBALU_rr;
def PseudoCCORN  : SFBALU_rr;
def PseudoCCXNOR : SFBALU_rr;
}

// imm -> bcc_opcode
def RISCVCCtoRISCVBCC : SDNodeXForm<imm, [{
  auto CCImm = cast<ConstantSDNode>(N)->getZExtValue();
  auto RvCC = static_cast<RISCVCC::CondCode>(CCImm);
  RvCC = RISCVCC::getInverseBranchCondition(RvCC);
  unsigned BccOpcode = RISCVCC::getBrCond(RvCC);
  return CurDAG->getTargetConstant(BccOpcode, SDLoc(N), MVT::i32);
}]>;

let Predicates = [HasShortForwardBranchIALU] in
def : Pat<(XLenVT (abs GPR:$rs1)),
          (PseudoCCSUB (XLenVT GPR:$rs1), (XLenVT X0), (RISCVCCtoRISCVBCC (i32 /* COND_LT */ 2)),
           (XLenVT GPR:$rs1), (XLenVT X0), (XLenVT GPR:$rs1))>;
let Predicates = [HasShortForwardBranchIALU, IsRV64] in
def : Pat<(sext_inreg (abs 33signbits_node:$rs1), i32),
          (PseudoCCSUBW (i64 GPR:$rs1), (i64 X0), (RISCVCCtoRISCVBCC (i32 /* COND_LT */ 2)),
           (i64 GPR:$rs1), (i64 X0), (i64 GPR:$rs1))>;

let Predicates = [HasShortForwardBranchIMinMax] in {
def PseudoCCMAX : SFBALU_rr;
def PseudoCCMIN : SFBALU_rr;
def PseudoCCMAXU : SFBALU_rr;
def PseudoCCMINU : SFBALU_rr;
}

let Predicates = [HasShortForwardBranchIMul] in
def PseudoCCMUL : SFBALU_rr;

let Predicates = [HasShortForwardBranchILoad] in {
def PseudoCCLB : SFBLoad;
def PseudoCCLH : SFBLoad;
def PseudoCCLW : SFBLoad;
def PseudoCCLHU : SFBLoad;
def PseudoCCLBU : SFBLoad;
def PseudoCCLWU : SFBLoad;
def PseudoCCLD : SFBLoad;
}
