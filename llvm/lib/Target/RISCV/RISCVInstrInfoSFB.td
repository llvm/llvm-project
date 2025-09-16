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

let Predicates = [HasShortForwardBranchOpt], isSelect = 1,
    Constraints = "$dst = $falsev", isCommutable = 1, Size = 8 in {
// This instruction moves $truev to $dst when the condition is true. It will
// be expanded to control flow in RISCVExpandPseudoInsts.
def PseudoCCMOVGPR : Pseudo<(outs GPR:$dst),
                            (ins GPR:$lhs, GPR:$rhs, cond_code:$cc,
                             GPR:$falsev, GPR:$truev),
                            [(set GPR:$dst,
                              (riscv_selectcc_frag:$cc (XLenVT GPR:$lhs),
                                                       GPR:$rhs, cond,
                                                       (XLenVT GPR:$truev),
                                                       GPR:$falsev))]>,
                     Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp,
                            ReadSFBALU, ReadSFBALU]>;
}

// This should always expand to a branch+c.mv so the size is 6 or 4 if the
// branch is compressible.
let Predicates = [HasConditionalMoveFusion, NoShortForwardBranchOpt],
    Constraints = "$dst = $falsev", isCommutable = 1, Size = 6 in {
// This instruction moves $truev to $dst when the condition is true. It will
// be expanded to control flow in RISCVExpandPseudoInsts.
// We use GPRNoX0 because c.mv cannot encode X0.
def PseudoCCMOVGPRNoX0 : Pseudo<(outs GPRNoX0:$dst),
                                (ins GPR:$lhs, GPR:$rhs, cond_code:$cc,
                                 GPRNoX0:$falsev, GPRNoX0:$truev),
                                [(set GPRNoX0:$dst,
                                  (riscv_selectcc_frag:$cc (XLenVT GPR:$lhs),
                                                           (XLenVT GPR:$rhs),
                                                           cond, (XLenVT GPRNoX0:$truev),
                                                           (XLenVT GPRNoX0:$falsev)))]>,
                         Sched<[]>;
}

class SFBALU_rr
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, GPR:$rhs, cond_code:$cc, GPR:$falsev, GPR:$rs1,
                  GPR:$rs2), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU,
             ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Size = 8;
  let Constraints = "$dst = $falsev";
}

class SFBALU_ri
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, GPR:$rhs, cond_code:$cc, GPR:$falsev, GPR:$rs1,
                  simm12:$imm), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Size = 8;
  let Constraints = "$dst = $falsev";
}

class SFBShift_ri
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, GPR:$rhs, cond_code:$cc, GPR:$falsev, GPR:$rs1,
                  uimmlog2xlen:$imm), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Size = 8;
  let Constraints = "$dst = $falsev";
}

class SFBShiftW_ri
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, GPR:$rhs, cond_code:$cc, GPR:$falsev, GPR:$rs1,
                  uimm5:$imm), []>,
      Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU, ReadSFBALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Size = 8;
  let Constraints = "$dst = $falsev";
}

// Conditional binops, that updates update $dst to (op rs1, rs2) when condition
// is true. Returns $falsev otherwise. Selected by optimizeSelect.
// TODO: Can we use DefaultOperands on the regular binop to accomplish this more
// like how ARM does predication?
let Predicates = [HasShortForwardBranchOpt] in {
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

let Predicates = [HasShortForwardBranchOpt] in
def : Pat<(XLenVT (abs GPR:$rs1)),
          (PseudoCCSUB (XLenVT GPR:$rs1), (XLenVT X0), /* COND_LT */ 2,
           (XLenVT GPR:$rs1), (XLenVT X0), (XLenVT GPR:$rs1))>;
let Predicates = [HasShortForwardBranchOpt, IsRV64] in
def : Pat<(sext_inreg (abs 33signbits_node:$rs1), i32),
          (PseudoCCSUBW (i64 GPR:$rs1), (i64 X0), /* COND_LT */ 2,
           (i64 GPR:$rs1), (i64 X0), (i64 GPR:$rs1))>;
