//===---------------- RISCVInstrInfoXqccmp.td --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Qualcomm's Xqccmp extension.
//
// Xqccmp is broadly equivalent to (and incompatible with) Zcmp except the
// following changes:
//
// - The registers are pushed in the opposite order, so `ra` and `fp` are
//   closest to the incoming stack pointer (to be compatible with the
//   frame-pointer convention), and
//
// - There is a new `qc.cm.pushfp` instruction which is `qc.cm.push` but it sets
//   `fp` to the incoming stack pointer value, as expected by the frame-pointer
//   convention.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def RegListS0AsmOperand : AsmOperandClass {
  let Name = "RegListS0";
  let ParserMethod = "parseRegListS0";
  let RenderMethod = "addRegListOperands";
  let DiagnosticType = "InvalidRegListS0";
  let DiagnosticString = "operand must be {ra, s0[-sN]} or {x1, x8[-x9][, x18[-xN]]}";
}

def reglist_s0 : RISCVOp<OtherVT> {
   let ParserMatchClass = RegListS0AsmOperand;
   let PrintMethod = "printRegList";
   let DecoderMethod = "decodeXqccmpRlistS0";
   let EncoderMethod = "getRlistS0OpValue";
   let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    // 0~4 invalid for `qc.cm.pushfp`
    return isUInt<4>(Imm) && Imm >= RISCVZC::RA_S0;
  }];

  string OperandType = "OPERAND_RLIST_S0";
}

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

class RVInstXqccmpCPPPFP<bits<5> funct5, string opcodestr,
                         DAGOperand immtype = stackadj>
    : RVInst16<(outs), (ins reglist_s0:$rlist, immtype:$stackadj),
               opcodestr, "$rlist, $stackadj", [], InstFormatOther> {
  bits<4> rlist;
  bits<16> stackadj;

  let Inst{1-0} = 0b10;
  let Inst{3-2} = stackadj{5-4};
  let Inst{7-4} = rlist;
  let Inst{12-8} = funct5;
  let Inst{15-13} = 0b101;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let DecoderNamespace = "Xqccmp", Predicates = [HasVendorXqccmp] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
let Defs = [X10, X11] in
def QC_CM_MVA01S : RVInst16CA<0b101011, 0b11, 0b10, (outs),
                              (ins SR07:$rs1, SR07:$rs2), "qc.cm.mva01s", "$rs1, $rs2">,
                   Sched<[WriteIALU, WriteIALU, ReadIALU, ReadIALU]>;

let Uses = [X10, X11] in
def QC_CM_MVSA01 : RVInst16CA<0b101011, 0b01, 0b10, (outs SR07:$rs1, SR07:$rs2),
                              (ins), "qc.cm.mvsa01", "$rs1, $rs2">,
                   Sched<[WriteIALU, WriteIALU, ReadIALU, ReadIALU]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, Uses = [X2], Defs = [X2] in
def QC_CM_PUSH : RVInstZcCPPP<0b11000, "qc.cm.push", negstackadj>,
                 Sched<[WriteIALU, ReadIALU, ReadStoreData, ReadStoreData,
                        ReadStoreData, ReadStoreData, ReadStoreData, ReadStoreData,
                        ReadStoreData, ReadStoreData, ReadStoreData, ReadStoreData,
                        ReadStoreData, ReadStoreData, ReadStoreData]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, Uses = [X2], Defs = [X2, X8] in
def QC_CM_PUSHFP : RVInstXqccmpCPPPFP<0b11001, "qc.cm.pushfp", negstackadj>,
                   Sched<[WriteIALU, WriteIALU, ReadIALU, ReadStoreData, ReadStoreData,
                          ReadStoreData, ReadStoreData, ReadStoreData, ReadStoreData,
                          ReadStoreData, ReadStoreData, ReadStoreData, ReadStoreData,
                          ReadStoreData, ReadStoreData, ReadStoreData]>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isReturn = 1,
    Uses = [X2], Defs = [X2] in
def QC_CM_POPRET : RVInstZcCPPP<0b11110, "qc.cm.popret">,
                   Sched<[WriteIALU, WriteLDW, WriteLDW, WriteLDW, WriteLDW,
                          WriteLDW, WriteLDW, WriteLDW, WriteLDW, WriteLDW,
                          WriteLDW, WriteLDW, WriteLDW, WriteLDW, ReadIALU]>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isReturn = 1,
    Uses = [X2], Defs = [X2, X10] in
def QC_CM_POPRETZ : RVInstZcCPPP<0b11100, "qc.cm.popretz">,
                    Sched<[WriteIALU, WriteIALU, WriteLDW, WriteLDW, WriteLDW,
                           WriteLDW, WriteLDW, WriteLDW, WriteLDW, WriteLDW,
                           WriteLDW, WriteLDW, WriteLDW, WriteLDW, WriteLDW,
                           ReadIALU]>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0,
    Uses = [X2], Defs = [X2] in
def QC_CM_POP : RVInstZcCPPP<0b11010, "qc.cm.pop">,
                Sched<[WriteIALU, WriteLDW, WriteLDW, WriteLDW, WriteLDW,
                       WriteLDW, WriteLDW, WriteLDW, WriteLDW, WriteLDW, WriteLDW,
                       WriteLDW, WriteLDW, WriteLDW, ReadIALU]>;

} // DecoderNamespace = "Xqccmp", Predicates = [HasVendorXqccmp]

//===----------------------------------------------------------------------===//
// Aliases
//===----------------------------------------------------------------------===//

