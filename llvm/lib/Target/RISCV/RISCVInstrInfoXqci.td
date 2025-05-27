//===---------------- RISCVInstrInfoXQci.td ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by QUALCOMM.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def uimm5nonzero : RISCVOp<XLenVT>,
                   ImmLeaf<XLenVT, [{return (Imm != 0) && isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5, "NonZero">;
  let DecoderMethod = "decodeUImmNonZeroOperand<5>";
  let OperandType = "OPERAND_UIMM5_NONZERO";
}

def uimm5gt3 : RISCVOp<XLenVT>, ImmLeaf<XLenVT,
  [{return (Imm > 3) && isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5, "GT3">;
  let DecoderMethod = "decodeUImmOperandGE<5, 4>";
  let OperandType = "OPERAND_UIMM5_GT3";
}

def UImm5Plus1AsmOperand : AsmOperandClass {
  let Name = "UImm5Plus1";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImm5Plus1";
}

def uimm5_plus1 : RISCVOp, ImmLeaf<XLenVT,
  [{return (isUInt<5>(Imm) && (Imm != 0)) || (Imm == 32);}]> {
  let ParserMatchClass = UImm5Plus1AsmOperand;
  let EncoderMethod = "getImmOpValueMinus1";
  let DecoderMethod = "decodeUImmPlus1Operand<5>";
  let OperandType = "OPERAND_UIMM5_PLUS1";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return (isUInt<5>(Imm) && (Imm != 0)) || (Imm == 32);
  }];
}

def uimm5ge6_plus1 : RISCVOp<XLenVT>, ImmLeaf<XLenVT,
  [{return (Imm >= 6) && (isUInt<5>(Imm) || (Imm == 32));}]> {
  let ParserMatchClass = UImmAsmOperand<5, "GE6Plus1">;
  let EncoderMethod = "getImmOpValueMinus1";
  let DecoderMethod = "decodeUImmPlus1OperandGE<5,6>";
  let OperandType = "OPERAND_UIMM5_GE6_PLUS1";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return (Imm >= 6) && (isUInt<5>(Imm) || (Imm == 32));
  }];
}

def uimm5slist : RISCVOp<XLenVT>, ImmLeaf<XLenVT,
  [{return ((Imm == 0)  ||
            (Imm == 1)  ||
            (Imm == 2)  ||
            (Imm == 4)  ||
            (Imm == 8)  ||
            (Imm == 16) ||
            (Imm == 15) ||
            (Imm == 31));}]> {
  let ParserMatchClass = UImmAsmOperand<5, "Slist">;
  let EncoderMethod = "getImmOpValueSlist";
  let DecoderMethod = "decodeUImmSlistOperand";
  let OperandType = "OPERAND_UIMM5_SLIST";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return ((Imm == 0)  || (Imm == 1)  ||
            (Imm == 2)  || (Imm == 4)  ||
            (Imm == 8)  || (Imm == 16) ||
            (Imm == 15) || (Imm == 31));
  }];
}

def uimm10 : RISCVUImmLeafOp<10>;

def uimm11 : RISCVUImmLeafOp<11>;

// A 14-bit unsigned immediate where the least significant two bits are zero.
def uimm14lsb00 : RISCVOp,
                  ImmLeaf<XLenVT, [{return isShiftedUInt<12, 2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<14, "Lsb00">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<14>";
  let OperandType = "OPERAND_UIMM14_LSB00";
}

def uimm16nonzero : RISCVOp<XLenVT>,
                    ImmLeaf<XLenVT, [{return (Imm != 0) && isUInt<16>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<16, "NonZero">;
  let DecoderMethod = "decodeUImmNonZeroOperand<16>";
  let OperandType = "OPERAND_UIMM16_NONZERO";
}

def simm5nonzero : RISCVOp<XLenVT>,
                   ImmLeaf<XLenVT, [{return (Imm != 0) && isInt<5>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<5, "NonZero">;
  let DecoderMethod = "decodeSImmNonZeroOperand<5>";
  let OperandType = "OPERAND_SIMM5_NONZERO";
}

def simm11 : RISCVSImmLeafOp<11>;

def simm16 : RISCVSImmOp<16>;

def simm16nonzero : RISCVOp<XLenVT>,
                    ImmLeaf<XLenVT, [{return (Imm != 0) && isInt<16>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<16, "NonZero">;
  let DecoderMethod = "decodeSImmNonZeroOperand<16>";
  let OperandType = "OPERAND_SIMM16_NONZERO";
}

def simm20_li : RISCVOp<XLenVT> {
  let ParserMatchClass = SImmAsmOperand<20, "LI">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<20>";
  let OperandType = "OPERAND_SIMM20_LI";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<20>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

def simm26 : RISCVSImmLeafOp<26>;

def simm26_nosimm12 : ImmLeaf<XLenVT, [{
  return isInt<26>(Imm) && !isInt<12>(Imm);}]>;

def simm32_nosimm26 : ImmLeaf<XLenVT, [{
  return isInt<32>(Imm) && !isInt<26>(Imm);}]>;

class BareSImmNAsmOperand<int width>
    : ImmAsmOperand<"BareS", width, ""> {
  let PredicateMethod = "isBareSimmN<" # width # ">";
}

// 32-bit Immediate, used by RV32 Instructions in 32-bit operations, so no
// sign-/zero-extension. This is represented internally as a signed 32-bit value.
def bare_simm32 : RISCVOp<XLenVT> {
  let ParserMatchClass = BareSImmNAsmOperand<32>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<32>";
  let OperandType = "OPERAND_BARE_SIMM32";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<32>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

// A 32-bit signed immediate where the least significant bit is zero.
def bare_simm32_lsb0 : Operand<OtherVT> {
  let ParserMatchClass = BareSImmNLsb0AsmOperand<32>;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValueAsrN<1>";
  let DecoderMethod = "decodeSImmOperandAndLslN<32, 1>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isShiftedInt<31, 1>(Imm);
    return MCOp.isBareSymbolRef();
  }];
  let OperandType = "OPERAND_PCREL";
}

def AddLike: PatFrags<(ops node:$A, node:$B),
                      [(add node:$A, node:$B), (or node:$A, node:$B)], [{
    return CurDAG->isBaseWithConstantOffset(SDValue(N, 0));
}]>;

def AddShl : PatFrag<(ops node:$Ra, node:$Rb, node:$SH3),
                     (add node:$Ra, (shl node:$Rb, node:$SH3))>;

def IntCCtoQCRISCVCC : SDNodeXForm<riscv_selectcc, [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  int64_t Imm = cast<ConstantSDNode>(N->getOperand(1))->getSExtValue();
  RISCVCC::CondCode BrCC;
  switch (CC) {
  default:
   report_fatal_error("Unexpected CondCode for Xqcibi branch instructions");
  case ISD::SETEQ:
    BrCC = isInt<5>(Imm) ? RISCVCC::COND_QC_BEQI :  RISCVCC::COND_QC_E_BEQI;
    break;
  case ISD::SETNE:
    BrCC = isInt<5>(Imm) ? RISCVCC::COND_QC_BNEI :  RISCVCC::COND_QC_E_BNEI;
    break;
  case ISD::SETLT:
    BrCC = isInt<5>(Imm) ? RISCVCC::COND_QC_BLTI :  RISCVCC::COND_QC_E_BLTI;
    break;
  case ISD::SETGE:
    BrCC = isInt<5>(Imm) ? RISCVCC::COND_QC_BGEI :  RISCVCC::COND_QC_E_BGEI;
    break;
  case ISD::SETULT:
    BrCC = isUInt<5>(Imm) ? RISCVCC::COND_QC_BLTUI :  RISCVCC::COND_QC_E_BLTUI;
    break;
  case ISD::SETUGE:
    BrCC = isUInt<5>(Imm) ? RISCVCC::COND_QC_BGEUI :  RISCVCC::COND_QC_E_BGEUI;
    break;
  }
  return CurDAG->getTargetConstant(BrCC, SDLoc(N), Subtarget->getXLenVT());
}]>;


//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//


class DirectiveInsnQC_EAI<dag outs, dag ins, string argstr>
  : RVInst48<outs, ins, "", "", [], InstFormatQC_EAI> {
  bits<7> opcode;
  bits<3> func3;
  bits<1> func1;

  bits<5> rd;
  bits<32> imm32;

  let Inst{47-16} = imm32;
  let Inst{15}    = func1;
  let Inst{14-12} = func3;
  let Inst{11-7}  = rd;
  let Inst{6-0}   = opcode;

  let AsmString = ".insn qc.eai " # argstr;
}

class DirectiveInsnQC_EI<dag outs, dag ins, string argstr>
  : RVInst48<outs, ins, "", "", [], InstFormatQC_EI> {
  bits<7> opcode;
  bits<3> func3;
  bits<2> func2;

  bits<5> rd;
  bits<5> rs1;
  bits<26> imm26;

  let Inst{47-32} = imm26{25-10};
  let Inst{31-30} = func2;
  let Inst{29-20} = imm26{9-0};
  let Inst{19-15} = rs1;
  let Inst{14-12} = func3;
  let Inst{11-7}  = rd;
  let Inst{6-0}   = opcode;

  let AsmString = ".insn qc.ei " # argstr;
}

class DirectiveInsnQC_EB<dag outs, dag ins, string argstr>
  : RVInst48<outs, ins, "", "", [], InstFormatQC_EB> {
  bits<7> opcode;
  bits<3> func3;
  bits<5> func5;

  bits<5> rs1;
  bits<12> imm12; // This one is the PC-relative offset
  bits<16> imm16;

  let Inst{47-32} = imm16;
  let Inst{31}    = imm12{11};
  let Inst{30-25} = imm12{9-4};
  let Inst{24-20} = func5;
  let Inst{19-15} = rs1;
  let Inst{14-12} = func3;
  let Inst{11-8}  = imm12{3-0};
  let Inst{7}     = imm12{10};
  let Inst{6-0}   = opcode;

  let AsmString = ".insn qc.eb " # argstr;
}

class DirectiveInsnQC_EJ<dag outs, dag ins, string argstr>
  : RVInst48<outs, ins, "", "", [], InstFormatQC_EJ> {
  bits<7> opcode;
  bits<3> func3;
  bits<2> func2;
  bits<5> func5;

  bits<31> imm31;

  let Inst{47-32} = imm31{30-15};
  let Inst{31}    = imm31{11};
  let Inst{30-25} = imm31{9-4};
  let Inst{24-20} = func5;
  let Inst{19-17} = imm31{14-12};
  let Inst{16-15} = func2;
  let Inst{14-12} = func3;
  let Inst{11-8}  = imm31{3-0};
  let Inst{7}     = imm31{10};
  let Inst{6-0}   = opcode;

  let AsmString = ".insn qc.ej " # argstr;
}

class DirectiveInsnQC_ES<dag outs, dag ins, string argstr>
  : RVInst48<outs, ins, "", "", [], InstFormatQC_ES> {
  bits<7> opcode;
  bits<3> func3;
  bits<2> func2;

  bits<5> rs1;
  bits<5> rs2;
  bits<26> imm26;

  let Inst{47-32} = imm26{25-10};
  let Inst{31-30} = func2;
  let Inst{29-25} = imm26{9-5};
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = func3;
  let Inst{11-7}  = imm26{4-0};
  let Inst{6-0}   = opcode;

  let AsmString = ".insn qc.es " # argstr;
}


let isCodeGenOnly = true, hasSideEffects = true, mayLoad = true,
    mayStore = true, hasNoSchedulingInfo = true, Predicates=[IsRV32] in {
def InsnQC_EAI : DirectiveInsnQC_EAI<(outs AnyReg:$rd),
                                     (ins uimm7_opcode:$opcode,
                                          uimm3:$func3,
                                          uimm1:$func1,
                                          bare_simm32:$imm32),
                                     "$opcode, $func3, $func1, $rd, $imm32">;
def InsnQC_EI : DirectiveInsnQC_EI<(outs AnyReg:$rd),
                                   (ins uimm7_opcode:$opcode,
                                        uimm3:$func3,
                                        uimm2:$func2,
                                        AnyReg:$rs1,
                                        simm26:$imm26),
                                   "$opcode, $func3, $func2, $rd, $rs1, $imm26">;
def InsnQC_EI_Mem : DirectiveInsnQC_EI<(outs AnyReg:$rd),
                                       (ins uimm7_opcode:$opcode,
                                            uimm3:$func3,
                                            uimm2:$func2,
                                            AnyReg:$rs1,
                                            simm26:$imm26),
                                       "$opcode, $func3, $func2, $rd, ${imm26}(${rs1})">;
def InsnQC_EB : DirectiveInsnQC_EB<(outs),
                                   (ins uimm7_opcode:$opcode,
                                        uimm3:$func3,
                                        uimm5:$func5,
                                        AnyReg:$rs1,
                                        simm16:$imm16,
                                        bare_simm13_lsb0:$imm12),
                                   "$opcode, $func3, $func5, $rs1, $imm16, $imm12">;
def InsnQC_EJ : DirectiveInsnQC_EJ<(outs),
                                   (ins uimm7_opcode:$opcode,
                                        uimm3:$func3,
                                        uimm2:$func2,
                                        uimm5:$func5,
                                        bare_simm32_lsb0:$imm31),
                                   "$opcode, $func3, $func2, $func5, $imm31">;
def InsnQC_ES : DirectiveInsnQC_ES<(outs),
                                   (ins uimm7_opcode:$opcode,
                                        uimm3:$func3,
                                        uimm2:$func2,
                                        AnyReg:$rs2,
                                        AnyReg:$rs1,
                                        simm26:$imm26),
                                   "$opcode, $func3, $func2, $rs2, ${imm26}(${rs1})">;
} // isCodeGenOnly, hasSideEffects, mayLoad, mayStore, hasNoSchedulingInfo, Predicates

let EmitPriority = 0, Predicates = [IsRV32] in {
def : InstAlias<".insn_qc.eai $opcode, $func3, $func1, $rd, $imm32",
                (InsnQC_EAI AnyReg:$rd,
                            uimm7_opcode:$opcode,
                            uimm3:$func3,
                            uimm1:$func1,
                            bare_simm32:$imm32)>;
def : InstAlias<".insn_qc.ei $opcode, $func3, $func2, $rd, $rs1, $imm26",
                (InsnQC_EI AnyReg:$rd,
                           uimm7_opcode:$opcode,
                           uimm3:$func3,
                           uimm2:$func2,
                           AnyReg:$rs1,
                           simm26:$imm26)>;
def : InstAlias<".insn_qc.ei $opcode, $func3, $func2, $rd, ${imm26}(${rs1})",
                (InsnQC_EI_Mem AnyReg:$rd,
                               uimm7_opcode:$opcode,
                               uimm3:$func3,
                               uimm2:$func2,
                               AnyReg:$rs1,
                               simm26:$imm26)>;
def : InstAlias<".insn_qc.ei $opcode, $func3, $func2, $rd, (${rs1})",
                (InsnQC_EI_Mem AnyReg:$rd,
                               uimm7_opcode:$opcode,
                               uimm3:$func3,
                               uimm2:$func2,
                               AnyReg:$rs1,
                               0)>;
def : InstAlias<".insn_qc.eb $opcode, $func3, $func5, $rs1, $imm16, $imm12",
                 (InsnQC_EB uimm7_opcode:$opcode,
                            uimm3:$func3,
                            uimm5:$func5,
                            AnyReg:$rs1,
                            simm16:$imm16,
                            bare_simm13_lsb0:$imm12)>;
def : InstAlias<".insn_qc.ej $opcode, $func3, $func2, $func5, $imm31",
                 (InsnQC_EJ uimm7_opcode:$opcode,
                            uimm3:$func3,
                            uimm2:$func2,
                            uimm5:$func5,
                            bare_simm32_lsb0:$imm31)>;
def : InstAlias<".insn_qc.es $opcode, $func3, $func2, $rs2, ${imm26}(${rs1})",
                 (InsnQC_ES uimm7_opcode:$opcode,
                            uimm3:$func3,
                            uimm2:$func2,
                            AnyReg:$rs2,
                            AnyReg:$rs1,
                            simm26:$imm26)>;
def : InstAlias<".insn_qc.es $opcode, $func3, $func2, $rs2, (${rs1})",
                 (InsnQC_ES uimm7_opcode:$opcode,
                            uimm3:$func3,
                            uimm2:$func2,
                            AnyReg:$rs2,
                            AnyReg:$rs1,
                            0)>;
} // EmitPriority = 0, Predicates = [IsRV32]

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
class QCILoad_ScaleIdx<bits<4> funct4, string opcodestr>
    : RVInstRBase<0b111, OPC_CUSTOM_0,
                  (outs GPR:$rd), (ins GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt),
                  opcodestr, "$rd, $rs1, $rs2, $shamt"> {
  bits<3> shamt;
  let Inst{31-28} = funct4;
  let Inst{27-25} = shamt;
}
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
// rd corresponds to the source for the store 'rs3' described in the spec.
class QCIStore_ScaleIdx<bits<4> funct4, string opcodestr>
    : RVInstRBase<0b110, OPC_CUSTOM_1, (outs),
                  (ins GPR:$rd, GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt),
                  opcodestr, "$rd, $rs1, $rs2, $shamt"> {
  bits<3> shamt;
  let Inst{31-28} = funct4;
  let Inst{27-25} = shamt;
}
}

class QCIRVInstI<bits<4> funct4, string opcodestr>
    : RVInstIUnary<{0b000, funct4, 0b00000}, 0b011, OPC_CUSTOM_0,
                   (outs GPRNoX0:$rd), (ins GPRNoX0:$rs1), opcodestr,
                   "$rd, $rs1">;

class QCIRVInstR<bits<4> funct4, string opcodestr>
    : RVInstR<{0b000, funct4}, 0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
              (ins GPRNoX0:$rs1), opcodestr, "$rd, $rs1"> {
  let rs2 = 0;
}

class QCIRVInstRR<bits<5> funct5, DAGOperand InTyRs1, string opcodestr>
    : RVInstR<{0b00, funct5}, 0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
              (ins InTyRs1:$rs1, GPRNoX0:$rs2), opcodestr, "$rd, $rs1, $rs2">;

class QCIBitManipRII<bits<3> funct3, bits<2> funct2,
                     DAGOperand InTyRs1, string opcodestr>
    : RVInstIBase<funct3, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                  (ins InTyRs1:$rs1, uimm5_plus1:$width, uimm5:$shamt),
                  opcodestr, "$rd, $rs1, $width, $shamt"> {
  bits<5> shamt;
  bits<5> width;

  let Inst{31-30} = funct2;
  let Inst{29-25} = width;
  let Inst{24-20} = shamt;
}

class QCIRVInstRI<bits<1> funct1, DAGOperand InTyImm11,
                             string opcodestr>
    : RVInstIBase<0b000, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                  (ins GPRNoX0:$rs1, InTyImm11:$imm11), opcodestr,
                  "$rd, $rs1, $imm11"> {
  bits<11> imm11;

  let Inst{31-31} = funct1;
  let Inst{30-20} = imm11;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTIICC<bits<3> funct3, string opcodestr>
    : RVInstR4<0b00, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, GPRNoX0:$rs1, simm5:$simm1, simm5:$simm2),
               opcodestr, "$rd, $rs1, $simm1, $simm2"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> simm1;
  bits<5> simm2;

  let rs3 = simm2;
  let rs2 = simm1;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTICC<bits<3> funct3, string opcodestr>
    : RVInstR4<0b01, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, GPRNoX0:$rs1, GPRNoX0:$rs2, simm5:$simm2),
               opcodestr, "$rd, $rs1, $rs2, $simm2"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> simm2;

  let rs3 = simm2;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTCCI<bits<3> funct3, string opcodestr>
    : RVInstR4<0b10, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, simm5:$imm, GPRNoX0:$rs2, GPRNoX0:$rs3),
               opcodestr, "$rd, $imm, $rs2, $rs3"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> imm;

  let rs1 = imm;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTICCI<bits<3> funct3, string opcodestr>
    : RVInstR4<0b11, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, simm5:$imm, GPRNoX0:$rs2, simm5:$simm2),
               opcodestr, "$rd, $imm, $rs2, $simm2"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> imm;
  bits<5> simm2;

  let rs3 = simm2;
  let rs1 = imm;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class QCILoadMultiple<bits<2> funct2, DAGOperand InTyRs2, string opcodestr>
    : RVInstRBase<0b111, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                  (ins GPR:$rs1, InTyRs2:$rs2, uimm7_lsb00:$imm),
                  opcodestr, "$rd, $rs2, ${imm}(${rs1})"> {
  bits<7> imm;
  let Inst{31-25} = {funct2, imm{6-2}};
}


// rd corresponds to the source for the store 'rs3' described in the spec.
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class QCIStoreMultiple<bits<2> funct2, DAGOperand InTyRs2, string opcodestr>
    : RVInstRBase<0b111, OPC_CUSTOM_1, (outs),
                  (ins GPR:$rd, GPR:$rs1, InTyRs2:$rs2, uimm7_lsb00:$imm),
                  opcodestr, "$rd, $rs2, ${imm}(${rs1})"> {
  bits<7> imm;
  let Inst{31-25} = {funct2, imm{6-2}};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCILICC<bits<3> funct3, bits<2> funct2, DAGOperand InTyRs2, string opcodestr>
    : RVInstRBase<funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
                  (ins GPRNoX0:$rd, GPRNoX0:$rs1, InTyRs2:$rs2, simm5:$simm),
                  opcodestr, "$rd, $rs1, $rs2, $simm"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> simm;

  let Inst{31-25} = {simm, funct2};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCIMVCC<bits<3> funct3, string opcodestr>
    : RVInstR4<0b00, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd),
               (ins GPRNoX0:$rs1, GPRNoX0:$rs2, GPRNoX0:$rs3),
               opcodestr, "$rd, $rs1, $rs2, $rs3">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCIMVCCI<bits<3> funct3, string opcodestr, DAGOperand immType>
    : RVInstR4<0b10, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd),
               (ins GPRNoX0:$rs1, immType:$imm, GPRNoX0:$rs3),
               opcodestr, "$rd, $rs1, $imm, $rs3"> {
  bits<5> imm;

  let rs2 = imm;
}

class QCI_RVInst16CB_BM<bits<2> funct2, string opcodestr>
    : RVInst16CB<0b100, 0b01, (outs GPRC:$rd),
                 (ins GPRC:$rs1, uimmlog2xlennonzero:$shamt),
                 opcodestr, "$rs1, $shamt"> {
  bits<5> shamt;
  let Constraints = "$rs1 = $rd";
  let Inst{12} = 0b1;
  let Inst{11-10} = funct2;
  let Inst{6-2} = shamt{4-0};
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class QCIRVInst16CI_RS1<bits<5> funct5, string OpcodeStr>
    : RVInst16CI<0b000, 0b10, (outs), (ins GPRNoX0:$rs1), OpcodeStr, "$rs1"> {
  bits<5> rs1;

  let Inst{12} = 0b1;
  let Inst{11-7} = rs1;
  let Inst{6-2} = funct5{4-0};
}

class QCIBranchInst_rii<bits<3> funct3, DAGOperand InTyImm5, string opcodestr>
    : RVInstB<funct3, OPC_CUSTOM_3, (outs),
              (ins GPRNoX0:$rs1, InTyImm5:$rs2, bare_simm13_lsb0:$imm12),
              opcodestr, "$rs1, $rs2, $imm12"> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

class QCIBranchInst48_rii<bits<5> funct5, DAGOperand InTyImm16, string opcodestr>
    : RVInst48<(outs),
               (ins GPRNoX0:$rs1, InTyImm16:$imm16, bare_simm13_lsb0:$imm12),
               opcodestr, "$rs1, $imm16, $imm12", [], InstFormatQC_EB> {
  bits<5> rs1;
  bits<16> imm16;
  bits<12> imm12;

  let Inst{47-32} = imm16;
  let Inst{31} = imm12{11};
  let Inst{30-25} = imm12{9-4};
  let Inst{24-20} = funct5;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b100;
  let Inst{11-8} = imm12{3-0};
  let Inst{7} = imm12{10};
  let Inst{6-0} = 0b0011111;
  let isBranch = 1;
  let isTerminator = 1;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let hasSideEffects = 1 in
class QCIRVInst16CI_NONE<bits<5> funct5, string OpcodeStr>
    : RVInst16CI<0b000, 0b10, (outs), (ins), OpcodeStr, ""> {
  let Inst{12} = 0b1;
  let Inst{11-7} = funct5;
  let Inst{6-2} = 0b00100;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class QCIInt_IMM<bits<1> funct1, string opcodestr>
    : RVInstIBase<0b000, OPC_SYSTEM, (outs), (ins uimm10:$imm10), opcodestr,
                  "$imm10"> {
  bits<10> imm10;

  let rd = 0;
  let rs1 = imm10{4-0};
  let Inst{31-25} = {0b110011, funct1};
  let Inst{24-20} = imm10{9-5};
}

class QCISim_NONE<bits<4> imm11_8, string opcodestr>
    : RVInstI<0b010, OPC_OP_IMM, (outs), (ins), opcodestr, ""> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = {imm11_8, 0b00000000};
}

class QCISim_RS1<bits<4> imm11_8, string opcodestr>
    : RVInstI<0b010, OPC_OP_IMM, (outs), (ins GPR:$rs1), opcodestr, "$rs1"> {
  let rd = 0;
  let imm12 = {imm11_8, 0b00000000};
}

let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in
class QCISync_UIMM5<bits<4> imm11_8, string opcodestr>
    : RVInstI<0b011, OPC_OP_IMM, (outs), (ins uimm5:$imm5), opcodestr, "$imm5">
{
  bits<5> imm5;

  let rs1 = 0;
  let rd = 0;
  let imm12 = {imm11_8, 0b000, imm5};
}

let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in
class QCIRVInst16CBSYNC<bits<3> imm5_func2, string OpcodeStr>
    : RVInst16CB<0b100, 0b01, (outs), (ins uimm5slist:$slist), OpcodeStr, "$slist"> {
    bits<3> slist;

    let Inst{6-2} = 0;
    let Inst{9-7} = slist;
    let Inst{12-10} = imm5_func2;
}

class QCIRVInstEIBase<bits<3> funct3, bits<2> funct2, dag outs,
                      dag ins, string opcodestr, string argstr>
    : RVInst48<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rd;
  bits<5> rs1;
  bits<26> imm;

  let Inst{47-32} = imm{25-10};
  let Inst{31-30} = funct2;
  let Inst{29-20} = imm{9-0};
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = 0b0011111;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class QCIRVInstEILoad<bits<3> funct3, bits<2> funct2, string opcodestr>
    : QCIRVInstEIBase<funct3, funct2, (outs GPR:$rd),
                      (ins GPRMem:$rs1, simm26:$imm), opcodestr,
                      "$rd, ${imm}(${rs1})">;

class QCIRVInstESBase<bits<3> funct3, bits<2> funct2, dag outs,
                      dag ins, string opcodestr, string argstr>
    : RVInst48<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs1;
  bits<5> rs2;
  bits<26> imm;

  let Inst{47-32} = imm{25-10};
  let Inst{31-30} = funct2;
  let Inst{29-25} = imm{9-5};
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = imm{4-0};
  let Inst{6-0} = 0b0011111;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class QCIRVInstESStore<bits<3> funct3, bits<2> funct2, string opcodestr>
    : QCIRVInstESBase<funct3, funct2, (outs),
                      (ins GPR:$rs2, GPRMem:$rs1, simm26:$imm),
                      opcodestr, "$rs2, ${imm}(${rs1})">;

class QCIRVInstEAI<bits<3> funct3, bits<1> funct1, string opcodestr>
    : RVInst48<(outs GPRNoX0:$rd_wb), (ins GPRNoX0:$rd, bare_simm32:$imm),
               opcodestr, "$rd, $imm", [], InstFormatOther> {
  bits<5> rd;
  bits<32> imm;

  let Constraints = "$rd = $rd_wb";
  let Inst{47-16} = imm{31-0};
  let Inst{15} = funct1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = 0b0011111;
}

class QCIRVInstEI<bits<3> funct3, bits<2> funct2, string opcodestr>
    : QCIRVInstEIBase<funct3, funct2, (outs GPRNoX0:$rd),
                      (ins GPRNoX0:$rs1, simm26:$imm), opcodestr,
                      "$rd, $rs1, $imm">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCIRVInst48EJ<bits<2> func2, string opcodestr>
    : RVInst48<(outs), (ins bare_simm32_lsb0:$imm31),
               opcodestr, "$imm31", [], InstFormatQC_EJ> {
  bits<31> imm31;

  let Inst{47-32} = imm31{30-15};
  let Inst{31} = imm31{11};
  let Inst{30-25} = imm31{9-4};
  let Inst{24-20} = 0b00000;
  let Inst{19-17} = imm31{14-12};
  let Inst{16-15} = func2;
  let Inst{14-12} = 0b100;
  let Inst{11-8} = imm31{3-0};
  let Inst{7} = imm31{10};
  let Inst{6-0} = 0b0011111;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let DecoderNamespace = "Xqci" in {

let Predicates = [HasVendorXqcicsr, IsRV32] in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  def QC_CSRRWR : RVInstR<0b1000110, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                          (ins GPR:$rs1, GPRNoX0:$rs2), "qc.csrrwr",
                          "$rd, $rs1, $rs2">;

  def QC_CSRRWRI : RVInstR<0b1000111, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                           (ins uimm5:$rs1, GPRNoX0:$rs2), "qc.csrrwri",
                           "$rd, $rs1, $rs2">;
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcicsr, IsRV32]

let Predicates = [HasVendorXqcisls, IsRV32] in {
  def  QC_LRB  : QCILoad_ScaleIdx<0b1000, "qc.lrb">;
  def  QC_LRH  : QCILoad_ScaleIdx<0b1001, "qc.lrh">;
  def  QC_LRW  : QCILoad_ScaleIdx<0b1010, "qc.lrw">;
  def  QC_LRBU : QCILoad_ScaleIdx<0b1011, "qc.lrbu">;
  def  QC_LRHU : QCILoad_ScaleIdx<0b1100, "qc.lrhu">;

  def  QC_SRB  : QCIStore_ScaleIdx<0b1101, "qc.srb">;
  def  QC_SRH  : QCIStore_ScaleIdx<0b1110, "qc.srh">;
  def  QC_SRW  : QCIStore_ScaleIdx<0b1111, "qc.srw">;
} // Predicates = [HasVendorXqcisls, IsRV32]

let Predicates = [HasVendorXqcia, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_SHLSAT : QCIRVInstRR<0b01010, GPRNoX0, "qc.shlsat">;
  def QC_SHLUSAT : QCIRVInstRR<0b01100, GPRNoX0, "qc.shlusat">;
  def QC_ADDSAT : QCIRVInstRR<0b01110, GPRNoX0, "qc.addsat">;
  def QC_ADDUSAT : QCIRVInstRR<0b01111, GPRNoX0, "qc.addusat">;
  def QC_SUBSAT : QCIRVInstRR<0b10000, GPRNoX0, "qc.subsat">;
  def QC_SUBUSAT : QCIRVInstRR<0b10001, GPRNoX0, "qc.subusat">;

  def QC_WRAP : QCIRVInstRR<0b10010, GPR, "qc.wrap">;
  def QC_WRAPI : RVInstI<0b000, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                         (ins GPRNoX0:$rs1, uimm11:$imm11), "qc.wrapi",
                         "$rd, $rs1, $imm11"> {
    bits<11> imm11;

    let imm12 = {0b0, imm11};
  }

  def QC_NORM : QCIRVInstR<0b0111, "qc.norm">;
  def QC_NORMU : QCIRVInstR<0b1000, "qc.normu">;
  def QC_NORMEU : QCIRVInstR<0b1001, "qc.normeu">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcia, IsRV32]

let Predicates = [HasVendorXqcibi, IsRV32] in {
  def QC_BEQI  : QCIBranchInst_rii<0b000, simm5nonzero, "qc.beqi">;
  def QC_BNEI  : QCIBranchInst_rii<0b001, simm5nonzero, "qc.bnei">;
  def QC_BLTI  : QCIBranchInst_rii<0b100, simm5nonzero, "qc.blti">;
  def QC_BGEI  : QCIBranchInst_rii<0b101, simm5nonzero, "qc.bgei">;
  def QC_BLTUI : QCIBranchInst_rii<0b110, uimm5nonzero, "qc.bltui">;
  def QC_BGEUI : QCIBranchInst_rii<0b111, uimm5nonzero, "qc.bgeui">;

  def QC_E_BEQI  : QCIBranchInst48_rii<0b11000, simm16nonzero, "qc.e.beqi">;
  def QC_E_BNEI  : QCIBranchInst48_rii<0b11001, simm16nonzero, "qc.e.bnei">;
  def QC_E_BLTI  : QCIBranchInst48_rii<0b11100, simm16nonzero, "qc.e.blti">;
  def QC_E_BGEI  : QCIBranchInst48_rii<0b11101, simm16nonzero, "qc.e.bgei">;
  def QC_E_BLTUI : QCIBranchInst48_rii<0b11110, uimm16nonzero, "qc.e.bltui">;
  def QC_E_BGEUI : QCIBranchInst48_rii<0b11111, uimm16nonzero, "qc.e.bgeui">;
} // Predicates = [HasVendorXqcibi, IsRV32]

let Predicates = [HasVendorXqcibm, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_INSBRI : QCIRVInstRI<0b1, simm11, "qc.insbri">;
  def QC_INSBI : RVInstIBase<0b001, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                             (ins simm5:$imm5, uimm5_plus1:$width,
                             uimm5:$shamt), "qc.insbi",
                             "$rd, $imm5, $width, $shamt"> {
    bits<5> imm5;
    bits<5> shamt;
    bits<5> width;
    let rs1 = imm5;
    let Inst{31-30} = 0b00;
    let Inst{29-25} = width;
    let Inst{24-20} = shamt;
  }
  def QC_INSB : QCIBitManipRII<0b001, 0b01, GPR, "qc.insb">;
  def QC_INSBH : QCIBitManipRII<0b001, 0b10, GPR, "qc.insbh">;
  def QC_INSBR : QCIRVInstRR<0b00000, GPR, "qc.insbr">;
  def QC_INSBHR : QCIRVInstRR<0b00001, GPR, "qc.insbhr">;
  def QC_INSBPR : QCIRVInstRR<0b00010, GPR, "qc.insbpr">;
  def QC_INSBPRH : QCIRVInstRR<0b00011, GPR, "qc.insbprh">;
  def QC_EXTU : QCIBitManipRII<0b010, 0b00, GPRNoX0, "qc.extu">;
  def QC_EXTDU : QCIBitManipRII<0b010, 0b10, GPRNoX31, "qc.extdu">;
  def QC_EXTDUR : QCIRVInstRR<0b00100, GPRNoX31, "qc.extdur">;
  def QC_EXTDUPR : QCIRVInstRR<0b00110, GPRNoX31, "qc.extdupr">;
  def QC_EXTDUPRH : QCIRVInstRR<0b00111, GPRNoX31, "qc.extduprh">;
  def QC_EXT : QCIBitManipRII<0b010, 0b01, GPRNoX0, "qc.ext">;
  def QC_EXTD : QCIBitManipRII<0b010, 0b11, GPRNoX31, "qc.extd">;
  def QC_EXTDR : QCIRVInstRR<0b00101, GPRNoX31, "qc.extdr">;
  def QC_EXTDPR : QCIRVInstRR<0b01000, GPRNoX31, "qc.extdpr">;
  def QC_EXTDPRH : QCIRVInstRR<0b01001, GPRNoX31, "qc.extdprh">;
  def QC_COMPRESS2 : QCIRVInstI<0b0000, "qc.compress2">;
  def QC_COMPRESS3 : QCIRVInstI<0b0001, "qc.compress3">;
  def QC_EXPAND2 : QCIRVInstI<0b0010, "qc.expand2">;
  def QC_EXPAND3 : QCIRVInstI<0b0011, "qc.expand3">;
  def QC_CLO : QCIRVInstI<0b0100, "qc.clo">;
  def QC_CTO : QCIRVInstI<0b0101, "qc.cto">;
  def QC_BREV32 : QCIRVInstI<0b0110, "qc.brev32">;
  def QC_C_BEXTI  : QCI_RVInst16CB_BM<0b00, "qc.c.bexti">;
  def QC_C_BSETI  : QCI_RVInst16CB_BM<0b01, "qc.c.bseti">;
  def QC_C_EXTU : RVInst16CI<0b000, 0b10, (outs GPRNoX0:$rd_wb),
                             (ins GPRNoX0:$rd, uimm5ge6_plus1:$width),
                             "qc.c.extu", "$rd, $width"> {
    bits<5> rd;
    bits<5> width;
    let Constraints = "$rd = $rd_wb";
    let Inst{6-2} = width;
    let Inst{11-7} = rd;
    let Inst{12} = 0b1;
  }
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcibm, IsRV32]

let Predicates = [HasVendorXqciac, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_C_MULIADD : RVInst16CL<0b001, 0b10, (outs GPRC:$rd_wb),
                               (ins GPRC:$rd, GPRC:$rs1, uimm5:$uimm),
                               "qc.c.muliadd", "$rd, $rs1, $uimm"> {
    let Constraints = "$rd = $rd_wb";
    bits<5> uimm;

    let Inst{12-10} = uimm{3-1};
    let Inst{6} = uimm{0};
    let Inst{5} = uimm{4};
  }

  def QC_MULIADD : RVInstI<0b110, OPC_CUSTOM_0, (outs GPRNoX0:$rd_wb),
                           (ins GPRNoX0:$rd, GPRNoX0:$rs1, simm12:$imm12),
                           "qc.muliadd", "$rd, $rs1, $imm12"> {
    let Constraints = "$rd = $rd_wb";
  }

  def QC_SHLADD : RVInstRBase<0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                              (ins GPRNoX0:$rs1, GPRNoX0:$rs2, uimm5gt3:$shamt),
                              "qc.shladd", "$rd, $rs1, $rs2, $shamt"> {
    bits<5> shamt;

    let Inst{31-30} = 0b01;
    let Inst{29-25} = shamt;
  }

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqciac, IsRV32]

let Predicates = [HasVendorXqcics, IsRV32] in {
  def QC_SELECTIIEQ : QCISELECTIICC <0b010, "qc.selectiieq">;
  def QC_SELECTIINE : QCISELECTIICC <0b011, "qc.selectiine">;
  def QC_SELECTIEQ  : QCISELECTICC  <0b010, "qc.selectieq">;
  def QC_SELECTINE  : QCISELECTICC  <0b011, "qc.selectine">;
  def QC_SELECTEQI  : QCISELECTCCI  <0b010, "qc.selecteqi">;
  def QC_SELECTNEI  : QCISELECTCCI  <0b011, "qc.selectnei">;
  def QC_SELECTIEQI : QCISELECTICCI <0b010, "qc.selectieqi">;
  def QC_SELECTINEI : QCISELECTICCI <0b011, "qc.selectinei">;
} // Predicates = [HasVendorXqcics, IsRV32]

let Predicates = [HasVendorXqcilsm, IsRV32] in {
    def QC_SWM : QCIStoreMultiple<0b00, GPRNoX0, "qc.swm">;
    def QC_SWMI : QCIStoreMultiple<0b01, uimm5nonzero, "qc.swmi">;
    def QC_SETWM : QCIStoreMultiple<0b10, GPRNoX0, "qc.setwm">;
    def QC_SETWMI : QCIStoreMultiple<0b11, uimm5nonzero, "qc.setwmi">;

    def QC_LWM : QCILoadMultiple<0b00, GPRNoX0, "qc.lwm">;
    def QC_LWMI : QCILoadMultiple<0b01, uimm5nonzero, "qc.lwmi">;
} // Predicates = [HasVendorXqcilsm, IsRV32]

let Predicates = [HasVendorXqcicli, IsRV32] in {
  def QC_LIEQ    : QCILICC<0b000, 0b01, GPRNoX0, "qc.lieq">;
  def QC_LINE    : QCILICC<0b001, 0b01, GPRNoX0, "qc.line">;
  def QC_LILT    : QCILICC<0b100, 0b01, GPRNoX0, "qc.lilt">;
  def QC_LIGE    : QCILICC<0b101, 0b01, GPRNoX0, "qc.lige">;
  def QC_LILTU   : QCILICC<0b110, 0b01, GPRNoX0, "qc.liltu">;
  def QC_LIGEU   : QCILICC<0b111, 0b01, GPRNoX0, "qc.ligeu">;

  def QC_LIEQI   : QCILICC<0b000, 0b11, simm5, "qc.lieqi">;
  def QC_LINEI   : QCILICC<0b001, 0b11, simm5, "qc.linei">;
  def QC_LILTI   : QCILICC<0b100, 0b11, simm5, "qc.lilti">;
  def QC_LIGEI   : QCILICC<0b101, 0b11, simm5, "qc.ligei">;
  def QC_LILTUI  : QCILICC<0b110, 0b11, uimm5, "qc.liltui">;
  def QC_LIGEUI  : QCILICC<0b111, 0b11, uimm5, "qc.ligeui">;
} // Predicates = [HasVendorXqcicli, IsRV32]

let Predicates = [HasVendorXqcicm, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
  def QC_C_MVEQZ   : RVInst16CL<0b101, 0b10, (outs GPRC:$rd_wb),
                              (ins GPRC:$rd, GPRC:$rs1),
                              "qc.c.mveqz", "$rd, $rs1"> {
    let Constraints = "$rd = $rd_wb";

    let Inst{12-10} = 0b011;
    let Inst{6-5} = 0b00;
  }

  def QC_MVEQ    : QCIMVCC<0b000, "qc.mveq">;
  def QC_MVNE    : QCIMVCC<0b001, "qc.mvne">;
  def QC_MVLT    : QCIMVCC<0b100, "qc.mvlt">;
  def QC_MVGE    : QCIMVCC<0b101, "qc.mvge">;
  def QC_MVLTU   : QCIMVCC<0b110, "qc.mvltu">;
  def QC_MVGEU   : QCIMVCC<0b111, "qc.mvgeu">;

  def QC_MVEQI   : QCIMVCCI<0b000, "qc.mveqi", simm5>;
  def QC_MVNEI   : QCIMVCCI<0b001, "qc.mvnei", simm5>;
  def QC_MVLTI   : QCIMVCCI<0b100, "qc.mvlti", simm5>;
  def QC_MVGEI   : QCIMVCCI<0b101, "qc.mvgei", simm5>;
  def QC_MVLTUI  : QCIMVCCI<0b110, "qc.mvltui", uimm5>;
  def QC_MVGEUI  : QCIMVCCI<0b111, "qc.mvgeui", uimm5>;
} // Predicates = [HasVendorXqcicm, IsRV32]

let Predicates = [HasVendorXqciint, IsRV32], hasSideEffects = 1 in {

let mayLoad = 0, mayStore = 0 in {
def QC_C_DIR : RVInst16CI<0b000, 0b10, (outs GPRNoX0:$rd), (ins),
                          "qc.c.dir", "$rd"> {
  bits<5> rd;

  let Inst{12} = 0b1;
  let Inst{11-7} = rd;
  let Inst{6-2} = 0b00000;
}

def QC_SETINTI : QCIInt_IMM<0b0, "qc.setinti">;
def QC_CLRINTI : QCIInt_IMM<0b1, "qc.clrinti">;

def QC_C_EIR    : QCIRVInst16CI_RS1<0b00001, "qc.c.eir">;
def QC_C_SETINT : QCIRVInst16CI_RS1<0b00010, "qc.c.setint">;
def QC_C_CLRINT : QCIRVInst16CI_RS1<0b00011, "qc.c.clrint">;

def QC_C_DI     : QCIRVInst16CI_NONE<0b10110, "qc.c.di">;
def QC_C_EI     : QCIRVInst16CI_NONE<0b10111, "qc.c.ei">;

let isBarrier = 1, isReturn = 1, isTerminator = 1 in {
def QC_C_MRET   : QCIRVInst16CI_NONE<0b10010, "qc.c.mret">;
def QC_C_MNRET  : QCIRVInst16CI_NONE<0b10011, "qc.c.mnret">;
} // isBarrier = 1, isReturn = 1, isTerminator = 1
} // mayLoad = 0, mayStore = 0

let mayLoad = 0, mayStore = 1,
  Uses = [X1, X2, X5, X6, X7, X8, X10, X11, X12, X13, X14, X15, X16, X17, X28, X29, X30, X31],
  Defs = [X2, X8] in {
def QC_C_MIENTER      : QCIRVInst16CI_NONE<0b10000, "qc.c.mienter">;
def QC_C_MIENTER_NEST : QCIRVInst16CI_NONE<0b10001, "qc.c.mienter.nest">;
} // mayLoad = 1, mayStore = 1, Uses = [...], Defs = [...]

let mayLoad = 1, mayStore = 0, isBarrier = 1, isReturn = 1, isTerminator = 1,
  Uses = [X2],
  Defs = [X1, X2, X5, X6, X7, X8, X10, X11, X12, X13, X14, X15, X16, X17, X28, X29, X30, X31] in
def QC_C_MILEAVERET   : QCIRVInst16CI_NONE<0b10100, "qc.c.mileaveret">;

} // Predicates = [HasVendorXqciint, IsRV32], hasSideEffects = 1

let Predicates = [HasVendorXqciio, IsRV32] in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  def QC_OUTW : RVInstI<0b100, OPC_CUSTOM_0, (outs),
                        (ins GPR:$rs2, GPR:$rs1, uimm14lsb00:$imm14),
                        "qc.outw", "$rs2, ${imm14}(${rs1})"> {
    bits<5> rs2;
    bits<14> imm14;

    let rd = rs2;
    let imm12 = imm14{13-2};
  }

  def QC_INW : RVInstI<0b101, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                       (ins GPR:$rs1, uimm14lsb00:$imm14),
                       "qc.inw", "$rd, ${imm14}(${rs1})"> {
    bits<14> imm14;

    let imm12 = imm14{13-2};
  }
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqciio, IsRV32]

let Predicates = [HasVendorXqcilo, IsRV32] in {
  def QC_E_LB    : QCIRVInstEILoad<0b101, 0b00, "qc.e.lb">;
  def QC_E_LBU   : QCIRVInstEILoad<0b101, 0b01, "qc.e.lbu">;
  def QC_E_LH    : QCIRVInstEILoad<0b101, 0b10, "qc.e.lh">;
  def QC_E_LHU   : QCIRVInstEILoad<0b101, 0b11, "qc.e.lhu">;
  def QC_E_LW    : QCIRVInstEILoad<0b110, 0b00, "qc.e.lw">;

  def QC_E_SB    : QCIRVInstESStore<0b110, 0b01, "qc.e.sb">;
  def QC_E_SH    : QCIRVInstESStore<0b110, 0b10, "qc.e.sh">;
  def QC_E_SW    : QCIRVInstESStore<0b110, 0b11, "qc.e.sw">;
} // Predicates = [HasVendorXqcilo, IsRV32]

let Predicates = [HasVendorXqcilb, IsRV32] in {
  let isCall = 1, Defs = [X1] in
  def QC_E_JAL   : QCIRVInst48EJ<0b01, "qc.e.jal">;
  let isBranch = 1, isTerminator = 1, isBarrier = 1 in
  def QC_E_J     : QCIRVInst48EJ<0b00, "qc.e.j">;
} // Predicates = [HasVendorXqcilb, IsRV32]

let Predicates = [HasVendorXqcili, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_LI : RVInstU<OPC_OP_IMM_32, (outs GPRNoX0:$rd), (ins simm20_li:$imm20),
                      "qc.li", "$rd, $imm20"> {
    let Inst{31} = imm20{19};
    let Inst{30-16} = imm20{14-0};
    let Inst{15-12} = imm20{18-15};
  }

  def QC_E_LI : RVInst48<(outs GPRNoX0:$rd), (ins bare_simm32:$imm),
                         "qc.e.li", "$rd, $imm", [], InstFormatQC_EAI> {
    bits<5> rd;
    bits<32> imm;

    let Inst{47-16} = imm;
    let Inst{15-12} = 0b0000;
    let Inst{11-7} = rd;
    let Inst{6-0} = 0b0011111;
  }
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcili, IsRV32]

let Predicates = [HasVendorXqcilia, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_E_XORAI : QCIRVInstEAI<0b001, 0b0, "qc.e.xorai">;
  def QC_E_ORAI  : QCIRVInstEAI<0b001, 0b1, "qc.e.orai" >;
  def QC_E_ADDAI : QCIRVInstEAI<0b010, 0b0, "qc.e.addai">;
  def QC_E_ANDAI : QCIRVInstEAI<0b010, 0b1, "qc.e.andai">;

  def QC_E_XORI  : QCIRVInstEI<0b011, 0b00, "qc.e.xori">;
  def QC_E_ORI   : QCIRVInstEI<0b011, 0b01, "qc.e.ori" >;
  def QC_E_ADDI  : QCIRVInstEI<0b011, 0b10, "qc.e.addi">;
  def QC_E_ANDI  : QCIRVInstEI<0b011, 0b11, "qc.e.andi">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcilia, IsRV32]

let Predicates = [HasVendorXqcisync, IsRV32] in {
  def QC_SYNC : QCISync_UIMM5<0b0001, "qc.sync">;
  def QC_SYNCR : QCISync_UIMM5<0b0010, "qc.syncr">;
  def QC_SYNCWF : QCISync_UIMM5<0b0100, "qc.syncwf">;
  def QC_SYNCWL : QCISync_UIMM5<0b1000, "qc.syncwl">;

  def QC_C_SYNC   : QCIRVInst16CBSYNC<0b000, "qc.c.sync">;
  def QC_C_SYNCR  : QCIRVInst16CBSYNC<0b001, "qc.c.syncr">;
  def QC_C_SYNCWF : QCIRVInst16CBSYNC<0b100, "qc.c.syncwf">;
  def QC_C_SYNCWL : QCIRVInst16CBSYNC<0b101, "qc.c.syncwl">;

  let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in
  def QC_C_DELAY : RVInst16CI<0b000, 0b10, (outs),
                              (ins uimm5nonzero:$imm),
                              "qc.c.delay", "$imm"> {
    let Inst{12} = 0;
    let Inst{11-7} = 0;
    let Inst{6-2} = imm{4-0};
  }
} // Predicates = [HasVendorXqcisync, IsRV32]

let Predicates = [HasVendorXqcisim, IsRV32] in {
let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in {
  def QC_PSYSCALLI : RVInstI<0b010, OPC_OP_IMM, (outs), (ins uimm10:$imm10),
                             "qc.psyscalli", "$imm10"> {
    bits<10> imm10;

    let rs1 = 0;
    let rd = 0;
    let imm12 = {0b00, imm10};
  }

  def QC_PPUTCI : RVInstI<0b010, OPC_OP_IMM, (outs), (ins uimm8:$imm8),
                          "qc.pputci", "$imm8"> {
    bits<8> imm8;

    let rs1 = 0;
    let rd = 0;
    let imm12 = {0b0100, imm8};
  }

  def QC_PCOREDUMP : QCISim_NONE<0b0110, "qc.pcoredump">;
  def QC_PPREGS : QCISim_NONE<0b0111, "qc.ppregs">;
  def QC_PPREG : QCISim_RS1<0b1000, "qc.ppreg">;
  def QC_PPUTC : QCISim_RS1<0b1001, "qc.pputc">;
  def QC_PPUTS : QCISim_RS1<0b1010, "qc.pputs">;
  def QC_PEXIT : QCISim_RS1<0b1011, "qc.pexit">;
  def QC_PSYSCALL : QCISim_RS1<0b1100, "qc.psyscall">;

  def QC_C_PTRACE : RVInst16CI<0b000, 0b10, (outs), (ins), "qc.c.ptrace", ""> {
    let rd = 0;
    let imm = 0;
  }
} // mayLoad = 0, mayStore = 0, hasSideEffects = 1
} // Predicates = [HasVendorXqcisim, IsRV32]

} // DecoderNamespace = "Xqci"

//===----------------------------------------------------------------------===//
// Aliases
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXqciio, IsRV32] in {
let EmitPriority = 0 in {
  def : InstAlias<"qc.outw $rs2, (${rs1})",
                  (QC_OUTW GPR:$rs2, GPR:$rs1, 0)>;
  def : InstAlias<"qc.inw $rd, (${rs1})",
                  (QC_INW GPRNoX0:$rd, GPR:$rs1, 0)>;
} // EmitPriority = 0
} // Predicates = [HasVendorXqciio, IsRV32]

let Predicates = [HasVendorXqcilsm, IsRV32] in {
let EmitPriority = 0 in {
  def : InstAlias<"qc.swm $rs3, $rs2, (${rs1})",
                  (QC_SWM GPR:$rs3, GPR:$rs1, GPRNoX0:$rs2, 0)>;
  def : InstAlias<"qc.swmi $rs3, $length, (${rs1})",
                  (QC_SWMI GPR:$rs3, GPR:$rs1, uimm5nonzero:$length, 0)>;
  def : InstAlias<"qc.setwm $rs3, $rs2, (${rs1})",
                  (QC_SETWM GPR:$rs3, GPR:$rs1, GPRNoX0:$rs2, 0)>;
  def : InstAlias<"qc.setwmi $rs3, $length, (${rs1})",
                  (QC_SETWMI GPR:$rs3, GPR:$rs1, uimm5nonzero:$length, 0)>;
  def : InstAlias<"qc.lwm $rd, $rs2, (${rs1})",
                  (QC_LWM GPRNoX0:$rd, GPR:$rs1, GPRNoX0:$rs2, 0)>;
  def : InstAlias<"qc.lwmi $rd, $length, (${rs1})",
                  (QC_LWMI GPRNoX0:$rd, GPR:$rs1, uimm5nonzero:$length, 0)>;
} // EmitPriority = 0
} // Predicates = [HasVendorXqcilsm, IsRV32]

let Predicates = [HasVendorXqcilo, IsRV32] in {
let EmitPriority = 0 in {
  def  : InstAlias<"qc.e.lb  $rd, (${rs1})",
                   (QC_E_LB  GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lbu $rd, (${rs1})",
                   (QC_E_LBU GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lh  $rd, (${rs1})",
                   (QC_E_LH  GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lhu $rd, (${rs1})",
                   (QC_E_LHU GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lw  $rd, (${rs1})",
                   (QC_E_LW  GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.sb  $rs2, (${rs1})",
                   (QC_E_SB GPR:$rs2, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.sh  $rs2, (${rs1})",
                   (QC_E_SH GPR:$rs2, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.sw  $rs2, (${rs1})",
                   (QC_E_SW GPR:$rs2, GPR:$rs1, 0)>;
} // EmitPriority = 0
} // Predicates = [HasVendorXqcilo, IsRV32]

//===----------------------------------------------------------------------===//
// Pseudo-instructions
//===----------------------------------------------------------------------===//

class LongBcciPseudo<DAGOperand InTyImm, int size>
    : Pseudo<(outs), (ins GPR:$rs1, InTyImm:$imm, simm21_lsb0_jal:$imm20), []>
{
  let Size = size;
  let isBarrier = 1;
  let isBranch = 1;
  let hasSideEffects = 0;
  let mayStore = 0;
  let mayLoad = 0;
  let isAsmParserOnly = 1;
  let hasNoSchedulingInfo = 1;
}

// This will be expanded into either:
// QC.BXXX(4 bytes) + JAL(4 bytes)
// or
// QC.E.BXXX(6 bytes) + JAL(4 bytes)
def PseudoLongQC_BEQI : LongBcciPseudo<simm5nonzero, 8>;
def PseudoLongQC_BNEI : LongBcciPseudo<simm5nonzero, 8>;
def PseudoLongQC_BLTI : LongBcciPseudo<simm5nonzero, 8>;
def PseudoLongQC_BGEI : LongBcciPseudo<simm5nonzero, 8>;
def PseudoLongQC_BLTUI : LongBcciPseudo<uimm5nonzero, 8>;
def PseudoLongQC_BGEUI : LongBcciPseudo<uimm5nonzero, 8>;
def PseudoLongQC_E_BEQI : LongBcciPseudo<simm16nonzero, 10>;
def PseudoLongQC_E_BNEI : LongBcciPseudo<simm16nonzero, 10>;
def PseudoLongQC_E_BLTI : LongBcciPseudo<simm16nonzero, 10>;
def PseudoLongQC_E_BGEI : LongBcciPseudo<simm16nonzero, 10>;
def PseudoLongQC_E_BLTUI : LongBcciPseudo<uimm16nonzero, 10>;
def PseudoLongQC_E_BGEUI : LongBcciPseudo<uimm16nonzero, 10>;

// Load/Store pseudos with QC.E.* Mnemonics. These expand to an AUIPC +
// (Standard) Load/Store sequence, as this can materialize all 32-bit addresses,
// and is shorter than e.g. an AUIPC + Xqcilo Load/Store sequence. These
// sequences can be turned back into a single Xqcilo instruction using linker
// relaxation.
let Predicates = [HasVendorXqcilo, IsRV32] in {
def PseudoQC_E_LB : PseudoLoad<"qc.e.lb">;
def PseudoQC_E_LBU : PseudoLoad<"qc.e.lbu">;
def PseudoQC_E_LH : PseudoLoad<"qc.e.lh">;
def PseudoQC_E_LHU : PseudoLoad<"qc.e.lhu">;
def PseudoQC_E_LW : PseudoLoad<"qc.e.lw">;

def PseudoQC_E_SB : PseudoStore<"qc.e.sb">;
def PseudoQC_E_SH : PseudoStore<"qc.e.sh">;
def PseudoQC_E_SW : PseudoStore<"qc.e.sw">;
} // Predicates = [HasVendorXqcilo, IsRV32]

//===----------------------------------------------------------------------===//
// Code Gen Patterns
//===----------------------------------------------------------------------===//

/// Generic pattern classes

class PatGprNoX0Simm26NoSimm12<SDPatternOperator OpNode, RVInst48 Inst>
    : Pat<(i32 (OpNode (i32 GPRNoX0:$rs1), simm26_nosimm12:$imm)),
          (Inst GPRNoX0:$rs1, simm26_nosimm12:$imm)>;

class PatGprNoX0Simm32NoSimm26<SDPatternOperator OpNode, RVInst48 Inst>
    : Pat<(i32 (OpNode (i32 GPRNoX0:$rs1), simm32_nosimm26:$imm)),
          (Inst GPRNoX0:$rs1, simm32_nosimm26:$imm)>;

class PatGprNoX0GprNoX0<SDPatternOperator OpNode, RVInstR Inst>
    : Pat<(i32 (OpNode (i32 GPRNoX0:$rs1), (i32 GPRNoX0:$rs2))),
          (Inst GPRNoX0:$rs1, GPRNoX0:$rs2)>;

class QC48LdPat<PatFrag LoadOp, RVInst48 Inst>
    : Pat<(i32 (LoadOp (AddLike (i32 GPR:$rs1), simm26_nosimm12:$imm26))),
          (Inst GPR:$rs1, simm26_nosimm12:$imm26)>;

class QC48StPat<PatFrag StoreOp, RVInst48 Inst>
    : Pat<(StoreOp (i32 GPR:$rs2), (AddLike (i32 GPR:$rs1), simm26_nosimm12:$imm26)),
          (Inst GPR:$rs2, GPR:$rs1, simm26_nosimm12:$imm26)>;

class QCScaledLdPat<PatFrag LoadOp, RVInst Inst>
    : Pat<(i32 (LoadOp (AddShl (i32 GPRMem:$rs1), (i32 GPRNoX0:$rs2), uimm3:$shamt))),
          (Inst GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt)>;

class QCScaledStPat<PatFrag StoreOp, RVInst Inst>
    : Pat<(StoreOp (i32 GPR:$rd), (AddShl (i32 GPRMem:$rs1), (i32 GPRNoX0:$rs2), uimm3:$shamt)),
          (Inst GPR:$rd, GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt)>;

// Match `riscv_brcc` and lower to the appropriate XQCIBI branch instruction.
class BcciPat<CondCode Cond, QCIBranchInst_rii Inst, DAGOperand InTyImm>
    : Pat<(riscv_brcc (XLenVT GPRNoX0:$rs1), InTyImm:$rs2, Cond, bb:$imm12),
          (Inst GPRNoX0:$rs1, InTyImm:$rs2, bare_simm13_lsb0:$imm12)>;

class Bcci48Pat<CondCode Cond, QCIBranchInst48_rii Inst, DAGOperand InTyImm>
    : Pat<(riscv_brcc (XLenVT GPRNoX0:$rs1), InTyImm:$rs2, Cond, bb:$imm12),
          (Inst GPRNoX0:$rs1, InTyImm:$rs2, bare_simm13_lsb0:$imm12)>;

defm CC_SImm5NonZero_QC  : SelectCC_GPR_riirr<GPRNoX0, simm5nonzero>;
defm CC_UImm5NonZero_QC  : SelectCC_GPR_riirr<GPRNoX0, uimm5nonzero>;
defm CC_SImm16NonZero_QC : SelectCC_GPR_riirr<GPRNoX0, simm16nonzero>;
defm CC_UImm16NonZero_QC : SelectCC_GPR_riirr<GPRNoX0, uimm16nonzero>;

class SelectQCbi<CondCode Cond, DAGOperand InTyImm, Pseudo OpNode >
    : Pat<(riscv_selectcc_frag:$cc (i32 GPRNoX0:$lhs), InTyImm:$Constant, Cond,
                                   (i32 GPRNoX0:$truev), GPRNoX0:$falsev),
          (OpNode GPRNoX0:$lhs, InTyImm:$Constant,
           (IntCCtoQCRISCVCC $cc), GPRNoX0:$truev, GPRNoX0:$falsev)>;

/// Simple arithmetic operations

let Predicates = [HasVendorXqcilia, IsRV32] in {
def : PatGprNoX0Simm32NoSimm26<add, QC_E_ADDAI>;
def : PatGprNoX0Simm32NoSimm26<and, QC_E_ANDAI>;
def : PatGprNoX0Simm32NoSimm26<or, QC_E_ORAI>;
def : PatGprNoX0Simm32NoSimm26<xor, QC_E_XORAI>;

def : PatGprNoX0Simm26NoSimm12<add, QC_E_ADDI>;
def : PatGprNoX0Simm26NoSimm12<and, QC_E_ANDI>;
def : PatGprNoX0Simm26NoSimm12<or, QC_E_ORI>;
def : PatGprNoX0Simm26NoSimm12<xor, QC_E_XORI>;
} // Predicates = [HasVendorXqcilia, IsRV32]

/// Load/Store operations

let Predicates = [HasVendorXqcilo, IsRV32], AddedComplexity = 2 in {
  def : QC48LdPat<sextloadi8, QC_E_LB>;
  def : QC48LdPat<extloadi8, QC_E_LBU>; // Prefer unsigned due to no c.lb in Zcb.
  def : QC48LdPat<sextloadi16, QC_E_LH>;
  def : QC48LdPat<extloadi16, QC_E_LH>;
  def : QC48LdPat<load, QC_E_LW>;
  def : QC48LdPat<zextloadi8, QC_E_LBU>;
  def : QC48LdPat<zextloadi16, QC_E_LHU>;

  def : QC48StPat<truncstorei8, QC_E_SB>;
  def : QC48StPat<truncstorei16, QC_E_SH>;
  def : QC48StPat<store, QC_E_SW>;
} // Predicates = [HasVendorXqcilo, IsRV32], AddedComplexity = 2


let Predicates = [HasVendorXqcisls, IsRV32], AddedComplexity = 1 in {
  def : QCScaledLdPat<sextloadi8, QC_LRB>;
  def : QCScaledLdPat<extloadi8, QC_LRBU>;
  def : QCScaledLdPat<sextloadi16, QC_LRH>;
  def : QCScaledLdPat<extloadi16, QC_LRH>;
  def : QCScaledLdPat<load, QC_LRW>;
  def : QCScaledLdPat<zextloadi8, QC_LRBU>;
  def : QCScaledLdPat<zextloadi16, QC_LRHU>;

  def : QCScaledStPat<truncstorei8, QC_SRB>;
  def : QCScaledStPat<truncstorei16, QC_SRH>;
  def : QCScaledStPat<store, QC_SRW>;
} // Predicates = [HasVendorXqcisls, IsRV32], AddedComplexity = 1

let Predicates = [HasVendorXqcia, IsRV32] in {
def : PatGprNoX0GprNoX0<saddsat, QC_ADDSAT>;
def : PatGprNoX0GprNoX0<uaddsat, QC_ADDUSAT>;
def : PatGprNoX0GprNoX0<ssubsat, QC_SUBSAT>;
def : PatGprNoX0GprNoX0<usubsat, QC_SUBUSAT>;
def : PatGprNoX0GprNoX0<ushlsat, QC_SHLUSAT>;
def : PatGprNoX0GprNoX0<sshlsat, QC_SHLSAT>;
} // Predicates = [HasVendorXqcia, IsRV32]

/// Branches

let Predicates = [HasVendorXqcibi, IsRV32], AddedComplexity = 2 in {
def : BcciPat<SETEQ, QC_BEQI, simm5nonzero>;
def : BcciPat<SETNE, QC_BNEI, simm5nonzero>;
def : BcciPat<SETLT, QC_BLTI, simm5nonzero>;
def : BcciPat<SETGE, QC_BGEI, simm5nonzero>;
def : BcciPat<SETULT, QC_BLTUI, uimm5nonzero>;
def : BcciPat<SETUGE, QC_BGEUI, uimm5nonzero>;

def : Bcci48Pat<SETEQ, QC_E_BEQI, simm16nonzero>;
def : Bcci48Pat<SETNE, QC_E_BNEI, simm16nonzero>;
def : Bcci48Pat<SETLT, QC_E_BLTI, simm16nonzero>;
def : Bcci48Pat<SETGE, QC_E_BGEI, simm16nonzero>;
def : Bcci48Pat<SETULT, QC_E_BLTUI, uimm16nonzero>;
def : Bcci48Pat<SETUGE, QC_E_BGEUI, uimm16nonzero>;

def : SelectQCbi<SETEQ, simm5nonzero, Select_GPRNoX0_Using_CC_SImm5NonZero_QC>;
def : SelectQCbi<SETNE, simm5nonzero, Select_GPRNoX0_Using_CC_SImm5NonZero_QC>;
def : SelectQCbi<SETLT, simm5nonzero, Select_GPRNoX0_Using_CC_SImm5NonZero_QC>;
def : SelectQCbi<SETGE, simm5nonzero, Select_GPRNoX0_Using_CC_SImm5NonZero_QC>;
def : SelectQCbi<SETULT, uimm5nonzero, Select_GPRNoX0_Using_CC_UImm5NonZero_QC>;
def : SelectQCbi<SETUGE, uimm5nonzero, Select_GPRNoX0_Using_CC_UImm5NonZero_QC>;

def : SelectQCbi<SETEQ, simm16nonzero, Select_GPRNoX0_Using_CC_SImm16NonZero_QC>;
def : SelectQCbi<SETNE, simm16nonzero, Select_GPRNoX0_Using_CC_SImm16NonZero_QC>;
def : SelectQCbi<SETLT, simm16nonzero, Select_GPRNoX0_Using_CC_SImm16NonZero_QC>;
def : SelectQCbi<SETGE, simm16nonzero, Select_GPRNoX0_Using_CC_SImm16NonZero_QC>;
def : SelectQCbi<SETULT, uimm16nonzero, Select_GPRNoX0_Using_CC_UImm16NonZero_QC>;
def : SelectQCbi<SETUGE, uimm16nonzero, Select_GPRNoX0_Using_CC_UImm16NonZero_QC>;
} // let Predicates = [HasVendorXqcibi, IsRV32], AddedComplexity = 2

let Predicates = [HasVendorXqcibm, IsRV32] in {
def : Pat<(sext_inreg (i32 GPR:$rs1), i16), (QC_EXT GPR:$rs1, 16, 0)>;
def : Pat<(sext_inreg (i32 GPR:$rs1), i8), (QC_EXT GPR:$rs1, 8, 0)>;
def : Pat<(sext_inreg (i32 GPR:$rs1), i1), (QC_EXT GPR:$rs1, 1, 0)>;
} // Predicates = [HasVendorXqcibm, IsRV32]

let Predicates = [HasVendorXqciint, IsRV32] in
def : Pat<(riscv_mileaveret_glue), (QC_C_MILEAVERET)>;


//===----------------------------------------------------------------------===/i
// Compress Instruction tablegen backend.
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXqcisync, IsRV32] in {
def : CompressPat<(QC_SYNC uimm5slist:$imm5), (QC_C_SYNC uimm5slist:$imm5)>;
def : CompressPat<(QC_SYNCR uimm5slist:$imm5), (QC_C_SYNCR uimm5slist:$imm5)>;
def : CompressPat<(QC_SYNCWL uimm5slist:$imm5), (QC_C_SYNCWL uimm5slist:$imm5)>;
def : CompressPat<(QC_SYNCWF uimm5slist:$imm5), (QC_C_SYNCWF uimm5slist:$imm5)>;
} // Predicates = [HasVendorXqcisync, IsRV32]

let isCompressOnly = true, Predicates = [HasVendorXqcilo, HasStdExtZcb, IsRV32] in {
def : CompressPat<(QC_E_LBU GPRC:$rd, GPRCMem:$rs1, uimm2:$imm),
                  (C_LBU GPRC:$rd, GPRCMem:$rs1, uimm2:$imm)>;
def : CompressPat<(QC_E_LHU GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm),
                  (C_LHU GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm)>;
def : CompressPat<(QC_E_LH GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm),
                  (C_LH GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm)>;
def : CompressPat<(QC_E_SB GPRC:$rs2, GPRCMem:$rs1, uimm2:$imm),
                  (C_SB GPRC:$rs2, GPRCMem:$rs1, uimm2:$imm)>;
def : CompressPat<(QC_E_SH GPRC:$rs2, GPRCMem:$rs1, uimm2_lsb0:$imm),
                  (C_SH GPRC:$rs2, GPRCMem:$rs1, uimm2_lsb0:$imm)>;
} // isCompressOnly = true, Predicates = [HasVendorXqcilo, HasStdExtZcb, IsRV32]

let isCompressOnly = true, Predicates = [HasVendorXqcilo, IsRV32] in {
def : CompressPat<(QC_E_LW GPRC:$rd, GPRCMem:$rs1, uimm7_lsb00:$imm),
                  (C_LW GPRC:$rd, GPRCMem:$rs1, uimm7_lsb00:$imm)>;
def : CompressPat<(QC_E_LW GPRNoX0:$rd, SPMem:$rs1,  uimm8_lsb00:$imm),
                  (C_LWSP GPRNoX0:$rd, SPMem:$rs1, uimm8_lsb00:$imm)>;
def : CompressPat<(QC_E_LB GPR:$rd, GPRMem:$rs1, simm12:$imm12),
                  (LB GPR:$rd, GPRMem:$rs1, simm12:$imm12)>;
def : CompressPat<(QC_E_LBU GPR:$rd, GPRMem:$rs1, simm12:$imm12),
                  (LBU GPR:$rd, GPRMem:$rs1, simm12:$imm12)>;
def : CompressPat<(QC_E_LH GPR:$rd, GPRMem:$rs1, simm12:$imm12),
                  (LH GPR:$rd, GPRMem:$rs1, simm12:$imm12)>;
def : CompressPat<(QC_E_LHU GPR:$rd, GPRMem:$rs1, simm12:$imm12),
                  (LHU GPR:$rd, GPRMem:$rs1, simm12:$imm12)>;
def : CompressPat<(QC_E_LW GPR:$rd, GPRMem:$rs1, simm12:$imm12),
                  (LW GPR:$rd, GPRMem:$rs1, simm12:$imm12)>;

def : CompressPat<(QC_E_SW GPRC:$rs2, GPRCMem:$rs1, uimm7_lsb00:$imm),
                  (C_SW GPRC:$rs2, GPRCMem:$rs1, uimm7_lsb00:$imm)>;
def : CompressPat<(QC_E_SW GPR:$rs2, SPMem:$rs1, uimm8_lsb00:$imm),
                  (C_SWSP GPR:$rs2, SPMem:$rs1, uimm8_lsb00:$imm)>;
def : CompressPat<(QC_E_SB GPR:$rs2, GPRMem:$rs1, simm12:$imm12),
                  (SB GPR:$rs2, GPRMem:$rs1, simm12:$imm12)>;
def : CompressPat<(QC_E_SH GPR:$rs2, GPRMem:$rs1, simm12:$imm12),
                  (SH GPR:$rs2, GPRMem:$rs1, simm12:$imm12)>;
def : CompressPat<(QC_E_SW GPR:$rs2, GPRMem:$rs1, simm12:$imm12),
                  (SW GPR:$rs2, GPRMem:$rs1, simm12:$imm12)>;
} // isCompressOnly = true, Predicates = [HasVendorXqcilo, IsRV32]

let Predicates = [HasVendorXqcicm, IsRV32] in {
def : CompressPat<(QC_MVEQI GPRC:$rd, GPRC:$rd, 0, GPRC:$rs1),
                  (QC_C_MVEQZ GPRC:$rd, GPRC:$rs1)>;
let isCompressOnly = true in
def : CompressPat<(QC_MVLTUI GPRC:$rd, GPRC:$rd, 1, GPRC:$rs1),
                  (QC_C_MVEQZ GPRC:$rd, GPRC:$rs1)>;
}

let Predicates = [HasVendorXqcibm, IsRV32] in {
def : CompressPat<(QC_EXTU GPRNoX0:$rd, GPRNoX0:$rd, uimm5ge6_plus1:$width, 0),
                  (QC_C_EXTU GPRNoX0:$rd, uimm5ge6_plus1:$width)>;
}

let Predicates = [HasVendorXqcibm, HasStdExtZbs, IsRV32] in {
def : CompressPat<(BSETI GPRC:$rs1, GPRC:$rs1, uimmlog2xlennonzero:$shamt),
                  (QC_C_BSETI GPRC:$rs1, uimmlog2xlennonzero:$shamt)>;
def : CompressPat<(BEXTI GPRC:$rs1, GPRC:$rs1, uimmlog2xlennonzero:$shamt),
                  (QC_C_BEXTI GPRC:$rs1, uimmlog2xlennonzero:$shamt)>;
let isCompressOnly = true in
def : CompressPat<(QC_EXTU GPRC:$rd, GPRC:$rd, 1, uimmlog2xlennonzero:$shamt),
                  (QC_C_BEXTI GPRC:$rd, uimmlog2xlennonzero:$shamt)>;
} // Predicates = [HasVendorXqcibm, HasStdExtZbs, IsRV32]

let isCompressOnly = true, Predicates = [HasVendorXqcilb, IsRV32] in {
def : CompressPat<(QC_E_J bare_simm12_lsb0:$offset),
                  (C_J bare_simm12_lsb0:$offset)>;
def : CompressPat<(QC_E_JAL bare_simm12_lsb0:$offset),
                  (C_JAL bare_simm12_lsb0:$offset)>;
} // isCompressOnly = true, Predicates = [HasVendorXqcilb, IsRV32]

let Predicates = [HasVendorXqcili, IsRV32] in {
def : CompressPat<(QC_LI GPRNoX0:$rd, simm6:$imm),
                  (C_LI GPRNoX0:$rd, simm6:$imm)>;

let isCompressOnly = true in
def : CompressPat<(QC_E_LI GPRNoX0:$rd, simm6:$imm),
                  (C_LI GPRNoX0:$rd, simm6:$imm)>;
} // Predicates = [HasVendorXqcili, IsRV32]

let isCompressOnly = true, Predicates = [HasVendorXqcilia, IsRV32] in {
def : CompressPat<(QC_E_ADDI GPRNoX0:$rs1, GPRNoX0:$rs1, simm6nonzero:$imm),
                  (C_ADDI GPRNoX0:$rs1, simm6nonzero:$imm)>;
def : CompressPat<(QC_E_ANDI GPRC:$rs1, GPRC:$rs1, simm6:$imm),
                  (C_ANDI GPRC:$rs1, simm6:$imm)>;
def : CompressPat<(QC_E_ADDI GPRNoX0:$rs1, GPRNoX0:$rs2, 0),
                  (C_MV GPRNoX0:$rs1, GPRNoX0:$rs2)>;
def : CompressPat<(QC_E_ADDI X2, X2, simm10_lsb0000nonzero:$imm),
                  (C_ADDI16SP X2, simm10_lsb0000nonzero:$imm)>;
} // let isCompressOnly = true, Predicates = [HasVendorXqcilia, IsRV32]
