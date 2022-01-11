//=-- SVEInstrFormats.td -  AArch64 SVE Instruction classes -*- tablegen -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AArch64 Scalable Vector Extension (SVE) Instruction Class Definitions.
//
//===----------------------------------------------------------------------===//

def SDT_AArch64Setcc : SDTypeProfile<1, 4, [
  SDTCisVec<0>, SDTCisVec<1>, SDTCisVec<2>, SDTCisVec<3>,
  SDTCVecEltisVT<0, i1>, SDTCVecEltisVT<1, i1>, SDTCisSameAs<2, 3>,
  SDTCisVT<4, OtherVT>
]>;

def AArch64setcc_z : SDNode<"AArch64ISD::SETCC_MERGE_ZERO", SDT_AArch64Setcc>;

def SVEPatternOperand : AsmOperandClass {
  let Name = "SVEPattern";
  let ParserMethod = "tryParseSVEPattern";
  let PredicateMethod = "isSVEPattern";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidSVEPattern";
}

def sve_pred_enum : Operand<i32>, TImmLeaf<i32, [{
  return (((uint32_t)Imm) < 32);
  }]> {

  let PrintMethod = "printSVEPattern";
  let ParserMatchClass = SVEPatternOperand;
}

def SVEPrefetchOperand : AsmOperandClass {
  let Name = "SVEPrefetch";
  let ParserMethod = "tryParsePrefetch<true>";
  let PredicateMethod = "isPrefetch";
  let RenderMethod = "addPrefetchOperands";
}

def sve_prfop : Operand<i32>, TImmLeaf<i32, [{
    return (((uint32_t)Imm) <= 15);
  }]> {
  let PrintMethod = "printPrefetchOp<true>";
  let ParserMatchClass = SVEPrefetchOperand;
}

class SVELogicalImmOperand<int Width> : AsmOperandClass {
  let Name = "SVELogicalImm" # Width;
  let DiagnosticType = "LogicalSecondSource";
  let PredicateMethod = "isLogicalImm<int" # Width # "_t>";
  let RenderMethod = "addLogicalImmOperands<int" # Width # "_t>";
}

def sve_logical_imm8 : Operand<i64> {
  let ParserMatchClass = SVELogicalImmOperand<8>;
  let PrintMethod = "printLogicalImm<int8_t>";

  let MCOperandPredicate = [{
    if (!MCOp.isImm())
      return false;
    int64_t Val = AArch64_AM::decodeLogicalImmediate(MCOp.getImm(), 64);
    return AArch64_AM::isSVEMaskOfIdenticalElements<int8_t>(Val);
  }];
}

def sve_logical_imm16 : Operand<i64> {
  let ParserMatchClass = SVELogicalImmOperand<16>;
  let PrintMethod = "printLogicalImm<int16_t>";

  let MCOperandPredicate = [{
    if (!MCOp.isImm())
      return false;
    int64_t Val = AArch64_AM::decodeLogicalImmediate(MCOp.getImm(), 64);
    return AArch64_AM::isSVEMaskOfIdenticalElements<int16_t>(Val);
  }];
}

def sve_logical_imm32 : Operand<i64> {
  let ParserMatchClass = SVELogicalImmOperand<32>;
  let PrintMethod = "printLogicalImm<int32_t>";

  let MCOperandPredicate = [{
    if (!MCOp.isImm())
      return false;
    int64_t Val = AArch64_AM::decodeLogicalImmediate(MCOp.getImm(), 64);
    return AArch64_AM::isSVEMaskOfIdenticalElements<int32_t>(Val);
  }];
}

class SVEPreferredLogicalImmOperand<int Width> : AsmOperandClass {
  let Name = "SVEPreferredLogicalImm" # Width;
  let PredicateMethod = "isSVEPreferredLogicalImm<int" # Width # "_t>";
  let RenderMethod = "addLogicalImmOperands<int" # Width # "_t>";
}

def sve_preferred_logical_imm16 : Operand<i64> {
  let ParserMatchClass = SVEPreferredLogicalImmOperand<16>;
  let PrintMethod = "printSVELogicalImm<int16_t>";

  let MCOperandPredicate = [{
    if (!MCOp.isImm())
      return false;
    int64_t Val = AArch64_AM::decodeLogicalImmediate(MCOp.getImm(), 64);
    return AArch64_AM::isSVEMaskOfIdenticalElements<int16_t>(Val) &&
           AArch64_AM::isSVEMoveMaskPreferredLogicalImmediate(Val);
  }];
}

def sve_preferred_logical_imm32 : Operand<i64> {
  let ParserMatchClass =  SVEPreferredLogicalImmOperand<32>;
  let PrintMethod = "printSVELogicalImm<int32_t>";

  let MCOperandPredicate = [{
    if (!MCOp.isImm())
      return false;
    int64_t Val = AArch64_AM::decodeLogicalImmediate(MCOp.getImm(), 64);
    return AArch64_AM::isSVEMaskOfIdenticalElements<int32_t>(Val) &&
           AArch64_AM::isSVEMoveMaskPreferredLogicalImmediate(Val);
  }];
}

def sve_preferred_logical_imm64 : Operand<i64> {
  let ParserMatchClass = SVEPreferredLogicalImmOperand<64>;
  let PrintMethod = "printSVELogicalImm<int64_t>";

  let MCOperandPredicate = [{
    if (!MCOp.isImm())
      return false;
    int64_t Val = AArch64_AM::decodeLogicalImmediate(MCOp.getImm(), 64);
    return AArch64_AM::isSVEMaskOfIdenticalElements<int64_t>(Val) &&
           AArch64_AM::isSVEMoveMaskPreferredLogicalImmediate(Val);
  }];
}

class SVELogicalImmNotOperand<int Width> : AsmOperandClass {
  let Name = "SVELogicalImm" # Width # "Not";
  let DiagnosticType = "LogicalSecondSource";
  let PredicateMethod = "isLogicalImm<int" # Width # "_t>";
  let RenderMethod = "addLogicalImmNotOperands<int" # Width # "_t>";
}

def sve_logical_imm8_not : Operand<i64> {
  let ParserMatchClass = SVELogicalImmNotOperand<8>;
}

def sve_logical_imm16_not : Operand<i64> {
  let ParserMatchClass = SVELogicalImmNotOperand<16>;
}

def sve_logical_imm32_not : Operand<i64> {
  let ParserMatchClass = SVELogicalImmNotOperand<32>;
}

class SVEShiftedImmOperand<int ElementWidth, string Infix, string Predicate>
    : AsmOperandClass {
  let Name = "SVE" # Infix # "Imm" # ElementWidth;
  let DiagnosticType = "Invalid" # Name;
  let RenderMethod = "addImmWithOptionalShiftOperands<8>";
  let ParserMethod = "tryParseImmWithOptionalShift";
  let PredicateMethod = Predicate;
}

def SVECpyImmOperand8  : SVEShiftedImmOperand<8,  "Cpy", "isSVECpyImm<int8_t>">;
def SVECpyImmOperand16 : SVEShiftedImmOperand<16, "Cpy", "isSVECpyImm<int16_t>">;
def SVECpyImmOperand32 : SVEShiftedImmOperand<32, "Cpy", "isSVECpyImm<int32_t>">;
def SVECpyImmOperand64 : SVEShiftedImmOperand<64, "Cpy", "isSVECpyImm<int64_t>">;

def SVEAddSubImmOperand8  : SVEShiftedImmOperand<8,  "AddSub", "isSVEAddSubImm<int8_t>">;
def SVEAddSubImmOperand16 : SVEShiftedImmOperand<16, "AddSub", "isSVEAddSubImm<int16_t>">;
def SVEAddSubImmOperand32 : SVEShiftedImmOperand<32, "AddSub", "isSVEAddSubImm<int32_t>">;
def SVEAddSubImmOperand64 : SVEShiftedImmOperand<64, "AddSub", "isSVEAddSubImm<int64_t>">;

class imm8_opt_lsl<int ElementWidth, string printType,
                   AsmOperandClass OpndClass>
    : Operand<i32> {
  let EncoderMethod = "getImm8OptLsl";
  let DecoderMethod = "DecodeImm8OptLsl<" # ElementWidth # ">";
  let PrintMethod = "printImm8OptLsl<" # printType # ">";
  let ParserMatchClass = OpndClass;
  let MIOperandInfo = (ops i32imm, i32imm);
}

def cpy_imm8_opt_lsl_i8  : imm8_opt_lsl<8,  "int8_t",  SVECpyImmOperand8>;
def cpy_imm8_opt_lsl_i16 : imm8_opt_lsl<16, "int16_t", SVECpyImmOperand16>;
def cpy_imm8_opt_lsl_i32 : imm8_opt_lsl<32, "int32_t", SVECpyImmOperand32>;
def cpy_imm8_opt_lsl_i64 : imm8_opt_lsl<64, "int64_t", SVECpyImmOperand64>;

def addsub_imm8_opt_lsl_i8  : imm8_opt_lsl<8,  "uint8_t",  SVEAddSubImmOperand8>;
def addsub_imm8_opt_lsl_i16 : imm8_opt_lsl<16, "uint16_t", SVEAddSubImmOperand16>;
def addsub_imm8_opt_lsl_i32 : imm8_opt_lsl<32, "uint32_t", SVEAddSubImmOperand32>;
def addsub_imm8_opt_lsl_i64 : imm8_opt_lsl<64, "uint64_t", SVEAddSubImmOperand64>;

def SVEAddSubImm8Pat  : ComplexPattern<i32, 2, "SelectSVEAddSubImm<MVT::i8>", []>;
def SVEAddSubImm16Pat : ComplexPattern<i32, 2, "SelectSVEAddSubImm<MVT::i16>", []>;
def SVEAddSubImm32Pat : ComplexPattern<i32, 2, "SelectSVEAddSubImm<MVT::i32>", []>;
def SVEAddSubImm64Pat : ComplexPattern<i64, 2, "SelectSVEAddSubImm<MVT::i64>", []>;

def SVELogicalImm8Pat  : ComplexPattern<i32, 1, "SelectSVELogicalImm<MVT::i8>", []>;
def SVELogicalImm16Pat : ComplexPattern<i32, 1, "SelectSVELogicalImm<MVT::i16>", []>;
def SVELogicalImm32Pat : ComplexPattern<i32, 1, "SelectSVELogicalImm<MVT::i32>", []>;
def SVELogicalImm64Pat : ComplexPattern<i64, 1, "SelectSVELogicalImm<MVT::i64>", []>;

def SVELogicalImm8NotPat  : ComplexPattern<i32, 1, "SelectSVELogicalImm<MVT::i8, true>", []>;
def SVELogicalImm16NotPat : ComplexPattern<i32, 1, "SelectSVELogicalImm<MVT::i16, true>", []>;
def SVELogicalImm32NotPat : ComplexPattern<i32, 1, "SelectSVELogicalImm<MVT::i32, true>", []>;
def SVELogicalImm64NotPat : ComplexPattern<i64, 1, "SelectSVELogicalImm<MVT::i64, true>", []>;

def SVE8BitLslImm32 : ComplexPattern<i32, 2, "SelectSVE8BitLslImm", [imm]>;
def SVE8BitLslImm64 : ComplexPattern<i64, 2, "SelectSVE8BitLslImm", [imm]>;
class SVE8BitLslImm<ValueType ty> {
  ComplexPattern Pat = !cond(
    !eq(ty, i32): SVE8BitLslImm32,
    !eq(ty, i64): SVE8BitLslImm64);
}

def SVEArithUImm8Pat  : ComplexPattern<i32, 1, "SelectSVEArithImm<MVT::i8>", []>;
def SVEArithUImm16Pat  : ComplexPattern<i32, 1, "SelectSVEArithImm<MVT::i16>", []>;
def SVEArithUImm32Pat  : ComplexPattern<i32, 1, "SelectSVEArithImm<MVT::i32>", []>;
def SVEArithUImm64Pat  : ComplexPattern<i64, 1, "SelectSVEArithImm<MVT::i64>", []>;

def SVEArithSImmPat32 : ComplexPattern<i32, 1, "SelectSVESignedArithImm", []>;
def SVEArithSImmPat64 : ComplexPattern<i64, 1, "SelectSVESignedArithImm", []>;

def SVEShiftImmL8  : ComplexPattern<i32, 1, "SelectSVEShiftImm<0, 7>",  []>;
def SVEShiftImmL16 : ComplexPattern<i32, 1, "SelectSVEShiftImm<0, 15>", []>;
def SVEShiftImmL32 : ComplexPattern<i32, 1, "SelectSVEShiftImm<0, 31>", []>;
def SVEShiftImmL64 : ComplexPattern<i64, 1, "SelectSVEShiftImm<0, 63>", []>;
def SVEShiftImmR8  : ComplexPattern<i32, 1, "SelectSVEShiftImm<1, 8,  true>", []>;
def SVEShiftImmR16 : ComplexPattern<i32, 1, "SelectSVEShiftImm<1, 16, true>", []>;
def SVEShiftImmR32 : ComplexPattern<i32, 1, "SelectSVEShiftImm<1, 32, true>", []>;
def SVEShiftImmR64 : ComplexPattern<i64, 1, "SelectSVEShiftImm<1, 64, true>", []>;

def SVEAllActive : ComplexPattern<untyped, 0, "SelectAllActivePredicate", []>;

class SVEExactFPImm<string Suffix, string ValA, string ValB> : AsmOperandClass {
  let Name = "SVEExactFPImmOperand" # Suffix;
  let DiagnosticType = "Invalid" # Name;
  let ParserMethod = "tryParseFPImm<false>";
  let PredicateMethod = "isExactFPImm<" # ValA # ", " # ValB # ">";
  let RenderMethod = "addExactFPImmOperands<" # ValA # ", " # ValB # ">";
}

class SVEExactFPImmOperand<string Suffix, string ValA, string ValB> : Operand<i32> {
  let PrintMethod = "printExactFPImm<" # ValA # ", " # ValB # ">";
  let ParserMatchClass = SVEExactFPImm<Suffix, ValA, ValB>;
}

def sve_fpimm_half_one
    : SVEExactFPImmOperand<"HalfOne", "AArch64ExactFPImm::half",
                           "AArch64ExactFPImm::one">;
def sve_fpimm_half_two
    : SVEExactFPImmOperand<"HalfTwo", "AArch64ExactFPImm::half",
                           "AArch64ExactFPImm::two">;
def sve_fpimm_zero_one
    : SVEExactFPImmOperand<"ZeroOne", "AArch64ExactFPImm::zero",
                           "AArch64ExactFPImm::one">;

def sve_incdec_imm : Operand<i32>, TImmLeaf<i32, [{
  return (((uint32_t)Imm) > 0) && (((uint32_t)Imm) < 17);
}]> {
  let ParserMatchClass = Imm1_16Operand;
  let EncoderMethod = "getSVEIncDecImm";
  let DecoderMethod = "DecodeSVEIncDecImm";
}

// This allows i32 immediate extraction from i64 based arithmetic.
def sve_cnt_mul_imm_i32 : ComplexPattern<i32, 1, "SelectCntImm<1, 16, 1, false>">;
def sve_cnt_mul_imm_i64 : ComplexPattern<i64, 1, "SelectCntImm<1, 16, 1, false>">;
def sve_cnt_shl_imm     : ComplexPattern<i64, 1, "SelectCntImm<1, 16, 1, true>">;

def sve_ext_imm_0_31  : ComplexPattern<i64, 1, "SelectEXTImm<31, 8>">;
def sve_ext_imm_0_63  : ComplexPattern<i64, 1, "SelectEXTImm<63, 4>">;
def sve_ext_imm_0_127 : ComplexPattern<i64, 1, "SelectEXTImm<127, 2>">;
def sve_ext_imm_0_255 : ComplexPattern<i64, 1, "SelectEXTImm<255, 1>">;

def int_aarch64_sve_cntp_oneuse : PatFrag<(ops node:$pred, node:$src2),
                                          (int_aarch64_sve_cntp node:$pred, node:$src2), [{
  return N->hasOneUse();
}]>;

def step_vector_oneuse : PatFrag<(ops node:$idx),
                                 (step_vector node:$idx), [{
  return N->hasOneUse();
}]>;


//===----------------------------------------------------------------------===//
// SVE PTrue - These are used extensively throughout the pattern matching so
//             it's important we define them first.
//===----------------------------------------------------------------------===//

class sve_int_ptrue<bits<2> sz8_64, bits<3> opc, string asm, PPRRegOp pprty,
                    ValueType vt, SDPatternOperator op>
: I<(outs pprty:$Pd), (ins sve_pred_enum:$pattern),
  asm, "\t$Pd, $pattern",
  "",
  [(set (vt pprty:$Pd), (op sve_pred_enum:$pattern))]>, Sched<[]> {
  bits<4> Pd;
  bits<5> pattern;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-19} = 0b011;
  let Inst{18-17} = opc{2-1};
  let Inst{16}    = opc{0};
  let Inst{15-10} = 0b111000;
  let Inst{9-5}   = pattern;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pd;

  let Defs = !if(!eq (opc{0}, 1), [NZCV], []);
  let ElementSize = pprty.ElementSize;
  let isReMaterializable = 1;
}

multiclass sve_int_ptrue<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_ptrue<0b00, opc, asm, PPR8, nxv16i1, op>;
  def _H : sve_int_ptrue<0b01, opc, asm, PPR16, nxv8i1, op>;
  def _S : sve_int_ptrue<0b10, opc, asm, PPR32, nxv4i1, op>;
  def _D : sve_int_ptrue<0b11, opc, asm, PPR64, nxv2i1, op>;

  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _B) PPR8:$Pd, 0b11111), 1>;
  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _H) PPR16:$Pd, 0b11111), 1>;
  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _S) PPR32:$Pd, 0b11111), 1>;
  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _D) PPR64:$Pd, 0b11111), 1>;
}

def SDT_AArch64PTrue : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVT<1, i32>]>;
def AArch64ptrue : SDNode<"AArch64ISD::PTRUE", SDT_AArch64PTrue>;

let Predicates = [HasSVEorStreamingSVE] in {
  defm PTRUE  : sve_int_ptrue<0b000, "ptrue", AArch64ptrue>;
  defm PTRUES : sve_int_ptrue<0b001, "ptrues", null_frag>;
}

//===----------------------------------------------------------------------===//
// SVE pattern match helpers.
//===----------------------------------------------------------------------===//

class SVE_1_Op_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                   Instruction inst>
: Pat<(vtd (op vt1:$Op1)),
      (inst $Op1)>;

class SVE_1_Op_Passthru_Pat<ValueType vtd, SDPatternOperator op, ValueType pg,
                            ValueType vts, Instruction inst>
: Pat<(vtd (op pg:$Op1, vts:$Op2, vtd:$Op3)),
      (inst $Op3, $Op1, $Op2)>;


multiclass SVE_1_Op_PassthruUndef_Pat<ValueType vtd, SDPatternOperator op, ValueType pg,
                                 ValueType vts, Instruction inst> {
  def : Pat<(vtd (op pg:$Op1, vts:$Op2, (vtd undef))),
            (inst (IMPLICIT_DEF), $Op1, $Op2)>;
  def : Pat<(vtd (op (pg (SVEAllActive:$Op1)), vts:$Op2, vtd:$Op3)),
            (inst $Op3, $Op1, $Op2)>;
}

// Used to match FP_ROUND_MERGE_PASSTHRU, which has an additional flag for the
// type of rounding. This is matched by timm0_1 in pattern below and ignored.
class SVE_1_Op_Passthru_Round_Pat<ValueType vtd, SDPatternOperator op, ValueType pg,
                                  ValueType vts, Instruction inst>
: Pat<(vtd (op pg:$Op1, vts:$Op2, (i64 timm0_1), vtd:$Op3)),
      (inst $Op3, $Op1, $Op2)>;

class SVE_1_Op_Imm_OptLsl_Reverse_Pat<ValueType vt, SDPatternOperator op, ZPRRegOp zprty,
                                      ValueType it, ComplexPattern cpx, Instruction inst>
  : Pat<(vt (op (vt (AArch64dup (it (cpx i32:$imm, i32:$shift)))), (vt zprty:$Op1))),
        (inst $Op1, i32:$imm, i32:$shift)>;

class SVE_1_Op_Imm_OptLsl_Pat<ValueType vt, SDPatternOperator op, ZPRRegOp zprty,
                              ValueType it, ComplexPattern cpx, Instruction inst>
  : Pat<(vt (op (vt zprty:$Op1), (vt (AArch64dup (it (cpx i32:$imm, i32:$shift)))))),
        (inst $Op1, i32:$imm, i32:$shift)>;

class SVE_1_Op_Imm_Arith_All_Active<ValueType vt, ValueType pt, SDPatternOperator op,
                                  ZPRRegOp zprty, ValueType it, ComplexPattern cpx, Instruction inst>
  : Pat<(vt (op (pt (SVEAllActive)), (vt zprty:$Op1), (vt (AArch64dup (it (cpx i32:$imm)))))),
        (inst $Op1, i32:$imm)>;

class SVE_1_Op_Imm_Log_Pat<ValueType vt, SDPatternOperator op, ZPRRegOp zprty,
                           ValueType it, ComplexPattern cpx, Instruction inst>
  : Pat<(vt (op (vt zprty:$Op1), (vt (AArch64dup (it (cpx i64:$imm)))))),
        (inst $Op1, i64:$imm)>;

class SVE_2_Op_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                   ValueType vt2, Instruction inst>
: Pat<(vtd (op vt1:$Op1, vt2:$Op2)),
      (inst $Op1, $Op2)>;

class SVE_2_Op_Pred_All_Active<ValueType vtd, SDPatternOperator op,
                               ValueType pt, ValueType vt1, ValueType vt2,
                               Instruction inst>
: Pat<(vtd (op (pt (SVEAllActive)), vt1:$Op1, vt2:$Op2)),
      (inst $Op1, $Op2)>;

class SVE_2_Op_Pred_All_Active_Pt<ValueType vtd, SDPatternOperator op,
                                  ValueType pt, ValueType vt1, ValueType vt2,
                                  Instruction inst>
: Pat<(vtd (op (pt (SVEAllActive:$Op1)), vt1:$Op2, vt2:$Op3)),
      (inst $Op1, $Op2, $Op3)>;

class SVE_3_Op_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                   ValueType vt2, ValueType vt3, Instruction inst>
: Pat<(vtd (op vt1:$Op1, vt2:$Op2, vt3:$Op3)),
      (inst $Op1, $Op2, $Op3)>;

multiclass SVE_3_Op_Undef_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                              ValueType vt2, ValueType vt3, Instruction inst> {
  def : Pat<(vtd (op (vt1 undef), vt2:$Op1, vt3:$Op2)),
            (inst (IMPLICIT_DEF), $Op1, $Op2)>;
  def : Pat<(vtd (op vt1:$Op1, (vt2 (SVEAllActive:$Op2)), vt3:$Op3)),
            (inst $Op1, $Op2, $Op3)>;
}

class SVE_4_Op_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                   ValueType vt2, ValueType vt3, ValueType vt4,
                   Instruction inst>
: Pat<(vtd (op vt1:$Op1, vt2:$Op2, vt3:$Op3, vt4:$Op4)),
      (inst $Op1, $Op2, $Op3, $Op4)>;

class SVE_2_Op_Imm_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                       ValueType vt2, Operand ImmTy, Instruction inst>
: Pat<(vtd (op vt1:$Op1, (vt2 ImmTy:$Op2))),
      (inst $Op1, ImmTy:$Op2)>;

class SVE_3_Op_Imm_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                       ValueType vt2, ValueType vt3, Operand ImmTy,
                       Instruction inst>
: Pat<(vtd (op vt1:$Op1, vt2:$Op2, (vt3 ImmTy:$Op3))),
      (inst $Op1, $Op2, ImmTy:$Op3)>;

class SVE_4_Op_Imm_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                       ValueType vt2, ValueType vt3, ValueType vt4,
                       Operand ImmTy, Instruction inst>
: Pat<(vtd (op vt1:$Op1, vt2:$Op2, vt3:$Op3, (vt4 ImmTy:$Op4))),
      (inst $Op1, $Op2, $Op3, ImmTy:$Op4)>;

def SVEDup0 : ComplexPattern<vAny, 0, "SelectDupZero", []>;
def SVEDup0Undef : ComplexPattern<vAny, 0, "SelectDupZeroOrUndef", []>;

let AddedComplexity = 1 in {
class SVE_3_Op_Pat_SelZero<ValueType vtd, SDPatternOperator op, ValueType vt1,
                   ValueType vt2, ValueType vt3, Instruction inst>
: Pat<(vtd (vtd (op vt1:$Op1, (vselect vt1:$Op1, vt2:$Op2, (SVEDup0)), vt3:$Op3))),
      (inst $Op1, $Op2, $Op3)>;

class SVE_3_Op_Pat_Shift_Imm_SelZero<ValueType vtd, SDPatternOperator op,
                                     ValueType vt1, ValueType vt2,
                                     Operand vt3, Instruction inst>
: Pat<(vtd (op vt1:$Op1, (vselect vt1:$Op1, vt2:$Op2, (SVEDup0)), (i32 (vt3:$Op3)))),
      (inst $Op1, $Op2, vt3:$Op3)>;
}

//
// Common but less generic patterns.
//

class SVE_1_Op_AllActive_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                             Instruction inst, Instruction ptrue>
: Pat<(vtd (op vt1:$Op1)),
      (inst (IMPLICIT_DEF), (ptrue 31), $Op1)>;

class SVE_2_Op_AllActive_Pat<ValueType vtd, SDPatternOperator op, ValueType vt1,
                             ValueType vt2, Instruction inst, Instruction ptrue>
: Pat<(vtd (op vt1:$Op1, vt2:$Op2)),
      (inst (ptrue 31), $Op1, $Op2)>;

class SVE_InReg_Extend<ValueType vt, SDPatternOperator op, ValueType pt,
                       ValueType inreg_vt, Instruction inst>
: Pat<(vt (op pt:$Pg, vt:$Src, inreg_vt, vt:$PassThru)),
      (inst $PassThru, $Pg, $Src)>;

multiclass SVE_InReg_Extend_PassthruUndef<ValueType vt, SDPatternOperator op, ValueType pt,
                                          ValueType inreg_vt, Instruction inst> {
  def : Pat<(vt (op pt:$Pg, vt:$Src, inreg_vt, (vt undef))),
            (inst (IMPLICIT_DEF), $Pg, $Src)>;
  def : Pat<(vt (op (pt (SVEAllActive:$Pg)), vt:$Src, inreg_vt, vt:$PassThru)),
            (inst $PassThru, $Pg, $Src)>;
}

class SVE_Shift_DupImm_Pred_Pat<ValueType vt, SDPatternOperator op,
                                ValueType pt, ValueType it,
                                ComplexPattern cast, Instruction inst>
: Pat<(vt (op pt:$Pg, vt:$Rn, (vt (AArch64dup (it (cast i32:$imm)))))),
      (inst $Pg, $Rn, i32:$imm)>;

class SVE_Shift_DupImm_All_Active_Pat<ValueType vt, SDPatternOperator op,
                                      ValueType pt, ValueType it,
                                      ComplexPattern cast, Instruction inst>
: Pat<(vt (op (pt (SVEAllActive)), vt:$Rn, (vt (AArch64dup (it (cast i32:$imm)))))),
      (inst $Rn, i32:$imm)>;

class SVE_2_Op_Fp_Imm_Pat<ValueType vt, SDPatternOperator op,
                          ValueType pt, ValueType it,
                          FPImmLeaf immL, int imm,
                          Instruction inst>
: Pat<(vt (op (pt PPR_3b:$Pg), (vt ZPR:$Zs1), (vt (AArch64dup (it immL))))),
      (inst $Pg, $Zs1, imm)>;

class SVE_2_Op_Fp_Imm_Pat_Zero<ValueType vt, SDPatternOperator op,
                              ValueType pt, ValueType it,
                              FPImmLeaf immL, int imm,
                              Instruction inst>
: Pat<(vt (op pt:$Pg, (vselect pt:$Pg, vt:$Zs1, (SVEDup0)),
                      (vt (AArch64dup (it immL))))),
      (inst $Pg, $Zs1, imm)>;

//
// Pseudo -> Instruction mappings
//
def getSVEPseudoMap : InstrMapping {
  let FilterClass = "SVEPseudo2Instr";
  let RowFields = ["PseudoName"];
  let ColFields = ["IsInstr"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

class SVEPseudo2Instr<string name, bit instr> {
  string PseudoName = name;
  bit IsInstr = instr;
}

// Lookup e.g. DIV -> DIVR
def getSVERevInstr : InstrMapping {
  let FilterClass = "SVEInstr2Rev";
  let RowFields = ["InstrName"];
  let ColFields = ["isReverseInstr"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Lookup e.g. DIVR -> DIV
def getSVENonRevInstr : InstrMapping {
  let FilterClass = "SVEInstr2Rev";
  let RowFields = ["InstrName"];
  let ColFields = ["isReverseInstr"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

class SVEInstr2Rev<string name1, string name2, bit name1IsReverseInstr> {
  string InstrName = !if(name1IsReverseInstr, name1, name2);
  bit isReverseInstr = name1IsReverseInstr;
}

//
// Pseudos for destructive operands
//
let hasNoSchedulingInfo = 1 in {
  class PredTwoOpPseudo<string name, ZPRRegOp zprty,
                        FalseLanesEnum flags = FalseLanesNone>
  : SVEPseudo2Instr<name, 0>,
    Pseudo<(outs zprty:$Zd), (ins PPR3bAny:$Pg, zprty:$Zs1, zprty:$Zs2), []> {
    let FalseLanes = flags;
  }

  class PredTwoOpImmPseudo<string name, ZPRRegOp zprty, Operand immty,
                           FalseLanesEnum flags = FalseLanesNone>
  : SVEPseudo2Instr<name, 0>,
    Pseudo<(outs zprty:$Zd), (ins PPR3bAny:$Pg, zprty:$Zs1, immty:$imm), []> {
    let FalseLanes = flags;
  }

  class PredThreeOpPseudo<string name, ZPRRegOp zprty,
                          FalseLanesEnum flags = FalseLanesNone>
  : SVEPseudo2Instr<name, 0>,
    Pseudo<(outs zprty:$Zd), (ins PPR3bAny:$Pg, zprty:$Zs1, zprty:$Zs2, zprty:$Zs3), []> {
    let FalseLanes = flags;
  }
}

//
// Pseudos for passthru operands
//
let hasNoSchedulingInfo = 1 in {
  class PredOneOpPassthruPseudo<string name, ZPRRegOp zprty>
  : SVEPseudo2Instr<name, 0>,
    Pseudo<(outs zprty:$Zd), (ins zprty:$Passthru, PPR3bAny:$Pg, zprty:$Zs), []>;
}

//===----------------------------------------------------------------------===//
// SVE Predicate Misc Group
//===----------------------------------------------------------------------===//

class sve_int_pfalse<bits<6> opc, string asm>
: I<(outs PPR8:$Pd), (ins),
  asm, "\t$Pd",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = opc{5-4};
  let Inst{21-19} = 0b011;
  let Inst{18-16} = opc{3-1};
  let Inst{15-10} = 0b111001;
  let Inst{9}     = opc{0};
  let Inst{8-4}   = 0b00000;
  let Inst{3-0}   = Pd;

  let isReMaterializable = 1;
}

class sve_int_ptest<bits<6> opc, string asm>
: I<(outs), (ins PPRAny:$Pg, PPR8:$Pn),
  asm, "\t$Pg, $Pn",
  "",
  []>, Sched<[]> {
  bits<4> Pg;
  bits<4> Pn;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = opc{5-4};
  let Inst{21-19} = 0b010;
  let Inst{18-16} = opc{3-1};
  let Inst{15-14} = 0b11;
  let Inst{13-10} = Pg;
  let Inst{9}     = opc{0};
  let Inst{8-5}   = Pn;
  let Inst{4-0}   = 0b00000;

  let Defs = [NZCV];
  let isCompare = 1;
}

class sve_int_pfirst_next<bits<2> sz8_64, bits<5> opc, string asm,
                          PPRRegOp pprty>
: I<(outs pprty:$Pdn), (ins PPRAny:$Pg, pprty:$_Pdn),
  asm, "\t$Pdn, $Pg, $_Pdn",
  "",
  []>, Sched<[]> {
  bits<4> Pdn;
  bits<4> Pg;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-19} = 0b011;
  let Inst{18-16} = opc{4-2};
  let Inst{15-11} = 0b11000;
  let Inst{10-9}  = opc{1-0};
  let Inst{8-5}   = Pg;
  let Inst{4}     = 0;
  let Inst{3-0}   = Pdn;

  let Constraints = "$Pdn = $_Pdn";
  let Defs = [NZCV];
  let isPTestLike = 1;
  let ElementSize = pprty.ElementSize;
}

multiclass sve_int_pfirst<bits<5> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_pfirst_next<0b01, opc, asm, PPR8>;

  def : SVE_2_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, !cast<Instruction>(NAME # _B)>;
}

multiclass sve_int_pnext<bits<5> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_pfirst_next<0b00, opc, asm, PPR8>;
  def _H : sve_int_pfirst_next<0b01, opc, asm, PPR16>;
  def _S : sve_int_pfirst_next<0b10, opc, asm, PPR32>;
  def _D : sve_int_pfirst_next<0b11, opc, asm, PPR64>;

  def : SVE_2_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i1, op, nxv8i1, nxv8i1, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i1, op, nxv4i1, nxv4i1, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i1, op, nxv2i1, nxv2i1, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Predicate Count Group
//===----------------------------------------------------------------------===//

class sve_int_count_r<bits<2> sz8_64, bits<5> opc, string asm,
                      RegisterOperand dty, PPRRegOp pprty, RegisterOperand sty>
: I<(outs dty:$Rdn), (ins pprty:$Pg, sty:$_Rdn),
  asm, "\t$Rdn, $Pg",
  "",
  []>, Sched<[]> {
  bits<5> Rdn;
  bits<4> Pg;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-19} = 0b101;
  let Inst{18-16} = opc{4-2};
  let Inst{15-11} = 0b10001;
  let Inst{10-9}  = opc{1-0};
  let Inst{8-5}   = Pg;
  let Inst{4-0}   = Rdn;

  // Signed 32bit forms require their GPR operand printed.
  let AsmString = !if(!eq(opc{4,2-0}, 0b0000),
                      !strconcat(asm, "\t$Rdn, $Pg, $_Rdn"),
                      !strconcat(asm, "\t$Rdn, $Pg"));
  let Constraints = "$Rdn = $_Rdn";
}

multiclass sve_int_count_r_s32<bits<5> opc, string asm,
                               SDPatternOperator op> {
  def _B : sve_int_count_r<0b00, opc, asm, GPR64z, PPR8, GPR64as32>;
  def _H : sve_int_count_r<0b01, opc, asm, GPR64z, PPR16, GPR64as32>;
  def _S : sve_int_count_r<0b10, opc, asm, GPR64z, PPR32, GPR64as32>;
  def _D : sve_int_count_r<0b11, opc, asm, GPR64z, PPR64, GPR64as32>;

  def : Pat<(i32 (op GPR32:$Rn, (nxv16i1 PPRAny:$Pg))),
            (EXTRACT_SUBREG (!cast<Instruction>(NAME # _B) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32)), sub_32)>;
  def : Pat<(i64 (sext (i32 (op GPR32:$Rn, (nxv16i1 PPRAny:$Pg))))),
            (!cast<Instruction>(NAME # _B) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32))>;

  def : Pat<(i32 (op GPR32:$Rn, (nxv8i1 PPRAny:$Pg))),
            (EXTRACT_SUBREG (!cast<Instruction>(NAME # _H) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32)), sub_32)>;
  def : Pat<(i64 (sext (i32 (op GPR32:$Rn, (nxv8i1 PPRAny:$Pg))))),
            (!cast<Instruction>(NAME # _H) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32))>;

  def : Pat<(i32 (op GPR32:$Rn, (nxv4i1 PPRAny:$Pg))),
            (EXTRACT_SUBREG (!cast<Instruction>(NAME # _S) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32)), sub_32)>;
  def : Pat<(i64 (sext (i32 (op GPR32:$Rn, (nxv4i1 PPRAny:$Pg))))),
            (!cast<Instruction>(NAME # _S) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32))>;

  def : Pat<(i32 (op GPR32:$Rn, (nxv2i1 PPRAny:$Pg))),
            (EXTRACT_SUBREG (!cast<Instruction>(NAME # _D) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32)), sub_32)>;
  def : Pat<(i64 (sext (i32 (op GPR32:$Rn, (nxv2i1 PPRAny:$Pg))))),
            (!cast<Instruction>(NAME # _D) PPRAny:$Pg, (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32))>;
}

multiclass sve_int_count_r_u32<bits<5> opc, string asm,
                               SDPatternOperator op> {
  def _B : sve_int_count_r<0b00, opc, asm, GPR32z, PPR8, GPR32z>;
  def _H : sve_int_count_r<0b01, opc, asm, GPR32z, PPR16, GPR32z>;
  def _S : sve_int_count_r<0b10, opc, asm, GPR32z, PPR32, GPR32z>;
  def _D : sve_int_count_r<0b11, opc, asm, GPR32z, PPR64, GPR32z>;

  def : Pat<(i32 (op GPR32:$Rn, (nxv16i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _B) PPRAny:$Pg, $Rn)>;
  def : Pat<(i32 (op GPR32:$Rn, (nxv8i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _H) PPRAny:$Pg, $Rn)>;
  def : Pat<(i32 (op GPR32:$Rn, (nxv4i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _S) PPRAny:$Pg, $Rn)>;
  def : Pat<(i32 (op GPR32:$Rn, (nxv2i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _D) PPRAny:$Pg, $Rn)>;
}

multiclass sve_int_count_r_x64<bits<5> opc, string asm,
                               SDPatternOperator op,
                               SDPatternOperator combine_op = null_frag> {
  def _B : sve_int_count_r<0b00, opc, asm, GPR64z, PPR8, GPR64z>;
  def _H : sve_int_count_r<0b01, opc, asm, GPR64z, PPR16, GPR64z>;
  def _S : sve_int_count_r<0b10, opc, asm, GPR64z, PPR32, GPR64z>;
  def _D : sve_int_count_r<0b11, opc, asm, GPR64z, PPR64, GPR64z>;

  def : Pat<(i64 (op GPR64:$Rn, (nxv16i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _B) PPRAny:$Pg, $Rn)>;
  def : Pat<(i64 (op GPR64:$Rn, (nxv8i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _H) PPRAny:$Pg, $Rn)>;
  def : Pat<(i64 (op GPR64:$Rn, (nxv4i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _S) PPRAny:$Pg, $Rn)>;
  def : Pat<(i64 (op GPR64:$Rn, (nxv2i1 PPRAny:$Pg))),
            (!cast<Instruction>(NAME # _D) PPRAny:$Pg, $Rn)>;

  // Combine cntp with combine_op
  def : Pat<(i64 (combine_op GPR64:$Rn, (int_aarch64_sve_cntp_oneuse (nxv16i1 (SVEAllActive)), (nxv16i1 PPRAny:$pred)))),
            (!cast<Instruction>(NAME # _B) PPRAny:$pred, $Rn)>;
  def : Pat<(i64 (combine_op GPR64:$Rn, (int_aarch64_sve_cntp_oneuse (nxv8i1 (SVEAllActive)), (nxv8i1 PPRAny:$pred)))),
            (!cast<Instruction>(NAME # _H) PPRAny:$pred, $Rn)>;
  def : Pat<(i64 (combine_op GPR64:$Rn, (int_aarch64_sve_cntp_oneuse (nxv4i1 (SVEAllActive)), (nxv4i1 PPRAny:$pred)))),
            (!cast<Instruction>(NAME # _S) PPRAny:$pred, $Rn)>;
  def : Pat<(i64 (combine_op GPR64:$Rn, (int_aarch64_sve_cntp_oneuse (nxv2i1 (SVEAllActive)), (nxv2i1 PPRAny:$pred)))),
            (!cast<Instruction>(NAME # _D) PPRAny:$pred, $Rn)>;
}

class sve_int_count_v<bits<2> sz8_64, bits<5> opc, string asm,
                      ZPRRegOp zprty, PPRRegOp pprty>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, pprty:$Pm),
  asm, "\t$Zdn, $Pm",
  "",
  []>, Sched<[]> {
  bits<4> Pm;
  bits<5> Zdn;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-19} = 0b101;
  let Inst{18-16} = opc{4-2};
  let Inst{15-11} = 0b10000;
  let Inst{10-9}  = opc{1-0};
  let Inst{8-5}   = Pm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_count_v<bits<5> opc, string asm,
                           SDPatternOperator op = null_frag> {
  def _H : sve_int_count_v<0b01, opc, asm, ZPR16, PPR16>;
  def _S : sve_int_count_v<0b10, opc, asm, ZPR32, PPR32>;
  def _D : sve_int_count_v<0b11, opc, asm, ZPR64, PPR64>;

  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16,  nxv8i1, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32,  nxv4i1, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64,  nxv2i1, !cast<Instruction>(NAME # _D)>;

  def : InstAlias<asm # "\t$Zdn, $Pm",
                 (!cast<Instruction>(NAME # "_H") ZPR16:$Zdn, PPRAny:$Pm), 0>;
  def : InstAlias<asm # "\t$Zdn, $Pm",
                 (!cast<Instruction>(NAME # "_S") ZPR32:$Zdn, PPRAny:$Pm), 0>;
  def : InstAlias<asm # "\t$Zdn, $Pm",
                  (!cast<Instruction>(NAME # "_D") ZPR64:$Zdn, PPRAny:$Pm), 0>;
}

class sve_int_pcount_pred<bits<2> sz8_64, bits<4> opc, string asm,
                          PPRRegOp pprty>
: I<(outs GPR64:$Rd), (ins PPRAny:$Pg, pprty:$Pn),
  asm, "\t$Rd, $Pg, $Pn",
  "",
  []>, Sched<[]> {
  bits<4> Pg;
  bits<4> Pn;
  bits<5> Rd;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-19} = 0b100;
  let Inst{18-16} = opc{3-1};
  let Inst{15-14} = 0b10;
  let Inst{13-10} = Pg;
  let Inst{9}     = opc{0};
  let Inst{8-5}   = Pn;
  let Inst{4-0}   = Rd;
}

multiclass sve_int_pcount_pred<bits<4> opc, string asm,
                               SDPatternOperator int_op> {
  def _B : sve_int_pcount_pred<0b00, opc, asm, PPR8>;
  def _H : sve_int_pcount_pred<0b01, opc, asm, PPR16>;
  def _S : sve_int_pcount_pred<0b10, opc, asm, PPR32>;
  def _D : sve_int_pcount_pred<0b11, opc, asm, PPR64>;

  def : SVE_2_Op_Pat<i64, int_op, nxv16i1, nxv16i1, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<i64, int_op, nxv8i1,  nxv8i1,  !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<i64, int_op, nxv4i1,  nxv4i1,  !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<i64, int_op, nxv2i1,  nxv2i1,  !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Element Count Group
//===----------------------------------------------------------------------===//

class sve_int_count<bits<3> opc, string asm>
: I<(outs GPR64:$Rd), (ins sve_pred_enum:$pattern, sve_incdec_imm:$imm4),
  asm, "\t$Rd, $pattern, mul $imm4",
  "",
  []>, Sched<[]> {
  bits<5> Rd;
  bits<4> imm4;
  bits<5> pattern;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc{2-1};
  let Inst{21-20} = 0b10;
  let Inst{19-16} = imm4;
  let Inst{15-11} = 0b11100;
  let Inst{10}    = opc{0};
  let Inst{9-5}   = pattern;
  let Inst{4-0}   = Rd;
}

multiclass sve_int_count<bits<3> opc, string asm, SDPatternOperator op> {
  def NAME : sve_int_count<opc, asm>;

  def : InstAlias<asm # "\t$Rd, $pattern",
                  (!cast<Instruction>(NAME) GPR64:$Rd, sve_pred_enum:$pattern, 1), 1>;
  def : InstAlias<asm # "\t$Rd",
                  (!cast<Instruction>(NAME) GPR64:$Rd, 0b11111, 1), 2>;

  def : Pat<(i64 (mul (op sve_pred_enum:$pattern), (sve_cnt_mul_imm_i64 i32:$imm))),
            (!cast<Instruction>(NAME) sve_pred_enum:$pattern, sve_incdec_imm:$imm)>;

  def : Pat<(i64 (shl (op sve_pred_enum:$pattern), (sve_cnt_shl_imm i32:$imm))),
            (!cast<Instruction>(NAME) sve_pred_enum:$pattern, sve_incdec_imm:$imm)>;

  def : Pat<(i64 (op sve_pred_enum:$pattern)),
            (!cast<Instruction>(NAME) sve_pred_enum:$pattern, 1)>;
}

class sve_int_countvlv<bits<5> opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, sve_pred_enum:$pattern, sve_incdec_imm:$imm4),
  asm, "\t$Zdn, $pattern, mul $imm4",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> pattern;
  bits<4> imm4;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc{4-3};
  let Inst{21}    = 0b1;
  let Inst{20}    = opc{2};
  let Inst{19-16} = imm4;
  let Inst{15-12} = 0b1100;
  let Inst{11-10} = opc{1-0};
  let Inst{9-5}   = pattern;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_countvlv<bits<5> opc, string asm, ZPRRegOp zprty,
                            SDPatternOperator op = null_frag,
                            ValueType vt = OtherVT> {
  def NAME : sve_int_countvlv<opc, asm, zprty>;

  def : InstAlias<asm # "\t$Zdn, $pattern",
                  (!cast<Instruction>(NAME) zprty:$Zdn, sve_pred_enum:$pattern, 1), 1>;
  def : InstAlias<asm # "\t$Zdn",
                  (!cast<Instruction>(NAME) zprty:$Zdn, 0b11111, 1), 2>;

  def : Pat<(vt (op (vt zprty:$Zn), (sve_pred_enum:$pattern), (sve_incdec_imm:$imm4))),
            (!cast<Instruction>(NAME) $Zn, sve_pred_enum:$pattern, sve_incdec_imm:$imm4)>;
}

class sve_int_pred_pattern_a<bits<3> opc, string asm>
: I<(outs GPR64:$Rdn), (ins GPR64:$_Rdn, sve_pred_enum:$pattern, sve_incdec_imm:$imm4),
  asm, "\t$Rdn, $pattern, mul $imm4",
  "",
  []>, Sched<[]> {
  bits<5> Rdn;
  bits<5> pattern;
  bits<4> imm4;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc{2-1};
  let Inst{21-20} = 0b11;
  let Inst{19-16} = imm4;
  let Inst{15-11} = 0b11100;
  let Inst{10}    = opc{0};
  let Inst{9-5}   = pattern;
  let Inst{4-0}   = Rdn;

  let Constraints = "$Rdn = $_Rdn";
}

multiclass sve_int_pred_pattern_a<bits<3> opc, string asm,
                                  SDPatternOperator op,
                                  SDPatternOperator opcnt> {
  let Predicates = [HasSVEorStreamingSVE] in {
    def NAME : sve_int_pred_pattern_a<opc, asm>;

    def : InstAlias<asm # "\t$Rdn, $pattern",
                    (!cast<Instruction>(NAME) GPR64:$Rdn, sve_pred_enum:$pattern, 1), 1>;
    def : InstAlias<asm # "\t$Rdn",
                    (!cast<Instruction>(NAME) GPR64:$Rdn, 0b11111, 1), 2>;
  }

  let Predicates = [HasSVEorStreamingSVE, UseScalarIncVL] in {
    def : Pat<(i64 (op GPR64:$Rdn, (opcnt sve_pred_enum:$pattern))),
              (!cast<Instruction>(NAME) GPR64:$Rdn, sve_pred_enum:$pattern, 1)>;

    def : Pat<(i64 (op GPR64:$Rdn, (mul (opcnt sve_pred_enum:$pattern), (sve_cnt_mul_imm_i64 i32:$imm)))),
              (!cast<Instruction>(NAME) GPR64:$Rdn, sve_pred_enum:$pattern, $imm)>;

    def : Pat<(i64 (op GPR64:$Rdn, (shl (opcnt sve_pred_enum:$pattern), (sve_cnt_shl_imm i32:$imm)))),
              (!cast<Instruction>(NAME) GPR64:$Rdn, sve_pred_enum:$pattern, $imm)>;

    def : Pat<(i32 (op GPR32:$Rdn, (i32 (trunc (opcnt (sve_pred_enum:$pattern)))))),
              (i32 (EXTRACT_SUBREG (!cast<Instruction>(NAME) (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                               GPR32:$Rdn, sub_32), sve_pred_enum:$pattern, 1),
                                    sub_32))>;

    def : Pat<(i32 (op GPR32:$Rdn, (mul (i32 (trunc (opcnt (sve_pred_enum:$pattern)))), (sve_cnt_mul_imm_i32 i32:$imm)))),
              (i32 (EXTRACT_SUBREG (!cast<Instruction>(NAME) (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                               GPR32:$Rdn, sub_32), sve_pred_enum:$pattern, $imm),
                                    sub_32))>;

    def : Pat<(i32 (op GPR32:$Rdn, (shl (i32 (trunc (opcnt (sve_pred_enum:$pattern)))), (sve_cnt_shl_imm i32:$imm)))),
              (i32 (EXTRACT_SUBREG (!cast<Instruction>(NAME) (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                               GPR32:$Rdn, sub_32), sve_pred_enum:$pattern, $imm),
                                    sub_32))>;
  }
}

class sve_int_pred_pattern_b<bits<5> opc, string asm, RegisterOperand dt,
                             RegisterOperand st>
: I<(outs dt:$Rdn), (ins st:$_Rdn, sve_pred_enum:$pattern, sve_incdec_imm:$imm4),
  asm, "\t$Rdn, $pattern, mul $imm4",
  "",
  []>, Sched<[]> {
  bits<5> Rdn;
  bits<5> pattern;
  bits<4> imm4;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc{4-3};
  let Inst{21}    = 0b1;
  let Inst{20}    = opc{2};
  let Inst{19-16} = imm4;
  let Inst{15-12} = 0b1111;
  let Inst{11-10} = opc{1-0};
  let Inst{9-5}   = pattern;
  let Inst{4-0}   = Rdn;

  // Signed 32bit forms require their GPR operand printed.
  let AsmString = !if(!eq(opc{2,0}, 0b00),
                      !strconcat(asm, "\t$Rdn, $_Rdn, $pattern, mul $imm4"),
                      !strconcat(asm, "\t$Rdn, $pattern, mul $imm4"));

  let Constraints = "$Rdn = $_Rdn";
}

multiclass sve_int_pred_pattern_b_s32<bits<5> opc, string asm,
                                      SDPatternOperator op> {
  def NAME : sve_int_pred_pattern_b<opc, asm, GPR64z, GPR64as32>;

  def : InstAlias<asm # "\t$Rd, $Rn, $pattern",
                  (!cast<Instruction>(NAME) GPR64z:$Rd, GPR64as32:$Rn, sve_pred_enum:$pattern, 1), 1>;
  def : InstAlias<asm # "\t$Rd, $Rn",
                  (!cast<Instruction>(NAME) GPR64z:$Rd, GPR64as32:$Rn, 0b11111, 1), 2>;

  // NOTE: Register allocation doesn't like tied operands of differing register
  //       class, hence the extra INSERT_SUBREG complication.

  def : Pat<(i32 (op GPR32:$Rn, (sve_pred_enum:$pattern), (sve_incdec_imm:$imm4))),
            (EXTRACT_SUBREG (!cast<Instruction>(NAME) (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32), sve_pred_enum:$pattern, sve_incdec_imm:$imm4), sub_32)>;
  def : Pat<(i64 (sext (i32 (op GPR32:$Rn, (sve_pred_enum:$pattern), (sve_incdec_imm:$imm4))))),
            (!cast<Instruction>(NAME) (INSERT_SUBREG (IMPLICIT_DEF), $Rn, sub_32), sve_pred_enum:$pattern, sve_incdec_imm:$imm4)>;
}

multiclass sve_int_pred_pattern_b_u32<bits<5> opc, string asm,
                                      SDPatternOperator op> {
  def NAME : sve_int_pred_pattern_b<opc, asm, GPR32z, GPR32z>;

  def : InstAlias<asm # "\t$Rdn, $pattern",
                  (!cast<Instruction>(NAME) GPR32z:$Rdn, sve_pred_enum:$pattern, 1), 1>;
  def : InstAlias<asm # "\t$Rdn",
                  (!cast<Instruction>(NAME) GPR32z:$Rdn, 0b11111, 1), 2>;

  def : Pat<(i32 (op GPR32:$Rn, (sve_pred_enum:$pattern), (sve_incdec_imm:$imm4))),
            (!cast<Instruction>(NAME) $Rn, sve_pred_enum:$pattern, sve_incdec_imm:$imm4)>;
}

multiclass sve_int_pred_pattern_b_x64<bits<5> opc, string asm,
                                      SDPatternOperator op> {
  def NAME : sve_int_pred_pattern_b<opc, asm, GPR64z, GPR64z>;

  def : InstAlias<asm # "\t$Rdn, $pattern",
                  (!cast<Instruction>(NAME) GPR64z:$Rdn, sve_pred_enum:$pattern, 1), 1>;
  def : InstAlias<asm # "\t$Rdn",
                  (!cast<Instruction>(NAME) GPR64z:$Rdn, 0b11111, 1), 2>;

  def : Pat<(i64 (op GPR64:$Rn, (sve_pred_enum:$pattern), (sve_incdec_imm:$imm4))),
            (!cast<Instruction>(NAME) $Rn, sve_pred_enum:$pattern, sve_incdec_imm:$imm4)>;
}


//===----------------------------------------------------------------------===//
// SVE Permute - Cross Lane Group
//===----------------------------------------------------------------------===//

class sve_int_perm_dup_r<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                         ValueType vt, RegisterClass srcRegType,
                         SDPatternOperator op>
: I<(outs zprty:$Zd), (ins srcRegType:$Rn),
  asm, "\t$Zd, $Rn",
  "",
  [(set (vt zprty:$Zd), (op srcRegType:$Rn))]>, Sched<[]> {
  bits<5> Rn;
  bits<5> Zd;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-10} = 0b100000001110;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_dup_r<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_dup_r<0b00, asm, ZPR8, nxv16i8, GPR32sp, op>;
  def _H : sve_int_perm_dup_r<0b01, asm, ZPR16, nxv8i16, GPR32sp, op>;
  def _S : sve_int_perm_dup_r<0b10, asm, ZPR32, nxv4i32, GPR32sp, op>;
  def _D : sve_int_perm_dup_r<0b11, asm, ZPR64, nxv2i64, GPR64sp, op>;

  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, GPR64sp:$Rn), 1>;
}

class sve_int_perm_dup_i<bits<5> tsz, Operand immtype, string asm,
                         ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn, immtype:$idx),
  asm, "\t$Zd, $Zn$idx",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<7> idx;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = {?,?}; // imm3h
  let Inst{21}    = 0b1;
  let Inst{20-16} = tsz;
  let Inst{15-10} = 0b001000;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_dup_i<string asm> {
  def _B : sve_int_perm_dup_i<{?,?,?,?,1}, sve_elm_idx_extdup_b, asm, ZPR8> {
    let Inst{23-22} = idx{5-4};
    let Inst{20-17} = idx{3-0};
  }
  def _H : sve_int_perm_dup_i<{?,?,?,1,0}, sve_elm_idx_extdup_h, asm, ZPR16> {
    let Inst{23-22} = idx{4-3};
    let Inst{20-18} = idx{2-0};
  }
  def _S : sve_int_perm_dup_i<{?,?,1,0,0}, sve_elm_idx_extdup_s, asm, ZPR32> {
    let Inst{23-22} = idx{3-2};
    let Inst{20-19}    = idx{1-0};
  }
  def _D : sve_int_perm_dup_i<{?,1,0,0,0}, sve_elm_idx_extdup_d, asm, ZPR64> {
    let Inst{23-22} = idx{2-1};
    let Inst{20}    = idx{0};
  }
  def _Q : sve_int_perm_dup_i<{1,0,0,0,0}, sve_elm_idx_extdup_q, asm, ZPR128> {
    let Inst{23-22} = idx{1-0};
  }

  def : InstAlias<"mov $Zd, $Zn$idx",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, ZPR8:$Zn, sve_elm_idx_extdup_b:$idx), 1>;
  def : InstAlias<"mov $Zd, $Zn$idx",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, ZPR16:$Zn, sve_elm_idx_extdup_h:$idx), 1>;
  def : InstAlias<"mov $Zd, $Zn$idx",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, ZPR32:$Zn, sve_elm_idx_extdup_s:$idx), 1>;
  def : InstAlias<"mov $Zd, $Zn$idx",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, ZPR64:$Zn, sve_elm_idx_extdup_d:$idx), 1>;
  def : InstAlias<"mov $Zd, $Zn$idx",
                  (!cast<Instruction>(NAME # _Q) ZPR128:$Zd, ZPR128:$Zn, sve_elm_idx_extdup_q:$idx), 1>;
  def : InstAlias<"mov $Zd, $Bn",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, FPR8asZPR:$Bn, 0), 2>;
  def : InstAlias<"mov $Zd, $Hn",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, FPR16asZPR:$Hn, 0), 2>;
  def : InstAlias<"mov $Zd, $Sn",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, FPR32asZPR:$Sn, 0), 2>;
  def : InstAlias<"mov $Zd, $Dn",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, FPR64asZPR:$Dn, 0), 2>;
  def : InstAlias<"mov $Zd, $Qn",
                  (!cast<Instruction>(NAME # _Q) ZPR128:$Zd, FPR128asZPR:$Qn, 0), 2>;

  // Duplicate extracted element of vector into all vector elements
  def : Pat<(nxv16i8 (AArch64dup (i32 (vector_extract (nxv16i8 ZPR:$vec), sve_elm_idx_extdup_b:$index)))),
            (!cast<Instruction>(NAME # _B) ZPR:$vec, sve_elm_idx_extdup_b:$index)>;
  def : Pat<(nxv8i16 (AArch64dup (i32 (vector_extract (nxv8i16 ZPR:$vec), sve_elm_idx_extdup_h:$index)))),
            (!cast<Instruction>(NAME # _H) ZPR:$vec, sve_elm_idx_extdup_h:$index)>;
  def : Pat<(nxv4i32 (AArch64dup (i32 (vector_extract (nxv4i32 ZPR:$vec), sve_elm_idx_extdup_s:$index)))),
            (!cast<Instruction>(NAME # _S) ZPR:$vec, sve_elm_idx_extdup_s:$index)>;
  def : Pat<(nxv2i64 (AArch64dup (i64 (vector_extract (nxv2i64 ZPR:$vec), sve_elm_idx_extdup_d:$index)))),
            (!cast<Instruction>(NAME # _D) ZPR:$vec, sve_elm_idx_extdup_d:$index)>;
  def : Pat<(nxv8f16 (AArch64dup (f16 (vector_extract (nxv8f16 ZPR:$vec), sve_elm_idx_extdup_h:$index)))),
            (!cast<Instruction>(NAME # _H) ZPR:$vec, sve_elm_idx_extdup_h:$index)>;
  def : Pat<(nxv8bf16 (AArch64dup (bf16 (vector_extract (nxv8bf16 ZPR:$vec), sve_elm_idx_extdup_h:$index)))),
            (!cast<Instruction>(NAME # _H) ZPR:$vec, sve_elm_idx_extdup_h:$index)>;
  def : Pat<(nxv4f16 (AArch64dup (f16 (vector_extract (nxv4f16 ZPR:$vec), sve_elm_idx_extdup_s:$index)))),
            (!cast<Instruction>(NAME # _S) ZPR:$vec, sve_elm_idx_extdup_s:$index)>;
  def : Pat<(nxv2f16 (AArch64dup (f16 (vector_extract (nxv2f16 ZPR:$vec), sve_elm_idx_extdup_d:$index)))),
            (!cast<Instruction>(NAME # _D) ZPR:$vec, sve_elm_idx_extdup_d:$index)>;
  def : Pat<(nxv4f32 (AArch64dup (f32 (vector_extract (nxv4f32 ZPR:$vec), sve_elm_idx_extdup_s:$index)))),
            (!cast<Instruction>(NAME # _S) ZPR:$vec, sve_elm_idx_extdup_s:$index)>;
  def : Pat<(nxv2f32 (AArch64dup (f32 (vector_extract (nxv2f32 ZPR:$vec), sve_elm_idx_extdup_d:$index)))),
            (!cast<Instruction>(NAME # _D) ZPR:$vec, sve_elm_idx_extdup_d:$index)>;
  def : Pat<(nxv2f64 (AArch64dup (f64 (vector_extract (nxv2f64 ZPR:$vec), sve_elm_idx_extdup_d:$index)))),
            (!cast<Instruction>(NAME # _D) ZPR:$vec, sve_elm_idx_extdup_d:$index)>;
}

class sve_int_perm_tbl<bits<2> sz8_64, bits<2> opc, string asm, ZPRRegOp zprty,
                       RegisterOperand VecList>
: I<(outs zprty:$Zd), (ins VecList:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b001;
  let Inst{12-11} = opc;
  let Inst{10}    = 0b0;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_tbl<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_tbl<0b00, 0b10, asm, ZPR8,  Z_b>;
  def _H : sve_int_perm_tbl<0b01, 0b10, asm, ZPR16, Z_h>;
  def _S : sve_int_perm_tbl<0b10, 0b10, asm, ZPR32, Z_s>;
  def _D : sve_int_perm_tbl<0b11, 0b10, asm, ZPR64, Z_d>;

  def : InstAlias<asm # "\t$Zd, $Zn, $Zm",
                 (!cast<Instruction>(NAME # _B) ZPR8:$Zd, ZPR8:$Zn, ZPR8:$Zm), 0>;
  def : InstAlias<asm # "\t$Zd, $Zn, $Zm",
                 (!cast<Instruction>(NAME # _H) ZPR16:$Zd, ZPR16:$Zn, ZPR16:$Zm), 0>;
  def : InstAlias<asm # "\t$Zd, $Zn, $Zm",
                 (!cast<Instruction>(NAME # _S) ZPR32:$Zd, ZPR32:$Zn, ZPR32:$Zm), 0>;
  def : InstAlias<asm # "\t$Zd, $Zn, $Zm",
                 (!cast<Instruction>(NAME # _D) ZPR64:$Zd, ZPR64:$Zn, ZPR64:$Zm), 0>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_2_Op_Pat<nxv8f16, op, nxv8f16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4f32, op, nxv4f32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2f64, op, nxv2f64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_2_Op_Pat<nxv8bf16, op, nxv8bf16, nxv8i16, !cast<Instruction>(NAME # _H)>;
}

multiclass sve2_int_perm_tbl<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_tbl<0b00, 0b01, asm, ZPR8,  ZZ_b>;
  def _H : sve_int_perm_tbl<0b01, 0b01, asm, ZPR16, ZZ_h>;
  def _S : sve_int_perm_tbl<0b10, 0b01, asm, ZPR32, ZZ_s>;
  def _D : sve_int_perm_tbl<0b11, 0b01, asm, ZPR64, ZZ_d>;

  def : Pat<(nxv16i8 (op nxv16i8:$Op1, nxv16i8:$Op2, nxv16i8:$Op3)),
            (nxv16i8 (!cast<Instruction>(NAME # _B) (REG_SEQUENCE ZPR2, nxv16i8:$Op1, zsub0,
                                                                        nxv16i8:$Op2, zsub1),
                                                     nxv16i8:$Op3))>;

  def : Pat<(nxv8i16 (op nxv8i16:$Op1, nxv8i16:$Op2, nxv8i16:$Op3)),
            (nxv8i16 (!cast<Instruction>(NAME # _H) (REG_SEQUENCE ZPR2, nxv8i16:$Op1, zsub0,
                                                                        nxv8i16:$Op2, zsub1),
                                                     nxv8i16:$Op3))>;

  def : Pat<(nxv4i32 (op nxv4i32:$Op1, nxv4i32:$Op2, nxv4i32:$Op3)),
            (nxv4i32 (!cast<Instruction>(NAME # _S) (REG_SEQUENCE ZPR2, nxv4i32:$Op1, zsub0,
                                                                        nxv4i32:$Op2, zsub1),
                                                     nxv4i32:$Op3))>;

  def : Pat<(nxv2i64 (op nxv2i64:$Op1, nxv2i64:$Op2, nxv2i64:$Op3)),
            (nxv2i64 (!cast<Instruction>(NAME # _D) (REG_SEQUENCE ZPR2, nxv2i64:$Op1, zsub0,
                                                                        nxv2i64:$Op2, zsub1),
                                                     nxv2i64:$Op3))>;

  def : Pat<(nxv8f16 (op nxv8f16:$Op1, nxv8f16:$Op2, nxv8i16:$Op3)),
            (nxv8f16 (!cast<Instruction>(NAME # _H) (REG_SEQUENCE ZPR2, nxv8f16:$Op1, zsub0,
                                                                        nxv8f16:$Op2, zsub1),
                                                     nxv8i16:$Op3))>;

  def : Pat<(nxv4f32 (op nxv4f32:$Op1, nxv4f32:$Op2, nxv4i32:$Op3)),
            (nxv4f32 (!cast<Instruction>(NAME # _S) (REG_SEQUENCE ZPR2, nxv4f32:$Op1, zsub0,
                                                                        nxv4f32:$Op2, zsub1),
                                                     nxv4i32:$Op3))>;

  def : Pat<(nxv2f64 (op nxv2f64:$Op1, nxv2f64:$Op2, nxv2i64:$Op3)),
            (nxv2f64 (!cast<Instruction>(NAME # _D) (REG_SEQUENCE ZPR2, nxv2f64:$Op1, zsub0,
                                                                        nxv2f64:$Op2, zsub1),
                                                     nxv2i64:$Op3))>;

  def : Pat<(nxv8bf16 (op nxv8bf16:$Op1, nxv8bf16:$Op2, nxv8i16:$Op3)),
            (nxv8bf16 (!cast<Instruction>(NAME # _H) (REG_SEQUENCE ZPR2, nxv8bf16:$Op1, zsub0,
                                                                         nxv8bf16:$Op2, zsub1),
                                                      nxv8i16:$Op3))>;
}

class sve2_int_perm_tbx<bits<2> sz8_64, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b001011;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
}

multiclass sve2_int_perm_tbx<string asm, SDPatternOperator op> {
  def _B : sve2_int_perm_tbx<0b00, asm, ZPR8>;
  def _H : sve2_int_perm_tbx<0b01, asm, ZPR16>;
  def _S : sve2_int_perm_tbx<0b10, asm, ZPR32>;
  def _D : sve2_int_perm_tbx<0b11, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8f16, nxv8f16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4f32, nxv4f32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2f64, nxv2f64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8bf16, op, nxv8bf16, nxv8bf16, nxv8i16, !cast<Instruction>(NAME # _H)>;
}

class sve_int_perm_reverse_z<bits<2> sz8_64, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn),
  asm, "\t$Zd, $Zn",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-10} = 0b111000001110;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_reverse_z<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_reverse_z<0b00, asm, ZPR8>;
  def _H : sve_int_perm_reverse_z<0b01, asm, ZPR16>;
  def _S : sve_int_perm_reverse_z<0b10, asm, ZPR32>;
  def _D : sve_int_perm_reverse_z<0b11, asm, ZPR64>;

  def : SVE_1_Op_Pat<nxv16i8, op, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Pat<nxv8i16, op, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Pat<nxv4i32, op, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv2i64, op, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_1_Op_Pat<nxv2f16, op, nxv2f16, !cast<Instruction>(NAME # _D)>;
  def : SVE_1_Op_Pat<nxv4f16, op, nxv4f16, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv8f16, op, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Pat<nxv2f32, op, nxv2f32, !cast<Instruction>(NAME # _D)>;
  def : SVE_1_Op_Pat<nxv4f32, op, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv2f64, op, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_1_Op_Pat<nxv2bf16, op, nxv2bf16, !cast<Instruction>(NAME # _D)>;
  def : SVE_1_Op_Pat<nxv4bf16, op, nxv4bf16, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv8bf16, op, nxv8bf16, !cast<Instruction>(NAME # _H)>;
}

class sve_int_perm_reverse_p<bits<2> sz8_64, string asm, PPRRegOp pprty>
: I<(outs pprty:$Pd), (ins pprty:$Pn),
  asm, "\t$Pd, $Pn",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<4> Pn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-9}  = 0b1101000100000;
  let Inst{8-5}   = Pn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pd;
}

multiclass sve_int_perm_reverse_p<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_reverse_p<0b00, asm, PPR8>;
  def _H : sve_int_perm_reverse_p<0b01, asm, PPR16>;
  def _S : sve_int_perm_reverse_p<0b10, asm, PPR32>;
  def _D : sve_int_perm_reverse_p<0b11, asm, PPR64>;

  def : SVE_1_Op_Pat<nxv16i1, op, nxv16i1, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Pat<nxv8i1, op, nxv8i1, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Pat<nxv4i1, op, nxv4i1, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv2i1, op, nxv2i1, !cast<Instruction>(NAME # _D)>;
}

class sve_int_perm_unpk<bits<2> sz16_64, bits<2> opc, string asm,
                        ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn),
  asm, "\t$Zd, $Zn",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz16_64;
  let Inst{21-18} = 0b1100;
  let Inst{17-16} = opc;
  let Inst{15-10} = 0b001110;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_unpk<bits<2> opc, string asm, SDPatternOperator op> {
  def _H : sve_int_perm_unpk<0b01, opc, asm, ZPR16, ZPR8>;
  def _S : sve_int_perm_unpk<0b10, opc, asm, ZPR32, ZPR16>;
  def _D : sve_int_perm_unpk<0b11, opc, asm, ZPR64, ZPR32>;

  def : SVE_1_Op_Pat<nxv8i16, op, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Pat<nxv4i32, op, nxv8i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv2i64, op, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

class sve_int_perm_insrs<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                         RegisterClass srcRegType>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, srcRegType:$Rm),
  asm, "\t$Zdn, $Rm",
  "",
  []>, Sched<[]> {
  bits<5> Rm;
  bits<5> Zdn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-10} = 0b100100001110;
  let Inst{9-5}   = Rm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
}

multiclass sve_int_perm_insrs<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_insrs<0b00, asm, ZPR8, GPR32>;
  def _H : sve_int_perm_insrs<0b01, asm, ZPR16, GPR32>;
  def _S : sve_int_perm_insrs<0b10, asm, ZPR32, GPR32>;
  def _D : sve_int_perm_insrs<0b11, asm, ZPR64, GPR64>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, i32, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, i32, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, i64, !cast<Instruction>(NAME # _D)>;
}

class sve_int_perm_insrv<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                         FPRasZPROperand srcOpType>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, srcOpType:$Vm),
  asm, "\t$Zdn, $Vm",
  "",
  []>, Sched<[]> {
  bits<5> Vm;
  bits<5> Zdn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-10} = 0b110100001110;
  let Inst{9-5}   = Vm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
}

multiclass sve_int_perm_insrv<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_insrv<0b00, asm, ZPR8, FPR8asZPR>;
  def _H : sve_int_perm_insrv<0b01, asm, ZPR16, FPR16asZPR>;
  def _S : sve_int_perm_insrv<0b10, asm, ZPR32, FPR32asZPR>;
  def _D : sve_int_perm_insrv<0b11, asm, ZPR64, FPR64asZPR>;

  def : Pat<(nxv8f16 (op nxv8f16:$Zn, f16:$Vm)),
            (!cast<Instruction>(NAME # _H) $Zn, (INSERT_SUBREG (IMPLICIT_DEF), $Vm, hsub))>;
  def : Pat<(nxv4f32 (op nxv4f32:$Zn, f32:$Vm)),
            (!cast<Instruction>(NAME # _S) $Zn, (INSERT_SUBREG (IMPLICIT_DEF), $Vm, ssub))>;
  def : Pat<(nxv2f64 (op nxv2f64:$Zn, f64:$Vm)),
            (!cast<Instruction>(NAME # _D) $Zn, (INSERT_SUBREG (IMPLICIT_DEF), $Vm, dsub))>;

  def : Pat<(nxv8bf16 (op nxv8bf16:$Zn, bf16:$Vm)),
            (!cast<Instruction>(NAME # _H) $Zn, (INSERT_SUBREG (IMPLICIT_DEF), $Vm, hsub))>;

  // Keep integer insertions within the vector unit.
  def : Pat<(nxv16i8 (op (nxv16i8 ZPR:$Zn), (i32 (vector_extract (nxv16i8 ZPR:$Vm), 0)))),
            (!cast<Instruction>(NAME # _B) $Zn, ZPR:$Vm)>;
  def : Pat<(nxv8i16 (op (nxv8i16 ZPR:$Zn), (i32 (vector_extract (nxv8i16 ZPR:$Vm), 0)))),
            (!cast<Instruction>(NAME # _H) $Zn, ZPR:$Vm)>;
  def : Pat<(nxv4i32 (op (nxv4i32 ZPR:$Zn), (i32 (vector_extract (nxv4i32 ZPR:$Vm), 0)))),
            (!cast<Instruction>(NAME # _S) $Zn, ZPR: $Vm)>;
  def : Pat<(nxv2i64 (op (nxv2i64 ZPR:$Zn), (i64 (vector_extract (nxv2i64 ZPR:$Vm), 0)))),
            (!cast<Instruction>(NAME # _D) $Zn, ZPR:$Vm)>;

}

//===----------------------------------------------------------------------===//
// SVE Permute - Extract Group
//===----------------------------------------------------------------------===//

class sve_int_perm_extract_i<string asm>
: I<(outs ZPR8:$Zdn), (ins ZPR8:$_Zdn, ZPR8:$Zm, imm0_255:$imm8),
  asm, "\t$Zdn, $_Zdn, $Zm, $imm8",
  "", []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> Zm;
  bits<8> imm8;
  let Inst{31-21} = 0b00000101001;
  let Inst{20-16} = imm8{7-3};
  let Inst{15-13} = 0b000;
  let Inst{12-10} = imm8{2-0};
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_perm_extract_i<string asm, SDPatternOperator op> {
  def NAME : sve_int_perm_extract_i<asm>;

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv16i8, i32, imm0_255,
                         !cast<Instruction>(NAME)>;
}

class sve2_int_perm_extract_i_cons<string asm>
: I<(outs ZPR8:$Zd), (ins ZZ_b:$Zn, imm0_255:$imm8),
  asm, "\t$Zd, $Zn, $imm8",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<8> imm8;
  let Inst{31-21} = 0b00000101011;
  let Inst{20-16} = imm8{7-3};
  let Inst{15-13} = 0b000;
  let Inst{12-10} = imm8{2-0};
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

//===----------------------------------------------------------------------===//
// SVE Vector Select Group
//===----------------------------------------------------------------------===//

class sve_int_sel_vvv<bits<2> sz8_64, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins PPRAny:$Pg, zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Pg, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<4> Pg;
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-14} = 0b11;
  let Inst{13-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_sel_vvv<string asm, SDPatternOperator op> {
  def _B : sve_int_sel_vvv<0b00, asm, ZPR8>;
  def _H : sve_int_sel_vvv<0b01, asm, ZPR16>;
  def _S : sve_int_sel_vvv<0b10, asm, ZPR32>;
  def _D : sve_int_sel_vvv<0b11, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1,  nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f16, op, nxv4i1,  nxv4f16, nxv4f16, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1,  nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f16, op, nxv2i1,  nxv2f16, nxv2f16, !cast<Instruction>(NAME # _D)>;
  def : SVE_3_Op_Pat<nxv2f32, op, nxv2i1,  nxv2f32, nxv2f32, !cast<Instruction>(NAME # _D)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1,  nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8bf16, op, nxv8i1,  nxv8bf16, nxv8bf16, !cast<Instruction>(NAME # _H)>;

  def : InstAlias<"mov $Zd, $Pg/m, $Zn",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, PPRAny:$Pg, ZPR8:$Zn, ZPR8:$Zd), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Zn",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, PPRAny:$Pg, ZPR16:$Zn, ZPR16:$Zd), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Zn",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, PPRAny:$Pg, ZPR32:$Zn, ZPR32:$Zd), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Zn",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, PPRAny:$Pg, ZPR64:$Zn, ZPR64:$Zd), 1>;
}


//===----------------------------------------------------------------------===//
// SVE Predicate Logical Operations Group
//===----------------------------------------------------------------------===//

class sve_int_pred_log<bits<4> opc, string asm>
: I<(outs PPR8:$Pd), (ins PPRAny:$Pg, PPR8:$Pn, PPR8:$Pm),
  asm, "\t$Pd, $Pg/z, $Pn, $Pm",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<4> Pg;
  bits<4> Pm;
  bits<4> Pn;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = opc{3-2};
  let Inst{21-20} = 0b00;
  let Inst{19-16} = Pm;
  let Inst{15-14} = 0b01;
  let Inst{13-10} = Pg;
  let Inst{9}     = opc{1};
  let Inst{8-5}   = Pn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;

  // SEL has no predication qualifier.
  let AsmString = !if(!eq(opc, 0b0011),
                      !strconcat(asm, "\t$Pd, $Pg, $Pn, $Pm"),
                      !strconcat(asm, "\t$Pd, $Pg/z, $Pn, $Pm"));

  let Defs = !if(!eq (opc{2}, 1), [NZCV], []);

}

multiclass sve_int_pred_log<bits<4> opc, string asm, SDPatternOperator op,
                            SDPatternOperator op_nopred = null_frag> {
  def NAME : sve_int_pred_log<opc, asm>;

  def : SVE_3_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, nxv16i1, !cast<Instruction>(NAME)>;
  def : SVE_3_Op_Pat<nxv8i1, op, nxv8i1, nxv8i1, nxv8i1, !cast<Instruction>(NAME)>;
  def : SVE_3_Op_Pat<nxv4i1, op, nxv4i1, nxv4i1, nxv4i1, !cast<Instruction>(NAME)>;
  def : SVE_3_Op_Pat<nxv2i1, op, nxv2i1, nxv2i1, nxv2i1, !cast<Instruction>(NAME)>;
  def : SVE_2_Op_AllActive_Pat<nxv16i1, op_nopred, nxv16i1, nxv16i1,
                               !cast<Instruction>(NAME), PTRUE_B>;
  def : SVE_2_Op_AllActive_Pat<nxv8i1, op_nopred, nxv8i1, nxv8i1,
                               !cast<Instruction>(NAME), PTRUE_H>;
  def : SVE_2_Op_AllActive_Pat<nxv4i1, op_nopred, nxv4i1, nxv4i1,
                               !cast<Instruction>(NAME), PTRUE_S>;
  def : SVE_2_Op_AllActive_Pat<nxv2i1, op_nopred, nxv2i1, nxv2i1,
                               !cast<Instruction>(NAME), PTRUE_D>;
}

//===----------------------------------------------------------------------===//
// SVE Logical Mask Immediate Group
//===----------------------------------------------------------------------===//

class sve_int_log_imm<bits<2> opc, string asm>
: I<(outs ZPR64:$Zdn), (ins ZPR64:$_Zdn, logical_imm64:$imms13),
  asm, "\t$Zdn, $_Zdn, $imms13",
  "", []>, Sched<[]> {
  bits<5> Zdn;
  bits<13> imms13;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = opc;
  let Inst{21-18} = 0b0000;
  let Inst{17-5}  = imms13;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DecoderMethod = "DecodeSVELogicalImmInstruction";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_log_imm<bits<2> opc, string asm, string alias, SDPatternOperator op> {
  def NAME : sve_int_log_imm<opc, asm>;

  def : SVE_1_Op_Imm_Log_Pat<nxv16i8, op, ZPR8,  i32, SVELogicalImm8Pat,  !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Imm_Log_Pat<nxv8i16, op, ZPR16, i32, SVELogicalImm16Pat, !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Imm_Log_Pat<nxv4i32, op, ZPR32, i32, SVELogicalImm32Pat, !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Imm_Log_Pat<nxv2i64, op, ZPR64, i64, SVELogicalImm64Pat, !cast<Instruction>(NAME)>;

  def : InstAlias<asm # "\t$Zdn, $Zdn, $imm",
                  (!cast<Instruction>(NAME) ZPR8:$Zdn, sve_logical_imm8:$imm), 4>;
  def : InstAlias<asm # "\t$Zdn, $Zdn, $imm",
                  (!cast<Instruction>(NAME) ZPR16:$Zdn, sve_logical_imm16:$imm), 3>;
  def : InstAlias<asm # "\t$Zdn, $Zdn, $imm",
                  (!cast<Instruction>(NAME) ZPR32:$Zdn, sve_logical_imm32:$imm), 2>;

  def : InstAlias<alias # "\t$Zdn, $Zdn, $imm",
                  (!cast<Instruction>(NAME) ZPR8:$Zdn, sve_logical_imm8_not:$imm), 0>;
  def : InstAlias<alias # "\t$Zdn, $Zdn, $imm",
                  (!cast<Instruction>(NAME) ZPR16:$Zdn, sve_logical_imm16_not:$imm), 0>;
  def : InstAlias<alias # "\t$Zdn, $Zdn, $imm",
                  (!cast<Instruction>(NAME) ZPR32:$Zdn, sve_logical_imm32_not:$imm), 0>;
  def : InstAlias<alias # "\t$Zdn, $Zdn, $imm",
                  (!cast<Instruction>(NAME) ZPR64:$Zdn, logical_imm64_not:$imm), 0>;
}

multiclass sve_int_log_imm_bic<SDPatternOperator op> {
  def : SVE_1_Op_Imm_Log_Pat<nxv16i8, op, ZPR8,  i32, SVELogicalImm8NotPat,  !cast<Instruction>("AND_ZI")>;
  def : SVE_1_Op_Imm_Log_Pat<nxv8i16, op, ZPR16, i32, SVELogicalImm16NotPat, !cast<Instruction>("AND_ZI")>;
  def : SVE_1_Op_Imm_Log_Pat<nxv4i32, op, ZPR32, i32, SVELogicalImm32NotPat, !cast<Instruction>("AND_ZI")>;
  def : SVE_1_Op_Imm_Log_Pat<nxv2i64, op, ZPR64, i64, SVELogicalImm64NotPat, !cast<Instruction>("AND_ZI")>;
}

class sve_int_dup_mask_imm<string asm>
: I<(outs ZPR64:$Zd), (ins logical_imm64:$imms),
  asm, "\t$Zd, $imms",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<13> imms;
  let Inst{31-18} = 0b00000101110000;
  let Inst{17-5} = imms;
  let Inst{4-0} = Zd;

  let isReMaterializable = 1;
  let DecoderMethod = "DecodeSVELogicalImmInstruction";
}

multiclass sve_int_dup_mask_imm<string asm> {
  def NAME : sve_int_dup_mask_imm<asm>;

  def : InstAlias<"dupm $Zd, $imm",
                  (!cast<Instruction>(NAME) ZPR8:$Zd, sve_logical_imm8:$imm), 4>;
  def : InstAlias<"dupm $Zd, $imm",
                  (!cast<Instruction>(NAME) ZPR16:$Zd, sve_logical_imm16:$imm), 3>;
  def : InstAlias<"dupm $Zd, $imm",
                  (!cast<Instruction>(NAME) ZPR32:$Zd, sve_logical_imm32:$imm), 2>;

  // All Zd.b forms have a CPY/DUP equivalent, hence no byte alias here.
  def : InstAlias<"mov $Zd, $imm",
                  (!cast<Instruction>(NAME) ZPR16:$Zd, sve_preferred_logical_imm16:$imm), 7>;
  def : InstAlias<"mov $Zd, $imm",
                  (!cast<Instruction>(NAME) ZPR32:$Zd, sve_preferred_logical_imm32:$imm), 6>;
  def : InstAlias<"mov $Zd, $imm",
                  (!cast<Instruction>(NAME) ZPR64:$Zd, sve_preferred_logical_imm64:$imm), 5>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Arithmetic -  Unpredicated Group.
//===----------------------------------------------------------------------===//

class sve_int_bin_cons_arit_0<bits<2> sz8_64, bits<3> opc, string asm,
                              ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b000;
  let Inst{12-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_bin_cons_arit_0<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_bin_cons_arit_0<0b00, opc, asm, ZPR8>;
  def _H : sve_int_bin_cons_arit_0<0b01, opc, asm, ZPR16>;
  def _S : sve_int_bin_cons_arit_0<0b10, opc, asm, ZPR32>;
  def _D : sve_int_bin_cons_arit_0<0b11, opc, asm, ZPR64>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Arithmetic - Predicated Group
//===----------------------------------------------------------------------===//

class sve_fp_2op_i_p_zds<bits<2> sz, bits<3> opc, string asm,
                         ZPRRegOp zprty,
                         Operand imm_ty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, imm_ty:$i1),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $i1",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bit i1;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21-19} = 0b011;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-6}   = 0b0000;
  let Inst{5}     = i1;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_fp_2op_i_p_zds<bits<3> opc, string asm, string Ps, Operand imm_ty, FPImmLeaf A, FPImmLeaf B, SDPatternOperator op> {
  let DestructiveInstType = DestructiveBinaryImm in {
  def _H : SVEPseudo2Instr<Ps # _H, 1>, sve_fp_2op_i_p_zds<0b01, opc, asm, ZPR16, imm_ty>;
  def _S : SVEPseudo2Instr<Ps # _S, 1>, sve_fp_2op_i_p_zds<0b10, opc, asm, ZPR32, imm_ty>;
  def _D : SVEPseudo2Instr<Ps # _D, 1>, sve_fp_2op_i_p_zds<0b11, opc, asm, ZPR64, imm_ty>;
  }

  def : SVE_2_Op_Fp_Imm_Pat<nxv8f16, op, nxv8i1, f16, A, 0, !cast<Instruction>(NAME # "_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv8f16, op, nxv8i1, f16, B, 1, !cast<Instruction>(NAME # "_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv4f32, op, nxv4i1, f32, A, 0, !cast<Instruction>(NAME # "_S")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv4f32, op, nxv4i1, f32, B, 1, !cast<Instruction>(NAME # "_S")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f64, op, nxv2i1, f64, A, 0, !cast<Instruction>(NAME # "_D")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f64, op, nxv2i1, f64, B, 1, !cast<Instruction>(NAME # "_D")>;
}

class sve_fp_2op_p_zds<bits<2> sz, bits<4> opc, string asm,
                       ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b00;
  let Inst{19-16} = opc;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_fp_2op_p_zds<bits<4> opc, string asm, string Ps,
                            SDPatternOperator op, DestructiveInstTypeEnum flags,
                            string revname="", bit isReverseInstr=0> {
  let DestructiveInstType = flags in {
  def _H : sve_fp_2op_p_zds<0b01, opc, asm, ZPR16>,
           SVEPseudo2Instr<Ps # _H, 1>, SVEInstr2Rev<NAME # _H, revname # _H, isReverseInstr>;
  def _S : sve_fp_2op_p_zds<0b10, opc, asm, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>, SVEInstr2Rev<NAME # _S, revname # _S, isReverseInstr>;
  def _D : sve_fp_2op_p_zds<0b11, opc, asm, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>, SVEInstr2Rev<NAME # _D, revname # _D, isReverseInstr>;
  }

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_fp_2op_p_zds_fscale<bits<4> opc, string asm,
                                   SDPatternOperator op> {
  def _H : sve_fp_2op_p_zds<0b01, opc, asm, ZPR16>;
  def _S : sve_fp_2op_p_zds<0b10, opc, asm, ZPR32>;
  def _D : sve_fp_2op_p_zds<0b11, opc, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_fp_2op_p_zds_zeroing_hsd<SDPatternOperator op> {
  def _ZERO_H : PredTwoOpPseudo<NAME # _H, ZPR16, FalseLanesZero>;
  def _ZERO_S : PredTwoOpPseudo<NAME # _S, ZPR32, FalseLanesZero>;
  def _ZERO_D : PredTwoOpPseudo<NAME # _D, ZPR64, FalseLanesZero>;

  def : SVE_3_Op_Pat_SelZero<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, !cast<Pseudo>(NAME # _ZERO_H)>;
  def : SVE_3_Op_Pat_SelZero<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, !cast<Pseudo>(NAME # _ZERO_S)>;
  def : SVE_3_Op_Pat_SelZero<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, !cast<Pseudo>(NAME # _ZERO_D)>;
}

class sve_fp_ftmad<bits<2> sz, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, zprty:$Zm, timm32_0_7:$imm3),
  asm, "\t$Zdn, $_Zdn, $Zm, $imm3",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> Zm;
  bits<3> imm3;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21-19} = 0b010;
  let Inst{18-16} = imm3;
  let Inst{15-10} = 0b100000;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_fp_ftmad<string asm, SDPatternOperator op> {
  def _H : sve_fp_ftmad<0b01, asm, ZPR16>;
  def _S : sve_fp_ftmad<0b10, asm, ZPR32>;
  def _D : sve_fp_ftmad<0b11, asm, ZPR64>;

  def : Pat<(nxv8f16 (op (nxv8f16 ZPR16:$Zn), (nxv8f16 ZPR16:$Zm), (i32 timm32_0_7:$imm))),
            (!cast<Instruction>(NAME # _H) ZPR16:$Zn, ZPR16:$Zm, timm32_0_7:$imm)>;
  def : Pat<(nxv4f32 (op (nxv4f32 ZPR32:$Zn), (nxv4f32 ZPR32:$Zm), (i32 timm32_0_7:$imm))),
            (!cast<Instruction>(NAME # _S) ZPR32:$Zn, ZPR32:$Zm, timm32_0_7:$imm)>;
  def : Pat<(nxv2f64 (op (nxv2f64 ZPR64:$Zn), (nxv2f64 ZPR64:$Zm), (i32 timm32_0_7:$imm))),
            (!cast<Instruction>(NAME # _D) ZPR64:$Zn, ZPR64:$Zm, timm32_0_7:$imm)>;
}

multiclass sve_fp_2op_i_p_zds_hfd<Operand imm_ty, FPImmLeaf A, FPImmLeaf B, SDPatternOperator ir_op = null_frag> {
  def _UNDEF_H : PredTwoOpImmPseudo<NAME # _H, ZPR16, imm_ty, FalseLanesUndef>;
  def _UNDEF_S : PredTwoOpImmPseudo<NAME # _S, ZPR32, imm_ty, FalseLanesUndef>;
  def _UNDEF_D : PredTwoOpImmPseudo<NAME # _D, ZPR64, imm_ty, FalseLanesUndef>;

  def : SVE_2_Op_Fp_Imm_Pat<nxv8f16, ir_op, nxv8i1, f16, A, 0, !cast<Instruction>(NAME # "_UNDEF_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv8f16, ir_op, nxv8i1, f16, B, 1, !cast<Instruction>(NAME # "_UNDEF_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv4f16, ir_op, nxv4i1, f16, A, 0, !cast<Instruction>(NAME # "_UNDEF_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv4f16, ir_op, nxv4i1, f16, B, 1, !cast<Instruction>(NAME # "_UNDEF_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f16, ir_op, nxv2i1, f16, A, 0, !cast<Instruction>(NAME # "_UNDEF_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f16, ir_op, nxv2i1, f16, B, 1, !cast<Instruction>(NAME # "_UNDEF_H")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv4f32, ir_op, nxv4i1, f32, A, 0, !cast<Instruction>(NAME # "_UNDEF_S")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv4f32, ir_op, nxv4i1, f32, B, 1, !cast<Instruction>(NAME # "_UNDEF_S")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f32, ir_op, nxv2i1, f32, A, 0, !cast<Instruction>(NAME # "_UNDEF_S")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f32, ir_op, nxv2i1, f32, B, 1, !cast<Instruction>(NAME # "_UNDEF_S")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f64, ir_op, nxv2i1, f64, A, 0, !cast<Instruction>(NAME # "_UNDEF_D")>;
  def : SVE_2_Op_Fp_Imm_Pat<nxv2f64, ir_op, nxv2i1, f64, B, 1, !cast<Instruction>(NAME # "_UNDEF_D")>;
}

multiclass sve_fp_2op_i_p_zds_zeroing_hfd<Operand imm_ty, FPImmLeaf A, FPImmLeaf B, SDPatternOperator op> {
  def _ZERO_H : PredTwoOpImmPseudo<NAME # _H, ZPR16, imm_ty, FalseLanesZero>;
  def _ZERO_S : PredTwoOpImmPseudo<NAME # _S, ZPR32, imm_ty, FalseLanesZero>;
  def _ZERO_D : PredTwoOpImmPseudo<NAME # _D, ZPR64, imm_ty, FalseLanesZero>;

  let AddedComplexity = 2 in {
    def : SVE_2_Op_Fp_Imm_Pat_Zero<nxv8f16, op, nxv8i1, f16, A, 0, !cast<Instruction>(NAME # "_ZERO_H")>;
    def : SVE_2_Op_Fp_Imm_Pat_Zero<nxv8f16, op, nxv8i1, f16, B, 1, !cast<Instruction>(NAME # "_ZERO_H")>;
    def : SVE_2_Op_Fp_Imm_Pat_Zero<nxv4f32, op, nxv4i1, f32, A, 0, !cast<Instruction>(NAME # "_ZERO_S")>;
    def : SVE_2_Op_Fp_Imm_Pat_Zero<nxv4f32, op, nxv4i1, f32, B, 1, !cast<Instruction>(NAME # "_ZERO_S")>;
    def : SVE_2_Op_Fp_Imm_Pat_Zero<nxv2f64, op, nxv2i1, f64, A, 0, !cast<Instruction>(NAME # "_ZERO_D")>;
    def : SVE_2_Op_Fp_Imm_Pat_Zero<nxv2f64, op, nxv2i1, f64, B, 1, !cast<Instruction>(NAME # "_ZERO_D")>;
  }
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Arithmetic - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve_fp_3op_u_zd<bits<2> sz, bits<3> opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins  zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b000;
  let Inst{12-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_fp_3op_u_zd<bits<3> opc, string asm, SDPatternOperator op,
                           SDPatternOperator predicated_op = null_frag> {
  def _H : sve_fp_3op_u_zd<0b01, opc, asm, ZPR16>;
  def _S : sve_fp_3op_u_zd<0b10, opc, asm, ZPR32>;
  def _D : sve_fp_3op_u_zd<0b11, opc, asm, ZPR64>;

  def : SVE_2_Op_Pat<nxv8f16, op, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4f32, op, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2f64, op, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_2_Op_Pred_All_Active<nxv8f16, predicated_op, nxv8i1, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pred_All_Active<nxv4f32, predicated_op, nxv4i1, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pred_All_Active<nxv2f64, predicated_op, nxv2i1, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_fp_3op_u_zd_ftsmul<bits<3> opc, string asm, SDPatternOperator op> {
  def _H : sve_fp_3op_u_zd<0b01, opc, asm, ZPR16>;
  def _S : sve_fp_3op_u_zd<0b10, opc, asm, ZPR32>;
  def _D : sve_fp_3op_u_zd<0b11, opc, asm, ZPR64>;

  def : SVE_2_Op_Pat<nxv8f16, op, nxv8f16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4f32, op, nxv4f32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2f64, op, nxv2f64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Fused Multiply-Add Group
//===----------------------------------------------------------------------===//

class sve_fp_3op_p_zds_a<bits<2> sz, bits<2> opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zda), (ins PPR3bAny:$Pg, zprty:$_Zda, zprty:$Zn, zprty:$Zm),
  asm, "\t$Zda, $Pg/m, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zda;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = opc;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let ElementSize = zprty.ElementSize;
}

multiclass sve_fp_3op_p_zds_a<bits<2> opc, string asm, string Ps,
                              SDPatternOperator op, string revname,
                              bit isReverseInstr=0> {
  let DestructiveInstType = DestructiveTernaryCommWithRev in {
  def _H : sve_fp_3op_p_zds_a<0b01, opc, asm, ZPR16>,
           SVEPseudo2Instr<Ps # _H, 1>, SVEInstr2Rev<NAME # _H, revname # _H, isReverseInstr>;
  def _S : sve_fp_3op_p_zds_a<0b10, opc, asm, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>, SVEInstr2Rev<NAME # _S, revname # _S, isReverseInstr>;
  def _D : sve_fp_3op_p_zds_a<0b11, opc, asm, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>, SVEInstr2Rev<NAME # _D, revname # _D, isReverseInstr>;
  }

  def : SVE_4_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_4_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

class sve_fp_3op_p_zds_b<bits<2> sz, bits<2> opc, string asm,
                         ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm, zprty:$Za),
  asm, "\t$Zdn, $Pg/m, $Zm, $Za",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Za;
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Za;
  let Inst{15}    = 0b1;
  let Inst{14-13} = opc;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_fp_3op_p_zds_b<bits<2> opc, string asm, SDPatternOperator op,
                              string revname, bit isReverseInstr> {
  def _H : sve_fp_3op_p_zds_b<0b01, opc, asm, ZPR16>,
           SVEInstr2Rev<NAME # _H, revname # _H, isReverseInstr>;
  def _S : sve_fp_3op_p_zds_b<0b10, opc, asm, ZPR32>,
           SVEInstr2Rev<NAME # _S, revname # _S, isReverseInstr>;
  def _D : sve_fp_3op_p_zds_b<0b11, opc, asm, ZPR64>,
           SVEInstr2Rev<NAME # _D, revname # _D, isReverseInstr>;

  def : SVE_4_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_4_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_fp_3op_p_zds_zx {
  def _UNDEF_H : PredThreeOpPseudo<NAME # _H, ZPR16, FalseLanesUndef>;
  def _UNDEF_S : PredThreeOpPseudo<NAME # _S, ZPR32, FalseLanesUndef>;
  def _UNDEF_D : PredThreeOpPseudo<NAME # _D, ZPR64, FalseLanesUndef>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Multiply-Add - Indexed Group
//===----------------------------------------------------------------------===//

class sve_fp_fma_by_indexed_elem<bits<2> sz, bit opc, string asm,
                                 ZPRRegOp zprty1,
                                 ZPRRegOp zprty2, Operand itype>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty1:$Zn, zprty2:$Zm, itype:$iop),
  asm, "\t$Zda, $Zn, $Zm$iop", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  let Inst{31-24} = 0b01100100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{15-11} = 0;
  let Inst{10}    = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_fp_fma_by_indexed_elem<bit opc, string asm,
                                      SDPatternOperator op> {
  def _H : sve_fp_fma_by_indexed_elem<{0, ?}, opc, asm, ZPR16, ZPR3b16, VectorIndexH32b> {
    bits<3> Zm;
    bits<3> iop;
    let Inst{22} = iop{2};
    let Inst{20-19} = iop{1-0};
    let Inst{18-16} = Zm;
  }
  def _S : sve_fp_fma_by_indexed_elem<0b10, opc, asm, ZPR32, ZPR3b32, VectorIndexS32b> {
    bits<3> Zm;
    bits<2> iop;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _D : sve_fp_fma_by_indexed_elem<0b11, opc, asm, ZPR64, ZPR4b64, VectorIndexD32b> {
    bits<4> Zm;
    bit iop;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : Pat<(nxv8f16 (op nxv8f16:$Op1, nxv8f16:$Op2, nxv8f16:$Op3, (i32 VectorIndexH32b_timm:$idx))),
            (!cast<Instruction>(NAME # _H) $Op1, $Op2, $Op3, VectorIndexH32b_timm:$idx)>;
  def : Pat<(nxv4f32 (op nxv4f32:$Op1, nxv4f32:$Op2, nxv4f32:$Op3, (i32 VectorIndexS32b_timm:$idx))),
            (!cast<Instruction>(NAME # _S) $Op1, $Op2, $Op3, VectorIndexS32b_timm:$idx)>;
  def : Pat<(nxv2f64 (op nxv2f64:$Op1, nxv2f64:$Op2, nxv2f64:$Op3, (i32 VectorIndexD32b_timm:$idx))),
            (!cast<Instruction>(NAME # _D) $Op1, $Op2, $Op3, VectorIndexD32b_timm:$idx)>;
}


//===----------------------------------------------------------------------===//
// SVE Floating Point Multiply - Indexed Group
//===----------------------------------------------------------------------===//

class sve_fp_fmul_by_indexed_elem<bits<2> sz, string asm, ZPRRegOp zprty,
                                      ZPRRegOp zprty2, Operand itype>
: I<(outs zprty:$Zd), (ins zprty:$Zn, zprty2:$Zm, itype:$iop),
  asm, "\t$Zd, $Zn, $Zm$iop", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b01100100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{15-10} = 0b001000;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_fp_fmul_by_indexed_elem<string asm, SDPatternOperator op> {
  def _H : sve_fp_fmul_by_indexed_elem<{0, ?}, asm, ZPR16, ZPR3b16, VectorIndexH32b> {
    bits<3> Zm;
    bits<3> iop;
    let Inst{22} = iop{2};
    let Inst{20-19} = iop{1-0};
    let Inst{18-16} = Zm;
  }
  def _S : sve_fp_fmul_by_indexed_elem<0b10, asm, ZPR32, ZPR3b32, VectorIndexS32b> {
    bits<3> Zm;
    bits<2> iop;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _D : sve_fp_fmul_by_indexed_elem<0b11, asm, ZPR64, ZPR4b64, VectorIndexD32b> {
    bits<4> Zm;
    bit iop;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : Pat<(nxv8f16 (op nxv8f16:$Op1, nxv8f16:$Op2, (i32 VectorIndexH32b_timm:$idx))),
            (!cast<Instruction>(NAME # _H) $Op1, $Op2, VectorIndexH32b_timm:$idx)>;
  def : Pat<(nxv4f32 (op nxv4f32:$Op1, nxv4f32:$Op2, (i32 VectorIndexS32b_timm:$idx))),
            (!cast<Instruction>(NAME # _S) $Op1, $Op2, VectorIndexS32b_timm:$idx)>;
  def : Pat<(nxv2f64 (op nxv2f64:$Op1, nxv2f64:$Op2, (i32 VectorIndexD32b_timm:$idx))),
            (!cast<Instruction>(NAME # _D) $Op1, $Op2, VectorIndexD32b_timm:$idx)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Complex Multiply-Add Group
//===----------------------------------------------------------------------===//

class sve_fp_fcmla<bits<2> sz, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zda), (ins PPR3bAny:$Pg, zprty:$_Zda, zprty:$Zn, zprty:$Zm,
                        complexrotateop:$imm),
  asm, "\t$Zda, $Pg/m, $Zn, $Zm, $imm",
  "", []>, Sched<[]> {
  bits<5> Zda;
  bits<3> Pg;
  bits<5> Zn;
  bits<5> Zm;
  bits<2> imm;
  let Inst{31-24} = 0b01100100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0;
  let Inst{20-16} = Zm;
  let Inst{15}    = 0;
  let Inst{14-13} = imm;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_fp_fcmla<string asm, SDPatternOperator op> {
  def _H : sve_fp_fcmla<0b01, asm, ZPR16>;
  def _S : sve_fp_fcmla<0b10, asm, ZPR32>;
  def _D : sve_fp_fcmla<0b11, asm, ZPR64>;

  def : Pat<(nxv8f16 (op nxv8i1:$Op1, nxv8f16:$Op2, nxv8f16:$Op3, nxv8f16:$Op4, (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # _H) $Op1, $Op2, $Op3, $Op4, complexrotateop:$imm)>;
  def : Pat<(nxv4f32 (op nxv4i1:$Op1, nxv4f32:$Op2, nxv4f32:$Op3, nxv4f32:$Op4, (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # _S) $Op1, $Op2, $Op3, $Op4, complexrotateop:$imm)>;
  def : Pat<(nxv2f64 (op nxv2i1:$Op1, nxv2f64:$Op2, nxv2f64:$Op3, nxv2f64:$Op4, (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # _D) $Op1, $Op2, $Op3, $Op4, complexrotateop:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Complex Multiply-Add - Indexed Group
//===----------------------------------------------------------------------===//

class sve_fp_fcmla_by_indexed_elem<bits<2> sz, string asm,
                                   ZPRRegOp zprty,
                                   ZPRRegOp zprty2, Operand itype>
: I<(outs zprty:$Zda), (ins zprty:$_Zda, zprty:$Zn, zprty2:$Zm, itype:$iop,
                        complexrotateop:$imm),
  asm, "\t$Zda, $Zn, $Zm$iop, $imm",
  "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<2> imm;
  let Inst{31-24} = 0b01100100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{15-12} = 0b0001;
  let Inst{11-10} = imm;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_fp_fcmla_by_indexed_elem<string asm, SDPatternOperator op> {
  def _H : sve_fp_fcmla_by_indexed_elem<0b10, asm, ZPR16, ZPR3b16, VectorIndexS32b> {
    bits<3> Zm;
    bits<2> iop;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _S : sve_fp_fcmla_by_indexed_elem<0b11, asm, ZPR32, ZPR4b32, VectorIndexD32b> {
    bits<4> Zm;
    bits<1> iop;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : Pat<(nxv8f16 (op nxv8f16:$Op1, nxv8f16:$Op2, nxv8f16:$Op3, (i32 VectorIndexS32b_timm:$idx), (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # _H) $Op1, $Op2, $Op3, VectorIndexS32b_timm:$idx, complexrotateop:$imm)>;
  def : Pat<(nxv4f32 (op nxv4f32:$Op1, nxv4f32:$Op2, nxv4f32:$Op3, (i32 VectorIndexD32b_timm:$idx), (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # _S) $Op1, $Op2, $Op3, VectorIndexD32b_timm:$idx, complexrotateop:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Complex Addition Group
//===----------------------------------------------------------------------===//

class sve_fp_fcadd<bits<2> sz, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm,
                        complexrotateopodd:$imm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $Zm, $imm",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> Zm;
  bits<3> Pg;
  bit imm;
  let Inst{31-24} = 0b01100100;
  let Inst{23-22} = sz;
  let Inst{21-17} = 0;
  let Inst{16}    = imm;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_fp_fcadd<string asm, SDPatternOperator op> {
  def _H : sve_fp_fcadd<0b01, asm, ZPR16>;
  def _S : sve_fp_fcadd<0b10, asm, ZPR32>;
  def _D : sve_fp_fcadd<0b11, asm, ZPR64>;

  def : Pat<(nxv8f16 (op nxv8i1:$Op1, nxv8f16:$Op2, nxv8f16:$Op3, (i32 complexrotateopodd:$imm))),
            (!cast<Instruction>(NAME # _H) $Op1, $Op2, $Op3, complexrotateopodd:$imm)>;
  def : Pat<(nxv4f32 (op nxv4i1:$Op1, nxv4f32:$Op2, nxv4f32:$Op3, (i32 complexrotateopodd:$imm))),
            (!cast<Instruction>(NAME # _S) $Op1, $Op2, $Op3, complexrotateopodd:$imm)>;
  def : Pat<(nxv2f64 (op nxv2i1:$Op1, nxv2f64:$Op2, nxv2f64:$Op3, (i32 complexrotateopodd:$imm))),
            (!cast<Instruction>(NAME # _D) $Op1, $Op2, $Op3, complexrotateopodd:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Floating Point Convert Group
//===----------------------------------------------------------------------===//

class sve2_fp_convert_precision<bits<4> opc, string asm,
                                ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty1:$_Zd, PPR3bAny:$Pg, zprty2:$Zn),
  asm, "\t$Zd, $Pg/m, $Zn",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<3> Pg;
  let Inst{31-24} = 0b01100100;
  let Inst{23-22} = opc{3-2};
  let Inst{21-18} = 0b0010;
  let Inst{17-16} = opc{1-0};
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
}

multiclass sve2_fp_convert_down_narrow<string asm, string op> {
  def _StoH : sve2_fp_convert_precision<0b1000, asm, ZPR16, ZPR32>;
  def _DtoS : sve2_fp_convert_precision<0b1110, asm, ZPR32, ZPR64>;

  def : SVE_3_Op_Pat<nxv8f16, !cast<SDPatternOperator>(op # _f16f32), nxv8f16, nxv4i1, nxv4f32, !cast<Instruction>(NAME # _StoH)>;
  def : SVE_3_Op_Pat<nxv4f32, !cast<SDPatternOperator>(op # _f32f64), nxv4f32, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _DtoS)>;
}

multiclass sve2_fp_convert_up_long<string asm, string op> {
  def _HtoS : sve2_fp_convert_precision<0b1001, asm, ZPR32, ZPR16>;
  def _StoD : sve2_fp_convert_precision<0b1111, asm, ZPR64, ZPR32>;

  def : SVE_3_Op_Pat<nxv4f32, !cast<SDPatternOperator>(op # _f32f16), nxv4f32, nxv4i1, nxv8f16, !cast<Instruction>(NAME # _HtoS)>;
  def : SVE_3_Op_Pat<nxv2f64, !cast<SDPatternOperator>(op # _f64f32), nxv2f64, nxv2i1, nxv4f32, !cast<Instruction>(NAME # _StoD)>;
}

multiclass sve2_fp_convert_down_odd_rounding_top<string asm, string op> {
  def _DtoS : sve2_fp_convert_precision<0b0010, asm, ZPR32, ZPR64>;

  def : SVE_3_Op_Pat<nxv4f32, !cast<SDPatternOperator>(op # _f32f64), nxv4f32, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _DtoS)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Floating Point Pairwise Group
//===----------------------------------------------------------------------===//

class sve2_fp_pairwise_pred<bits<2> sz, bits<3> opc, string asm,
                            ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zm;
  bits<5> Zdn;
  let Inst{31-24} = 0b01100100;
  let Inst{23-22} = sz;
  let Inst{21-19} = 0b010;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve2_fp_pairwise_pred<bits<3> opc, string asm,
                                 SDPatternOperator op> {
  def _H : sve2_fp_pairwise_pred<0b01, opc, asm, ZPR16>;
  def _S : sve2_fp_pairwise_pred<0b10, opc, asm, ZPR32>;
  def _D : sve2_fp_pairwise_pred<0b11, opc, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Floating Point Widening Multiply-Add - Indexed Group
//===----------------------------------------------------------------------===//

class sve2_fp_mla_long_by_indexed_elem<bits<2> opc, string asm>
: I<(outs ZPR32:$Zda), (ins ZPR32:$_Zda, ZPR16:$Zn, ZPR3b16:$Zm,
                        VectorIndexH32b:$iop),
  asm, "\t$Zda, $Zn, $Zm$iop",
  "",
  []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<3> Zm;
  bits<3> iop;
  let Inst{31-21} = 0b01100100101;
  let Inst{20-19} = iop{2-1};
  let Inst{18-16} = Zm;
  let Inst{15-14} = 0b01;
  let Inst{13}    = opc{1};
  let Inst{12}    = 0b0;
  let Inst{11}    = iop{0};
  let Inst{10}    = opc{0};
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_fp_mla_long_by_indexed_elem<bits<2> opc, string asm,
                                            SDPatternOperator op> {
  def NAME : sve2_fp_mla_long_by_indexed_elem<opc, asm>;
  def : SVE_4_Op_Imm_Pat<nxv4f32, op, nxv4f32, nxv8f16, nxv8f16, i32, VectorIndexH32b_timm, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Floating Point Widening Multiply-Add Group
//===----------------------------------------------------------------------===//

class sve2_fp_mla_long<bits<2> opc, string asm>
: I<(outs ZPR32:$Zda), (ins ZPR32:$_Zda, ZPR16:$Zn, ZPR16:$Zm),
  asm, "\t$Zda, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-21} = 0b01100100101;
  let Inst{20-16} = Zm;
  let Inst{15-14} = 0b10;
  let Inst{13}    = opc{1};
  let Inst{12-11} = 0b00;
  let Inst{10}    = opc{0};
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_fp_mla_long<bits<2> opc, string asm, SDPatternOperator op> {
  def NAME : sve2_fp_mla_long<opc, asm>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4f32, nxv8f16, nxv8f16, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE Stack Allocation Group
//===----------------------------------------------------------------------===//

class sve_int_arith_vl<bit opc, string asm>
: I<(outs GPR64sp:$Rd), (ins GPR64sp:$Rn, simm6_32b:$imm6),
  asm, "\t$Rd, $Rn, $imm6",
  "",
  []>, Sched<[]> {
  bits<5> Rd;
  bits<5> Rn;
  bits<6> imm6;
  let Inst{31-23} = 0b000001000;
  let Inst{22}    = opc;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Rn;
  let Inst{15-11} = 0b01010;
  let Inst{10-5}  = imm6;
  let Inst{4-0}   = Rd;
}

class sve_int_read_vl_a<bit op, bits<5> opc2, string asm>
: I<(outs GPR64:$Rd), (ins simm6_32b:$imm6),
  asm, "\t$Rd, $imm6",
  "",
  []>, Sched<[]> {
  bits<5> Rd;
  bits<6> imm6;
  let Inst{31-23} = 0b000001001;
  let Inst{22}    = op;
  let Inst{21}    = 0b1;
  let Inst{20-16} = opc2{4-0};
  let Inst{15-11} = 0b01010;
  let Inst{10-5}  = imm6;
  let Inst{4-0}   = Rd;
}

//===----------------------------------------------------------------------===//
// SVE Permute - In Lane Group
//===----------------------------------------------------------------------===//

class sve_int_perm_bin_perm_zz<bits<3> opc, bits<2> sz8_64, string asm,
                               ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b011;
  let Inst{12-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_bin_perm_zz<bits<3> opc, string asm,
                                    SDPatternOperator op> {
  def _B : sve_int_perm_bin_perm_zz<opc, 0b00, asm, ZPR8>;
  def _H : sve_int_perm_bin_perm_zz<opc, 0b01, asm, ZPR16>;
  def _S : sve_int_perm_bin_perm_zz<opc, 0b10, asm, ZPR32>;
  def _D : sve_int_perm_bin_perm_zz<opc, 0b11, asm, ZPR64>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_2_Op_Pat<nxv8f16, op, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4f16, op, nxv4f16, nxv4f16, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv4f32, op, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2f16, op, nxv2f16, nxv2f16, !cast<Instruction>(NAME # _D)>;
  def : SVE_2_Op_Pat<nxv2f32, op, nxv2f32, nxv2f32, !cast<Instruction>(NAME # _D)>;
  def : SVE_2_Op_Pat<nxv2f64, op, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_2_Op_Pat<nxv8bf16, op, nxv8bf16, nxv8bf16, !cast<Instruction>(NAME # _H)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Unary Operations Group
//===----------------------------------------------------------------------===//

class sve_fp_2op_p_zd<bits<7> opc, string asm, RegisterOperand i_zprtype,
                      RegisterOperand o_zprtype, ElementSizeEnum Sz>
: I<(outs o_zprtype:$Zd), (ins i_zprtype:$_Zd, PPR3bAny:$Pg, i_zprtype:$Zn),
  asm, "\t$Zd, $Pg/m, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = opc{6-5};
  let Inst{21}    = 0b0;
  let Inst{20-16} = opc{4-0};
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveUnaryPassthru;
  let ElementSize = Sz;
}

multiclass sve_fp_2op_p_zd<bits<7> opc, string asm,
                           RegisterOperand i_zprtype,
                           RegisterOperand o_zprtype,
                           SDPatternOperator int_op,
                           SDPatternOperator ir_op, ValueType vt1,
                           ValueType vt2, ValueType vt3, ElementSizeEnum Sz> {
  def NAME : sve_fp_2op_p_zd<opc, asm, i_zprtype, o_zprtype, Sz>;

  // convert vt1 to a packed type for the intrinsic patterns
  defvar packedvt1 = !cond(!eq(!cast<string>(vt1), "nxv2f16"): nxv8f16,
                           !eq(!cast<string>(vt1), "nxv4f16"): nxv8f16,
                           !eq(!cast<string>(vt1), "nxv2f32"): nxv4f32,
                           1 : vt1);

  // convert vt3 to a packed type for the intrinsic patterns
  defvar packedvt3 = !cond(!eq(!cast<string>(vt3), "nxv2f16"): nxv8f16,
                           !eq(!cast<string>(vt3), "nxv4f16"): nxv8f16,
                           !eq(!cast<string>(vt3), "nxv2f32"): nxv4f32,
                           1 : vt3);

  def : SVE_3_Op_Pat<packedvt1, int_op, packedvt1, vt2, packedvt3, !cast<Instruction>(NAME)>;

  def : SVE_1_Op_Passthru_Pat<vt1, ir_op, vt2, vt3, !cast<Instruction>(NAME)>;
}

multiclass sve_fp_2op_p_zdr<bits<7> opc, string asm,
                            RegisterOperand i_zprtype,
                            RegisterOperand o_zprtype,
                            SDPatternOperator int_op,
                            SDPatternOperator ir_op, ValueType vt1,
                            ValueType vt2, ValueType vt3, ElementSizeEnum Sz> {
  def NAME : sve_fp_2op_p_zd<opc, asm, i_zprtype, o_zprtype, Sz>;

  // convert vt1 to a packed type for the intrinsic patterns
  defvar packedvt1 = !cond(!eq(!cast<string>(vt1), "nxv2f16"): nxv8f16,
                           !eq(!cast<string>(vt1), "nxv4f16"): nxv8f16,
                           !eq(!cast<string>(vt1), "nxv2f32"): nxv4f32,
                           1 : vt1);

  def : SVE_3_Op_Pat<packedvt1, int_op, packedvt1, vt2, vt3, !cast<Instruction>(NAME)>;

  def : SVE_1_Op_Passthru_Round_Pat<vt1, ir_op, vt2, vt3, !cast<Instruction>(NAME)>;
}

multiclass sve_fp_2op_p_zd_HSD<bits<5> opc, string asm, SDPatternOperator op> {
  def _H : sve_fp_2op_p_zd<{ 0b01, opc }, asm, ZPR16, ZPR16, ElementSizeH>,
           SVEPseudo2Instr<NAME # _H, 1>;
  def _S : sve_fp_2op_p_zd<{ 0b10, opc }, asm, ZPR32, ZPR32, ElementSizeS>,
           SVEPseudo2Instr<NAME # _S, 1>;
  def _D : sve_fp_2op_p_zd<{ 0b11, opc }, asm, ZPR64, ZPR64, ElementSizeD>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_1_Op_Passthru_Pat<nxv8f16, op, nxv8i1, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4f16, op, nxv4i1, nxv4f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv2f16, op, nxv2i1, nxv2f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4f32, op, nxv4i1, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2f32, op, nxv2i1, nxv2f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2f64, op, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_H : PredOneOpPassthruPseudo<NAME # _H, ZPR16>;
  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;
  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_1_Op_PassthruUndef_Pat<nxv8f16, op, nxv8i1, nxv8f16, !cast<Instruction>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv4f16, op, nxv4i1, nxv4f16, !cast<Instruction>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2f16, op, nxv2i1, nxv2f16, !cast<Instruction>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv4f32, op, nxv4i1, nxv4f32, !cast<Instruction>(NAME # _UNDEF_S)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2f32, op, nxv2i1, nxv2f32, !cast<Instruction>(NAME # _UNDEF_S)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2f64, op, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _UNDEF_D)>;
}

multiclass sve2_fp_flogb<string asm, SDPatternOperator op> {
  def _H : sve_fp_2op_p_zd<0b0011010, asm, ZPR16, ZPR16, ElementSizeH>;
  def _S : sve_fp_2op_p_zd<0b0011100, asm, ZPR32, ZPR32, ElementSizeS>;
  def _D : sve_fp_2op_p_zd<0b0011110, asm, ZPR64, ZPR64, ElementSizeD>;

  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i1, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i1, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_fp_convert_down_odd_rounding<string asm, string op> {
  def _DtoS : sve_fp_2op_p_zd<0b0001010, asm, ZPR64, ZPR32, ElementSizeD>;
  def : SVE_3_Op_Pat<nxv4f32, !cast<SDPatternOperator>(op # _f32f64), nxv4f32, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _DtoS)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Unary Operations - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve_fp_2op_u_zd<bits<2> sz, bits<3> opc, string asm,
                      ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn),
  asm, "\t$Zd, $Zn",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21-19} = 0b001;
  let Inst{18-16} = opc;
  let Inst{15-10} = 0b001100;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_fp_2op_u_zd<bits<3> opc, string asm, SDPatternOperator op> {
  def _H : sve_fp_2op_u_zd<0b01, opc, asm, ZPR16>;
  def _S : sve_fp_2op_u_zd<0b10, opc, asm, ZPR32>;
  def _D : sve_fp_2op_u_zd<0b11, opc, asm, ZPR64>;

  def : SVE_1_Op_Pat<nxv8f16, op, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Pat<nxv4f32, op, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv2f64, op, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Arithmetic - Binary Predicated Group
//===----------------------------------------------------------------------===//

class sve_int_bin_pred_arit_log<bits<2> sz8_64, bits<2> fmt, bits<3> opc,
                                string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $Zm", "", []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b0;
  let Inst{20-19} = fmt;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b000;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_bin_pred_log<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_bin_pred_arit_log<0b00, 0b11, opc, asm, ZPR8>;
  def _H : sve_int_bin_pred_arit_log<0b01, 0b11, opc, asm, ZPR16>;
  def _S : sve_int_bin_pred_arit_log<0b10, 0b11, opc, asm, ZPR32>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b11, opc, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_bin_pred_arit_0<bits<3> opc, string asm, string Ps,
                                   SDPatternOperator op,
                                   DestructiveInstTypeEnum flags,
                                   string revname="", bit isReverseInstr=0> {
  let DestructiveInstType = flags in {
  def _B : sve_int_bin_pred_arit_log<0b00, 0b00, opc, asm, ZPR8>,
           SVEPseudo2Instr<Ps # _B, 1>, SVEInstr2Rev<NAME # _B, revname # _B, isReverseInstr>;
  def _H : sve_int_bin_pred_arit_log<0b01, 0b00, opc, asm, ZPR16>,
           SVEPseudo2Instr<Ps # _H, 1>, SVEInstr2Rev<NAME # _H, revname # _H, isReverseInstr>;
  def _S : sve_int_bin_pred_arit_log<0b10, 0b00, opc, asm, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>, SVEInstr2Rev<NAME # _S, revname # _S, isReverseInstr>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b00, opc, asm, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>, SVEInstr2Rev<NAME # _D, revname # _D, isReverseInstr>;
  }

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_bin_pred_arit_1<bits<3> opc, string asm, string Ps,
                                   SDPatternOperator op,
                                   DestructiveInstTypeEnum flags> {
  let DestructiveInstType = flags in {
  def _B : sve_int_bin_pred_arit_log<0b00, 0b01, opc, asm, ZPR8>,
           SVEPseudo2Instr<Ps # _B, 1>;
  def _H : sve_int_bin_pred_arit_log<0b01, 0b01, opc, asm, ZPR16>,
           SVEPseudo2Instr<Ps # _H, 1>;
  def _S : sve_int_bin_pred_arit_log<0b10, 0b01, opc, asm, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b01, opc, asm, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>;
  }

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_bin_pred_arit_2<bits<3> opc, string asm, string Ps,
                                   SDPatternOperator op,
                                   DestructiveInstTypeEnum flags> {
  let DestructiveInstType = flags in {
  def _B : sve_int_bin_pred_arit_log<0b00, 0b10, opc, asm, ZPR8>,
           SVEPseudo2Instr<Ps # _B, 1>;
  def _H : sve_int_bin_pred_arit_log<0b01, 0b10, opc, asm, ZPR16>,
           SVEPseudo2Instr<Ps # _H, 1>;
  def _S : sve_int_bin_pred_arit_log<0b10, 0b10, opc, asm, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b10, opc, asm, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>;
  }

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

// Special case for divides which are not defined for 8b/16b elements.
multiclass sve_int_bin_pred_arit_2_div<bits<3> opc, string asm, string Ps,
                                       SDPatternOperator op,
                                       DestructiveInstTypeEnum flags,
                                       string revname="", bit isReverseInstr=0> {
  let DestructiveInstType = flags in {
  def _S : sve_int_bin_pred_arit_log<0b10, 0b10, opc, asm, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>, SVEInstr2Rev<NAME # _S, revname # _S, isReverseInstr>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b10, opc, asm, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>, SVEInstr2Rev<NAME # _D, revname # _D, isReverseInstr>;
  }

  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Multiply-Add Group
//===----------------------------------------------------------------------===//

class sve_int_mladdsub_vvv_pred<bits<2> sz8_64, bits<1> opc, string asm,
                                ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm, zprty:$Za),
  asm, "\t$Zdn, $Pg/m, $Zm, $Za",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<5> Za;
  bits<5> Zm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-14} = 0b11;
  let Inst{13}    = opc;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Za;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_mladdsub_vvv_pred<bits<1> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_mladdsub_vvv_pred<0b00, opc, asm, ZPR8>;
  def _H : sve_int_mladdsub_vvv_pred<0b01, opc, asm, ZPR16>;
  def _S : sve_int_mladdsub_vvv_pred<0b10, opc, asm, ZPR32>;
  def _D : sve_int_mladdsub_vvv_pred<0b11, opc, asm, ZPR64>;

  def : SVE_4_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_4_Op_Pat<nxv8i16, op, nxv8i1, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_4_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve_int_mlas_vvv_pred<bits<2> sz8_64, bits<1> opc, string asm,
                            ZPRRegOp zprty>
: I<(outs zprty:$Zda), (ins PPR3bAny:$Pg, zprty:$_Zda, zprty:$Zn, zprty:$Zm),
  asm, "\t$Zda, $Pg/m, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zda;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-14} = 0b01;
  let Inst{13}    = opc;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_mlas_vvv_pred<bits<1> opc, string asm, SDPatternOperator op,
                                 SDPatternOperator outerop, SDPatternOperator mulop> {
  def _B : sve_int_mlas_vvv_pred<0b00, opc, asm, ZPR8>;
  def _H : sve_int_mlas_vvv_pred<0b01, opc, asm, ZPR16>;
  def _S : sve_int_mlas_vvv_pred<0b10, opc, asm, ZPR32>;
  def _D : sve_int_mlas_vvv_pred<0b11, opc, asm, ZPR64>;

  def : SVE_4_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_4_Op_Pat<nxv8i16, op, nxv8i1, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_4_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : Pat<(outerop nxv16i8:$Op1, (mulop nxv16i1:$pred, nxv16i8:$Op2, nxv16i8:$Op3)),
            (!cast<Instruction>(NAME # _B) $pred, $Op1, $Op2, $Op3)>;
  def : Pat<(outerop nxv8i16:$Op1, (mulop nxv8i1:$pred, nxv8i16:$Op2, nxv8i16:$Op3)),
            (!cast<Instruction>(NAME # _H) $pred, $Op1, $Op2, $Op3)>;
  def : Pat<(outerop nxv4i32:$Op1, (mulop nxv4i1:$pred, nxv4i32:$Op2, nxv4i32:$Op3)),
            (!cast<Instruction>(NAME # _S) $pred, $Op1, $Op2, $Op3)>;
  def : Pat<(outerop nxv2i64:$Op1, (mulop nxv2i1:$pred, nxv2i64:$Op2, nxv2i64:$Op3)),
            (!cast<Instruction>(NAME # _D) $pred, $Op1, $Op2, $Op3)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Integer Multiply-Add - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve2_int_mla<bits<2> sz, bits<5> opc, string asm,
                   ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty2:$Zn, zprty2:$Zm),
  asm, "\t$Zda, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_int_mla<bit S, string asm, SDPatternOperator op> {
  def _B : sve2_int_mla<0b00, { 0b1110, S }, asm, ZPR8, ZPR8>;
  def _H : sve2_int_mla<0b01, { 0b1110, S }, asm, ZPR16, ZPR16>;
  def _S : sve2_int_mla<0b10, { 0b1110, S }, asm, ZPR32, ZPR32>;
  def _D : sve2_int_mla<0b11, { 0b1110, S }, asm, ZPR64, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_int_mla_long<bits<5> opc, string asm, SDPatternOperator op> {
  def _H : sve2_int_mla<0b01, opc, asm, ZPR16, ZPR8>;
  def _S : sve2_int_mla<0b10, opc, asm, ZPR32, ZPR16>;
  def _D : sve2_int_mla<0b11, opc, asm, ZPR64, ZPR32>;

  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Integer Multiply-Add - Indexed Group
//===----------------------------------------------------------------------===//

class sve2_int_mla_by_indexed_elem<bits<2> sz, bits<6> opc, string asm,
                                   ZPRRegOp zprty1, ZPRRegOp zprty2,
                                   ZPRRegOp zprty3, Operand itype>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty2:$Zn, zprty3:$Zm, itype:$iop),
  asm, "\t$Zda, $Zn, $Zm$iop", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{15-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_int_mla_by_indexed_elem<bits<2> opc, bit S, string asm,
                                        SDPatternOperator op> {
  def _H : sve2_int_mla_by_indexed_elem<{0, ?}, { 0b000, opc, S }, asm, ZPR16, ZPR16, ZPR3b16, VectorIndexH32b> {
    bits<3> Zm;
    bits<3> iop;
    let Inst{22} = iop{2};
    let Inst{20-19} = iop{1-0};
    let Inst{18-16} = Zm;
  }
  def _S : sve2_int_mla_by_indexed_elem<0b10, { 0b000, opc, S }, asm, ZPR32, ZPR32, ZPR3b32, VectorIndexS32b> {
    bits<3> Zm;
    bits<2> iop;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _D : sve2_int_mla_by_indexed_elem<0b11, { 0b000, opc, S }, asm, ZPR64, ZPR64, ZPR4b64, VectorIndexD32b> {
    bits<4> Zm;
    bit iop;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : SVE_4_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, i32, VectorIndexH32b_timm, !cast<Instruction>(NAME # _H)>;
  def : SVE_4_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, i32, VectorIndexS32b_timm, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, i32, VectorIndexD32b_timm, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Integer Multiply-Add Long - Indexed Group
//===----------------------------------------------------------------------===//

multiclass sve2_int_mla_long_by_indexed_elem<bits<4> opc, string asm,
                                             SDPatternOperator op> {
  def _S : sve2_int_mla_by_indexed_elem<0b10, { opc{3}, 0b0, opc{2-1}, ?, opc{0} },
                                        asm, ZPR32, ZPR16, ZPR3b16, VectorIndexH32b> {
    bits<3> Zm;
    bits<3> iop;
    let Inst{20-19} = iop{2-1};
    let Inst{18-16} = Zm;
    let Inst{11} = iop{0};
  }
  def _D : sve2_int_mla_by_indexed_elem<0b11, { opc{3}, 0b0, opc{2-1}, ?, opc{0} },
                                        asm, ZPR64, ZPR32, ZPR4b32, VectorIndexS32b> {
    bits<4> Zm;
    bits<2> iop;
    let Inst{20} = iop{1};
    let Inst{19-16} = Zm;
    let Inst{11} = iop{0};
  }

  def : SVE_4_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv8i16, nxv8i16, i32, VectorIndexH32b_timm, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv4i32, nxv4i32, i32, VectorIndexS32b_timm, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Dot Product Group
//===----------------------------------------------------------------------===//

class sve_intx_dot<bit sz, bit U, string asm, ZPRRegOp zprty1,
                   ZPRRegOp zprty2>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty2:$Zn, zprty2:$Zm), asm,
  "\t$Zda, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-23} = 0b010001001;
  let Inst{22}    = sz;
  let Inst{21}    = 0;
  let Inst{20-16} = Zm;
  let Inst{15-11} = 0;
  let Inst{10}    = U;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
}

multiclass sve_intx_dot<bit opc, string asm, SDPatternOperator op> {
  def _S : sve_intx_dot<0b0, opc, asm, ZPR32, ZPR8>;
  def _D : sve_intx_dot<0b1, opc, asm, ZPR64, ZPR16>;

  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32,  nxv16i8, nxv16i8, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64,  nxv8i16, nxv8i16, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Dot Product Group - Indexed Group
//===----------------------------------------------------------------------===//

class sve_intx_dot_by_indexed_elem<bit sz, bit U, string asm,
                                   ZPRRegOp zprty1, ZPRRegOp zprty2,
                                   ZPRRegOp zprty3, Operand itype>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty2:$Zn, zprty3:$Zm, itype:$iop),
  asm, "\t$Zda, $Zn, $Zm$iop",
  "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  let Inst{31-23} = 0b010001001;
  let Inst{22}    = sz;
  let Inst{21}    = 0b1;
  let Inst{15-11} = 0;
  let Inst{10}    = U;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
}

multiclass sve_intx_dot_by_indexed_elem<bit opc, string asm,
                                        SDPatternOperator op> {
  def _S : sve_intx_dot_by_indexed_elem<0b0, opc, asm, ZPR32, ZPR8, ZPR3b8, VectorIndexS32b_timm> {
    bits<2> iop;
    bits<3> Zm;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _D : sve_intx_dot_by_indexed_elem<0b1, opc, asm, ZPR64, ZPR16, ZPR4b16, VectorIndexD32b_timm> {
    bits<1> iop;
    bits<4> Zm;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : SVE_4_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv16i8, nxv16i8, i32, VectorIndexS32b_timm, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv8i16, nxv8i16, i32, VectorIndexD32b_timm, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Complex Integer Dot Product Group
//===----------------------------------------------------------------------===//

class sve2_complex_int_arith<bits<2> sz, bits<4> opc, string asm,
                             ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty2:$Zn, zprty2:$Zm,
                         complexrotateop:$rot),
  asm, "\t$Zda, $Zn, $Zm, $rot", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  bits<2> rot;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-12} = opc;
  let Inst{11-10} = rot;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_cintx_dot<string asm, SDPatternOperator op> {
  def _S : sve2_complex_int_arith<0b10, 0b0001, asm, ZPR32, ZPR8>;
  def _D : sve2_complex_int_arith<0b11, 0b0001, asm, ZPR64, ZPR16>;

  def : Pat<(nxv4i32 (op (nxv4i32 ZPR32:$Op1), (nxv16i8 ZPR8:$Op2), (nxv16i8 ZPR8:$Op3),
                         (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # "_S") ZPR32:$Op1, ZPR8:$Op2, ZPR8:$Op3, complexrotateop:$imm)>;
  def : Pat<(nxv2i64 (op (nxv2i64 ZPR64:$Op1), (nxv8i16 ZPR16:$Op2), (nxv8i16 ZPR16:$Op3),
                         (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # "_D") ZPR64:$Op1, ZPR16:$Op2, ZPR16:$Op3, complexrotateop:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Complex Multiply-Add Group
//===----------------------------------------------------------------------===//

multiclass sve2_int_cmla<bit opc, string asm, SDPatternOperator op> {
  def _B : sve2_complex_int_arith<0b00, { 0b001, opc }, asm, ZPR8, ZPR8>;
  def _H : sve2_complex_int_arith<0b01, { 0b001, opc }, asm, ZPR16, ZPR16>;
  def _S : sve2_complex_int_arith<0b10, { 0b001, opc }, asm, ZPR32, ZPR32>;
  def _D : sve2_complex_int_arith<0b11, { 0b001, opc }, asm, ZPR64, ZPR64>;

  def : SVE_4_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv16i8, nxv16i8, i32, complexrotateop, !cast<Instruction>(NAME # _B)>;
  def : SVE_4_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, i32, complexrotateop, !cast<Instruction>(NAME # _H)>;
  def : SVE_4_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, i32, complexrotateop, !cast<Instruction>(NAME # _S)>;
  def : SVE_4_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, i32, complexrotateop, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Complex Integer Dot Product - Indexed Group
//===----------------------------------------------------------------------===//

class sve2_complex_int_arith_indexed<bits<2> sz, bits<4> opc, string asm,
                                     ZPRRegOp zprty1, ZPRRegOp zprty2,
                                     ZPRRegOp zprty3, Operand itype>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty2:$Zn, zprty3:$Zm, itype:$iop,
                         complexrotateop:$rot),
  asm, "\t$Zda, $Zn, $Zm$iop, $rot", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<2> rot;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{15-12} = opc;
  let Inst{11-10} = rot;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_cintx_dot_by_indexed_elem<string asm, SDPatternOperator op> {
  def _S : sve2_complex_int_arith_indexed<0b10, 0b0100, asm, ZPR32, ZPR8, ZPR3b8, VectorIndexS32b> {
    bits<2> iop;
    bits<3> Zm;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _D : sve2_complex_int_arith_indexed<0b11, 0b0100, asm, ZPR64, ZPR16, ZPR4b16, VectorIndexD32b> {
    bit iop;
    bits<4> Zm;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : Pat<(nxv4i32 (op (nxv4i32 ZPR32:$Op1), (nxv16i8 ZPR8:$Op2), (nxv16i8 ZPR8:$Op3),
                         (i32 VectorIndexS32b_timm:$idx), (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # "_S") ZPR32:$Op1, ZPR8:$Op2, ZPR8:$Op3, VectorIndexS32b_timm:$idx, complexrotateop:$imm)>;
  def : Pat<(nxv2i64 (op (nxv2i64 ZPR64:$Op1), (nxv8i16 ZPR16:$Op2), (nxv8i16 ZPR16:$Op3),
                         (i32 VectorIndexD32b_timm:$idx), (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # "_D") ZPR64:$Op1, ZPR16:$Op2, ZPR16:$Op3, VectorIndexD32b_timm:$idx, complexrotateop:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Complex Multiply-Add - Indexed Group
//===----------------------------------------------------------------------===//

multiclass sve2_cmla_by_indexed_elem<bit opc, string asm,
                                     SDPatternOperator op> {
  def _H : sve2_complex_int_arith_indexed<0b10, { 0b011, opc }, asm, ZPR16, ZPR16, ZPR3b16, VectorIndexS32b> {
    bits<2> iop;
    bits<3> Zm;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _S : sve2_complex_int_arith_indexed<0b11, { 0b011, opc }, asm, ZPR32, ZPR32, ZPR4b32, VectorIndexD32b> {
    bit iop;
    bits<4> Zm;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : Pat<(nxv8i16 (op (nxv8i16 ZPR16:$Op1), (nxv8i16 ZPR16:$Op2), (nxv8i16 ZPR16:$Op3),
                         (i32 VectorIndexS32b_timm:$idx), (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # "_H") ZPR16:$Op1, ZPR16:$Op2, ZPR16:$Op3, VectorIndexS32b_timm:$idx, complexrotateop:$imm)>;

  def : Pat<(nxv4i32 (op (nxv4i32 ZPR32:$Op1), (nxv4i32 ZPR32:$Op2), (nxv4i32 ZPR32:$Op3),
                         (i32 VectorIndexD32b_timm:$idx), (i32 complexrotateop:$imm))),
            (!cast<Instruction>(NAME # "_S") ZPR32:$Op1, ZPR32:$Op2, ZPR32:$Op3, VectorIndexD32b_timm:$idx, complexrotateop:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Integer Multiply - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve2_int_mul<bits<2> sz, bits<3> opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b011;
  let Inst{12-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_int_mul<bits<3> opc, string asm, SDPatternOperator op,
                        SDPatternOperator op_pred = null_frag> {
  def _B : sve2_int_mul<0b00, opc, asm, ZPR8>;
  def _H : sve2_int_mul<0b01, opc, asm, ZPR16>;
  def _S : sve2_int_mul<0b10, opc, asm, ZPR32>;
  def _D : sve2_int_mul<0b11, opc, asm, ZPR64>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_2_Op_Pred_All_Active<nxv16i8, op_pred, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pred_All_Active<nxv8i16, op_pred, nxv8i1, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pred_All_Active<nxv4i32, op_pred, nxv4i1, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pred_All_Active<nxv2i64, op_pred, nxv2i1, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_int_mul_single<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve2_int_mul<0b00, opc, asm, ZPR8>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Integer Multiply - Indexed Group
//===----------------------------------------------------------------------===//

class sve2_int_mul_by_indexed_elem<bits<2> sz, bits<4> opc, string asm,
                                   ZPRRegOp zprty1, ZPRRegOp zprty2,
                                   ZPRRegOp zprty3, Operand itype>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn, zprty3:$Zm, itype:$iop),
  asm, "\t$Zd, $Zn, $Zm$iop", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{15-14} = 0b11;
  let Inst{13-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_int_mul_by_indexed_elem<bits<4> opc, string asm,
                                        SDPatternOperator op> {
  def _H : sve2_int_mul_by_indexed_elem<{0, ?}, opc, asm, ZPR16, ZPR16, ZPR3b16, VectorIndexH32b> {
    bits<3> Zm;
    bits<3> iop;
    let Inst{22} = iop{2};
    let Inst{20-19} = iop{1-0};
    let Inst{18-16} = Zm;
  }
  def _S : sve2_int_mul_by_indexed_elem<0b10, opc, asm, ZPR32, ZPR32, ZPR3b32, VectorIndexS32b> {
    bits<3> Zm;
    bits<2> iop;
    let Inst{20-19} = iop;
    let Inst{18-16} = Zm;
  }
  def _D : sve2_int_mul_by_indexed_elem<0b11, opc, asm, ZPR64, ZPR64, ZPR4b64, VectorIndexD32b> {
    bits<4> Zm;
    bit iop;
    let Inst{20} = iop;
    let Inst{19-16} = Zm;
  }

  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, i32, VectorIndexH32b_timm, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, i32, VectorIndexS32b_timm, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, i32, VectorIndexD32b_timm, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_int_mul_long_by_indexed_elem<bits<3> opc, string asm,
                                             SDPatternOperator op> {
  def _S : sve2_int_mul_by_indexed_elem<0b10, { opc{2-1}, ?, opc{0} }, asm,
                                        ZPR32, ZPR16, ZPR3b16, VectorIndexH32b> {
    bits<3> Zm;
    bits<3> iop;
    let Inst{20-19} = iop{2-1};
    let Inst{18-16} = Zm;
    let Inst{11} = iop{0};
  }
  def _D : sve2_int_mul_by_indexed_elem<0b11, { opc{2-1}, ?, opc{0} }, asm,
                                        ZPR64, ZPR32, ZPR4b32, VectorIndexS32b> {
    bits<4> Zm;
    bits<2> iop;
    let Inst{20} = iop{1};
    let Inst{19-16} = Zm;
    let Inst{11} = iop{0};
  }

  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv8i16, nxv8i16, i32, VectorIndexH32b_timm, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv4i32, nxv4i32, i32, VectorIndexS32b_timm, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Integer - Predicated Group
//===----------------------------------------------------------------------===//

class sve2_int_arith_pred<bits<2> sz, bits<6> opc, string asm,
                          ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $Zm", "", []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zm;
  bits<5> Zdn;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b01;
  let Inst{20-16} = opc{5-1};
  let Inst{15-14} = 0b10;
  let Inst{13}    = opc{0};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve2_int_arith_pred<bits<6> opc, string asm, SDPatternOperator op,
                               string Ps = "",
                               DestructiveInstTypeEnum flags=DestructiveOther,
                               string revname="", bit isReverseInstr=0> {
  let DestructiveInstType = flags in {
  def _B : sve2_int_arith_pred<0b00, opc, asm, ZPR8>,
           SVEPseudo2Instr<Ps # _B, 1>, SVEInstr2Rev<NAME # _B, revname # _B, isReverseInstr>;
  def _H : sve2_int_arith_pred<0b01, opc, asm, ZPR16>,
           SVEPseudo2Instr<Ps # _H, 1>, SVEInstr2Rev<NAME # _H, revname # _H, isReverseInstr>;
  def _S : sve2_int_arith_pred<0b10, opc, asm, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>, SVEInstr2Rev<NAME # _S, revname # _S, isReverseInstr>;
  def _D : sve2_int_arith_pred<0b11, opc, asm, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>, SVEInstr2Rev<NAME # _D, revname # _D, isReverseInstr>;
  }

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve2_int_sadd_long_accum_pairwise<bits<2> sz, bit U, string asm,
                                        ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zda), (ins PPR3bAny:$Pg, zprty1:$_Zda, zprty2:$Zn),
  asm, "\t$Zda, $Pg/m, $Zn", "", []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zn;
  bits<5> Zda;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21-17} = 0b00010;
  let Inst{16}    = U;
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty1.ElementSize;
}

multiclass sve2_int_sadd_long_accum_pairwise<bit U, string asm, SDPatternOperator op> {
  def _H : sve2_int_sadd_long_accum_pairwise<0b01, U, asm, ZPR16, ZPR8>;
  def _S : sve2_int_sadd_long_accum_pairwise<0b10, U, asm, ZPR32, ZPR16>;
  def _D : sve2_int_sadd_long_accum_pairwise<0b11, U, asm, ZPR64, ZPR32>;

  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1, nxv8i16, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv8i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

class sve2_int_un_pred_arit<bits<2> sz, bit Q, bits<2> opc,
                            string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Zd, $Pg/m, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b01000100;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b00;
  let Inst{19}    = Q;
  let Inst{18}    = 0b0;
  let Inst{17-16} = opc;
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveUnaryPassthru;
  let ElementSize = zprty.ElementSize;
}

multiclass sve2_int_un_pred_arit_s<bits<3> opc, string asm,
                                   SDPatternOperator op> {
  def _S : sve2_int_un_pred_arit<0b10, opc{2}, opc{1-0}, asm, ZPR32>,
           SVEPseudo2Instr<NAME # _S, 1>;

  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i1, nxv4i32, !cast<Instruction>(NAME # _S)>;

  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;

  defm : SVE_3_Op_Undef_Pat<nxv4i32, op, nxv4i32, nxv4i1,  nxv4i32, !cast<Pseudo>(NAME # _UNDEF_S)>;
}

multiclass sve2_int_un_pred_arit<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve2_int_un_pred_arit<0b00, opc{2}, opc{1-0}, asm, ZPR8>,
           SVEPseudo2Instr<NAME # _B, 1>;
  def _H : sve2_int_un_pred_arit<0b01, opc{2}, opc{1-0}, asm, ZPR16>,
           SVEPseudo2Instr<NAME # _H, 1>;
  def _S : sve2_int_un_pred_arit<0b10, opc{2}, opc{1-0}, asm, ZPR32>,
           SVEPseudo2Instr<NAME # _S, 1>;
  def _D : sve2_int_un_pred_arit<0b11, opc{2}, opc{1-0}, asm, ZPR64>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_B : PredOneOpPassthruPseudo<NAME # _B, ZPR8>;
  def _UNDEF_H : PredOneOpPassthruPseudo<NAME # _H, ZPR16>;
  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;
  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_3_Op_Undef_Pat<nxv16i8, op, nxv16i8, nxv16i1, nxv16i8, !cast<Pseudo>(NAME # _UNDEF_B)>;
  defm : SVE_3_Op_Undef_Pat<nxv8i16, op, nxv8i16, nxv8i1,  nxv8i16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  defm : SVE_3_Op_Undef_Pat<nxv4i32, op, nxv4i32, nxv4i1,  nxv4i32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  defm : SVE_3_Op_Undef_Pat<nxv2i64, op, nxv2i64, nxv2i1,  nxv2i64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Widening Integer Arithmetic Group
//===----------------------------------------------------------------------===//

class sve2_wide_int_arith<bits<2> sz, bits<5> opc, string asm,
                          ZPRRegOp zprty1, ZPRRegOp zprty2, ZPRRegOp zprty3>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn, zprty3:$Zm),
  asm, "\t$Zd, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_wide_int_arith_long<bits<5> opc, string asm,
                                    SDPatternOperator op> {
  def _H : sve2_wide_int_arith<0b01, opc, asm, ZPR16, ZPR8, ZPR8>;
  def _S : sve2_wide_int_arith<0b10, opc, asm, ZPR32, ZPR16, ZPR16>;
  def _D : sve2_wide_int_arith<0b11, opc, asm, ZPR64, ZPR32, ZPR32>;

  def : SVE_2_Op_Pat<nxv8i16, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_wide_int_arith_wide<bits<3> opc, string asm,
                                    SDPatternOperator op> {
  def _H : sve2_wide_int_arith<0b01, { 0b10, opc }, asm, ZPR16, ZPR16, ZPR8>;
  def _S : sve2_wide_int_arith<0b10, { 0b10, opc }, asm, ZPR32, ZPR32, ZPR16>;
  def _D : sve2_wide_int_arith<0b11, { 0b10, opc }, asm, ZPR64, ZPR64, ZPR32>;

  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv8i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_wide_int_arith_pmul<bits<2> sz, bits<5> opc, string asm,
                                     SDPatternOperator op> {
  def NAME : sve2_wide_int_arith<sz, opc, asm, ZPR128, ZPR64, ZPR64>;

  // To avoid using 128 bit elements in the IR, the pattern below works with
  // llvm intrinsics with the _pair suffix, to reflect that
  // _Q is implemented as a pair of _D.
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME)>;
}

multiclass sve2_pmul_long<bits<1> opc, string asm, SDPatternOperator op> {
  def _H : sve2_wide_int_arith<0b01, {0b1101, opc}, asm, ZPR16, ZPR8, ZPR8>;
  def _D : sve2_wide_int_arith<0b11, {0b1101, opc}, asm, ZPR64, ZPR32, ZPR32>;

  // To avoid using 128 bit elements in the IR, the patterns below work with
  // llvm intrinsics with the _pair suffix, to reflect that
  // _H is implemented as a pair of _B and _D is implemented as a pair of _S.
  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Misc Group
//===----------------------------------------------------------------------===//

class sve2_misc<bits<2> sz, bits<4> opc, string asm,
                ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn, zprty2:$Zm),
  asm, "\t$Zd, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-14} = 0b10;
  let Inst{13-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_misc_bitwise<bits<4> opc, string asm, SDPatternOperator op> {
  def _B : sve2_misc<0b00, opc, asm, ZPR8, ZPR8>;
  def _H : sve2_misc<0b01, opc, asm, ZPR16, ZPR16>;
  def _S : sve2_misc<0b10, opc, asm, ZPR32, ZPR32>;
  def _D : sve2_misc<0b11, opc, asm, ZPR64, ZPR64>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_misc_int_addsub_long_interleaved<bits<2> opc, string asm,
                                                 SDPatternOperator op> {
  def _H : sve2_misc<0b01, { 0b00, opc }, asm, ZPR16, ZPR8>;
  def _S : sve2_misc<0b10, { 0b00, opc }, asm, ZPR32, ZPR16>;
  def _D : sve2_misc<0b11, { 0b00, opc }, asm, ZPR64, ZPR32>;

  def : SVE_2_Op_Pat<nxv8i16, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

class sve2_bitwise_xor_interleaved<bits<2> sz, bits<1> opc, string asm,
                                   ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty1:$_Zd, zprty2:$Zn, zprty2:$Zm),
  asm, "\t$Zd, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-11} = 0b10010;
  let Inst{10}    = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_bitwise_xor_interleaved<bit opc, string asm,
                                        SDPatternOperator op> {
  def _B : sve2_bitwise_xor_interleaved<0b00, opc, asm, ZPR8,  ZPR8>;
  def _H : sve2_bitwise_xor_interleaved<0b01, opc, asm, ZPR16, ZPR16>;
  def _S : sve2_bitwise_xor_interleaved<0b10, opc, asm, ZPR32, ZPR32>;
  def _D : sve2_bitwise_xor_interleaved<0b11, opc, asm, ZPR64, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve2_bitwise_shift_left_long<bits<3> tsz8_64, bits<2> opc, string asm,
                                   ZPRRegOp zprty1, ZPRRegOp zprty2,
                                   Operand immtype>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn, immtype:$imm),
  asm, "\t$Zd, $Zn, $imm",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> imm;
  let Inst{31-23} = 0b010001010;
  let Inst{22}    = tsz8_64{2};
  let Inst{21}    = 0b0;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-16} = imm{2-0}; // imm3
  let Inst{15-12} = 0b1010;
  let Inst{11-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_bitwise_shift_left_long<bits<2> opc, string asm,
                                        SDPatternOperator op> {
  def _H : sve2_bitwise_shift_left_long<{0,0,1}, opc, asm,
                                        ZPR16, ZPR8, vecshiftL8>;
  def _S : sve2_bitwise_shift_left_long<{0,1,?}, opc, asm,
                                        ZPR32, ZPR16, vecshiftL16> {
    let Inst{19} = imm{3};
  }
  def _D : sve2_bitwise_shift_left_long<{1,?,?}, opc, asm,
                                        ZPR64, ZPR32, vecshiftL32> {
    let Inst{20-19} = imm{4-3};
  }
  def : SVE_2_Op_Imm_Pat<nxv8i16, op, nxv16i8, i32, tvecshiftL8,  !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Imm_Pat<nxv4i32, op, nxv8i16, i32, tvecshiftL16, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Imm_Pat<nxv2i64, op, nxv4i32, i32, tvecshiftL32, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Accumulate Group
//===----------------------------------------------------------------------===//

class sve2_int_bin_shift_imm<bits<4> tsz8_64, bit opc, string asm,
                             ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, zprty:$Zn, immtype:$imm),
  asm, "\t$Zd, $Zn, $imm",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<6> imm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = tsz8_64{3-2};
  let Inst{21}    = 0b0;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-16} = imm{2-0}; // imm3
  let Inst{15-11} = 0b11110;
  let Inst{10}    = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
}

multiclass sve2_int_bin_shift_imm_left<bit opc, string asm,
                                       SDPatternOperator op> {
  def _B : sve2_int_bin_shift_imm<{0,0,0,1}, opc, asm, ZPR8, vecshiftL8>;
  def _H : sve2_int_bin_shift_imm<{0,0,1,?}, opc, asm, ZPR16, vecshiftL16> {
    let Inst{19} = imm{3};
  }
  def _S : sve2_int_bin_shift_imm<{0,1,?,?}, opc, asm, ZPR32, vecshiftL32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve2_int_bin_shift_imm<{1,?,?,?}, opc, asm, ZPR64, vecshiftL64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv16i8, i32, tvecshiftL8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, i32, tvecshiftL16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, i32, tvecshiftL32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, i32, tvecshiftL64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_int_bin_shift_imm_right<bit opc, string asm,
                                        SDPatternOperator op> {
  def _B : sve2_int_bin_shift_imm<{0,0,0,1}, opc, asm, ZPR8, vecshiftR8>;
  def _H : sve2_int_bin_shift_imm<{0,0,1,?}, opc, asm, ZPR16, vecshiftR16> {
    let Inst{19} = imm{3};
  }
  def _S : sve2_int_bin_shift_imm<{0,1,?,?}, opc, asm, ZPR32, vecshiftR32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve2_int_bin_shift_imm<{1,?,?,?}, opc, asm, ZPR64, vecshiftR64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv16i8, i32, tvecshiftR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, i32, tvecshiftR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, i32, tvecshiftR32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, i32, tvecshiftR64, !cast<Instruction>(NAME # _D)>;
}

class sve2_int_bin_accum_shift_imm<bits<4> tsz8_64, bits<2> opc, string asm,
                                   ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zda), (ins zprty:$_Zda, zprty:$Zn, immtype:$imm),
  asm, "\t$Zda, $Zn, $imm",
  "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<6> imm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = tsz8_64{3-2};
  let Inst{21}    = 0b0;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-16} = imm{2-0}; // imm3
  let Inst{15-12} = 0b1110;
  let Inst{11-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_int_bin_accum_shift_imm_right<bits<2> opc, string asm,
                                              SDPatternOperator op> {
  def _B : sve2_int_bin_accum_shift_imm<{0,0,0,1}, opc, asm, ZPR8, vecshiftR8>;
  def _H : sve2_int_bin_accum_shift_imm<{0,0,1,?}, opc, asm, ZPR16, vecshiftR16> {
    let Inst{19} = imm{3};
  }
  def _S : sve2_int_bin_accum_shift_imm<{0,1,?,?}, opc, asm, ZPR32, vecshiftR32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve2_int_bin_accum_shift_imm<{1,?,?,?}, opc, asm, ZPR64, vecshiftR64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv16i8, i32, tvecshiftR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, i32, tvecshiftR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, i32, tvecshiftR32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, i32, tvecshiftR64, !cast<Instruction>(NAME # _D)>;
}

class sve2_int_cadd<bits<2> sz, bit opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, zprty:$Zm, complexrotateopodd:$rot),
  asm, "\t$Zdn, $_Zdn, $Zm, $rot", "", []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> Zm;
  bit rot;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = sz;
  let Inst{21-17} = 0b00000;
  let Inst{16}    = opc;
  let Inst{15-11} = 0b11011;
  let Inst{10}    = rot;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_int_cadd<bit opc, string asm, SDPatternOperator op> {
  def _B : sve2_int_cadd<0b00, opc, asm, ZPR8>;
  def _H : sve2_int_cadd<0b01, opc, asm, ZPR16>;
  def _S : sve2_int_cadd<0b10, opc, asm, ZPR32>;
  def _D : sve2_int_cadd<0b11, opc, asm, ZPR64>;

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv16i8, i32, complexrotateopodd, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, i32, complexrotateopodd, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, i32, complexrotateopodd, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, i32, complexrotateopodd, !cast<Instruction>(NAME # _D)>;
}

class sve2_int_absdiff_accum<bits<2> sz, bits<4> opc, string asm,
                             ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zda), (ins zprty1:$_Zda, zprty2:$Zn, zprty2:$Zm),
  asm, "\t$Zda, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15-14} = 0b11;
  let Inst{13-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_int_absdiff_accum<bit opc, string asm, SDPatternOperator op> {
  def _B : sve2_int_absdiff_accum<0b00, { 0b111, opc }, asm, ZPR8, ZPR8>;
  def _H : sve2_int_absdiff_accum<0b01, { 0b111, opc }, asm, ZPR16, ZPR16>;
  def _S : sve2_int_absdiff_accum<0b10, { 0b111, opc }, asm, ZPR32, ZPR32>;
  def _D : sve2_int_absdiff_accum<0b11, { 0b111, opc }, asm, ZPR64, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_int_absdiff_accum_long<bits<2> opc, string asm,
                                       SDPatternOperator op> {
  def _H : sve2_int_absdiff_accum<0b01, { 0b00, opc }, asm, ZPR16, ZPR8>;
  def _S : sve2_int_absdiff_accum<0b10, { 0b00, opc }, asm, ZPR32, ZPR16>;
  def _D : sve2_int_absdiff_accum<0b11, { 0b00, opc }, asm, ZPR64, ZPR32>;

  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _D)>;
}

multiclass sve2_int_addsub_long_carry<bits<2> opc, string asm,
                                      SDPatternOperator op> {
  def _S : sve2_int_absdiff_accum<{ opc{1}, 0b0 }, { 0b010, opc{0} }, asm,
                                  ZPR32, ZPR32>;
  def _D : sve2_int_absdiff_accum<{ opc{1}, 0b1 }, { 0b010, opc{0} }, asm,
                                  ZPR64, ZPR64>;

  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Narrowing Group
//===----------------------------------------------------------------------===//

class sve2_int_bin_shift_imm_narrow_bottom<bits<3> tsz8_64, bits<3> opc,
                                           string asm, ZPRRegOp zprty1,
                                           ZPRRegOp zprty2, Operand immtype>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn, immtype:$imm),
  asm, "\t$Zd, $Zn, $imm",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> imm;
  let Inst{31-23} = 0b010001010;
  let Inst{22}    = tsz8_64{2};
  let Inst{21}    = 0b1;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-16} = imm{2-0}; // imm3
  let Inst{15-14} = 0b00;
  let Inst{13-11} = opc;
  let Inst{10}    = 0b0;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_int_bin_shift_imm_right_narrow_bottom<bits<3> opc, string asm,
                                                      SDPatternOperator op> {
  def _B : sve2_int_bin_shift_imm_narrow_bottom<{0,0,1}, opc, asm, ZPR8, ZPR16,
                                                tvecshiftR8>;
  def _H : sve2_int_bin_shift_imm_narrow_bottom<{0,1,?}, opc, asm, ZPR16, ZPR32,
                                                tvecshiftR16> {
    let Inst{19} = imm{3};
  }
  def _S : sve2_int_bin_shift_imm_narrow_bottom<{1,?,?}, opc, asm, ZPR32, ZPR64,
                                                tvecshiftR32> {
    let Inst{20-19} = imm{4-3};
  }
  def : SVE_2_Op_Imm_Pat<nxv16i8, op, nxv8i16, i32, tvecshiftR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Imm_Pat<nxv8i16, op, nxv4i32, i32, tvecshiftR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Imm_Pat<nxv4i32, op, nxv2i64, i32, tvecshiftR32, !cast<Instruction>(NAME # _S)>;
}

class sve2_int_bin_shift_imm_narrow_top<bits<3> tsz8_64, bits<3> opc,
                                        string asm, ZPRRegOp zprty1,
                                        ZPRRegOp zprty2, Operand immtype>
: I<(outs zprty1:$Zd), (ins zprty1:$_Zd, zprty2:$Zn, immtype:$imm),
  asm, "\t$Zd, $Zn, $imm",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> imm;
  let Inst{31-23} = 0b010001010;
  let Inst{22}    = tsz8_64{2};
  let Inst{21}    = 0b1;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-16} = imm{2-0}; // imm3
  let Inst{15-14} = 0b00;
  let Inst{13-11} = opc;
  let Inst{10}    = 0b1;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
}

multiclass sve2_int_bin_shift_imm_right_narrow_top<bits<3> opc, string asm,
                                                   SDPatternOperator op> {
  def _B : sve2_int_bin_shift_imm_narrow_top<{0,0,1}, opc, asm, ZPR8, ZPR16,
                                             tvecshiftR8>;
  def _H : sve2_int_bin_shift_imm_narrow_top<{0,1,?}, opc, asm, ZPR16, ZPR32,
                                             tvecshiftR16> {
    let Inst{19} = imm{3};
  }
  def _S : sve2_int_bin_shift_imm_narrow_top<{1,?,?}, opc, asm, ZPR32, ZPR64,
                                             tvecshiftR32> {
    let Inst{20-19} = imm{4-3};
  }
  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv8i16, i32, tvecshiftR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv4i32, i32, tvecshiftR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv2i64, i32, tvecshiftR32, !cast<Instruction>(NAME # _S)>;
}

class sve2_int_addsub_narrow_high_bottom<bits<2> sz, bits<2> opc, string asm,
                                         ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn, zprty2:$Zm),
  asm, "\t$Zd, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b011;
  let Inst{12-11} = opc; // S, R
  let Inst{10}    = 0b0; // Top
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_int_addsub_narrow_high_bottom<bits<2> opc, string asm,
                                              SDPatternOperator op> {
  def _B : sve2_int_addsub_narrow_high_bottom<0b01, opc, asm, ZPR8, ZPR16>;
  def _H : sve2_int_addsub_narrow_high_bottom<0b10, opc, asm, ZPR16, ZPR32>;
  def _S : sve2_int_addsub_narrow_high_bottom<0b11, opc, asm, ZPR32, ZPR64>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _S)>;
}

class sve2_int_addsub_narrow_high_top<bits<2> sz, bits<2> opc, string asm,
                                      ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty1:$_Zd, zprty2:$Zn, zprty2:$Zm),
  asm, "\t$Zd, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b011;
  let Inst{12-11} = opc; // S, R
  let Inst{10}    = 0b1; // Top
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
}

multiclass sve2_int_addsub_narrow_high_top<bits<2> opc, string asm,
                                           SDPatternOperator op> {
  def _B : sve2_int_addsub_narrow_high_top<0b01, opc, asm, ZPR8, ZPR16>;
  def _H : sve2_int_addsub_narrow_high_top<0b10, opc, asm, ZPR16, ZPR32>;
  def _S : sve2_int_addsub_narrow_high_top<0b11, opc, asm, ZPR32, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv8i16, nxv8i16, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _S)>;
}

class sve2_int_sat_extract_narrow_bottom<bits<3> tsz8_64, bits<2> opc, string asm,
                                         ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty2:$Zn),
  asm, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-23} = 0b010001010;
  let Inst{22}    = tsz8_64{2};
  let Inst{21}    = 0b1;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-13} = 0b000010;
  let Inst{12-11} = opc;
  let Inst{10}    = 0b0;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_int_sat_extract_narrow_bottom<bits<2> opc, string asm,
                                              SDPatternOperator op> {
  def _B : sve2_int_sat_extract_narrow_bottom<0b001, opc, asm, ZPR8, ZPR16>;
  def _H : sve2_int_sat_extract_narrow_bottom<0b010, opc, asm, ZPR16, ZPR32>;
  def _S : sve2_int_sat_extract_narrow_bottom<0b100, opc, asm, ZPR32, ZPR64>;

  def : SVE_1_Op_Pat<nxv16i8, op, nxv8i16, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Pat<nxv8i16, op, nxv4i32, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Pat<nxv4i32, op, nxv2i64, !cast<Instruction>(NAME # _S)>;
}

class sve2_int_sat_extract_narrow_top<bits<3> tsz8_64, bits<2> opc, string asm,
                                      ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs zprty1:$Zd), (ins zprty1:$_Zd, zprty2:$Zn),
  asm, "\t$Zd, $Zn", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-23} = 0b010001010;
  let Inst{22}    = tsz8_64{2};
  let Inst{21}    = 0b1;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-13} = 0b000010;
  let Inst{12-11} = opc;
  let Inst{10}    = 0b1;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
}

multiclass sve2_int_sat_extract_narrow_top<bits<2> opc, string asm,
                                           SDPatternOperator op> {
  def _B : sve2_int_sat_extract_narrow_top<0b001, opc, asm, ZPR8, ZPR16>;
  def _H : sve2_int_sat_extract_narrow_top<0b010, opc, asm, ZPR16, ZPR32>;
  def _S : sve2_int_sat_extract_narrow_top<0b100, opc, asm, ZPR32, ZPR64>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv8i16, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv4i32, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv2i64, !cast<Instruction>(NAME # _S)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Arithmetic - Unary Predicated Group
//===----------------------------------------------------------------------===//

class sve_int_un_pred_arit<bits<2> sz8_64, bits<4> opc,
                             string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Zd, $Pg/m, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21-20} = 0b01;
  let Inst{19}    = opc{0};
  let Inst{18-16} = opc{3-1};
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveUnaryPassthru;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_un_pred_arit_0<bits<3> opc, string asm,
                                  SDPatternOperator op> {
  def _B : sve_int_un_pred_arit<0b00, { opc, 0b0 }, asm, ZPR8>,
           SVEPseudo2Instr<NAME # _B, 1>;
  def _H : sve_int_un_pred_arit<0b01, { opc, 0b0 }, asm, ZPR16>,
           SVEPseudo2Instr<NAME # _H, 1>;
  def _S : sve_int_un_pred_arit<0b10, { opc, 0b0 }, asm, ZPR32>,
           SVEPseudo2Instr<NAME # _S, 1>;
  def _D : sve_int_un_pred_arit<0b11, { opc, 0b0 }, asm, ZPR64>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_1_Op_Passthru_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Passthru_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_B : PredOneOpPassthruPseudo<NAME # _B, ZPR8>;
  def _UNDEF_H : PredOneOpPassthruPseudo<NAME # _H, ZPR16>;
  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;
  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_1_Op_PassthruUndef_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Pseudo>(NAME # _UNDEF_B)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

multiclass sve_int_un_pred_arit_0_h<bits<3> opc, string asm,
                                    SDPatternOperator op> {
  def _H : sve_int_un_pred_arit<0b01, { opc, 0b0 }, asm, ZPR16>,
           SVEPseudo2Instr<NAME # _H, 1>;
  def _S : sve_int_un_pred_arit<0b10, { opc, 0b0 }, asm, ZPR32>,
           SVEPseudo2Instr<NAME # _S, 1>;
  def _D : sve_int_un_pred_arit<0b11, { opc, 0b0 }, asm, ZPR64>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_InReg_Extend<nxv8i16, op, nxv8i1, nxv8i8, !cast<Instruction>(NAME # _H)>;
  def : SVE_InReg_Extend<nxv4i32, op, nxv4i1, nxv4i8, !cast<Instruction>(NAME # _S)>;
  def : SVE_InReg_Extend<nxv2i64, op, nxv2i1, nxv2i8, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_H : PredOneOpPassthruPseudo<NAME # _H, ZPR16>;
  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;
  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_InReg_Extend_PassthruUndef<nxv8i16, op, nxv8i1, nxv8i8, !cast<Pseudo>(NAME # _UNDEF_H)>;
  defm : SVE_InReg_Extend_PassthruUndef<nxv4i32, op, nxv4i1, nxv4i8, !cast<Pseudo>(NAME # _UNDEF_S)>;
  defm : SVE_InReg_Extend_PassthruUndef<nxv2i64, op, nxv2i1, nxv2i8, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

multiclass sve_int_un_pred_arit_0_w<bits<3> opc, string asm,
                                    SDPatternOperator op> {
  def _S : sve_int_un_pred_arit<0b10, { opc, 0b0 }, asm, ZPR32>,
           SVEPseudo2Instr<NAME # _S, 1>;
  def _D : sve_int_un_pred_arit<0b11, { opc, 0b0 }, asm, ZPR64>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_InReg_Extend<nxv4i32, op, nxv4i1, nxv4i16, !cast<Instruction>(NAME # _S)>;
  def : SVE_InReg_Extend<nxv2i64, op, nxv2i1, nxv2i16, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;
  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_InReg_Extend_PassthruUndef<nxv4i32, op, nxv4i1, nxv4i16, !cast<Pseudo>(NAME # _UNDEF_S)>;
  defm : SVE_InReg_Extend_PassthruUndef<nxv2i64, op, nxv2i1, nxv2i16, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

multiclass sve_int_un_pred_arit_0_d<bits<3> opc, string asm,
                                    SDPatternOperator op> {
  def _D : sve_int_un_pred_arit<0b11, { opc, 0b0 }, asm, ZPR64>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_InReg_Extend<nxv2i64, op, nxv2i1, nxv2i32, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_InReg_Extend_PassthruUndef<nxv2i64, op, nxv2i1, nxv2i32, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

multiclass sve_int_un_pred_arit_1<bits<3> opc, string asm,
                                  SDPatternOperator op> {
  def _B : sve_int_un_pred_arit<0b00, { opc, 0b1 }, asm, ZPR8>,
           SVEPseudo2Instr<NAME # _B, 1>;
  def _H : sve_int_un_pred_arit<0b01, { opc, 0b1 }, asm, ZPR16>,
           SVEPseudo2Instr<NAME # _H, 1>;
  def _S : sve_int_un_pred_arit<0b10, { opc, 0b1 }, asm, ZPR32>,
           SVEPseudo2Instr<NAME # _S, 1>;
  def _D : sve_int_un_pred_arit<0b11, { opc, 0b1 }, asm, ZPR64>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_1_Op_Passthru_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Passthru_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_B : PredOneOpPassthruPseudo<NAME # _B, ZPR8>;
  def _UNDEF_H : PredOneOpPassthruPseudo<NAME # _H, ZPR16>;
  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;
  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_1_Op_PassthruUndef_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Pseudo>(NAME # _UNDEF_B)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

multiclass sve_int_un_pred_arit_1_fp<bits<3> opc, string asm, SDPatternOperator op> {
  def _H : sve_int_un_pred_arit<0b01, { opc, 0b1 }, asm, ZPR16>,
           SVEPseudo2Instr<NAME # _H, 1>;
  def _S : sve_int_un_pred_arit<0b10, { opc, 0b1 }, asm, ZPR32>,
           SVEPseudo2Instr<NAME # _S, 1>;
  def _D : sve_int_un_pred_arit<0b11, { opc, 0b1 }, asm, ZPR64>,
           SVEPseudo2Instr<NAME # _D, 1>;

  def : SVE_1_Op_Passthru_Pat<nxv8f16, op, nxv8i1, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4f16, op, nxv4i1, nxv4f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv2f16, op, nxv2i1, nxv2f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4f32, op, nxv4i1, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2f32, op, nxv2i1, nxv2f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2f64, op, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def _UNDEF_H : PredOneOpPassthruPseudo<NAME # _H, ZPR16>;
  def _UNDEF_S : PredOneOpPassthruPseudo<NAME # _S, ZPR32>;
  def _UNDEF_D : PredOneOpPassthruPseudo<NAME # _D, ZPR64>;

  defm : SVE_1_Op_PassthruUndef_Pat<nxv8f16, op, nxv8i1, nxv8f16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv4f16, op, nxv4i1, nxv4f16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2f16, op, nxv2i1, nxv2f16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv4f32, op, nxv4i1, nxv4f32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2f32, op, nxv2i1, nxv2f32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  defm : SVE_1_Op_PassthruUndef_Pat<nxv2f64, op, nxv2i1, nxv2f64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Wide Immediate - Unpredicated Group
//===----------------------------------------------------------------------===//
class sve_int_dup_imm<bits<2> sz8_64, string asm,
                      ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zd), (ins immtype:$imm),
  asm, "\t$Zd, $imm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<9> imm;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-14} = 0b11100011;
  let Inst{13}    = imm{8};   // sh
  let Inst{12-5}  = imm{7-0}; // imm8
  let Inst{4-0}   = Zd;

  let isReMaterializable = 1;
}

multiclass sve_int_dup_imm<string asm> {
  def _B : sve_int_dup_imm<0b00, asm, ZPR8, cpy_imm8_opt_lsl_i8>;
  def _H : sve_int_dup_imm<0b01, asm, ZPR16, cpy_imm8_opt_lsl_i16>;
  def _S : sve_int_dup_imm<0b10, asm, ZPR32, cpy_imm8_opt_lsl_i32>;
  def _D : sve_int_dup_imm<0b11, asm, ZPR64, cpy_imm8_opt_lsl_i64>;

  def : InstAlias<"mov $Zd, $imm",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, cpy_imm8_opt_lsl_i8:$imm), 1>;
  def : InstAlias<"mov $Zd, $imm",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, cpy_imm8_opt_lsl_i16:$imm), 1>;
  def : InstAlias<"mov $Zd, $imm",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, cpy_imm8_opt_lsl_i32:$imm), 1>;
  def : InstAlias<"mov $Zd, $imm",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, cpy_imm8_opt_lsl_i64:$imm), 1>;

  def : InstAlias<"fmov $Zd, #0.0",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, 0, 0), 1>;
  def : InstAlias<"fmov $Zd, #0.0",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, 0, 0), 1>;
  def : InstAlias<"fmov $Zd, #0.0",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, 0, 0), 1>;
}

class sve_int_dup_fpimm<bits<2> sz8_64, Operand fpimmtype,
                        string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins fpimmtype:$imm8),
  asm, "\t$Zd, $imm8",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<8> imm8;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-14} = 0b11100111;
  let Inst{13}    = 0b0;
  let Inst{12-5}  = imm8;
  let Inst{4-0}   = Zd;

  let isReMaterializable = 1;
}

multiclass sve_int_dup_fpimm<string asm> {
  def _H : sve_int_dup_fpimm<0b01, fpimm16, asm, ZPR16>;
  def _S : sve_int_dup_fpimm<0b10, fpimm32, asm, ZPR32>;
  def _D : sve_int_dup_fpimm<0b11, fpimm64, asm, ZPR64>;

  def : InstAlias<"fmov $Zd, $imm8",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, fpimm16:$imm8), 1>;
  def : InstAlias<"fmov $Zd, $imm8",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, fpimm32:$imm8), 1>;
  def : InstAlias<"fmov $Zd, $imm8",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, fpimm64:$imm8), 1>;
}

class sve_int_arith_imm0<bits<2> sz8_64, bits<3> opc, string asm,
                         ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, immtype:$imm),
  asm, "\t$Zdn, $_Zdn, $imm",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<9> imm;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-19} = 0b100;
  let Inst{18-16} = opc;
  let Inst{15-14} = 0b11;
  let Inst{13}    = imm{8};   // sh
  let Inst{12-5}  = imm{7-0}; // imm8
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_arith_imm0<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_arith_imm0<0b00, opc, asm, ZPR8,  addsub_imm8_opt_lsl_i8>;
  def _H : sve_int_arith_imm0<0b01, opc, asm, ZPR16, addsub_imm8_opt_lsl_i16>;
  def _S : sve_int_arith_imm0<0b10, opc, asm, ZPR32, addsub_imm8_opt_lsl_i32>;
  def _D : sve_int_arith_imm0<0b11, opc, asm, ZPR64, addsub_imm8_opt_lsl_i64>;

  def : SVE_1_Op_Imm_OptLsl_Pat<nxv16i8, op, ZPR8,  i32, SVEAddSubImm8Pat,  !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Imm_OptLsl_Pat<nxv8i16, op, ZPR16, i32, SVEAddSubImm16Pat, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Imm_OptLsl_Pat<nxv4i32, op, ZPR32, i32, SVEAddSubImm32Pat, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Imm_OptLsl_Pat<nxv2i64, op, ZPR64, i64, SVEAddSubImm64Pat, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_arith_imm0_subr<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_arith_imm0<0b00, opc, asm, ZPR8,  addsub_imm8_opt_lsl_i8>;
  def _H : sve_int_arith_imm0<0b01, opc, asm, ZPR16, addsub_imm8_opt_lsl_i16>;
  def _S : sve_int_arith_imm0<0b10, opc, asm, ZPR32, addsub_imm8_opt_lsl_i32>;
  def _D : sve_int_arith_imm0<0b11, opc, asm, ZPR64, addsub_imm8_opt_lsl_i64>;

  def : SVE_1_Op_Imm_OptLsl_Reverse_Pat<nxv16i8, op, ZPR8,  i32, SVEAddSubImm8Pat,  !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Imm_OptLsl_Reverse_Pat<nxv8i16, op, ZPR16, i32, SVEAddSubImm16Pat, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Imm_OptLsl_Reverse_Pat<nxv4i32, op, ZPR32, i32, SVEAddSubImm32Pat, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Imm_OptLsl_Reverse_Pat<nxv2i64, op, ZPR64, i64, SVEAddSubImm64Pat, !cast<Instruction>(NAME # _D)>;
}

class sve_int_arith_imm<bits<2> sz8_64, bits<6> opc, string asm,
                        ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, immtype:$imm),
  asm, "\t$Zdn, $_Zdn, $imm",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<8> imm;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21-16} = opc;
  let Inst{15-13} = 0b110;
  let Inst{12-5} = imm;
  let Inst{4-0} = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_arith_imm1<bits<2> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_arith_imm<0b00, { 0b1010, opc }, asm, ZPR8, simm8>;
  def _H : sve_int_arith_imm<0b01, { 0b1010, opc }, asm, ZPR16, simm8>;
  def _S : sve_int_arith_imm<0b10, { 0b1010, opc }, asm, ZPR32, simm8>;
  def _D : sve_int_arith_imm<0b11, { 0b1010, opc }, asm, ZPR64, simm8>;

  def : SVE_1_Op_Imm_Arith_All_Active<nxv16i8, nxv16i1, op, ZPR8, i32, SVEArithSImmPat32, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv8i16, nxv8i1,  op, ZPR16, i32, SVEArithSImmPat32, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv4i32, nxv4i1,  op, ZPR32, i32, SVEArithSImmPat32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv2i64, nxv2i1,  op, ZPR64, i64, SVEArithSImmPat64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_arith_imm1_unsigned<bits<2> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_arith_imm<0b00, { 0b1010, opc }, asm, ZPR8, imm0_255>;
  def _H : sve_int_arith_imm<0b01, { 0b1010, opc }, asm, ZPR16, imm0_255>;
  def _S : sve_int_arith_imm<0b10, { 0b1010, opc }, asm, ZPR32, imm0_255>;
  def _D : sve_int_arith_imm<0b11, { 0b1010, opc }, asm, ZPR64, imm0_255>;

  def : SVE_1_Op_Imm_Arith_All_Active<nxv16i8, nxv16i1, op, ZPR8, i32, SVEArithUImm8Pat, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv8i16, nxv8i1, op, ZPR16, i32, SVEArithUImm16Pat, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv4i32, nxv4i1, op, ZPR32, i32, SVEArithUImm32Pat, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv2i64, nxv2i1, op, ZPR64, i64, SVEArithUImm64Pat, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_arith_imm2<string asm, SDPatternOperator op> {
  def _B : sve_int_arith_imm<0b00, 0b110000, asm, ZPR8,  simm8>;
  def _H : sve_int_arith_imm<0b01, 0b110000, asm, ZPR16, simm8>;
  def _S : sve_int_arith_imm<0b10, 0b110000, asm, ZPR32, simm8>;
  def _D : sve_int_arith_imm<0b11, 0b110000, asm, ZPR64, simm8>;

  def : SVE_1_Op_Imm_Arith_All_Active<nxv16i8, nxv16i1, op, ZPR8, i32, SVEArithSImmPat32, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv8i16, nxv8i1, op, ZPR16, i32, SVEArithSImmPat32, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv4i32, nxv4i1, op, ZPR32, i32, SVEArithSImmPat32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Imm_Arith_All_Active<nxv2i64, nxv2i1, op, ZPR64, i64, SVEArithSImmPat64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Bitwise Logical - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve_int_bin_cons_log<bits<2> opc, string asm>
: I<(outs ZPR64:$Zd), (ins ZPR64:$Zn, ZPR64:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc{1-0};
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b001100;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_bin_cons_log<bits<2> opc, string asm, SDPatternOperator op> {
  def NAME : sve_int_bin_cons_log<opc, asm>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, !cast<Instruction>(NAME)>;

  def : InstAlias<asm # "\t$Zd, $Zn, $Zm",
                  (!cast<Instruction>(NAME) ZPR8:$Zd,  ZPR8:$Zn,  ZPR8:$Zm),  1>;
  def : InstAlias<asm # "\t$Zd, $Zn, $Zm",
                  (!cast<Instruction>(NAME) ZPR16:$Zd, ZPR16:$Zn, ZPR16:$Zm), 1>;
  def : InstAlias<asm # "\t$Zd, $Zn, $Zm",
                  (!cast<Instruction>(NAME) ZPR32:$Zd, ZPR32:$Zn, ZPR32:$Zm), 1>;
}

class sve2_int_bitwise_ternary_op_d<bits<3> opc, string asm>
: I<(outs ZPR64:$Zdn), (ins ZPR64:$_Zdn, ZPR64:$Zm, ZPR64:$Zk),
  asm, "\t$Zdn, $_Zdn, $Zm, $Zk",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> Zk;
  bits<5> Zm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc{2-1};
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-11} = 0b00111;
  let Inst{10}    = opc{0};
  let Inst{9-5}   = Zk;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_int_bitwise_ternary_op<bits<3> opc, string asm, SDPatternOperator op> {
  def NAME : sve2_int_bitwise_ternary_op_d<opc, asm>;

  def : InstAlias<asm # "\t$Zdn, $Zdn, $Zm, $Zk",
                  (!cast<Instruction>(NAME) ZPR8:$Zdn,  ZPR8:$Zm,  ZPR8:$Zk),  1>;
  def : InstAlias<asm # "\t$Zdn, $Zdn, $Zm, $Zk",
                  (!cast<Instruction>(NAME) ZPR16:$Zdn, ZPR16:$Zm, ZPR16:$Zk), 1>;
  def : InstAlias<asm # "\t$Zdn, $Zdn, $Zm, $Zk",
                  (!cast<Instruction>(NAME) ZPR32:$Zdn, ZPR32:$Zm, ZPR32:$Zk), 1>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i8, nxv16i8, nxv16i8, !cast<Instruction>(NAME)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i16, nxv8i16, nxv8i16, !cast<Instruction>(NAME)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i32, nxv4i32, nxv4i32, !cast<Instruction>(NAME)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i64, nxv2i64, nxv2i64, !cast<Instruction>(NAME)>;
}

class sve2_int_rotate_right_imm<bits<4> tsz8_64, string asm,
                                ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, zprty:$Zm, immtype:$imm),
  asm, "\t$Zdn, $_Zdn, $Zm, $imm",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> Zm;
  bits<6> imm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = tsz8_64{3-2};
  let Inst{21}    = 0b1;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-16} = imm{2-0}; // imm3
  let Inst{15-10} = 0b001101;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve2_int_rotate_right_imm<string asm, SDPatternOperator op> {
  def _B : sve2_int_rotate_right_imm<{0,0,0,1}, asm, ZPR8, vecshiftR8>;
  def _H : sve2_int_rotate_right_imm<{0,0,1,?}, asm, ZPR16, vecshiftR16> {
    let Inst{19} = imm{3};
  }
  def _S : sve2_int_rotate_right_imm<{0,1,?,?}, asm, ZPR32, vecshiftR32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve2_int_rotate_right_imm<{1,?,?,?}, asm, ZPR64, vecshiftR64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i8, nxv16i8, i32, tvecshiftR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i16, nxv8i16, i32, tvecshiftR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv4i32, i32, tvecshiftR32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i64, nxv2i64, i32, tvecshiftR64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Wide Immediate - Predicated Group
//===----------------------------------------------------------------------===//

class sve_int_dup_fpimm_pred<bits<2> sz, Operand fpimmtype,
                             string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, PPRAny:$Pg, fpimmtype:$imm8),
  asm, "\t$Zd, $Pg/m, $imm8",
  "",
  []>, Sched<[]> {
  bits<4> Pg;
  bits<5> Zd;
  bits<8> imm8;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz;
  let Inst{21-20} = 0b01;
  let Inst{19-16} = Pg;
  let Inst{15-13} = 0b110;
  let Inst{12-5}  = imm8;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_dup_fpimm_pred<string asm> {
  def _H : sve_int_dup_fpimm_pred<0b01, fpimm16, asm, ZPR16>;
  def _S : sve_int_dup_fpimm_pred<0b10, fpimm32, asm, ZPR32>;
  def _D : sve_int_dup_fpimm_pred<0b11, fpimm64, asm, ZPR64>;

  def : InstAlias<"fmov $Zd, $Pg/m, $imm8",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, PPRAny:$Pg, fpimm16:$imm8), 1>;
  def : InstAlias<"fmov $Zd, $Pg/m, $imm8",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, PPRAny:$Pg, fpimm32:$imm8), 1>;
  def : InstAlias<"fmov $Zd, $Pg/m, $imm8",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, PPRAny:$Pg, fpimm64:$imm8), 1>;
}

class sve_int_dup_imm_pred<bits<2> sz8_64, bit m, string asm,
                           ZPRRegOp zprty, string pred_qual, dag iops>
: I<(outs zprty:$Zd), iops,
  asm, "\t$Zd, $Pg"#pred_qual#", $imm",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<4> Pg;
  bits<9> imm;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-20} = 0b01;
  let Inst{19-16} = Pg;
  let Inst{15}    = 0b0;
  let Inst{14}    = m;
  let Inst{13}    = imm{8};   // sh
  let Inst{12-5}  = imm{7-0}; // imm8
  let Inst{4-0}   = Zd;

  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_dup_imm_pred_merge_inst<
    bits<2> sz8_64, string asm, ZPRRegOp zprty, ValueType intty,
    ValueType predty, ValueType scalarty, imm8_opt_lsl cpyimm> {
  let Constraints = "$Zd = $_Zd" in
  def NAME : sve_int_dup_imm_pred<sz8_64, 1, asm, zprty,  "/m",
                                  (ins zprty:$_Zd, PPRAny:$Pg, cpyimm:$imm)>;
  def : InstAlias<"mov $Zd, $Pg/m, $imm",
                  (!cast<Instruction>(NAME) zprty:$Zd, PPRAny:$Pg, cpyimm:$imm), 1>;
  def : Pat<(intty
              (vselect predty:$Pg,
                (intty (AArch64dup (scalarty (SVE8BitLslImm<scalarty>.Pat i32:$imm, i32:$shift)))),
                intty:$Zd)),
            (!cast<Instruction>(NAME) zprty:$Zd, $Pg, i32:$imm, i32:$shift)>;
}

multiclass sve_int_dup_imm_pred_merge<string asm> {
  defm _B : sve_int_dup_imm_pred_merge_inst<0b00, asm, ZPR8, nxv16i8, nxv16i1,
                                            i32, cpy_imm8_opt_lsl_i8>;
  defm _H : sve_int_dup_imm_pred_merge_inst<0b01, asm, ZPR16, nxv8i16, nxv8i1,
                                            i32, cpy_imm8_opt_lsl_i16>;
  defm _S : sve_int_dup_imm_pred_merge_inst<0b10, asm, ZPR32, nxv4i32, nxv4i1,
                                            i32, cpy_imm8_opt_lsl_i32>;
  defm _D : sve_int_dup_imm_pred_merge_inst<0b11, asm, ZPR64, nxv2i64, nxv2i1,
                                            i64, cpy_imm8_opt_lsl_i64>;

  def : InstAlias<"fmov $Zd, $Pg/m, #0.0",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, PPRAny:$Pg, 0, 0), 0>;
  def : InstAlias<"fmov $Zd, $Pg/m, #0.0",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, PPRAny:$Pg, 0, 0), 0>;
  def : InstAlias<"fmov $Zd, $Pg/m, #0.0",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, PPRAny:$Pg, 0, 0), 0>;
}

multiclass sve_int_dup_imm_pred_zero_inst<
    bits<2> sz8_64, string asm, ZPRRegOp zprty, ValueType intty,
    ValueType predty, ValueType scalarty, imm8_opt_lsl cpyimm> {
  def NAME : sve_int_dup_imm_pred<sz8_64, 0, asm, zprty, "/z",
                                  (ins PPRAny:$Pg, cpyimm:$imm)>;
  def : InstAlias<"mov $Zd, $Pg/z, $imm",
                  (!cast<Instruction>(NAME) zprty:$Zd, PPRAny:$Pg, cpyimm:$imm), 1>;
  def : Pat<(intty (zext (predty PPRAny:$Ps1))),
            (!cast<Instruction>(NAME) PPRAny:$Ps1, 1, 0)>;
  def : Pat<(intty (sext (predty PPRAny:$Ps1))),
            (!cast<Instruction>(NAME) PPRAny:$Ps1, -1, 0)>;
  def : Pat<(intty (anyext (predty PPRAny:$Ps1))),
            (!cast<Instruction>(NAME) PPRAny:$Ps1, 1, 0)>;
  def : Pat<(intty
              (vselect predty:$Pg,
                (intty (AArch64dup (scalarty (SVE8BitLslImm<scalarty>.Pat i32:$imm, i32:$shift)))),
                (intty (AArch64dup (scalarty 0))))),
            (!cast<Instruction>(NAME) $Pg, i32:$imm, i32:$shift)>;
}

multiclass sve_int_dup_imm_pred_zero<string asm> {
  defm _B : sve_int_dup_imm_pred_zero_inst<0b00, asm, ZPR8,  nxv16i8, nxv16i1,
                                           i32, cpy_imm8_opt_lsl_i8>;
  defm _H : sve_int_dup_imm_pred_zero_inst<0b01, asm, ZPR16, nxv8i16, nxv8i1,
                                           i32, cpy_imm8_opt_lsl_i16>;
  defm _S : sve_int_dup_imm_pred_zero_inst<0b10, asm, ZPR32, nxv4i32, nxv4i1,
                                           i32, cpy_imm8_opt_lsl_i32>;
  defm _D : sve_int_dup_imm_pred_zero_inst<0b11, asm, ZPR64, nxv2i64, nxv2i1,
                                           i64, cpy_imm8_opt_lsl_i64>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Compare - Vectors Group
//===----------------------------------------------------------------------===//

class sve_int_cmp<bit cmp_1, bits<2> sz8_64, bits<3> opc, string asm,
                  PPRRegOp pprty, ZPRRegOp zprty1, ZPRRegOp zprty2>
: I<(outs pprty:$Pd), (ins PPR3bAny:$Pg, zprty1:$Zn, zprty2:$Zm),
  asm, "\t$Pd, $Pg/z, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<3> Pg;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00100100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15}    = opc{2};
  let Inst{14}    = cmp_1;
  let Inst{13}    = opc{1};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;

  let Defs = [NZCV];
  let ElementSize = pprty.ElementSize;
  let isPTestLike = 1;
}

multiclass SVE_SETCC_Pat<CondCode cc, CondCode invcc, ValueType predvt,
                         ValueType intvt, Instruction cmp> {
  def : Pat<(predvt (AArch64setcc_z predvt:$Op1, intvt:$Op2, intvt:$Op3, cc)),
            (cmp $Op1, $Op2, $Op3)>;
  def : Pat<(predvt (AArch64setcc_z predvt:$Op1, intvt:$Op2, intvt:$Op3, invcc)),
            (cmp $Op1, $Op3, $Op2)>;
  def : Pat<(predvt (and predvt:$Pg, (AArch64setcc_z (predvt (AArch64ptrue 31)), intvt:$Op2, intvt:$Op3, cc))),
            (cmp $Pg, $Op2, $Op3)>;
  def : Pat<(predvt (and predvt:$Pg, (AArch64setcc_z (predvt (AArch64ptrue 31)), intvt:$Op2, intvt:$Op3, invcc))),
            (cmp $Pg, $Op3, $Op2)>;
}

multiclass SVE_SETCC_Pat_With_Zero<CondCode cc, CondCode invcc, ValueType predvt,
                                   ValueType intvt, Instruction cmp> {
  def : Pat<(predvt (AArch64setcc_z predvt:$Op1, intvt:$Op2, (SVEDup0), cc)),
            (cmp $Op1, $Op2)>;
  def : Pat<(predvt (AArch64setcc_z predvt:$Op1, (SVEDup0), intvt:$Op2, invcc)),
            (cmp $Op1, $Op2)>;
}

multiclass sve_int_cmp_0<bits<3> opc, string asm, CondCode cc, CondCode invcc> {
  def _B : sve_int_cmp<0b0, 0b00, opc, asm, PPR8, ZPR8, ZPR8>;
  def _H : sve_int_cmp<0b0, 0b01, opc, asm, PPR16, ZPR16, ZPR16>;
  def _S : sve_int_cmp<0b0, 0b10, opc, asm, PPR32, ZPR32, ZPR32>;
  def _D : sve_int_cmp<0b0, 0b11, opc, asm, PPR64, ZPR64, ZPR64>;

  defm : SVE_SETCC_Pat<cc, invcc, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  defm : SVE_SETCC_Pat<cc, invcc, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat<cc, invcc, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat<cc, invcc, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_cmp_0_wide<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_cmp<0b0, 0b00, opc, asm, PPR8, ZPR8, ZPR64>;
  def _H : sve_int_cmp<0b0, 0b01, opc, asm, PPR16, ZPR16, ZPR64>;
  def _S : sve_int_cmp<0b0, 0b10, opc, asm, PPR32, ZPR32, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i1, op, nxv16i1, nxv16i8, nxv2i64, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i1,  op, nxv8i1,  nxv8i16, nxv2i64, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i1,  op, nxv4i1,  nxv4i32, nxv2i64, !cast<Instruction>(NAME # _S)>;
}

multiclass sve_int_cmp_1_wide<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_cmp<0b1, 0b00, opc, asm, PPR8, ZPR8, ZPR64>;
  def _H : sve_int_cmp<0b1, 0b01, opc, asm, PPR16, ZPR16, ZPR64>;
  def _S : sve_int_cmp<0b1, 0b10, opc, asm, PPR32, ZPR32, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i1, op, nxv16i1, nxv16i8, nxv2i64, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i1,  op, nxv8i1,  nxv8i16, nxv2i64, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i1,  op, nxv4i1,  nxv4i32, nxv2i64, !cast<Instruction>(NAME # _S)>;
}


//===----------------------------------------------------------------------===//
// SVE Integer Compare - Signed Immediate Group
//===----------------------------------------------------------------------===//

class sve_int_scmp_vi<bits<2> sz8_64, bits<3> opc, string asm, PPRRegOp pprty,
                      ZPRRegOp zprty,
                      Operand immtype>
: I<(outs pprty:$Pd), (ins PPR3bAny:$Pg, zprty:$Zn, immtype:$imm5),
  asm, "\t$Pd, $Pg/z, $Zn, $imm5",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<3> Pg;
  bits<5> Zn;
  bits<5> imm5;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b0;
  let Inst{20-16} = imm5;
  let Inst{15}    = opc{2};
  let Inst{14}    = 0b0;
  let Inst{13}    = opc{1};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;

  let Defs = [NZCV];
  let ElementSize = pprty.ElementSize;
  let isPTestLike = 1;
}

multiclass SVE_SETCC_Imm_Pat<CondCode cc, CondCode commuted_cc,
                             ValueType predvt, ValueType intvt,
                             Operand immtype, Instruction cmp> {
  def : Pat<(predvt (AArch64setcc_z (predvt PPR_3b:$Pg),
                                       (intvt ZPR:$Zs1),
                                       (intvt (AArch64dup (immtype:$imm))),
                                       cc)),
            (cmp $Pg, $Zs1, immtype:$imm)>;
  def : Pat<(predvt (AArch64setcc_z (predvt PPR_3b:$Pg),
                                       (intvt (AArch64dup (immtype:$imm))),
                                       (intvt ZPR:$Zs1),
                                       commuted_cc)),
            (cmp $Pg, $Zs1, immtype:$imm)>;
}

multiclass sve_int_scmp_vi<bits<3> opc, string asm, CondCode cc, CondCode commuted_cc> {
  def _B : sve_int_scmp_vi<0b00, opc, asm, PPR8, ZPR8, simm5_32b>;
  def _H : sve_int_scmp_vi<0b01, opc, asm, PPR16, ZPR16, simm5_32b>;
  def _S : sve_int_scmp_vi<0b10, opc, asm, PPR32, ZPR32, simm5_32b>;
  def _D : sve_int_scmp_vi<0b11, opc, asm, PPR64, ZPR64, simm5_64b>;

  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv16i1, nxv16i8, simm5_32b,
                           !cast<Instruction>(NAME # _B)>;
  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv8i1,  nxv8i16, simm5_32b,
                           !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv4i1,  nxv4i32, simm5_32b,
                           !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv2i1,  nxv2i64, simm5_64b,
                           !cast<Instruction>(NAME # _D)>;
}


//===----------------------------------------------------------------------===//
// SVE Integer Compare - Unsigned Immediate Group
//===----------------------------------------------------------------------===//

class sve_int_ucmp_vi<bits<2> sz8_64, bits<2> opc, string asm, PPRRegOp pprty,
                      ZPRRegOp zprty, Operand immtype>
: I<(outs pprty:$Pd), (ins PPR3bAny:$Pg, zprty:$Zn, immtype:$imm7),
  asm, "\t$Pd, $Pg/z, $Zn, $imm7",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<3> Pg;
  bits<5> Zn;
  bits<7> imm7;
  let Inst{31-24} = 0b00100100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 1;
  let Inst{20-14} = imm7;
  let Inst{13}    = opc{1};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;

  let Defs = [NZCV];
  let ElementSize = pprty.ElementSize;
  let isPTestLike = 1;
}

multiclass sve_int_ucmp_vi<bits<2> opc, string asm, CondCode cc,
                           CondCode commuted_cc> {
  def _B : sve_int_ucmp_vi<0b00, opc, asm, PPR8, ZPR8, imm0_127>;
  def _H : sve_int_ucmp_vi<0b01, opc, asm, PPR16, ZPR16, imm0_127>;
  def _S : sve_int_ucmp_vi<0b10, opc, asm, PPR32, ZPR32, imm0_127>;
  def _D : sve_int_ucmp_vi<0b11, opc, asm, PPR64, ZPR64, imm0_127_64b>;

  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv16i1, nxv16i8, imm0_127,
                           !cast<Instruction>(NAME # _B)>;
  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv8i1,  nxv8i16, imm0_127,
                           !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv4i1,  nxv4i32, imm0_127,
                           !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Imm_Pat<cc, commuted_cc, nxv2i1,  nxv2i64, imm0_127_64b,
                           !cast<Instruction>(NAME # _D)>;
}


//===----------------------------------------------------------------------===//
// SVE Integer Compare - Scalars Group
//===----------------------------------------------------------------------===//

class sve_int_cterm<bit sz, bit opc, string asm, RegisterClass rt>
: I<(outs), (ins rt:$Rn, rt:$Rm),
  asm, "\t$Rn, $Rm",
  "",
  []>, Sched<[]> {
  bits<5> Rm;
  bits<5> Rn;
  let Inst{31-23} = 0b001001011;
  let Inst{22}    = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-10} = 0b001000;
  let Inst{9-5}   = Rn;
  let Inst{4}     = opc;
  let Inst{3-0}   = 0b0000;

  let Defs = [NZCV];
}

class sve_int_while_rr<bits<2> sz8_64, bits<4> opc, string asm,
                       RegisterClass gprty, PPRRegOp pprty>
: I<(outs pprty:$Pd), (ins gprty:$Rn, gprty:$Rm),
  asm, "\t$Pd, $Rn, $Rm",
  "", []>, Sched<[]> {
  bits<4> Pd;
  bits<5> Rm;
  bits<5> Rn;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0b000;
  let Inst{12-10} = opc{3-1};
  let Inst{9-5}   = Rn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;

  let Defs = [NZCV];
  let ElementSize = pprty.ElementSize;
  let isWhile = 1;
}

multiclass sve_int_while4_rr<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_while_rr<0b00, { 0, opc }, asm, GPR32, PPR8>;
  def _H : sve_int_while_rr<0b01, { 0, opc }, asm, GPR32, PPR16>;
  def _S : sve_int_while_rr<0b10, { 0, opc }, asm, GPR32, PPR32>;
  def _D : sve_int_while_rr<0b11, { 0, opc }, asm, GPR32, PPR64>;

  def : SVE_2_Op_Pat<nxv16i1, op, i32, i32, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i1,  op, i32, i32, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i1,  op, i32, i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i1,  op, i32, i32, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_while8_rr<bits<3> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_while_rr<0b00, { 1, opc }, asm, GPR64, PPR8>;
  def _H : sve_int_while_rr<0b01, { 1, opc }, asm, GPR64, PPR16>;
  def _S : sve_int_while_rr<0b10, { 1, opc }, asm, GPR64, PPR32>;
  def _D : sve_int_while_rr<0b11, { 1, opc }, asm, GPR64, PPR64>;

  def : SVE_2_Op_Pat<nxv16i1, op, i64, i64, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i1,  op, i64, i64, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i1,  op, i64, i64, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i1,  op, i64, i64, !cast<Instruction>(NAME # _D)>;
}

class sve2_int_while_rr<bits<2> sz8_64, bits<1> rw, string asm,
                        PPRRegOp pprty>
: I<(outs pprty:$Pd), (ins GPR64:$Rn, GPR64:$Rm),
  asm, "\t$Pd, $Rn, $Rm",
  "", []>, Sched<[]> {
  bits<4> Pd;
  bits<5> Rm;
  bits<5> Rn;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-10} = 0b001100;
  let Inst{9-5}   = Rn;
  let Inst{4}     = rw;
  let Inst{3-0}   = Pd;

  let Defs = [NZCV];
  let ElementSize = pprty.ElementSize;
  let isWhile = 1;
}

multiclass sve2_int_while_rr<bits<1> rw, string asm, string op> {
  def _B : sve2_int_while_rr<0b00, rw, asm, PPR8>;
  def _H : sve2_int_while_rr<0b01, rw, asm, PPR16>;
  def _S : sve2_int_while_rr<0b10, rw, asm, PPR32>;
  def _D : sve2_int_while_rr<0b11, rw, asm, PPR64>;

  def : SVE_2_Op_Pat<nxv16i1, !cast<SDPatternOperator>(op # _b), i64, i64, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i1,  !cast<SDPatternOperator>(op # _h), i64, i64, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i1,  !cast<SDPatternOperator>(op # _s), i64, i64, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i1,  !cast<SDPatternOperator>(op # _d), i64, i64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Fast Reduction Group
//===----------------------------------------------------------------------===//

class sve_fp_fast_red<bits<2> sz, bits<3> opc, string asm,
                      ZPRRegOp zprty, FPRasZPROperand dstOpType>
: I<(outs dstOpType:$Vd), (ins PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Vd, $Pg, $Zn",
  "",
  []>, Sched<[]> {
  bits<5> Zn;
  bits<5> Vd;
  bits<3> Pg;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21-19} = 0b000;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Vd;
}

multiclass sve_fp_fast_red<bits<3> opc, string asm, SDPatternOperator op> {
  def _H : sve_fp_fast_red<0b01, opc, asm, ZPR16, FPR16asZPR>;
  def _S : sve_fp_fast_red<0b10, opc, asm, ZPR32, FPR32asZPR>;
  def _D : sve_fp_fast_red<0b11, opc, asm, ZPR64, FPR64asZPR>;

  def : SVE_2_Op_Pat<nxv2f16, op, nxv2i1, nxv2f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4f16, op, nxv4i1, nxv4f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv2f32, op, nxv2i1, nxv2f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Accumulating Reduction Group
//===----------------------------------------------------------------------===//

class sve_fp_2op_p_vd<bits<2> sz, bits<3> opc, string asm,
                      ZPRRegOp zprty, FPRasZPROperand dstOpType>
: I<(outs dstOpType:$Vdn), (ins PPR3bAny:$Pg, dstOpType:$_Vdn, zprty:$Zm),
  asm, "\t$Vdn, $Pg, $_Vdn, $Zm",
  "",
  []>,
  Sched<[]> {
  bits<3> Pg;
  bits<5> Vdn;
  bits<5> Zm;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21-19} = 0b011;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Vdn;

  let Constraints = "$Vdn = $_Vdn";
}

multiclass sve_fp_2op_p_vd<bits<3> opc, string asm, SDPatternOperator op> {
  def _H : sve_fp_2op_p_vd<0b01, opc, asm, ZPR16, FPR16asZPR>;
  def _S : sve_fp_2op_p_vd<0b10, opc, asm, ZPR32, FPR32asZPR>;
  def _D : sve_fp_2op_p_vd<0b11, opc, asm, ZPR64, FPR64asZPR>;

  def : SVE_3_Op_Pat<nxv2f16, op, nxv2i1, nxv2f16, nxv2f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f16, op, nxv4i1, nxv4f16, nxv4f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv2f32, op, nxv2i1, nxv2f32, nxv2f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Compare - Vectors Group
//===----------------------------------------------------------------------===//

class sve_fp_3op_p_pd<bits<2> sz, bits<3> opc, string asm, PPRRegOp pprty,
                      ZPRRegOp zprty>
: I<(outs pprty:$Pd), (ins PPR3bAny:$Pg, zprty:$Zn, zprty:$Zm),
  asm, "\t$Pd, $Pg/z, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<3> Pg;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Zm;
  let Inst{15}    = opc{2};
  let Inst{14}    = 0b1;
  let Inst{13}    = opc{1};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;
}

multiclass sve_fp_3op_p_pd<bits<3> opc, string asm, SDPatternOperator op> {
  def _H : sve_fp_3op_p_pd<0b01, opc, asm, PPR16, ZPR16>;
  def _S : sve_fp_3op_p_pd<0b10, opc, asm, PPR32, ZPR32>;
  def _D : sve_fp_3op_p_pd<0b11, opc, asm, PPR64, ZPR64>;

  def : SVE_3_Op_Pat<nxv8i1, op, nxv8i1, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i1, op, nxv4i1, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i1, op, nxv2i1, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_fp_3op_p_pd_cc<bits<3> opc, string asm,
                              CondCode cc1, CondCode cc2,
                              CondCode invcc1, CondCode invcc2> {
  def _H : sve_fp_3op_p_pd<0b01, opc, asm, PPR16, ZPR16>;
  def _S : sve_fp_3op_p_pd<0b10, opc, asm, PPR32, ZPR32>;
  def _D : sve_fp_3op_p_pd<0b11, opc, asm, PPR64, ZPR64>;

  defm : SVE_SETCC_Pat<cc1, invcc1, nxv8i1,  nxv8f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat<cc1, invcc1, nxv4i1,  nxv4f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat<cc1, invcc1, nxv2i1,  nxv2f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat<cc1, invcc1, nxv4i1,  nxv4f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat<cc1, invcc1, nxv2i1,  nxv2f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat<cc1, invcc1, nxv2i1,  nxv2f64, !cast<Instruction>(NAME # _D)>;

  defm : SVE_SETCC_Pat<cc2, invcc2, nxv8i1,  nxv8f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat<cc2, invcc2, nxv4i1,  nxv4f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat<cc2, invcc2, nxv2i1,  nxv2f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat<cc2, invcc2, nxv4i1,  nxv4f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat<cc2, invcc2, nxv2i1,  nxv2f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat<cc2, invcc2, nxv2i1,  nxv2f64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Compare - with Zero Group
//===----------------------------------------------------------------------===//

class sve_fp_2op_p_pd<bits<2> sz, bits<3> opc, string asm, PPRRegOp pprty,
                      ZPRRegOp zprty>
: I<(outs pprty:$Pd), (ins PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Pd, $Pg/z, $Zn, #0.0",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<3> Pg;
  bits<5> Zn;
  let Inst{31-24} = 0b01100101;
  let Inst{23-22} = sz;
  let Inst{21-18} = 0b0100;
  let Inst{17-16} = opc{2-1};
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;
}

multiclass sve_fp_2op_p_pd<bits<3> opc, string asm,
                           CondCode cc1, CondCode cc2,
                           CondCode invcc1, CondCode invcc2> {
  def _H : sve_fp_2op_p_pd<0b01, opc, asm, PPR16, ZPR16>;
  def _S : sve_fp_2op_p_pd<0b10, opc, asm, PPR32, ZPR32>;
  def _D : sve_fp_2op_p_pd<0b11, opc, asm, PPR64, ZPR64>;

  defm : SVE_SETCC_Pat_With_Zero<cc1, invcc1, nxv8i1,  nxv8f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat_With_Zero<cc1, invcc1, nxv4i1,  nxv4f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat_With_Zero<cc1, invcc1, nxv2i1,  nxv2f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat_With_Zero<cc1, invcc1, nxv4i1,  nxv4f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat_With_Zero<cc1, invcc1, nxv2i1,  nxv2f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat_With_Zero<cc1, invcc1, nxv2i1,  nxv2f64, !cast<Instruction>(NAME # _D)>;

  defm : SVE_SETCC_Pat_With_Zero<cc2, invcc2, nxv8i1,  nxv8f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat_With_Zero<cc2, invcc2, nxv4i1,  nxv4f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat_With_Zero<cc2, invcc2, nxv2i1,  nxv2f16, !cast<Instruction>(NAME # _H)>;
  defm : SVE_SETCC_Pat_With_Zero<cc2, invcc2, nxv4i1,  nxv4f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat_With_Zero<cc2, invcc2, nxv2i1,  nxv2f32, !cast<Instruction>(NAME # _S)>;
  defm : SVE_SETCC_Pat_With_Zero<cc2, invcc2, nxv2i1,  nxv2f64, !cast<Instruction>(NAME # _D)>;
}


//===----------------------------------------------------------------------===//
//SVE Index Generation Group
//===----------------------------------------------------------------------===//

def simm5_8b_tgt : TImmLeaf<i8, [{ return (int8_t)Imm >= -16 && (int8_t)Imm < 16; }]>;
def simm5_16b_tgt : TImmLeaf<i16, [{ return (int16_t)Imm >= -16 && (int16_t)Imm < 16; }]>;
def simm5_32b_tgt : TImmLeaf<i32, [{ return (int32_t)Imm >= -16 && (int32_t)Imm < 16; }]>;
def simm5_64b_tgt : TImmLeaf<i64, [{ return (int64_t)Imm >= -16 && (int64_t)Imm < 16; }]>;
def i64imm_32bit_tgt : TImmLeaf<i64, [{
  return (Imm & 0xffffffffULL) == static_cast<uint64_t>(Imm);
}]>;

class sve_int_index_ii<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                       Operand imm_ty>
: I<(outs zprty:$Zd), (ins imm_ty:$imm5, imm_ty:$imm5b),
  asm, "\t$Zd, $imm5, $imm5b",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> imm5;
  bits<5> imm5b;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = imm5b;
  let Inst{15-10} = 0b010000;
  let Inst{9-5}   = imm5;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_index_ii<string asm> {
  def _B : sve_int_index_ii<0b00, asm, ZPR8, simm5_8b>;
  def _H : sve_int_index_ii<0b01, asm, ZPR16, simm5_16b>;
  def _S : sve_int_index_ii<0b10, asm, ZPR32, simm5_32b>;
  def _D : sve_int_index_ii<0b11, asm, ZPR64, simm5_64b>;

  def : Pat<(nxv16i8 (step_vector simm5_8b_tgt:$imm5b)),
            (!cast<Instruction>(NAME # "_B") (i32 0), (!cast<SDNodeXForm>("trunc_imm") $imm5b))>;
  def : Pat<(nxv8i16 (step_vector simm5_16b_tgt:$imm5b)),
            (!cast<Instruction>(NAME # "_H") (i32 0), (!cast<SDNodeXForm>("trunc_imm") $imm5b))>;
  def : Pat<(nxv4i32 (step_vector simm5_32b_tgt:$imm5b)),
            (!cast<Instruction>(NAME # "_S") (i32 0), simm5_32b:$imm5b)>;
  def : Pat<(nxv2i64 (step_vector simm5_64b_tgt:$imm5b)),
            (!cast<Instruction>(NAME # "_D") (i64 0), simm5_64b:$imm5b)>;

  // add(step_vector(step), dup(X)) -> index(X, step).
  def : Pat<(add (nxv16i8 (step_vector_oneuse simm5_8b_tgt:$imm5b)), (nxv16i8 (AArch64dup(simm5_8b:$imm5)))),
            (!cast<Instruction>(NAME # "_B") simm5_8b:$imm5, (!cast<SDNodeXForm>("trunc_imm") $imm5b))>;
  def : Pat<(add (nxv8i16 (step_vector_oneuse simm5_16b_tgt:$imm5b)), (nxv8i16 (AArch64dup(simm5_16b:$imm5)))),
            (!cast<Instruction>(NAME # "_H") simm5_16b:$imm5, (!cast<SDNodeXForm>("trunc_imm") $imm5b))>;
  def : Pat<(add (nxv4i32 (step_vector_oneuse simm5_32b_tgt:$imm5b)), (nxv4i32 (AArch64dup(simm5_32b:$imm5)))),
            (!cast<Instruction>(NAME # "_S") simm5_32b:$imm5, simm5_32b:$imm5b)>;
  def : Pat<(add (nxv2i64 (step_vector_oneuse simm5_64b_tgt:$imm5b)), (nxv2i64 (AArch64dup(simm5_64b:$imm5)))),
            (!cast<Instruction>(NAME # "_D") simm5_64b:$imm5, simm5_64b:$imm5b)>;
}

class sve_int_index_ir<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                       RegisterClass srcRegType, Operand imm_ty>
: I<(outs zprty:$Zd), (ins imm_ty:$imm5, srcRegType:$Rm),
  asm, "\t$Zd, $imm5, $Rm",
  "", []>, Sched<[]> {
  bits<5> Rm;
  bits<5> Zd;
  bits<5> imm5;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-10} = 0b010010;
  let Inst{9-5}   = imm5;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_index_ir<string asm, SDPatternOperator mulop, SDPatternOperator muloneuseop> {
  def _B : sve_int_index_ir<0b00, asm, ZPR8, GPR32, simm5_8b>;
  def _H : sve_int_index_ir<0b01, asm, ZPR16, GPR32, simm5_16b>;
  def _S : sve_int_index_ir<0b10, asm, ZPR32, GPR32, simm5_32b>;
  def _D : sve_int_index_ir<0b11, asm, ZPR64, GPR64, simm5_64b>;

  def : Pat<(nxv16i8 (step_vector i8:$imm)),
            (!cast<Instruction>(NAME # "_B") (i32 0), (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)))>;
  def : Pat<(nxv8i16 (step_vector i16:$imm)),
            (!cast<Instruction>(NAME # "_H") (i32 0), (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)))>;
  def : Pat<(nxv4i32 (step_vector i32:$imm)),
            (!cast<Instruction>(NAME # "_S") (i32 0), (!cast<Instruction>("MOVi32imm") $imm))>;
  def : Pat<(nxv2i64 (step_vector i64:$imm)),
            (!cast<Instruction>(NAME # "_D") (i64 0), (!cast<Instruction>("MOVi64imm") $imm))>;
  def : Pat<(nxv2i64 (step_vector i64imm_32bit_tgt:$imm)),
            (!cast<Instruction>(NAME # "_D") (i64 0), (SUBREG_TO_REG (i64 0), (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)), sub_32))>;

  // add(step_vector(step), dup(X)) -> index(X, step).
  def : Pat<(add (nxv16i8 (step_vector_oneuse i8:$imm)), (nxv16i8 (AArch64dup(simm5_8b:$imm5)))),
            (!cast<Instruction>(NAME # "_B") simm5_8b:$imm5, (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)))>;
  def : Pat<(add (nxv8i16 (step_vector_oneuse i16:$imm)), (nxv8i16 (AArch64dup(simm5_16b:$imm5)))),
            (!cast<Instruction>(NAME # "_H") simm5_16b:$imm5, (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)))>;
  def : Pat<(add (nxv4i32 (step_vector_oneuse i32:$imm)), (nxv4i32 (AArch64dup(simm5_32b:$imm5)))),
            (!cast<Instruction>(NAME # "_S") simm5_32b:$imm5, (!cast<Instruction>("MOVi32imm") $imm))>;
  def : Pat<(add (nxv2i64 (step_vector_oneuse i64:$imm)), (nxv2i64 (AArch64dup(simm5_64b:$imm5)))),
            (!cast<Instruction>(NAME # "_D") simm5_64b:$imm5, (!cast<Instruction>("MOVi64imm") $imm))>;
  def : Pat<(add (nxv2i64 (step_vector_oneuse i64imm_32bit_tgt:$imm)), (nxv2i64 (AArch64dup(simm5_64b:$imm5)))),
            (!cast<Instruction>(NAME # "_D") simm5_64b:$imm5, (SUBREG_TO_REG (i64 0), (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)), sub_32))>;

  // mul(step_vector(1), dup(Y)) -> index(0, Y).
  def : Pat<(mulop (nxv16i1 (AArch64ptrue 31)), (nxv16i8 (step_vector_oneuse (i8 1))), (nxv16i8 (AArch64dup(i32 GPR32:$Rm)))),
            (!cast<Instruction>(NAME # "_B") (i32 0), GPR32:$Rm)>;
  def : Pat<(mulop (nxv8i1 (AArch64ptrue 31)), (nxv8i16 (step_vector_oneuse (i16 1))), (nxv8i16 (AArch64dup(i32 GPR32:$Rm)))),
            (!cast<Instruction>(NAME # "_H") (i32 0), GPR32:$Rm)>;
  def : Pat<(mulop (nxv4i1 (AArch64ptrue 31)), (nxv4i32 (step_vector_oneuse (i32 1))), (nxv4i32 (AArch64dup(i32 GPR32:$Rm)))),
            (!cast<Instruction>(NAME # "_S") (i32 0), GPR32:$Rm)>;
  def : Pat<(mulop (nxv2i1 (AArch64ptrue 31)), (nxv2i64 (step_vector_oneuse (i64 1))), (nxv2i64 (AArch64dup(i64 GPR64:$Rm)))),
            (!cast<Instruction>(NAME # "_D") (i64 0), GPR64:$Rm)>;

  // add(mul(step_vector(1), dup(Y)), dup(X)) -> index(X, Y).
  def : Pat<(add (muloneuseop (nxv16i1 (AArch64ptrue 31)), (nxv16i8 (step_vector_oneuse (i8 1))), (nxv16i8 (AArch64dup(i32 GPR32:$Rm)))), (nxv16i8 (AArch64dup(simm5_8b:$imm5)))),
            (!cast<Instruction>(NAME # "_B") simm5_8b:$imm5, GPR32:$Rm)>;
  def : Pat<(add (muloneuseop (nxv8i1 (AArch64ptrue 31)), (nxv8i16 (step_vector_oneuse (i16 1))), (nxv8i16 (AArch64dup(i32 GPR32:$Rm)))), (nxv8i16 (AArch64dup(simm5_16b:$imm5)))),
            (!cast<Instruction>(NAME # "_H") simm5_16b:$imm5, GPR32:$Rm)>;
  def : Pat<(add (muloneuseop (nxv4i1 (AArch64ptrue 31)), (nxv4i32 (step_vector_oneuse (i32 1))), (nxv4i32 (AArch64dup(i32 GPR32:$Rm)))), (nxv4i32 (AArch64dup(simm5_32b:$imm5)))),
            (!cast<Instruction>(NAME # "_S") simm5_32b:$imm5, GPR32:$Rm)>;
  def : Pat<(add (muloneuseop (nxv2i1 (AArch64ptrue 31)), (nxv2i64 (step_vector_oneuse (i64 1))), (nxv2i64 (AArch64dup(i64 GPR64:$Rm)))), (nxv2i64 (AArch64dup(simm5_64b:$imm5)))),
            (!cast<Instruction>(NAME # "_D") simm5_64b:$imm5, GPR64:$Rm)>;
}

class sve_int_index_ri<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                       RegisterClass srcRegType, Operand imm_ty>
: I<(outs zprty:$Zd), (ins srcRegType:$Rn, imm_ty:$imm5),
  asm, "\t$Zd, $Rn, $imm5",
  "", []>, Sched<[]> {
  bits<5> Rn;
  bits<5> Zd;
  bits<5> imm5;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = imm5;
  let Inst{15-10} = 0b010001;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_index_ri<string asm> {
  def _B : sve_int_index_ri<0b00, asm, ZPR8, GPR32, simm5_8b>;
  def _H : sve_int_index_ri<0b01, asm, ZPR16, GPR32, simm5_16b>;
  def _S : sve_int_index_ri<0b10, asm, ZPR32, GPR32, simm5_32b>;
  def _D : sve_int_index_ri<0b11, asm, ZPR64, GPR64, simm5_64b>;

  // add(step_vector(step), dup(X)) -> index(X, step).
  def : Pat<(add (nxv16i8 (step_vector_oneuse simm5_8b_tgt:$imm5)), (nxv16i8 (AArch64dup(i32 GPR32:$Rm)))),
            (!cast<Instruction>(NAME # "_B") GPR32:$Rm, (!cast<SDNodeXForm>("trunc_imm") $imm5))>;
  def : Pat<(add (nxv8i16 (step_vector_oneuse simm5_16b_tgt:$imm5)), (nxv8i16 (AArch64dup(i32 GPR32:$Rm)))),
            (!cast<Instruction>(NAME # "_H") GPR32:$Rm, (!cast<SDNodeXForm>("trunc_imm") $imm5))>;
  def : Pat<(add (nxv4i32 (step_vector_oneuse simm5_32b_tgt:$imm5)), (nxv4i32 (AArch64dup(i32 GPR32:$Rm)))),
            (!cast<Instruction>(NAME # "_S") GPR32:$Rm, simm5_32b:$imm5)>;
  def : Pat<(add (nxv2i64 (step_vector_oneuse simm5_64b_tgt:$imm5)), (nxv2i64 (AArch64dup(i64 GPR64:$Rm)))),
            (!cast<Instruction>(NAME # "_D") GPR64:$Rm, simm5_64b:$imm5)>;
}

class sve_int_index_rr<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                       RegisterClass srcRegType>
: I<(outs zprty:$Zd), (ins srcRegType:$Rn, srcRegType:$Rm),
  asm, "\t$Zd, $Rn, $Rm",
  "", []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Rm;
  bits<5> Rn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-10} = 0b010011;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_index_rr<string asm, SDPatternOperator mulop> {
  def _B : sve_int_index_rr<0b00, asm, ZPR8, GPR32>;
  def _H : sve_int_index_rr<0b01, asm, ZPR16, GPR32>;
  def _S : sve_int_index_rr<0b10, asm, ZPR32, GPR32>;
  def _D : sve_int_index_rr<0b11, asm, ZPR64, GPR64>;

  // add(step_vector(step), dup(X)) -> index(X, step).
  def : Pat<(add (nxv16i8 (step_vector_oneuse i8:$imm)), (nxv16i8 (AArch64dup(i32 GPR32:$Rn)))),
            (!cast<Instruction>(NAME # "_B") GPR32:$Rn, (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)))>;
  def : Pat<(add (nxv8i16 (step_vector_oneuse i16:$imm)), (nxv8i16 (AArch64dup(i32 GPR32:$Rn)))),
            (!cast<Instruction>(NAME # "_H") GPR32:$Rn, (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)))>;
  def : Pat<(add (nxv4i32 (step_vector_oneuse i32:$imm)), (nxv4i32 (AArch64dup(i32 GPR32:$Rn)))),
            (!cast<Instruction>(NAME # "_S") GPR32:$Rn, (!cast<Instruction>("MOVi32imm") $imm))>;
  def : Pat<(add (nxv2i64 (step_vector_oneuse i64:$imm)), (nxv2i64 (AArch64dup(i64 GPR64:$Rn)))),
            (!cast<Instruction>(NAME # "_D") GPR64:$Rn, (!cast<Instruction>("MOVi64imm") $imm))>;
  def : Pat<(add (nxv2i64 (step_vector_oneuse i64imm_32bit_tgt:$imm)), (nxv2i64 (AArch64dup(i64 GPR64:$Rn)))),
            (!cast<Instruction>(NAME # "_D") GPR64:$Rn, (SUBREG_TO_REG (i64 0), (!cast<Instruction>("MOVi32imm") (!cast<SDNodeXForm>("trunc_imm") $imm)), sub_32))>;

  // add(mul(step_vector(1), dup(Y)), dup(X)) -> index(X, Y).
  def : Pat<(add (mulop (nxv16i1 (AArch64ptrue 31)), (nxv16i8 (step_vector_oneuse (i8 1))), (nxv16i8 (AArch64dup(i32 GPR32:$Rm)))), (nxv16i8 (AArch64dup(i32 GPR32:$Rn)))),
            (!cast<Instruction>(NAME # "_B") GPR32:$Rn, GPR32:$Rm)>;
  def : Pat<(add (mulop (nxv8i1 (AArch64ptrue 31)), (nxv8i16 (step_vector_oneuse (i16 1))), (nxv8i16 (AArch64dup(i32 GPR32:$Rm)))),(nxv8i16 (AArch64dup(i32 GPR32:$Rn)))),
            (!cast<Instruction>(NAME # "_H") GPR32:$Rn, GPR32:$Rm)>;
  def : Pat<(add (mulop (nxv4i1 (AArch64ptrue 31)), (nxv4i32 (step_vector_oneuse (i32 1))), (nxv4i32 (AArch64dup(i32 GPR32:$Rm)))),(nxv4i32 (AArch64dup(i32 GPR32:$Rn)))),
            (!cast<Instruction>(NAME # "_S") GPR32:$Rn, GPR32:$Rm)>;
  def : Pat<(add (mulop (nxv2i1 (AArch64ptrue 31)), (nxv2i64 (step_vector_oneuse (i64 1))), (nxv2i64 (AArch64dup(i64 GPR64:$Rm)))),(nxv2i64 (AArch64dup(i64 GPR64:$Rn)))),
            (!cast<Instruction>(NAME # "_D") GPR64:$Rn, GPR64:$Rm)>;
}

//===----------------------------------------------------------------------===//
// SVE Bitwise Shift - Predicated Group
//===----------------------------------------------------------------------===//

class sve_int_bin_pred_shift_imm<bits<4> tsz8_64, bits<4> opc, string asm,
                                 ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, immtype:$imm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $imm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<6> imm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = tsz8_64{3-2};
  let Inst{21-20} = 0b00;
  let Inst{19-16} = opc;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-8}   = tsz8_64{1-0};
  let Inst{7-5}   = imm{2-0}; // imm3
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveBinaryImm;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_bin_pred_shift_imm_left<bits<4> opc, string asm, string Ps,
                                           SDPatternOperator op = null_frag> {
  def _B : SVEPseudo2Instr<Ps # _B, 1>,
           sve_int_bin_pred_shift_imm<{0,0,0,1}, opc, asm, ZPR8, vecshiftL8>;
  def _H : SVEPseudo2Instr<Ps # _H, 1>,
           sve_int_bin_pred_shift_imm<{0,0,1,?}, opc, asm, ZPR16, vecshiftL16> {
    let Inst{8} = imm{3};
  }
  def _S : SVEPseudo2Instr<Ps # _S, 1>,
           sve_int_bin_pred_shift_imm<{0,1,?,?}, opc, asm, ZPR32, vecshiftL32> {
    let Inst{9-8} = imm{4-3};
  }
  def _D : SVEPseudo2Instr<Ps # _D, 1>,
           sve_int_bin_pred_shift_imm<{1,?,?,?}, opc, asm, ZPR64, vecshiftL64> {
    let Inst{22}  = imm{5};
    let Inst{9-8} = imm{4-3};
  }

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i1, nxv16i8, i32, tvecshiftL8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i1,  nxv8i16, i32, tvecshiftL16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i1,  nxv4i32, i32, tvecshiftL32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i1,  nxv2i64, i32, tvecshiftL64, !cast<Instruction>(NAME # _D)>;
}

// As above but shift amount takes the form of a "vector immediate".
multiclass sve_int_bin_pred_shift_imm_left_dup<bits<4> opc, string asm,
                                               string Ps, SDPatternOperator op>
: sve_int_bin_pred_shift_imm_left<opc, asm, Ps, null_frag> {
  def : SVE_Shift_DupImm_Pred_Pat<nxv16i8, op, nxv16i1, i32, SVEShiftImmL8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv8i16, op, nxv8i1,  i32, SVEShiftImmL16, !cast<Instruction>(NAME # _H)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv4i32, op, nxv4i1,  i32, SVEShiftImmL32, !cast<Instruction>(NAME # _S)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv2i64, op, nxv2i1,  i64, SVEShiftImmL64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_bin_pred_shift_imm_left_zeroing_bhsd<SDPatternOperator op> {
  def _ZERO_B : PredTwoOpImmPseudo<NAME # _B, ZPR8,  tvecshiftL8,  FalseLanesZero>;
  def _ZERO_H : PredTwoOpImmPseudo<NAME # _H, ZPR16, tvecshiftL16, FalseLanesZero>;
  def _ZERO_S : PredTwoOpImmPseudo<NAME # _S, ZPR32, tvecshiftL32, FalseLanesZero>;
  def _ZERO_D : PredTwoOpImmPseudo<NAME # _D, ZPR64, tvecshiftL64, FalseLanesZero>;

  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv16i8, op, nxv16i1, nxv16i8, tvecshiftL8,  !cast<Pseudo>(NAME # _ZERO_B)>;
  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv8i16, op, nxv8i1,  nxv8i16, tvecshiftL16, !cast<Pseudo>(NAME # _ZERO_H)>;
  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv4i32, op, nxv4i1,  nxv4i32, tvecshiftL32, !cast<Pseudo>(NAME # _ZERO_S)>;
  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv2i64, op, nxv2i1,  nxv2i64, tvecshiftL64, !cast<Pseudo>(NAME # _ZERO_D)>;
}

multiclass sve_int_bin_pred_shift_imm_right<bits<4> opc, string asm, string Ps,
                                            SDPatternOperator op = null_frag> {
  def _B : SVEPseudo2Instr<Ps # _B, 1>,
           sve_int_bin_pred_shift_imm<{0,0,0,1}, opc, asm, ZPR8, vecshiftR8>;
  def _H : SVEPseudo2Instr<Ps # _H, 1>,
           sve_int_bin_pred_shift_imm<{0,0,1,?}, opc, asm, ZPR16, vecshiftR16> {
    let Inst{8} = imm{3};
  }
  def _S : SVEPseudo2Instr<Ps # _S, 1>,
           sve_int_bin_pred_shift_imm<{0,1,?,?}, opc, asm, ZPR32, vecshiftR32> {
    let Inst{9-8} = imm{4-3};
  }
  def _D : SVEPseudo2Instr<Ps # _D, 1>,
           sve_int_bin_pred_shift_imm<{1,?,?,?}, opc, asm, ZPR64, vecshiftR64> {
    let Inst{22}  = imm{5};
    let Inst{9-8} = imm{4-3};
  }

  def : SVE_3_Op_Imm_Pat<nxv16i8, op, nxv16i1, nxv16i8, i32, tvecshiftR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Imm_Pat<nxv8i16, op, nxv8i1,  nxv8i16, i32, tvecshiftR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Imm_Pat<nxv4i32, op, nxv4i1,  nxv4i32, i32, tvecshiftR32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Imm_Pat<nxv2i64, op, nxv2i1,  nxv2i64, i32, tvecshiftR64, !cast<Instruction>(NAME # _D)>;
}

// As above but shift amount takes the form of a "vector immediate".
multiclass sve_int_bin_pred_shift_imm_right_dup<bits<4> opc, string asm,
                                            string Ps, SDPatternOperator op>
: sve_int_bin_pred_shift_imm_right<opc, asm, Ps, null_frag> {
  def : SVE_Shift_DupImm_Pred_Pat<nxv16i8, op, nxv16i1, i32, SVEShiftImmR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv8i16, op, nxv8i1,  i32, SVEShiftImmR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv4i32, op, nxv4i1,  i32, SVEShiftImmR32, !cast<Instruction>(NAME # _S)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv2i64, op, nxv2i1,  i64, SVEShiftImmR64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_bin_pred_shift_imm_right_zeroing_bhsd<SDPatternOperator op = null_frag> {
  def _ZERO_B : PredTwoOpImmPseudo<NAME # _B, ZPR8, vecshiftR8, FalseLanesZero>;
  def _ZERO_H : PredTwoOpImmPseudo<NAME # _H, ZPR16, vecshiftR16, FalseLanesZero>;
  def _ZERO_S : PredTwoOpImmPseudo<NAME # _S, ZPR32, vecshiftR32, FalseLanesZero>;
  def _ZERO_D : PredTwoOpImmPseudo<NAME # _D, ZPR64, vecshiftR64, FalseLanesZero>;

  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv16i8, op, nxv16i1, nxv16i8, tvecshiftR8, !cast<Pseudo>(NAME # _ZERO_B)>;
  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv8i16, op, nxv8i1, nxv8i16, tvecshiftR16, !cast<Pseudo>(NAME # _ZERO_H)>;
  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv4i32, op, nxv4i1, nxv4i32, tvecshiftR32, !cast<Pseudo>(NAME # _ZERO_S)>;
  def : SVE_3_Op_Pat_Shift_Imm_SelZero<nxv2i64, op, nxv2i1, nxv2i64, tvecshiftR64, !cast<Pseudo>(NAME # _ZERO_D)>;
}

class sve_int_bin_pred_shift<bits<2> sz8_64, bit wide, bits<3> opc,
                             string asm, ZPRRegOp zprty, ZPRRegOp zprty2>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty2:$Zm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21-20} = 0b01;
  let Inst{19}    = wide;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_bin_pred_shift<bits<3> opc, string asm, string Ps,
                                  SDPatternOperator op, string revname, bit isReverseInstr = 0> {
  let DestructiveInstType = DestructiveBinaryCommWithRev in {
  def _B : sve_int_bin_pred_shift<0b00, 0b0, opc, asm, ZPR8, ZPR8>,
           SVEPseudo2Instr<Ps # _B, 1>, SVEInstr2Rev<NAME # _B, revname # _B, isReverseInstr>;
  def _H : sve_int_bin_pred_shift<0b01, 0b0, opc, asm, ZPR16, ZPR16>,
           SVEPseudo2Instr<Ps # _H, 1>, SVEInstr2Rev<NAME # _H, revname # _H, isReverseInstr>;
  def _S : sve_int_bin_pred_shift<0b10, 0b0, opc, asm, ZPR32, ZPR32>,
           SVEPseudo2Instr<Ps # _S, 1>, SVEInstr2Rev<NAME # _S, revname # _S, isReverseInstr>;
  def _D : sve_int_bin_pred_shift<0b11, 0b0, opc, asm, ZPR64, ZPR64>,
           SVEPseudo2Instr<Ps # _D, 1>, SVEInstr2Rev<NAME # _D, revname # _D, isReverseInstr>;
  }
  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_bin_pred_zeroing_bhsd<SDPatternOperator op> {
  def _ZERO_B : PredTwoOpPseudo<NAME # _B, ZPR8, FalseLanesZero>;
  def _ZERO_H : PredTwoOpPseudo<NAME # _H, ZPR16, FalseLanesZero>;
  def _ZERO_S : PredTwoOpPseudo<NAME # _S, ZPR32, FalseLanesZero>;
  def _ZERO_D : PredTwoOpPseudo<NAME # _D, ZPR64, FalseLanesZero>;

  def : SVE_3_Op_Pat_SelZero<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Pseudo>(NAME # _ZERO_B)>;
  def : SVE_3_Op_Pat_SelZero<nxv8i16, op, nxv8i1, nxv8i16, nxv8i16, !cast<Pseudo>(NAME # _ZERO_H)>;
  def : SVE_3_Op_Pat_SelZero<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Pseudo>(NAME # _ZERO_S)>;
  def : SVE_3_Op_Pat_SelZero<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Pseudo>(NAME # _ZERO_D)>;
}

multiclass sve_int_bin_pred_shift_wide<bits<3> opc, string asm,
                                  SDPatternOperator op> {
  def _B : sve_int_bin_pred_shift<0b00, 0b1, opc, asm, ZPR8, ZPR64>;
  def _H : sve_int_bin_pred_shift<0b01, 0b1, opc, asm, ZPR16, ZPR64>;
  def _S : sve_int_bin_pred_shift<0b10, 0b1, opc, asm, ZPR32, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv2i64, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, nxv2i64, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, nxv2i64, !cast<Instruction>(NAME # _S)>;
}

//===----------------------------------------------------------------------===//
// SVE Shift - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve_int_bin_cons_shift_wide<bits<2> sz8_64, bits<2> opc, string asm,
                               ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn, ZPR64:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-12} = 0b1000;
  let Inst{11-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_bin_cons_shift_wide<bits<2> opc, string asm, SDPatternOperator op> {
  def _B : sve_int_bin_cons_shift_wide<0b00, opc, asm, ZPR8>;
  def _H : sve_int_bin_cons_shift_wide<0b01, opc, asm, ZPR16>;
  def _S : sve_int_bin_cons_shift_wide<0b10, opc, asm, ZPR32>;

  def : SVE_2_Op_Pred_All_Active<nxv16i8, op, nxv16i1, nxv16i8, nxv2i64, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pred_All_Active<nxv8i16, op, nxv8i1, nxv8i16, nxv2i64, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pred_All_Active<nxv4i32, op, nxv4i1, nxv4i32, nxv2i64, !cast<Instruction>(NAME # _S)>;
}

class sve_int_bin_cons_shift_imm<bits<4> tsz8_64, bits<2> opc, string asm,
                               ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zd), (ins zprty:$Zn, immtype:$imm),
  asm, "\t$Zd, $Zn, $imm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<6> imm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = tsz8_64{3-2};
  let Inst{21}    = 0b1;
  let Inst{20-19} = tsz8_64{1-0};
  let Inst{18-16} = imm{2-0}; // imm3
  let Inst{15-12} = 0b1001;
  let Inst{11-10} = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_bin_cons_shift_imm_left<bits<2> opc, string asm,
                                           SDPatternOperator op> {
  def _B : sve_int_bin_cons_shift_imm<{0,0,0,1}, opc, asm, ZPR8, vecshiftL8>;
  def _H : sve_int_bin_cons_shift_imm<{0,0,1,?}, opc, asm, ZPR16, vecshiftL16> {
    let Inst{19} = imm{3};
  }
  def _S : sve_int_bin_cons_shift_imm<{0,1,?,?}, opc, asm, ZPR32, vecshiftL32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve_int_bin_cons_shift_imm<{1,?,?,?}, opc, asm, ZPR64, vecshiftL64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }

  def : SVE_Shift_DupImm_All_Active_Pat<nxv16i8, op, nxv16i1, i32, SVEShiftImmL8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_Shift_DupImm_All_Active_Pat<nxv8i16, op, nxv8i1,  i32, SVEShiftImmL16, !cast<Instruction>(NAME # _H)>;
  def : SVE_Shift_DupImm_All_Active_Pat<nxv4i32, op, nxv4i1,  i32, SVEShiftImmL32, !cast<Instruction>(NAME # _S)>;
  def : SVE_Shift_DupImm_All_Active_Pat<nxv2i64, op, nxv2i1,  i64, SVEShiftImmL64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_bin_cons_shift_imm_right<bits<2> opc, string asm,
                                            SDPatternOperator op> {
  def _B : sve_int_bin_cons_shift_imm<{0,0,0,1}, opc, asm, ZPR8, vecshiftR8>;
  def _H : sve_int_bin_cons_shift_imm<{0,0,1,?}, opc, asm, ZPR16, vecshiftR16> {
    let Inst{19} = imm{3};
  }
  def _S : sve_int_bin_cons_shift_imm<{0,1,?,?}, opc, asm, ZPR32, vecshiftR32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve_int_bin_cons_shift_imm<{1,?,?,?}, opc, asm, ZPR64, vecshiftR64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }

  def : SVE_Shift_DupImm_All_Active_Pat<nxv16i8, op, nxv16i1, i32, SVEShiftImmR8,  !cast<Instruction>(NAME # _B)>;
  def : SVE_Shift_DupImm_All_Active_Pat<nxv8i16, op, nxv8i1,  i32, SVEShiftImmR16, !cast<Instruction>(NAME # _H)>;
  def : SVE_Shift_DupImm_All_Active_Pat<nxv4i32, op, nxv4i1,  i32, SVEShiftImmR32, !cast<Instruction>(NAME # _S)>;
  def : SVE_Shift_DupImm_All_Active_Pat<nxv2i64, op, nxv2i1,  i64, SVEShiftImmR64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Memory - Store Group
//===----------------------------------------------------------------------===//

class sve_mem_cst_si<bits<2> msz, bits<2> esz, string asm,
                     RegisterOperand VecList>
: I<(outs), (ins VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4),
  asm, "\t$Zt, $Pg, [$Rn, $imm4, mul vl]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zt;
  bits<4> imm4;
  let Inst{31-25} = 0b1110010;
  let Inst{24-23} = msz;
  let Inst{22-21} = esz;
  let Inst{20}    = 0;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b111;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_cst_si<bits<2> msz, bits<2> esz, string asm,
                          RegisterOperand listty, ZPRRegOp zprty>
{
  def NAME : sve_mem_cst_si<msz, esz, asm, listty>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $imm4, mul vl]",
                 (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn]",
                  (!cast<Instruction>(NAME) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_est_si<bits<2> sz, bits<2> nregs, RegisterOperand VecList,
                     string asm, Operand immtype>
: I<(outs), (ins VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, immtype:$imm4),
  asm, "\t$Zt, $Pg, [$Rn, $imm4, mul vl]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zt;
  bits<4> imm4;
  let Inst{31-25} = 0b1110010;
  let Inst{24-23} = sz;
  let Inst{22-21} = nregs;
  let Inst{20}    = 1;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b111;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_est_si<bits<2> sz, bits<2> nregs, RegisterOperand VecList,
                          string asm, Operand immtype> {
  def NAME : sve_mem_est_si<sz, nregs, VecList, asm, immtype>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn]",
                  (!cast<Instruction>(NAME) VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_est_ss<bits<2> sz, bits<2> nregs, RegisterOperand VecList,
                     string asm, RegisterOperand gprty>
: I<(outs), (ins VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$Zt, $Pg, [$Rn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Rn;
  bits<5> Zt;
  let Inst{31-25} = 0b1110010;
  let Inst{24-23} = sz;
  let Inst{22-21} = nregs;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0b011;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

class sve_mem_cst_ss_base<bits<4> dtype, string asm,
                          RegisterOperand listty, RegisterOperand gprty>
: I<(outs), (ins listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$Zt, $Pg, [$Rn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Rn;
  bits<5> Zt;
  let Inst{31-25} = 0b1110010;
  let Inst{24-21} = dtype;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0b010;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_cst_ss<bits<4> dtype, string asm,
                          RegisterOperand listty, ZPRRegOp zprty,
                          RegisterOperand gprty> {
  def NAME : sve_mem_cst_ss_base<dtype, asm, listty, gprty>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Rm]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), 0>;
}

class sve_mem_cstnt_si<bits<2> msz, string asm, RegisterOperand VecList>
: I<(outs), (ins VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4),
  asm, "\t$Zt, $Pg, [$Rn, $imm4, mul vl]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zt;
  bits<4> imm4;
  let Inst{31-25} = 0b1110010;
  let Inst{24-23} = msz;
  let Inst{22-20} = 0b001;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b111;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_cstnt_si<bits<2> msz, string asm, RegisterOperand listty,
                            ZPRRegOp zprty> {
  def NAME : sve_mem_cstnt_si<msz, asm, listty>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $imm4, mul vl]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn]",
                  (!cast<Instruction>(NAME) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_cstnt_ss_base<bits<2> msz, string asm, RegisterOperand listty,
                            RegisterOperand gprty>
: I<(outs), (ins listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$Zt, $Pg, [$Rn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Rn;
  bits<5> Zt;
  let Inst{31-25} = 0b1110010;
  let Inst{24-23} = msz;
  let Inst{22-21} = 0b00;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0b011;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_cstnt_ss<bits<2> msz, string asm, RegisterOperand listty,
                            ZPRRegOp zprty, RegisterOperand gprty> {
  def NAME : sve_mem_cstnt_ss_base<msz, asm, listty, gprty>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Rm]",
                 (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), 0>;
}

class sve2_mem_sstnt_vs_base<bits<3> opc, string asm,
                             RegisterOperand listty, ZPRRegOp zprty>
: I<(outs), (ins listty:$Zt, PPR3bAny:$Pg, zprty:$Zn, GPR64:$Rm),
  asm, "\t$Zt, $Pg, [$Zn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Zn;
  bits<5> Zt;
  let Inst{31-25} = 0b1110010;
  let Inst{24-22} = opc;
  let Inst{21}    = 0b0;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve2_mem_sstnt_vs_32_ptrs<bits<3> opc, string asm,
                             SDPatternOperator op,
                             ValueType vt> {
  def _REAL : sve2_mem_sstnt_vs_base<opc, asm, Z_s, ZPR32>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn, $Rm]",
                 (!cast<Instruction>(NAME # _REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, GPR64:$Rm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, XZR), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) Z_s:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, XZR), 1>;

  def : Pat <(op (nxv4i32 ZPR32:$Zt), (nxv4i1 PPR3bAny:$Pg), (nxv4i32 ZPR32:$Zn), (i64 GPR64:$Rm), vt),
             (!cast<Instruction>(NAME # _REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, GPR64:$Rm)>;
}

multiclass sve2_mem_sstnt_vs_64_ptrs<bits<3> opc, string asm,
                             SDPatternOperator op,
                             ValueType vt> {
  def _REAL : sve2_mem_sstnt_vs_base<opc, asm, Z_d, ZPR64>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn, $Rm]",
                 (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, GPR64:$Rm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, XZR), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) Z_d:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, XZR), 1>;

  def : Pat <(op (nxv2i64 ZPR64:$Zt), (nxv2i1 PPR3bAny:$Pg), (nxv2i64 ZPR64:$Zn), (i64 GPR64:$Rm), vt),
             (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, GPR64:$Rm)>;
}

class sve_mem_sst_sv<bits<3> opc, bit xs, bit scaled, string asm,
                     RegisterOperand VecList, RegisterOperand zprext>
: I<(outs), (ins VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm),
  asm, "\t$Zt, $Pg, [$Rn, $Zm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zm;
  bits<5> Zt;
  let Inst{31-25} = 0b1110010;
  let Inst{24-22} = opc;
  let Inst{21}    = scaled;
  let Inst{20-16} = Zm;
  let Inst{15}    = 0b1;
  let Inst{14}    = xs;
  let Inst{13}    = 0;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_32b_sst_sv_32_scaled<bits<3> opc, string asm,
                                    SDPatternOperator sxtw_op,
                                    SDPatternOperator uxtw_op,
                                    RegisterOperand sxtw_opnd,
                                    RegisterOperand uxtw_opnd,
                                    ValueType vt > {
  def _UXTW_SCALED : sve_mem_sst_sv<opc, 0, 1, asm, Z_s, uxtw_opnd>;
  def _SXTW_SCALED : sve_mem_sst_sv<opc, 1, 1, asm, Z_s, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _UXTW_SCALED) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _SXTW_SCALED) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  def : Pat<(uxtw_op (nxv4i32 ZPR:$data), (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _UXTW_SCALED) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
  def : Pat<(sxtw_op (nxv4i32 ZPR:$data), (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _SXTW_SCALED) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}

multiclass sve_mem_64b_sst_sv_32_scaled<bits<3> opc, string asm,
                                    SDPatternOperator sxtw_op,
                                    SDPatternOperator uxtw_op,
                                    RegisterOperand sxtw_opnd,
                                    RegisterOperand uxtw_opnd,
                                    ValueType vt > {
  def _UXTW_SCALED : sve_mem_sst_sv<opc, 0, 1, asm, Z_d, uxtw_opnd>;
  def _SXTW_SCALED : sve_mem_sst_sv<opc, 1, 1, asm, Z_d, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _UXTW_SCALED) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _SXTW_SCALED) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  def : Pat<(uxtw_op (nxv2i64 ZPR:$data), (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _UXTW_SCALED) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
  def : Pat<(sxtw_op (nxv2i64 ZPR:$data), (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _SXTW_SCALED) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}

multiclass sve_mem_64b_sst_sv_32_unscaled<bits<3> opc, string asm,
                                         SDPatternOperator sxtw_op,
                                         SDPatternOperator uxtw_op,
                                         RegisterOperand sxtw_opnd,
                                         RegisterOperand uxtw_opnd,
                                         ValueType vt> {
  def _UXTW : sve_mem_sst_sv<opc, 0, 0, asm, Z_d, uxtw_opnd>;
  def _SXTW : sve_mem_sst_sv<opc, 1, 0, asm, Z_d, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _UXTW) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _SXTW) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  def : Pat<(uxtw_op (nxv2i64 ZPR:$data), (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _UXTW) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
  def : Pat<(sxtw_op (nxv2i64 ZPR:$data), (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _SXTW) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}

multiclass sve_mem_32b_sst_sv_32_unscaled<bits<3> opc, string asm,
                                          SDPatternOperator sxtw_op,
                                          SDPatternOperator uxtw_op,
                                          RegisterOperand sxtw_opnd,
                                          RegisterOperand uxtw_opnd,
                                          ValueType vt> {
  def _UXTW : sve_mem_sst_sv<opc, 0, 0, asm, Z_s, uxtw_opnd>;
  def _SXTW : sve_mem_sst_sv<opc, 1, 0, asm, Z_s, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _UXTW) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _SXTW) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  def : Pat<(uxtw_op (nxv4i32 ZPR:$data), (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _UXTW) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
  def : Pat<(sxtw_op (nxv4i32 ZPR:$data), (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _SXTW) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}

class sve_mem_sst_sv2<bits<2> msz, bit scaled, string asm,
                      RegisterOperand zprext>
: I<(outs), (ins Z_d:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm),
  asm, "\t$Zt, $Pg, [$Rn, $Zm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zm;
  bits<5> Zt;
  let Inst{31-25} = 0b1110010;
  let Inst{24-23} = msz;
  let Inst{22}    = 0b0;
  let Inst{21}    = scaled;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_sst_sv_64_scaled<bits<2> msz, string asm,
                                    SDPatternOperator op,
                                    RegisterOperand zprext,
                                    ValueType vt> {
  def _SCALED_REAL : sve_mem_sst_sv2<msz, 1, asm, zprext>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _SCALED_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm), 0>;

  def : Pat<(op (nxv2i64 ZPR:$data), (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$indices), vt),
            (!cast<Instruction>(NAME # _SCALED_REAL) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$indices)>;
}

multiclass sve_mem_sst_sv_64_unscaled<bits<2> msz, string asm,
                                      SDPatternOperator op,
                                      ValueType vt> {
  def _REAL : sve_mem_sst_sv2<msz, 0, asm, ZPR64ExtLSL8>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, ZPR64ExtLSL8:$Zm), 0>;

  def : Pat<(op (nxv2i64 ZPR:$data), (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt),
            (!cast<Instruction>(NAME # _REAL) ZPR:$data, PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}

class sve_mem_sst_vi<bits<3> opc, string asm, ZPRRegOp zprty,
                     RegisterOperand VecList, Operand imm_ty>
: I<(outs), (ins VecList:$Zt, PPR3bAny:$Pg, zprty:$Zn, imm_ty:$imm5),
  asm, "\t$Zt, $Pg, [$Zn, $imm5]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> imm5;
  bits<5> Zn;
  bits<5> Zt;
  let Inst{31-25} = 0b1110010;
  let Inst{24-23} = opc{2-1};
  let Inst{22}    = 0b1;
  let Inst{21}    = opc{0};
  let Inst{20-16} = imm5;
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_32b_sst_vi_ptrs<bits<3> opc, string asm,
                                   Operand imm_ty,
                                   SDPatternOperator op,
                                   ValueType vt> {
  def _IMM : sve_mem_sst_vi<opc, asm, ZPR32, Z_s, imm_ty>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn, $imm5]",
                  (!cast<Instruction>(NAME # _IMM) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, imm_ty:$imm5), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM) Z_s:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, 0), 1>;

  def : Pat<(op (nxv4i32 ZPR:$data), (nxv4i1 PPR:$gp), (nxv4i32 ZPR:$ptrs), imm_ty:$index, vt),
            (!cast<Instruction>(NAME # _IMM) ZPR:$data, PPR:$gp, ZPR:$ptrs, imm_ty:$index)>;
}

multiclass sve_mem_64b_sst_vi_ptrs<bits<3> opc, string asm,
                                   Operand imm_ty,
                                   SDPatternOperator op,
                                   ValueType vt> {
  def _IMM : sve_mem_sst_vi<opc, asm, ZPR64, Z_d, imm_ty>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn, $imm5]",
                  (!cast<Instruction>(NAME # _IMM) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, imm_ty:$imm5), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM) Z_d:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, 0), 1>;

  def : Pat<(op (nxv2i64 ZPR:$data), (nxv2i1 PPR:$gp), (nxv2i64 ZPR:$ptrs), imm_ty:$index, vt),
            (!cast<Instruction>(NAME # _IMM) ZPR:$data, PPR:$gp, ZPR:$ptrs, imm_ty:$index)>;
}

class sve_mem_z_spill<string asm>
: I<(outs), (ins ZPRAny:$Zt, GPR64sp:$Rn, simm9:$imm9),
  asm, "\t$Zt, [$Rn, $imm9, mul vl]",
  "",
  []>, Sched<[]> {
  bits<5> Rn;
  bits<5> Zt;
  bits<9> imm9;
  let Inst{31-22} = 0b1110010110;
  let Inst{21-16} = imm9{8-3};
  let Inst{15-13} = 0b010;
  let Inst{12-10} = imm9{2-0};
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayStore = 1;
}

multiclass sve_mem_z_spill<string asm> {
  def NAME : sve_mem_z_spill<asm>;

  def : InstAlias<asm # "\t$Zt, [$Rn]",
                  (!cast<Instruction>(NAME) ZPRAny:$Zt, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_p_spill<string asm>
: I<(outs), (ins PPRAny:$Pt, GPR64sp:$Rn, simm9:$imm9),
  asm, "\t$Pt, [$Rn, $imm9, mul vl]",
  "",
  []>, Sched<[]> {
  bits<4> Pt;
  bits<5> Rn;
  bits<9> imm9;
  let Inst{31-22} = 0b1110010110;
  let Inst{21-16} = imm9{8-3};
  let Inst{15-13} = 0b000;
  let Inst{12-10} = imm9{2-0};
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pt;

  let mayStore = 1;
}

multiclass sve_mem_p_spill<string asm> {
  def NAME : sve_mem_p_spill<asm>;

  def : InstAlias<asm # "\t$Pt, [$Rn]",
                  (!cast<Instruction>(NAME) PPRAny:$Pt, GPR64sp:$Rn, 0), 1>;
}

//===----------------------------------------------------------------------===//
// SVE Permute - Predicates Group
//===----------------------------------------------------------------------===//

class sve_int_perm_bin_perm_pp<bits<3> opc, bits<2> sz8_64, string asm,
                               PPRRegOp pprty>
: I<(outs pprty:$Pd), (ins pprty:$Pn, pprty:$Pm),
  asm, "\t$Pd, $Pn, $Pm",
  "", []>, Sched<[]> {
  bits<4> Pd;
  bits<4> Pm;
  bits<4> Pn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-20} = 0b10;
  let Inst{19-16} = Pm;
  let Inst{15-13} = 0b010;
  let Inst{12-10} = opc;
  let Inst{9}     = 0b0;
  let Inst{8-5}   = Pn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pd;
}

multiclass sve_int_perm_bin_perm_pp<bits<3> opc, string asm,
                                    SDPatternOperator op> {
  def _B : sve_int_perm_bin_perm_pp<opc, 0b00, asm, PPR8>;
  def _H : sve_int_perm_bin_perm_pp<opc, 0b01, asm, PPR16>;
  def _S : sve_int_perm_bin_perm_pp<opc, 0b10, asm, PPR32>;
  def _D : sve_int_perm_bin_perm_pp<opc, 0b11, asm, PPR64>;

  def : SVE_2_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i1, op, nxv8i1,  nxv8i1,  !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i1, op, nxv4i1,  nxv4i1,  !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i1, op, nxv2i1,  nxv2i1,  !cast<Instruction>(NAME # _D)>;
}

class sve_int_perm_punpk<bit opc, string asm>
: I<(outs PPR16:$Pd), (ins PPR8:$Pn),
  asm, "\t$Pd, $Pn",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<4> Pn;
  let Inst{31-17} = 0b000001010011000;
  let Inst{16}    = opc;
  let Inst{15-9}  = 0b0100000;
  let Inst{8-5}   = Pn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pd;
}

multiclass sve_int_perm_punpk<bit opc, string asm, SDPatternOperator op> {
  def NAME : sve_int_perm_punpk<opc, asm>;

  def : SVE_1_Op_Pat<nxv8i1, op, nxv16i1, !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Pat<nxv4i1, op, nxv8i1,  !cast<Instruction>(NAME)>;
  def : SVE_1_Op_Pat<nxv2i1, op, nxv4i1,  !cast<Instruction>(NAME)>;
}

class sve_int_rdffr_pred<bit s, string asm>
: I<(outs PPR8:$Pd), (ins PPRAny:$Pg),
  asm, "\t$Pd, $Pg/z",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<4> Pg;
  let Inst{31-23} = 0b001001010;
  let Inst{22}    = s;
  let Inst{21-9}  = 0b0110001111000;
  let Inst{8-5}   = Pg;
  let Inst{4}     = 0;
  let Inst{3-0}   = Pd;

  let Defs = !if(s, [NZCV], []);
  let Uses = [FFR];
}

multiclass sve_int_rdffr_pred<bit s, string asm, SDPatternOperator op> {
  def _REAL : sve_int_rdffr_pred<s, asm>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def "" : Pseudo<(outs PPR8:$Pd), (ins PPRAny:$Pg), [(set (nxv16i1 PPR8:$Pd), (op (nxv16i1 PPRAny:$Pg)))]>,
           PseudoInstExpansion<(!cast<Instruction>(NAME # _REAL) PPR8:$Pd, PPRAny:$Pg)>;
  }
}

class sve_int_rdffr_unpred<string asm> : I<
  (outs PPR8:$Pd), (ins),
  asm, "\t$Pd",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  let Inst{31-4} = 0b0010010100011001111100000000;
  let Inst{3-0}   = Pd;

  let Uses = [FFR];
}

multiclass sve_int_rdffr_unpred<string asm, SDPatternOperator op> {
  def _REAL : sve_int_rdffr_unpred<asm>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def "" : Pseudo<(outs PPR8:$Pd), (ins), [(set (nxv16i1 PPR8:$Pd), (op))]>,
           PseudoInstExpansion<(!cast<Instruction>(NAME # _REAL) PPR8:$Pd)>;
  }
}

class sve_int_wrffr<string asm, SDPatternOperator op>
: I<(outs), (ins PPR8:$Pn),
  asm, "\t$Pn",
  "",
  [(op (nxv16i1 PPR8:$Pn))]>, Sched<[]> {
  bits<4> Pn;
  let Inst{31-9} = 0b00100101001010001001000;
  let Inst{8-5}  = Pn;
  let Inst{4-0}  = 0b00000;

  let hasSideEffects = 1;
  let Defs = [FFR];
}

class sve_int_setffr<string asm, SDPatternOperator op>
: I<(outs), (ins),
  asm, "",
  "",
  [(op)]>, Sched<[]> {
  let Inst{31-0} = 0b00100101001011001001000000000000;

  let hasSideEffects = 1;
  let Defs = [FFR];
}

//===----------------------------------------------------------------------===//
// SVE Permute Vector - Predicated Group
//===----------------------------------------------------------------------===//

class sve_int_perm_clast_rz<bits<2> sz8_64, bit ab, string asm,
                            ZPRRegOp zprty, RegisterClass rt>
: I<(outs rt:$Rdn), (ins PPR3bAny:$Pg, rt:$_Rdn, zprty:$Zm),
  asm, "\t$Rdn, $Pg, $_Rdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rdn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-17} = 0b11000;
  let Inst{16}    = ab;
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Rdn;

  let Constraints = "$Rdn = $_Rdn";
}

multiclass sve_int_perm_clast_rz<bit ab, string asm, SDPatternOperator op> {
  def _B : sve_int_perm_clast_rz<0b00, ab, asm, ZPR8, GPR32>;
  def _H : sve_int_perm_clast_rz<0b01, ab, asm, ZPR16, GPR32>;
  def _S : sve_int_perm_clast_rz<0b10, ab, asm, ZPR32, GPR32>;
  def _D : sve_int_perm_clast_rz<0b11, ab, asm, ZPR64, GPR64>;

  def : SVE_3_Op_Pat<i32, op, nxv16i1, i32, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<i32, op, nxv8i1,  i32, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<i32, op, nxv4i1,  i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<i64, op, nxv2i1,  i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve_int_perm_clast_vz<bits<2> sz8_64, bit ab, string asm,
                            ZPRRegOp zprty, RegisterClass rt>
: I<(outs rt:$Vdn), (ins PPR3bAny:$Pg, rt:$_Vdn, zprty:$Zm),
  asm, "\t$Vdn, $Pg, $_Vdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Vdn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-17} = 0b10101;
  let Inst{16}    = ab;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Vdn;

  let Constraints = "$Vdn = $_Vdn";
}

multiclass sve_int_perm_clast_vz<bit ab, string asm, SDPatternOperator op> {
  def _B : sve_int_perm_clast_vz<0b00, ab, asm, ZPR8, FPR8>;
  def _H : sve_int_perm_clast_vz<0b01, ab, asm, ZPR16, FPR16>;
  def _S : sve_int_perm_clast_vz<0b10, ab, asm, ZPR32, FPR32>;
  def _D : sve_int_perm_clast_vz<0b11, ab, asm, ZPR64, FPR64>;

  def : SVE_3_Op_Pat<f16, op, nxv8i1, f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<f32, op, nxv4i1, f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<f64, op, nxv2i1, f64, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<bf16, op, nxv8i1, bf16, nxv8bf16, !cast<Instruction>(NAME # _H)>;
}

class sve_int_perm_clast_zz<bits<2> sz8_64, bit ab, string asm,
                            ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $Pg, $_Zdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-17} = 0b10100;
  let Inst{16}    = ab;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_perm_clast_zz<bit ab, string asm, SDPatternOperator op> {
  def _B : sve_int_perm_clast_zz<0b00, ab, asm, ZPR8>;
  def _H : sve_int_perm_clast_zz<0b01, ab, asm, ZPR16>;
  def _S : sve_int_perm_clast_zz<0b10, ab, asm, ZPR32>;
  def _D : sve_int_perm_clast_zz<0b11, ab, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8bf16, op, nxv8i1, nxv8bf16, nxv8bf16, !cast<Instruction>(NAME # _H)>;
}

class sve_int_perm_last_r<bits<2> sz8_64, bit ab, string asm,
                          ZPRRegOp zprty, RegisterClass resultRegType>
: I<(outs resultRegType:$Rd), (ins PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Rd, $Pg, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-17} = 0b10000;
  let Inst{16}    = ab;
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Rd;
}

multiclass sve_int_perm_last_r<bit ab, string asm, SDPatternOperator op> {
  def _B : sve_int_perm_last_r<0b00, ab, asm, ZPR8, GPR32>;
  def _H : sve_int_perm_last_r<0b01, ab, asm, ZPR16, GPR32>;
  def _S : sve_int_perm_last_r<0b10, ab, asm, ZPR32, GPR32>;
  def _D : sve_int_perm_last_r<0b11, ab, asm, ZPR64, GPR64>;

  def : SVE_2_Op_Pat<i32, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<i32, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve_int_perm_last_v<bits<2> sz8_64, bit ab, string asm,
                          ZPRRegOp zprty, RegisterClass dstRegtype>
: I<(outs dstRegtype:$Vd), (ins PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Vd, $Pg, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Vd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-17} = 0b10001;
  let Inst{16}    = ab;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Vd;
}

multiclass sve_int_perm_last_v<bit ab, string asm, SDPatternOperator op> {
  def _B : sve_int_perm_last_v<0b00, ab, asm, ZPR8, FPR8>;
  def _H : sve_int_perm_last_v<0b01, ab, asm, ZPR16, FPR16>;
  def _S : sve_int_perm_last_v<0b10, ab, asm, ZPR32, FPR32>;
  def _D : sve_int_perm_last_v<0b11, ab, asm, ZPR64, FPR64>;

  def : SVE_2_Op_Pat<f16, op, nxv8i1,  nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<f32, op, nxv4i1,  nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<f32, op, nxv2i1,  nxv2f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<f64, op, nxv2i1,  nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_2_Op_Pat<bf16, op, nxv8i1,  nxv8bf16, !cast<Instruction>(NAME # _H)>;
}

class sve_int_perm_splice<bits<2> sz8_64, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $Pg, $_Zdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-13} = 0b101100100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeNone;
}

multiclass sve_int_perm_splice<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_splice<0b00, asm, ZPR8>;
  def _H : sve_int_perm_splice<0b01, asm, ZPR16>;
  def _S : sve_int_perm_splice<0b10, asm, ZPR32>;
  def _D : sve_int_perm_splice<0b11, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1,  nxv8f16, nxv8f16, !cast<Instruction>(NAME # _H)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1,  nxv4f32, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1,  nxv2f64, nxv2f64, !cast<Instruction>(NAME # _D)>;

  def : SVE_3_Op_Pat<nxv8bf16, op, nxv8i1, nxv8bf16, nxv8bf16, !cast<Instruction>(NAME # _H)>;
}

class sve2_int_perm_splice_cons<bits<2> sz8_64, string asm,
                               ZPRRegOp zprty, RegisterOperand VecList>
: I<(outs zprty:$Zd), (ins PPR3bAny:$Pg, VecList:$Zn),
  asm, "\t$Zd, $Pg, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zn;
  bits<5> Zd;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-13} = 0b101101100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_int_perm_splice_cons<string asm> {
  def _B : sve2_int_perm_splice_cons<0b00, asm, ZPR8,  ZZ_b>;
  def _H : sve2_int_perm_splice_cons<0b01, asm, ZPR16, ZZ_h>;
  def _S : sve2_int_perm_splice_cons<0b10, asm, ZPR32, ZZ_s>;
  def _D : sve2_int_perm_splice_cons<0b11, asm, ZPR64, ZZ_d>;
}

class sve_int_perm_rev<bits<2> sz8_64, bits<2> opc, string asm,
                       ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Zd, $Pg/m, $Zn",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<3> Pg;
  bits<5> Zn;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-18} = 0b1001;
  let Inst{17-16} = opc;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_perm_rev_rbit<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_rev<0b00, 0b11, asm, ZPR8>;
  def _H : sve_int_perm_rev<0b01, 0b11, asm, ZPR16>;
  def _S : sve_int_perm_rev<0b10, 0b11, asm, ZPR32>;
  def _D : sve_int_perm_rev<0b11, 0b11, asm, ZPR64>;

  def : SVE_1_Op_Passthru_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_1_Op_Passthru_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_perm_rev_revb<string asm, SDPatternOperator op> {
  def _H : sve_int_perm_rev<0b01, 0b00, asm, ZPR16>;
  def _S : sve_int_perm_rev<0b10, 0b00, asm, ZPR32>;
  def _D : sve_int_perm_rev<0b11, 0b00, asm, ZPR64>;

  def : SVE_1_Op_Passthru_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Passthru_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_perm_rev_revh<string asm, SDPatternOperator op> {
  def _S : sve_int_perm_rev<0b10, 0b01, asm, ZPR32>;
  def _D : sve_int_perm_rev<0b11, 0b01, asm, ZPR64>;

  def : SVE_1_Op_Passthru_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Passthru_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_perm_rev_revw<string asm, SDPatternOperator op> {
  def _D : sve_int_perm_rev<0b11, 0b10, asm, ZPR64>;

  def : SVE_1_Op_Passthru_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve_int_perm_cpy_r<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                         RegisterClass srcRegType>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, PPR3bAny:$Pg, srcRegType:$Rn),
  asm, "\t$Zd, $Pg/m, $Rn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zd;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-13} = 0b101000101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_perm_cpy_r<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_cpy_r<0b00, asm, ZPR8, GPR32sp>;
  def _H : sve_int_perm_cpy_r<0b01, asm, ZPR16, GPR32sp>;
  def _S : sve_int_perm_cpy_r<0b10, asm, ZPR32, GPR32sp>;
  def _D : sve_int_perm_cpy_r<0b11, asm, ZPR64, GPR64sp>;

  def : InstAlias<"mov $Zd, $Pg/m, $Rn",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, PPR3bAny:$Pg, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Rn",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, PPR3bAny:$Pg, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Rn",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, PPR3bAny:$Pg, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Rn",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, PPR3bAny:$Pg, GPR64sp:$Rn), 1>;

  def : Pat<(nxv16i8 (op nxv16i1:$pg, i32:$splat, nxv16i8:$passthru)),
            (!cast<Instruction>(NAME # _B) $passthru, $pg, $splat)>;
  def : Pat<(nxv8i16 (op nxv8i1:$pg, i32:$splat, nxv8i16:$passthru)),
            (!cast<Instruction>(NAME # _H) $passthru, $pg, $splat)>;
  def : Pat<(nxv4i32 (op nxv4i1:$pg, i32:$splat, nxv4i32:$passthru)),
            (!cast<Instruction>(NAME # _S) $passthru, $pg, $splat)>;
  def : Pat<(nxv2i64 (op nxv2i1:$pg, i64:$splat, nxv2i64:$passthru)),
            (!cast<Instruction>(NAME # _D) $passthru, $pg, $splat)>;
}

class sve_int_perm_cpy_v<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                         RegisterClass srcRegtype>
: I<(outs zprty:$Zd), (ins zprty:$_Zd, PPR3bAny:$Pg, srcRegtype:$Vn),
  asm, "\t$Zd, $Pg/m, $Vn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Vn;
  bits<5> Zd;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-13} = 0b100000100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Vn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_perm_cpy_v<string asm, SDPatternOperator op> {
  def _B : sve_int_perm_cpy_v<0b00, asm, ZPR8, FPR8>;
  def _H : sve_int_perm_cpy_v<0b01, asm, ZPR16, FPR16>;
  def _S : sve_int_perm_cpy_v<0b10, asm, ZPR32, FPR32>;
  def _D : sve_int_perm_cpy_v<0b11, asm, ZPR64, FPR64>;

  def : InstAlias<"mov $Zd, $Pg/m, $Vn",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, PPR3bAny:$Pg, FPR8:$Vn), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Vn",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, PPR3bAny:$Pg, FPR16:$Vn), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Vn",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, PPR3bAny:$Pg, FPR32:$Vn), 1>;
  def : InstAlias<"mov $Zd, $Pg/m, $Vn",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, PPR3bAny:$Pg, FPR64:$Vn), 1>;

  def : Pat<(nxv8f16 (op nxv8i1:$pg, f16:$splat, nxv8f16:$passthru)),
            (!cast<Instruction>(NAME # _H) $passthru, $pg, $splat)>;
  def : Pat<(nxv2f32 (op nxv2i1:$pg, f32:$splat, nxv2f32:$passthru)),
            (!cast<Instruction>(NAME # _S) $passthru, $pg, $splat)>;
  def : Pat<(nxv4f32 (op nxv4i1:$pg, f32:$splat, nxv4f32:$passthru)),
            (!cast<Instruction>(NAME # _S) $passthru, $pg, $splat)>;
  def : Pat<(nxv2f64 (op nxv2i1:$pg, f64:$splat, nxv2f64:$passthru)),
            (!cast<Instruction>(NAME # _D) $passthru, $pg, $splat)>;

  def : Pat<(nxv8bf16 (op nxv8i1:$pg, bf16:$splat, nxv8bf16:$passthru)),
            (!cast<Instruction>(NAME # _H) $passthru, $pg, $splat)>;
}

class sve_int_perm_compact<bit sz, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Zd, $Pg, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-23} = 0b000001011;
  let Inst{22}    = sz;
  let Inst{21-13} = 0b100001100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_compact<string asm, SDPatternOperator op> {
  def _S : sve_int_perm_compact<0b0, asm, ZPR32>;
  def _D : sve_int_perm_compact<0b1, asm, ZPR64>;

  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, !cast<Instruction>(NAME # _D)>;
  def : SVE_2_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Memory - Contiguous Load Group
//===----------------------------------------------------------------------===//

class sve_mem_cld_si_base<bits<4> dtype, bit nf, string asm,
                          RegisterOperand VecList>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4),
  asm, "\t$Zt, $Pg/z, [$Rn, $imm4, mul vl]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zt;
  bits<4> imm4;
  let Inst{31-25} = 0b1010010;
  let Inst{24-21} = dtype;
  let Inst{20}    = nf;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
  let Uses = !if(nf, [FFR], []);
  let Defs = !if(nf, [FFR], []);
}

multiclass sve_mem_cld_si_base<bits<4> dtype, bit nf, string asm,
                               RegisterOperand listty, ZPRRegOp zprty> {
  def _REAL : sve_mem_cld_si_base<dtype, nf, asm, listty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME # _REAL) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $imm4, mul vl]",
                  (!cast<Instruction>(NAME # _REAL) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME # _REAL) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1, mayLoad = 1 in {
  def "" : Pseudo<(outs listty:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4), []>,
           PseudoInstExpansion<(!cast<Instruction>(NAME # _REAL) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4)>;
  }
}

multiclass sve_mem_cld_si<bits<4> dtype, string asm, RegisterOperand listty,
                          ZPRRegOp zprty>
: sve_mem_cld_si_base<dtype, 0, asm, listty, zprty>;

class sve_mem_cldnt_si_base<bits<2> msz, string asm, RegisterOperand VecList>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4),
  asm, "\t$Zt, $Pg/z, [$Rn, $imm4, mul vl]",
  "",
  []>, Sched<[]> {
  bits<5> Zt;
  bits<3> Pg;
  bits<5> Rn;
  bits<4> imm4;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = msz;
  let Inst{22-20} = 0b000;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b111;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_cldnt_si<bits<2> msz, string asm, RegisterOperand listty,
                            ZPRRegOp zprty> {
  def NAME : sve_mem_cldnt_si_base<msz, asm, listty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $imm4, mul vl]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s1:$imm4), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_cldnt_ss_base<bits<2> msz, string asm, RegisterOperand VecList,
                            RegisterOperand gprty>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$Zt, $Pg/z, [$Rn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Rn;
  bits<5> Zt;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = msz;
  let Inst{22-21} = 0b00;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0b110;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_cldnt_ss<bits<2> msz, string asm, RegisterOperand listty,
                            ZPRRegOp zprty, RegisterOperand gprty> {
  def NAME : sve_mem_cldnt_ss_base<msz, asm, listty, gprty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Rm]",
                 (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), 0>;
}

class sve_mem_ldqr_si<bits<2> sz, string asm, RegisterOperand VecList>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, simm4s16:$imm4),
  asm, "\t$Zt, $Pg/z, [$Rn, $imm4]", "", []>, Sched<[]> {
  bits<5> Zt;
  bits<5> Rn;
  bits<3> Pg;
  bits<4> imm4;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = sz;
  let Inst{22-20} = 0;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_ldqr_si<bits<2> sz, string asm, RegisterOperand listty,
                           ZPRRegOp zprty> {
  def NAME : sve_mem_ldqr_si<sz, asm, listty>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $imm4]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s16:$imm4), 0>;
}

class sve_mem_ldqr_ss<bits<2> sz, string asm, RegisterOperand VecList,
                      RegisterOperand gprty>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$Zt, $Pg/z, [$Rn, $Rm]", "", []>, Sched<[]> {
  bits<5> Zt;
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Rm;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = sz;
  let Inst{22-21} = 0;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_ldqr_ss<bits<2> sz, string asm, RegisterOperand listty,
                           ZPRRegOp zprty, RegisterOperand gprty> {
  def NAME : sve_mem_ldqr_ss<sz, asm, listty, gprty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Rm]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), 0>;
}

class sve_mem_ld_dup<bits<2> dtypeh, bits<2> dtypel, string asm,
                     RegisterOperand VecList, Operand immtype>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, immtype:$imm6),
  asm, "\t$Zt, $Pg/z, [$Rn, $imm6]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zt;
  bits<6> imm6;
  let Inst{31-25} = 0b1000010;
  let Inst{24-23} = dtypeh;
  let Inst{22}    = 1;
  let Inst{21-16} = imm6;
  let Inst{15}    = 0b1;
  let Inst{14-13} = dtypel;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_ld_dup<bits<2> dtypeh, bits<2> dtypel, string asm,
                          RegisterOperand zlistty, ZPRRegOp zprty, Operand immtype> {
  def NAME : sve_mem_ld_dup<dtypeh, dtypel, asm, zlistty, immtype>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $imm6]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, immtype:$imm6), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) zlistty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_cld_ss_base<bits<4> dtype, bit ff, dag iops, string asm,
                          RegisterOperand VecList>
: I<(outs VecList:$Zt), iops,
  asm, "\t$Zt, $Pg/z, [$Rn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<5> Zt;
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Rn;
  let Inst{31-25} = 0b1010010;
  let Inst{24-21} = dtype;
  let Inst{20-16} = Rm;
  let Inst{15-14} = 0b01;
  let Inst{13}    = ff;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
  let Uses = !if(ff, [FFR], []);
  let Defs = !if(ff, [FFR], []);
}

multiclass sve_mem_cld_ss<bits<4> dtype, string asm, RegisterOperand listty,
                          ZPRRegOp zprty, RegisterOperand gprty> {
  def "" : sve_mem_cld_ss_base<dtype, 0, (ins PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
                               asm, listty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Rm]",
                 (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), 0>;
}

multiclass sve_mem_cldff_ss<bits<4> dtype, string asm, RegisterOperand listty,
                            ZPRRegOp zprty, RegisterOperand gprty> {
  def _REAL : sve_mem_cld_ss_base<dtype, 1, (ins PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
                                  asm, listty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Rm]",
                 (!cast<Instruction>(NAME # _REAL) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), 0>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                 (!cast<Instruction>(NAME # _REAL) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 1>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                 (!cast<Instruction>(NAME # _REAL) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, XZR), 0>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def "" : Pseudo<(outs listty:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), []>,
           PseudoInstExpansion<(!cast<Instruction>(NAME # _REAL) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm)>;
  }
}

multiclass sve_mem_cldnf_si<bits<4> dtype, string asm, RegisterOperand listty,
                            ZPRRegOp zprty>
: sve_mem_cld_si_base<dtype, 1, asm, listty, zprty>;

class sve_mem_eld_si<bits<2> sz, bits<2> nregs, RegisterOperand VecList,
                     string asm, Operand immtype>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, immtype:$imm4),
  asm, "\t$Zt, $Pg/z, [$Rn, $imm4, mul vl]",
  "",
  []>, Sched<[]> {
  bits<5> Zt;
  bits<3> Pg;
  bits<5> Rn;
  bits<4> imm4;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = sz;
  let Inst{22-21} = nregs;
  let Inst{20}    = 0;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b111;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_eld_si<bits<2> sz, bits<2> nregs, RegisterOperand VecList,
                          string asm, Operand immtype> {
  def NAME : sve_mem_eld_si<sz, nregs, VecList, asm, immtype>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_eld_ss<bits<2> sz, bits<2> nregs, RegisterOperand VecList,
                     string asm, RegisterOperand gprty>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$Zt, $Pg/z, [$Rn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Rn;
  bits<5> Zt;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = sz;
  let Inst{22-21} = nregs;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0b110;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

//===----------------------------------------------------------------------===//
// SVE Memory - 32-bit Gather and Unsized Contiguous Group
//===----------------------------------------------------------------------===//

// bit xs      is '1' if offsets are signed
// bit scaled  is '1' if the offsets are scaled
class sve_mem_32b_gld_sv<bits<4> opc, bit xs, bit scaled, string asm,
                         RegisterOperand zprext>
: I<(outs Z_s:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm),
  asm, "\t$Zt, $Pg/z, [$Rn, $Zm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zm;
  bits<5> Zt;
  let Inst{31-25} = 0b1000010;
  let Inst{24-23} = opc{3-2};
  let Inst{22}    = xs;
  let Inst{21}    = scaled;
  let Inst{20-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = opc{1-0};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
  let Defs = !if(!eq(opc{0}, 1), [FFR], []);
  let Uses = !if(!eq(opc{0}, 1), [FFR], []);
}

multiclass sve_mem_32b_gld_sv_32_scaled<bits<4> opc, string asm,
                                        SDPatternOperator sxtw_op,
                                        SDPatternOperator uxtw_op,
                                        RegisterOperand sxtw_opnd,
                                        RegisterOperand uxtw_opnd,
                                        ValueType vt> {
  def _UXTW_SCALED_REAL : sve_mem_32b_gld_sv<opc, 0, 1, asm, uxtw_opnd>;
  def _SXTW_SCALED_REAL : sve_mem_32b_gld_sv<opc, 1, 1, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_SCALED_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_SCALED_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def _UXTW_SCALED : Pseudo<(outs Z_s:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), []>,
                     PseudoInstExpansion<(!cast<Instruction>(NAME # _UXTW_SCALED_REAL) Z_s:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm)>;
  def _SXTW_SCALED : Pseudo<(outs Z_s:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), []>,
                     PseudoInstExpansion<(!cast<Instruction>(NAME # _SXTW_SCALED_REAL) Z_s:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm)>;
  }

  def : Pat<(nxv4i32 (uxtw_op (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$indices), vt)),
            (!cast<Instruction>(NAME # _UXTW_SCALED) PPR:$gp, GPR64sp:$base, ZPR:$indices)>;
  def : Pat<(nxv4i32 (sxtw_op (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$indices), vt)),
            (!cast<Instruction>(NAME # _SXTW_SCALED) PPR:$gp, GPR64sp:$base, ZPR:$indices)>;
}

multiclass sve_mem_32b_gld_vs_32_unscaled<bits<4> opc, string asm,
                                          SDPatternOperator sxtw_op,
                                          SDPatternOperator uxtw_op,
                                          RegisterOperand sxtw_opnd,
                                          RegisterOperand uxtw_opnd,
                                          ValueType vt> {
  def _UXTW_REAL : sve_mem_32b_gld_sv<opc, 0, 0, asm, uxtw_opnd>;
  def _SXTW_REAL : sve_mem_32b_gld_sv<opc, 1, 0, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def _UXTW : Pseudo<(outs Z_s:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), []>,
              PseudoInstExpansion<(!cast<Instruction>(NAME # _UXTW_REAL) Z_s:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm)>;
  def _SXTW : Pseudo<(outs Z_s:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), []>,
              PseudoInstExpansion<(!cast<Instruction>(NAME # _SXTW_REAL) Z_s:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm)>;
  }

  def : Pat<(nxv4i32 (uxtw_op (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$offsets), vt)),
            (!cast<Instruction>(NAME # _UXTW) PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
  def : Pat<(nxv4i32 (sxtw_op (nxv4i1 PPR:$gp), GPR64sp:$base, (nxv4i32 ZPR:$offsets), vt)),
            (!cast<Instruction>(NAME # _SXTW) PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}


class sve_mem_32b_gld_vi<bits<4> opc, string asm, Operand imm_ty>
: I<(outs Z_s:$Zt), (ins PPR3bAny:$Pg, ZPR32:$Zn, imm_ty:$imm5),
  asm, "\t$Zt, $Pg/z, [$Zn, $imm5]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zn;
  bits<5> Zt;
  bits<5> imm5;
  let Inst{31-25} = 0b1000010;
  let Inst{24-23} = opc{3-2};
  let Inst{22-21} = 0b01;
  let Inst{20-16} = imm5;
  let Inst{15}    = 0b1;
  let Inst{14-13} = opc{1-0};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
  let Defs = !if(!eq(opc{0}, 1), [FFR], []);
  let Uses = !if(!eq(opc{0}, 1), [FFR], []);
}

multiclass sve_mem_32b_gld_vi_32_ptrs<bits<4> opc, string asm, Operand imm_ty,
                                      SDPatternOperator op, ValueType vt> {
  def _IMM_REAL : sve_mem_32b_gld_vi<opc, asm, imm_ty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn, $imm5]",
                  (!cast<Instruction>(NAME # _IMM_REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, imm_ty:$imm5), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) Z_s:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, 0), 1>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def _IMM : Pseudo<(outs Z_s:$Zt), (ins PPR3bAny:$Pg, ZPR32:$Zn, imm_ty:$imm5), []>,
             PseudoInstExpansion<(!cast<Instruction>(NAME # _IMM_REAL) Z_s:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, imm_ty:$imm5)>;
  }

  def : Pat<(nxv4i32 (op (nxv4i1 PPR:$gp), (nxv4i32 ZPR:$ptrs), imm_ty:$index, vt)),
            (!cast<Instruction>(NAME # _IMM) PPR:$gp, ZPR:$ptrs, imm_ty:$index)>;
}

class sve_mem_prfm_si<bits<2> msz, string asm>
: I<(outs), (ins sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, simm6s1:$imm6),
  asm, "\t$prfop, $Pg, [$Rn, $imm6, mul vl]",
  "",
  []>, Sched<[]> {
  bits<5> Rn;
  bits<3> Pg;
  bits<6> imm6;
  bits<4> prfop;
  let Inst{31-22} = 0b1000010111;
  let Inst{21-16} = imm6;
  let Inst{15}    = 0b0;
  let Inst{14-13} = msz;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = prfop;

  let hasSideEffects = 1;
}

multiclass sve_mem_prfm_si<bits<2> msz, string asm> {
  def NAME : sve_mem_prfm_si<msz, asm>;

  def : InstAlias<asm # "\t$prfop, $Pg, [$Rn]",
                  (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_prfm_ss<bits<3> opc, string asm, RegisterOperand gprty>
: I<(outs), (ins sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$prfop, $Pg, [$Rn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<5> Rm;
  bits<5> Rn;
  bits<3> Pg;
  bits<4> prfop;
  let Inst{31-25} = 0b1000010;
  let Inst{24-23} = opc{2-1};
  let Inst{22-21} = 0b00;
  let Inst{20-16} = Rm;
  let Inst{15}    = 0b1;
  let Inst{14}    = opc{0};
  let Inst{13}    = 0b0;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = prfop;

  let hasSideEffects = 1;
}

class sve_mem_32b_prfm_sv<bits<2> msz, bit xs, string asm,
                          RegisterOperand zprext>
: I<(outs), (ins sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm),
  asm, "\t$prfop, $Pg, [$Rn, $Zm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zm;
  bits<4> prfop;
  let Inst{31-23} = 0b100001000;
  let Inst{22}    = xs;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15}    = 0b0;
  let Inst{14-13} = msz;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = prfop;

  let hasSideEffects = 1;
}

multiclass sve_mem_32b_prfm_sv_scaled<bits<2> msz, string asm,
                                      RegisterOperand sxtw_opnd,
                                      RegisterOperand uxtw_opnd,
                                      SDPatternOperator op_sxtw,
                                      SDPatternOperator op_uxtw> {
  def _UXTW_SCALED : sve_mem_32b_prfm_sv<msz, 0, asm, uxtw_opnd>;
  def _SXTW_SCALED : sve_mem_32b_prfm_sv<msz, 1, asm, sxtw_opnd>;

  def : Pat<(op_uxtw (nxv4i1 PPR3bAny:$Pg), (i64 GPR64sp:$Rn), (nxv4i32 uxtw_opnd:$Zm), (i32 sve_prfop:$prfop)),
            (!cast<Instruction>(NAME # _UXTW_SCALED) sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm)>;

  def : Pat<(op_sxtw (nxv4i1 PPR3bAny:$Pg), (i64 GPR64sp:$Rn), (nxv4i32 sxtw_opnd:$Zm), (i32 sve_prfop:$prfop)),
            (!cast<Instruction>(NAME # _SXTW_SCALED) sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm)>;
}

class sve_mem_32b_prfm_vi<bits<2> msz, string asm, Operand imm_ty>
: I<(outs), (ins sve_prfop:$prfop, PPR3bAny:$Pg, ZPR32:$Zn, imm_ty:$imm5),
  asm, "\t$prfop, $Pg, [$Zn, $imm5]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zn;
  bits<5> imm5;
  bits<4> prfop;
  let Inst{31-25} = 0b1000010;
  let Inst{24-23} = msz;
  let Inst{22-21} = 0b00;
  let Inst{20-16} = imm5;
  let Inst{15-13} = 0b111;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = prfop;
}

multiclass sve_mem_32b_prfm_vi<bits<2> msz, string asm, Operand imm_ty, SDPatternOperator op> {
  def NAME : sve_mem_32b_prfm_vi<msz, asm, imm_ty>;

  def : InstAlias<asm # "\t$prfop, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR3bAny:$Pg, ZPR32:$Zn, 0), 1>;

  def : Pat<(op (nxv4i1 PPR_3b:$Pg), (nxv4i32 ZPR32:$Zn), (i64 imm_ty:$imm), (i32 sve_prfop:$prfop)),
            (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR_3b:$Pg, ZPR32:$Zn, imm_ty:$imm)>;
}

class sve_mem_z_fill<string asm>
: I<(outs ZPRAny:$Zt), (ins GPR64sp:$Rn, simm9:$imm9),
  asm, "\t$Zt, [$Rn, $imm9, mul vl]",
  "",
  []>, Sched<[]> {
  bits<5> Rn;
  bits<5> Zt;
  bits<9> imm9;
  let Inst{31-22} = 0b1000010110;
  let Inst{21-16} = imm9{8-3};
  let Inst{15-13} = 0b010;
  let Inst{12-10} = imm9{2-0};
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_z_fill<string asm> {
  def NAME : sve_mem_z_fill<asm>;

  def : InstAlias<asm # "\t$Zt, [$Rn]",
                  (!cast<Instruction>(NAME) ZPRAny:$Zt, GPR64sp:$Rn, 0), 1>;
}

class sve_mem_p_fill<string asm>
: I<(outs PPRAny:$Pt), (ins GPR64sp:$Rn, simm9:$imm9),
  asm, "\t$Pt, [$Rn, $imm9, mul vl]",
  "",
  []>, Sched<[]> {
  bits<4> Pt;
  bits<5> Rn;
  bits<9> imm9;
  let Inst{31-22} = 0b1000010110;
  let Inst{21-16} = imm9{8-3};
  let Inst{15-13} = 0b000;
  let Inst{12-10} = imm9{2-0};
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pt;

  let mayLoad = 1;
}

multiclass sve_mem_p_fill<string asm> {
  def NAME : sve_mem_p_fill<asm>;

  def : InstAlias<asm # "\t$Pt, [$Rn]",
                  (!cast<Instruction>(NAME) PPRAny:$Pt, GPR64sp:$Rn, 0), 1>;
}

class sve2_mem_gldnt_vs_base<bits<5> opc, dag iops, string asm,
                             RegisterOperand VecList>
: I<(outs VecList:$Zt), iops,
  asm, "\t$Zt, $Pg/z, [$Zn, $Rm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rm;
  bits<5> Zn;
  bits<5> Zt;
  let Inst{31}    = 0b1;
  let Inst{30}    = opc{4};
  let Inst{29-25} = 0b00010;
  let Inst{24-23} = opc{3-2};
  let Inst{22-21} = 0b00;
  let Inst{20-16} = Rm;
  let Inst{15}    = 0b1;
  let Inst{14-13} = opc{1-0};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve2_mem_gldnt_vs_32_ptrs<bits<5> opc, string asm,
                                  SDPatternOperator op,
                                  ValueType vt> {
  def _REAL : sve2_mem_gldnt_vs_base<opc, (ins PPR3bAny:$Pg, ZPR32:$Zn, GPR64:$Rm),
                                     asm, Z_s>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn, $Rm]",
                 (!cast<Instruction>(NAME # _REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, GPR64:$Rm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, XZR), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) Z_s:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, XZR), 1>;

  def : Pat <(nxv4i32 (op (nxv4i1 PPR3bAny:$Pg), (nxv4i32 ZPR32:$Zd), (i64 GPR64:$Rm), vt)),
             (!cast<Instruction>(NAME # _REAL) PPR3bAny:$Pg, ZPR32:$Zd, GPR64:$Rm)>;
}

multiclass sve2_mem_gldnt_vs_64_ptrs<bits<5> opc, string asm,
                                   SDPatternOperator op,
                                   ValueType vt> {
  def _REAL : sve2_mem_gldnt_vs_base<opc, (ins PPR3bAny:$Pg, ZPR64:$Zn, GPR64:$Rm),
                                     asm, Z_d>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn, $Rm]",
                 (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, GPR64:$Rm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, XZR), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                 (!cast<Instruction>(NAME # _REAL) Z_d:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, XZR), 1>;

  def : Pat <(nxv2i64 (op (nxv2i1 PPR3bAny:$Pg), (nxv2i64 ZPR64:$Zd), (i64 GPR64:$Rm), vt)),
             (!cast<Instruction>(NAME # _REAL) PPR3bAny:$Pg, ZPR64:$Zd, GPR64:$Rm)>;
}

//===----------------------------------------------------------------------===//
// SVE Memory - 64-bit Gather Group
//===----------------------------------------------------------------------===//

// bit xs      is '1' if offsets are signed
// bit scaled  is '1' if the offsets are scaled
// bit lsl     is '0' if the offsets are extended (uxtw/sxtw), '1' if shifted (lsl)
class sve_mem_64b_gld_sv<bits<4> opc, bit xs, bit scaled, bit lsl, string asm,
                         RegisterOperand zprext>
: I<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm),
  asm, "\t$Zt, $Pg/z, [$Rn, $Zm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zm;
  bits<5> Zt;
  let Inst{31-25} = 0b1100010;
  let Inst{24-23} = opc{3-2};
  let Inst{22}    = xs;
  let Inst{21}    = scaled;
  let Inst{20-16} = Zm;
  let Inst{15}    = lsl;
  let Inst{14-13} = opc{1-0};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
  let Defs = !if(!eq(opc{0}, 1), [FFR], []);
  let Uses = !if(!eq(opc{0}, 1), [FFR], []);
}

multiclass sve_mem_64b_gld_sv_32_scaled<bits<4> opc, string asm,
                                        SDPatternOperator sxtw_op,
                                        SDPatternOperator uxtw_op,
                                        RegisterOperand sxtw_opnd,
                                        RegisterOperand uxtw_opnd,
                                        ValueType vt> {
  def _UXTW_SCALED_REAL : sve_mem_64b_gld_sv<opc, 0, 1, 0, asm, uxtw_opnd>;
  def _SXTW_SCALED_REAL : sve_mem_64b_gld_sv<opc, 1, 1, 0, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_SCALED_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_SCALED_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def _UXTW_SCALED : Pseudo<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), []>,
                     PseudoInstExpansion<(!cast<Instruction>(NAME # _UXTW_SCALED_REAL) Z_d:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm)>;
  def _SXTW_SCALED : Pseudo<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), []>,
                     PseudoInstExpansion<(!cast<Instruction>(NAME # _SXTW_SCALED_REAL) Z_d:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm)>;
  }

  def : Pat<(nxv2i64 (uxtw_op (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$indices), vt)),
            (!cast<Instruction>(NAME # _UXTW_SCALED) PPR:$gp, GPR64sp:$base, ZPR:$indices)>;
  def : Pat<(nxv2i64 (sxtw_op (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$indices), vt)),
            (!cast<Instruction>(NAME # _SXTW_SCALED) PPR:$gp, GPR64sp:$base, ZPR:$indices)>;
}

multiclass sve_mem_64b_gld_vs_32_unscaled<bits<4> opc, string asm,
                                          SDPatternOperator sxtw_op,
                                          SDPatternOperator uxtw_op,
                                          RegisterOperand sxtw_opnd,
                                          RegisterOperand uxtw_opnd,
                                          ValueType vt> {
  def _UXTW_REAL : sve_mem_64b_gld_sv<opc, 0, 0, 0, asm, uxtw_opnd>;
  def _SXTW_REAL : sve_mem_64b_gld_sv<opc, 1, 0, 0, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def _UXTW : Pseudo<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), []>,
              PseudoInstExpansion<(!cast<Instruction>(NAME # _UXTW_REAL) Z_d:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm)>;
  def _SXTW : Pseudo<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), []>,
              PseudoInstExpansion<(!cast<Instruction>(NAME # _SXTW_REAL) Z_d:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm)>;
  }

  def : Pat<(nxv2i64 (uxtw_op (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt)),
            (!cast<Instruction>(NAME # _UXTW) PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
  def : Pat<(nxv2i64 (sxtw_op (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt)),
            (!cast<Instruction>(NAME # _SXTW) PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}

multiclass sve_mem_64b_gld_sv2_64_scaled<bits<4> opc, string asm,
                                         SDPatternOperator op,
                                         RegisterOperand zprext, ValueType vt> {
  def _SCALED_REAL : sve_mem_64b_gld_sv<opc, 1, 1, 1, asm, zprext>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SCALED_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm), 0>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def _SCALED : Pseudo<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm), []>,
                PseudoInstExpansion<(!cast<Instruction>(NAME # _SCALED_REAL) Z_d:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm)>;
  }

  def : Pat<(nxv2i64 (op (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$indices), vt)),
                     (!cast<Instruction>(NAME # _SCALED) PPR:$gp, GPR64sp:$base, ZPR:$indices)>;
}

multiclass sve_mem_64b_gld_vs2_64_unscaled<bits<4> opc, string asm,
                                           SDPatternOperator op, ValueType vt> {
  def _REAL : sve_mem_64b_gld_sv<opc, 1, 0, 1, asm, ZPR64ExtLSL8>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, ZPR64ExtLSL8:$Zm), 0>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def "" : Pseudo<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, ZPR64ExtLSL8:$Zm), []>,
           PseudoInstExpansion<(!cast<Instruction>(NAME # _REAL) Z_d:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, ZPR64ExtLSL8:$Zm)>;
  }

  def : Pat<(nxv2i64 (op (nxv2i1 PPR:$gp), GPR64sp:$base, (nxv2i64 ZPR:$offsets), vt)),
            (!cast<Instruction>(NAME) PPR:$gp, GPR64sp:$base, ZPR:$offsets)>;
}

class sve_mem_64b_gld_vi<bits<4> opc, string asm, Operand imm_ty>
: I<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, ZPR64:$Zn, imm_ty:$imm5),
  asm, "\t$Zt, $Pg/z, [$Zn, $imm5]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zn;
  bits<5> Zt;
  bits<5> imm5;
  let Inst{31-25} = 0b1100010;
  let Inst{24-23} = opc{3-2};
  let Inst{22-21} = 0b01;
  let Inst{20-16} = imm5;
  let Inst{15}    = 0b1;
  let Inst{14-13} = opc{1-0};
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
  let Defs = !if(!eq(opc{0}, 1), [FFR], []);
  let Uses = !if(!eq(opc{0}, 1), [FFR], []);
}

multiclass sve_mem_64b_gld_vi_64_ptrs<bits<4> opc, string asm, Operand imm_ty,
                                      SDPatternOperator op, ValueType vt> {
  def _IMM_REAL : sve_mem_64b_gld_vi<opc, asm, imm_ty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn, $imm5]",
                 (!cast<Instruction>(NAME # _IMM_REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, imm_ty:$imm5), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) Z_d:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, 0), 1>;

  // We need a layer of indirection because early machine code passes balk at
  // physical register (i.e. FFR) uses that have no previous definition.
  let hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def _IMM : Pseudo<(outs Z_d:$Zt), (ins PPR3bAny:$Pg, ZPR64:$Zn, imm_ty:$imm5), []>,
                  PseudoInstExpansion<(!cast<Instruction>(NAME # _IMM_REAL) Z_d:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, imm_ty:$imm5)>;
  }

  def : Pat<(nxv2i64 (op (nxv2i1 PPR:$gp), (nxv2i64 ZPR:$ptrs), imm_ty:$index, vt)),
            (!cast<Instruction>(NAME # _IMM) PPR:$gp, ZPR:$ptrs, imm_ty:$index)>;
}

// bit lsl is '0' if the offsets are extended (uxtw/sxtw), '1' if shifted (lsl)
class sve_mem_64b_prfm_sv<bits<2> msz, bit xs, bit lsl, string asm,
                          RegisterOperand zprext>
: I<(outs), (ins sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm),
  asm, "\t$prfop, $Pg, [$Rn, $Zm]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Zm;
  bits<4> prfop;
  let Inst{31-23} = 0b110001000;
  let Inst{22}    = xs;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15}    = lsl;
  let Inst{14-13} = msz;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = prfop;

  let hasSideEffects = 1;
}

multiclass sve_mem_64b_prfm_sv_ext_scaled<bits<2> msz, string asm,
                                          RegisterOperand sxtw_opnd,
                                          RegisterOperand uxtw_opnd,
                                          SDPatternOperator op_sxtw,
                                          SDPatternOperator op_uxtw> {
  def _UXTW_SCALED : sve_mem_64b_prfm_sv<msz, 0, 0, asm, uxtw_opnd>;
  def _SXTW_SCALED : sve_mem_64b_prfm_sv<msz, 1, 0, asm, sxtw_opnd>;

  def : Pat<(op_uxtw (nxv2i1 PPR3bAny:$Pg), (i64 GPR64sp:$Rn), (nxv2i64 uxtw_opnd:$Zm), (i32 sve_prfop:$prfop)),
            (!cast<Instruction>(NAME # _UXTW_SCALED) sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm)>;

  def : Pat<(op_sxtw (nxv2i1 PPR3bAny:$Pg), (i64 GPR64sp:$Rn), (nxv2i64 sxtw_opnd:$Zm), (i32 sve_prfop:$prfop)),
            (!cast<Instruction>(NAME # _SXTW_SCALED) sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm)>;

}

multiclass sve_mem_64b_prfm_sv_lsl_scaled<bits<2> msz, string asm,
                                          RegisterOperand zprext, SDPatternOperator frag> {
  def NAME : sve_mem_64b_prfm_sv<msz, 1, 1, asm, zprext>;

  def : Pat<(frag (nxv2i1 PPR3bAny:$Pg), (i64 GPR64sp:$Rn), (nxv2i64 zprext:$Zm), (i32 sve_prfop:$prfop)),
            (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm)>;

}

class sve_mem_64b_prfm_vi<bits<2> msz, string asm, Operand imm_ty>
: I<(outs), (ins sve_prfop:$prfop, PPR3bAny:$Pg, ZPR64:$Zn, imm_ty:$imm5),
  asm, "\t$prfop, $Pg, [$Zn, $imm5]",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zn;
  bits<5> imm5;
  bits<4> prfop;
  let Inst{31-25} = 0b1100010;
  let Inst{24-23} = msz;
  let Inst{22-21} = 0b00;
  let Inst{20-16} = imm5;
  let Inst{15-13} = 0b111;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = prfop;

  let hasSideEffects = 1;
}

multiclass sve_mem_64b_prfm_vi<bits<2> msz, string asm, Operand imm_ty, SDPatternOperator op> {
  def NAME : sve_mem_64b_prfm_vi<msz, asm, imm_ty>;

  def : InstAlias<asm # "\t$prfop, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR3bAny:$Pg, ZPR64:$Zn, 0), 1>;

  def : Pat<(op (nxv2i1 PPR_3b:$Pg), (nxv2i64 ZPR32:$Zn), (i64 imm_ty:$imm), (i32 sve_prfop:$prfop)),
            (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR_3b:$Pg, ZPR32:$Zn, imm_ty:$imm)>;
}

//===----------------------------------------------------------------------===//
// SVE Compute Vector Address Group
//===----------------------------------------------------------------------===//

class sve_int_bin_cons_misc_0_a<bits<2> opc, bits<2> msz, string asm,
                                ZPRRegOp zprty, RegisterOperand zprext>
: I<(outs zprty:$Zd), (ins zprty:$Zn, zprext:$Zm),
  asm, "\t$Zd, [$Zn, $Zm]",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-12} = 0b1010;
  let Inst{11-10} = msz;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_bin_cons_misc_0_a_uxtw<bits<2> opc, string asm> {
  def _0 : sve_int_bin_cons_misc_0_a<opc, 0b00, asm, ZPR64, ZPR64ExtUXTW8>;
  def _1 : sve_int_bin_cons_misc_0_a<opc, 0b01, asm, ZPR64, ZPR64ExtUXTW16>;
  def _2 : sve_int_bin_cons_misc_0_a<opc, 0b10, asm, ZPR64, ZPR64ExtUXTW32>;
  def _3 : sve_int_bin_cons_misc_0_a<opc, 0b11, asm, ZPR64, ZPR64ExtUXTW64>;
}

multiclass sve_int_bin_cons_misc_0_a_sxtw<bits<2> opc, string asm> {
  def _0 : sve_int_bin_cons_misc_0_a<opc, 0b00, asm, ZPR64, ZPR64ExtSXTW8>;
  def _1 : sve_int_bin_cons_misc_0_a<opc, 0b01, asm, ZPR64, ZPR64ExtSXTW16>;
  def _2 : sve_int_bin_cons_misc_0_a<opc, 0b10, asm, ZPR64, ZPR64ExtSXTW32>;
  def _3 : sve_int_bin_cons_misc_0_a<opc, 0b11, asm, ZPR64, ZPR64ExtSXTW64>;
}

multiclass sve_int_bin_cons_misc_0_a_32_lsl<bits<2> opc, string asm> {
  def _0 : sve_int_bin_cons_misc_0_a<opc, 0b00, asm, ZPR32, ZPR32ExtLSL8>;
  def _1 : sve_int_bin_cons_misc_0_a<opc, 0b01, asm, ZPR32, ZPR32ExtLSL16>;
  def _2 : sve_int_bin_cons_misc_0_a<opc, 0b10, asm, ZPR32, ZPR32ExtLSL32>;
  def _3 : sve_int_bin_cons_misc_0_a<opc, 0b11, asm, ZPR32, ZPR32ExtLSL64>;
}

multiclass sve_int_bin_cons_misc_0_a_64_lsl<bits<2> opc, string asm> {
  def _0 : sve_int_bin_cons_misc_0_a<opc, 0b00, asm, ZPR64, ZPR64ExtLSL8>;
  def _1 : sve_int_bin_cons_misc_0_a<opc, 0b01, asm, ZPR64, ZPR64ExtLSL16>;
  def _2 : sve_int_bin_cons_misc_0_a<opc, 0b10, asm, ZPR64, ZPR64ExtLSL32>;
  def _3 : sve_int_bin_cons_misc_0_a<opc, 0b11, asm, ZPR64, ZPR64ExtLSL64>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Misc - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve_int_bin_cons_misc_0_b<bits<2> sz, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b101100;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_bin_cons_misc_0_b<string asm, SDPatternOperator op> {
  def _H : sve_int_bin_cons_misc_0_b<0b01, asm, ZPR16>;
  def _S : sve_int_bin_cons_misc_0_b<0b10, asm, ZPR32>;
  def _D : sve_int_bin_cons_misc_0_b<0b11, asm, ZPR64>;

  def : SVE_2_Op_Pat<nxv8f16, op, nxv8f16, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4f32, op, nxv4f32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2f64, op, nxv2f64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve_int_bin_cons_misc_0_c<bits<8> opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn),
  asm, "\t$Zd, $Zn",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = opc{7-6};
  let Inst{21}    = 0b1;
  let Inst{20-16} = opc{5-1};
  let Inst{15-11} = 0b10111;
  let Inst{10}    = opc{0};
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_bin_cons_misc_0_c_fexpa<string asm, SDPatternOperator op> {
  def _H : sve_int_bin_cons_misc_0_c<0b01000000, asm, ZPR16>;
  def _S : sve_int_bin_cons_misc_0_c<0b10000000, asm, ZPR32>;
  def _D : sve_int_bin_cons_misc_0_c<0b11000000, asm, ZPR64>;

  def : SVE_1_Op_Pat<nxv8f16, op, nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_1_Op_Pat<nxv4f32, op, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_1_Op_Pat<nxv2f64, op, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Reduction Group
//===----------------------------------------------------------------------===//

class sve_int_reduce<bits<2> sz8_32, bits<2> fmt, bits<3> opc, string asm,
                     ZPRRegOp zprty, FPRasZPROperand dstOpType>
: I<(outs dstOpType:$Vd), (ins PPR3bAny:$Pg, zprty:$Zn),
  asm, "\t$Vd, $Pg, $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Vd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_32;
  let Inst{21}    = 0b0;
  let Inst{20-19} = fmt;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Vd;
}

multiclass sve_int_reduce_0_saddv<bits<3> opc, string asm,
                                  SDPatternOperator op> {
  def _B : sve_int_reduce<0b00, 0b00, opc, asm, ZPR8, FPR64asZPR>;
  def _H : sve_int_reduce<0b01, 0b00, opc, asm, ZPR16, FPR64asZPR>;
  def _S : sve_int_reduce<0b10, 0b00, opc, asm, ZPR32, FPR64asZPR>;

  def : SVE_2_Op_Pat<nxv2i64, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
}

multiclass sve_int_reduce_0_uaddv<bits<3> opc, string asm,
                                  SDPatternOperator op> {
  def _B : sve_int_reduce<0b00, 0b00, opc, asm, ZPR8, FPR64asZPR>;
  def _H : sve_int_reduce<0b01, 0b00, opc, asm, ZPR16, FPR64asZPR>;
  def _S : sve_int_reduce<0b10, 0b00, opc, asm, ZPR32, FPR64asZPR>;
  def _D : sve_int_reduce<0b11, 0b00, opc, asm, ZPR64, FPR64asZPR>;

  def : SVE_2_Op_Pat<nxv2i64, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_reduce_1<bits<3> opc, string asm,
                            SDPatternOperator op> {
  def _B : sve_int_reduce<0b00, 0b01, opc, asm, ZPR8, FPR8asZPR>;
  def _H : sve_int_reduce<0b01, 0b01, opc, asm, ZPR16, FPR16asZPR>;
  def _S : sve_int_reduce<0b10, 0b01, opc, asm, ZPR32, FPR32asZPR>;
  def _D : sve_int_reduce<0b11, 0b01, opc, asm, ZPR64, FPR64asZPR>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

multiclass sve_int_reduce_2<bits<3> opc, string asm,
                            SDPatternOperator op> {
  def _B : sve_int_reduce<0b00, 0b11, opc, asm, ZPR8, FPR8asZPR>;
  def _H : sve_int_reduce<0b01, 0b11, opc, asm, ZPR16, FPR16asZPR>;
  def _S : sve_int_reduce<0b10, 0b11, opc, asm, ZPR32, FPR32asZPR>;
  def _D : sve_int_reduce<0b11, 0b11, opc, asm, ZPR64, FPR64asZPR>;

  def : SVE_2_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_2_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, !cast<Instruction>(NAME # _H)>;
  def : SVE_2_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_2_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, !cast<Instruction>(NAME # _D)>;
}

class sve_int_movprfx_pred<bits<2> sz8_32, bits<3> opc, string asm,
                           ZPRRegOp zprty, string pg_suffix, dag iops>
: I<(outs zprty:$Zd), iops,
  asm, "\t$Zd, $Pg"#pg_suffix#", $Zn",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zd;
  bits<5> Zn;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_32;
  let Inst{21-19} = 0b010;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let ElementSize = zprty.ElementSize;
}

multiclass sve_int_movprfx_pred_merge<bits<3> opc, string asm> {
let Constraints = "$Zd = $_Zd" in {
  def _B : sve_int_movprfx_pred<0b00, opc, asm, ZPR8, "/m",
                                (ins ZPR8:$_Zd, PPR3bAny:$Pg, ZPR8:$Zn)>;
  def _H : sve_int_movprfx_pred<0b01, opc, asm, ZPR16, "/m",
                                (ins ZPR16:$_Zd, PPR3bAny:$Pg, ZPR16:$Zn)>;
  def _S : sve_int_movprfx_pred<0b10, opc, asm, ZPR32, "/m",
                                (ins ZPR32:$_Zd, PPR3bAny:$Pg, ZPR32:$Zn)>;
  def _D : sve_int_movprfx_pred<0b11, opc, asm, ZPR64, "/m",
                                (ins ZPR64:$_Zd, PPR3bAny:$Pg, ZPR64:$Zn)>;
}
}

multiclass sve_int_movprfx_pred_zero<bits<3> opc, string asm> {
  def _B : sve_int_movprfx_pred<0b00, opc, asm, ZPR8, "/z",
                                (ins PPR3bAny:$Pg, ZPR8:$Zn)>;
  def _H : sve_int_movprfx_pred<0b01, opc, asm, ZPR16, "/z",
                                (ins PPR3bAny:$Pg, ZPR16:$Zn)>;
  def _S : sve_int_movprfx_pred<0b10, opc, asm, ZPR32, "/z",
                                (ins PPR3bAny:$Pg, ZPR32:$Zn)>;
  def _D : sve_int_movprfx_pred<0b11, opc, asm, ZPR64, "/z",
                                (ins PPR3bAny:$Pg, ZPR64:$Zn)>;
}

//===----------------------------------------------------------------------===//
// SVE Propagate Break Group
//===----------------------------------------------------------------------===//

class sve_int_brkp<bits<2> opc, string asm>
: I<(outs PPR8:$Pd), (ins PPRAny:$Pg, PPR8:$Pn, PPR8:$Pm),
  asm, "\t$Pd, $Pg/z, $Pn, $Pm",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<4> Pg;
  bits<4> Pm;
  bits<4> Pn;
  let Inst{31-24} = 0b00100101;
  let Inst{23}    = 0b0;
  let Inst{22}    = opc{1};
  let Inst{21-20} = 0b00;
  let Inst{19-16} = Pm;
  let Inst{15-14} = 0b11;
  let Inst{13-10} = Pg;
  let Inst{9}     = 0b0;
  let Inst{8-5}   = Pn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;

  let Defs = !if(!eq (opc{1}, 1), [NZCV], []);
}

multiclass sve_int_brkp<bits<2> opc, string asm, SDPatternOperator op> {
  def NAME : sve_int_brkp<opc, asm>;

  def : SVE_3_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, nxv16i1, !cast<Instruction>(NAME)>;
}


//===----------------------------------------------------------------------===//
// SVE Partition Break Group
//===----------------------------------------------------------------------===//

class sve_int_brkn<bit S, string asm>
: I<(outs PPR8:$Pdm), (ins PPRAny:$Pg, PPR8:$Pn, PPR8:$_Pdm),
  asm, "\t$Pdm, $Pg/z, $Pn, $_Pdm",
  "",
  []>, Sched<[]> {
  bits<4> Pdm;
  bits<4> Pg;
  bits<4> Pn;
  let Inst{31-23} = 0b001001010;
  let Inst{22}    = S;
  let Inst{21-14} = 0b01100001;
  let Inst{13-10} = Pg;
  let Inst{9}     = 0b0;
  let Inst{8-5}   = Pn;
  let Inst{4}     = 0b0;
  let Inst{3-0}   = Pdm;

  let Constraints = "$Pdm = $_Pdm";
  let Defs = !if(S, [NZCV], []);
}

multiclass sve_int_brkn<bits<1> opc, string asm, SDPatternOperator op> {
  def NAME : sve_int_brkn<opc, asm>;

  def : SVE_3_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, nxv16i1, !cast<Instruction>(NAME)>;
}

class sve_int_break<bits<3> opc, string asm, string suffix, dag iops>
: I<(outs PPR8:$Pd), iops,
  asm, "\t$Pd, $Pg"#suffix#", $Pn",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<4> Pg;
  bits<4> Pn;
  let Inst{31-24} = 0b00100101;
  let Inst{23-22} = opc{2-1};
  let Inst{21-14} = 0b01000001;
  let Inst{13-10} = Pg;
  let Inst{9}     = 0b0;
  let Inst{8-5}   = Pn;
  let Inst{4}     = opc{0};
  let Inst{3-0}   = Pd;

  let Constraints = !if(!eq (opc{0}, 1), "$Pd = $_Pd", "");
  let Defs = !if(!eq (opc{1}, 1), [NZCV], []);

}

multiclass sve_int_break_m<bits<3> opc, string asm, SDPatternOperator op> {
  def NAME : sve_int_break<opc, asm, "/m", (ins PPR8:$_Pd, PPRAny:$Pg, PPR8:$Pn)>;

  def : SVE_3_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, nxv16i1, !cast<Instruction>(NAME)>;
}

multiclass sve_int_break_z<bits<3> opc, string asm, SDPatternOperator op> {
  def NAME : sve_int_break<opc, asm, "/z", (ins PPRAny:$Pg, PPR8:$Pn)>;

  def : SVE_2_Op_Pat<nxv16i1, op, nxv16i1, nxv16i1, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE2 String Processing Group
//===----------------------------------------------------------------------===//

class sve2_char_match<bit sz, bit opc, string asm,
                      PPRRegOp pprty, ZPRRegOp zprty>
: I<(outs pprty:$Pd), (ins PPR3bAny:$Pg, zprty:$Zn, zprty:$Zm),
  asm, "\t$Pd, $Pg/z, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<4> Pd;
  bits<3> Pg;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-23} = 0b010001010;
  let Inst{22}    = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4}     = opc;
  let Inst{3-0}   = Pd;

  let Defs = [NZCV];
  let isPTestLike = 1;
}

multiclass sve2_char_match<bit opc, string asm, SDPatternOperator op> {
  def _B : sve2_char_match<0b0, opc, asm, PPR8, ZPR8>;
  def _H : sve2_char_match<0b1, opc, asm, PPR16, ZPR16>;

  def : SVE_3_Op_Pat<nxv16i1, op, nxv16i1, nxv16i8, nxv16i8, !cast<Instruction>(NAME # _B)>;
  def : SVE_3_Op_Pat<nxv8i1,  op, nxv8i1,  nxv8i16, nxv8i16, !cast<Instruction>(NAME # _H)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Histogram Computation - Segment Group
//===----------------------------------------------------------------------===//

class sve2_hist_gen_segment<string asm, SDPatternOperator op>
: I<(outs ZPR8:$Zd), (ins ZPR8:$Zn, ZPR8:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  [(set nxv16i8:$Zd, (op nxv16i8:$Zn, nxv16i8:$Zm))]>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-21} = 0b01000101001;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b101000;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

//===----------------------------------------------------------------------===//
// SVE2 Histogram Computation - Vector Group
//===----------------------------------------------------------------------===//

class sve2_hist_gen_vector<bit sz, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins PPR3bAny:$Pg, zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Pg/z, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<3> Pg;
  bits<5> Zm;
  let Inst{31-23} = 0b010001011;
  let Inst{22}    = sz;
  let Inst{21}    = 0b1;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b110;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_hist_gen_vector<string asm, SDPatternOperator op> {
  def _S : sve2_hist_gen_vector<0b0, asm, ZPR32>;
  def _D : sve2_hist_gen_vector<0b1, asm, ZPR64>;

  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Instruction>(NAME # _S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Instruction>(NAME # _D)>;
}

//===----------------------------------------------------------------------===//
// SVE2 Crypto Extensions Group
//===----------------------------------------------------------------------===//

class sve2_crypto_cons_bin_op<bit opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zd), (ins zprty:$Zn, zprty:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-21} = 0b01000101001;
  let Inst{20-16} = Zm;
  let Inst{15-11} = 0b11110;
  let Inst{10}    = opc;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve2_crypto_cons_bin_op<bit opc, string asm, ZPRRegOp zprty,
                                   SDPatternOperator op, ValueType vt> {
  def NAME : sve2_crypto_cons_bin_op<opc, asm, zprty>;
  def : SVE_2_Op_Pat<vt, op, vt, vt, !cast<Instruction>(NAME)>;
}

class sve2_crypto_des_bin_op<bits<2> opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $_Zdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-17} = 0b010001010010001;
  let Inst{16}    = opc{1};
  let Inst{15-11} = 0b11100;
  let Inst{10}    = opc{0};
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
}

multiclass sve2_crypto_des_bin_op<bits<2> opc, string asm, ZPRRegOp zprty,
                                  SDPatternOperator op, ValueType vt> {
  def NAME : sve2_crypto_des_bin_op<opc, asm, zprty>;
  def : SVE_2_Op_Pat<vt, op, vt, vt, !cast<Instruction>(NAME)>;
}

class sve2_crypto_unary_op<bit opc, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins zprty:$_Zdn),
  asm, "\t$Zdn, $_Zdn",
  "",
  []>, Sched<[]> {
  bits<5> Zdn;
  let Inst{31-11} = 0b010001010010000011100;
  let Inst{10}    = opc;
  let Inst{9-5}   = 0b00000;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
}

multiclass sve2_crypto_unary_op<bit opc, string asm, SDPatternOperator op> {
  def NAME : sve2_crypto_unary_op<opc, asm, ZPR8>;
  def : SVE_1_Op_Pat<nxv16i8, op, nxv16i8, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE BFloat16 Group
//===----------------------------------------------------------------------===//

class sve_bfloat_dot_base<bits<2> opc, string asm, string ops, dag iops>
: I<(outs ZPR32:$Zda), iops, asm, ops, "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  let Inst{31-21} = 0b01100100011;
  let Inst{15-14} = opc;
  let Inst{13-10} = 0b0000;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeH;
}

class sve_bfloat_dot<string asm>
: sve_bfloat_dot_base<0b10, asm, "\t$Zda, $Zn, $Zm",
  (ins ZPR32:$_Zda, ZPR16:$Zn, ZPR16:$Zm)> {
  bits<5> Zm;
  let Inst{20-16} = Zm;
}

multiclass sve_bfloat_dot<string asm, SDPatternOperator op> {
  def NAME : sve_bfloat_dot<asm>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4f32, nxv8bf16, nxv8bf16 ,!cast<Instruction>(NAME)>;
}

class sve_bfloat_dot_indexed<string asm>
: sve_bfloat_dot_base<0b01, asm, "\t$Zda, $Zn, $Zm$iop",
  (ins ZPR32:$_Zda, ZPR16:$Zn, ZPR3b16:$Zm, VectorIndexS:$iop)> {
  bits<2> iop;
  bits<3> Zm;
  let Inst{20-19} = iop;
  let Inst{18-16} = Zm;
}

multiclass sve_bfloat_dot_indexed<string asm, SDPatternOperator op> {
  def NAME : sve_bfloat_dot_indexed<asm>;
  def : SVE_4_Op_Imm_Pat<nxv4f32, op, nxv4f32, nxv8bf16, nxv8bf16, i64, VectorIndexS_timm, !cast<Instruction>(NAME)>;
}

class sve_bfloat_matmul<string asm>
: I<(outs ZPR32:$Zda), (ins ZPR32:$_Zda, ZPR16:$Zn, ZPR16:$Zm),
  asm, "\t$Zda, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zm;
  bits<5> Zda;
  bits<5> Zn;
  let Inst{31-21} = 0b01100100011;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b111001;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ElementSizeH;
}

multiclass sve_bfloat_matmul<string asm, SDPatternOperator op> {
  def NAME : sve_bfloat_matmul<asm>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4f32, nxv8bf16, nxv8bf16 ,!cast<Instruction>(NAME)>;
}

class sve_bfloat_matmul_longvecl<bit BT, string asm>
: sve_bfloat_matmul<asm> {
  let Inst{23}    = 0b1;
  let Inst{14-13} = 0b00;
  let Inst{10}    = BT;
}

multiclass sve_bfloat_matmul_longvecl<bit BT, string asm, SDPatternOperator op> {
  def NAME : sve_bfloat_matmul_longvecl<BT, asm>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4f32, nxv8bf16, nxv8bf16 ,!cast<Instruction>(NAME)>;
}

class sve_bfloat_matmul_longvecl_idx<bit BT, string asm>
: sve_bfloat_dot_base<0b01, asm, "\t$Zda, $Zn, $Zm$iop",
  (ins ZPR32:$_Zda, ZPR16:$Zn, ZPR3b16:$Zm, VectorIndexH:$iop)> {
  bits<3> iop;
  bits<3> Zm;
  let Inst{23}    = 0b1;
  let Inst{20-19} = iop{2-1};
  let Inst{18-16} = Zm;
  let Inst{11}    = iop{0};
  let Inst{10}    = BT;
}

multiclass sve_bfloat_matmul_longvecl_idx<bit BT, string asm, SDPatternOperator op> {
  def NAME : sve_bfloat_matmul_longvecl_idx<BT, asm>;
  def : SVE_4_Op_Imm_Pat<nxv4f32, op, nxv4f32, nxv8bf16, nxv8bf16, i64, VectorIndexH_timm, !cast<Instruction>(NAME)>;
}

class sve_bfloat_convert<bit N, string asm>
: I<(outs ZPR16:$Zd), (ins ZPR16:$_Zd, PPR3bAny:$Pg, ZPR32:$Zn),
  asm, "\t$Zd, $Pg/m, $Zn", "", []>, Sched<[]> {
  bits<5> Zd;
  bits<3> Pg;
  bits<5> Zn;
  let Inst{31-25} = 0b0110010;
  let Inst{24}    = N;
  let Inst{23-13} = 0b10001010101;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;

  let Constraints = "$Zd = $_Zd";
  let DestructiveInstType = DestructiveOther;
  let hasSideEffects = 1;
  let ElementSize = ElementSizeS;
}

multiclass sve_bfloat_convert<bit N, string asm, SDPatternOperator op> {
  def NAME : sve_bfloat_convert<N, asm>;
  def : SVE_3_Op_Pat<nxv8bf16, op, nxv8bf16, nxv8i1, nxv4f32, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Matrix Multiply Group
//===----------------------------------------------------------------------===//

class sve_int_matmul<bits<2> uns, string asm>
: I<(outs ZPR32:$Zda), (ins ZPR32:$_Zda, ZPR8:$Zn, ZPR8:$Zm), asm,
  "\t$Zda, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-24} = 0b01000101;
  let Inst{23-22} = uns;
  let Inst{21}    = 0;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b100110;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ZPR32.ElementSize;
}

multiclass sve_int_matmul<bits<2> uns, string asm, SDPatternOperator op> {
  def NAME : sve_int_matmul<uns, asm>;

  def : SVE_3_Op_Pat<nxv4i32, op , nxv4i32, nxv16i8, nxv16i8, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Dot Product Mixed Sign Group
//===----------------------------------------------------------------------===//

class sve_int_dot_mixed<string asm>
: I<(outs ZPR32:$Zda), (ins ZPR32:$_Zda, ZPR8:$Zn, ZPR8:$Zm), asm,
  "\t$Zda, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-21} = 0b01000100100;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b011110;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ZPR32.ElementSize;
}

multiclass sve_int_dot_mixed<string asm, SDPatternOperator op> {
  def NAME : sve_int_dot_mixed<asm>;

  def : SVE_3_Op_Pat<nxv4i32, op , nxv4i32, nxv16i8, nxv16i8, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE Integer Dot Product Mixed Sign - Indexed Group
//===----------------------------------------------------------------------===//

class sve_int_dot_mixed_indexed<bit U, string asm>
: I<(outs ZPR32:$Zda), (ins ZPR32:$_Zda, ZPR8:$Zn, ZPR3b8:$Zm, VectorIndexS32b:$idx),
    asm, "\t$Zda, $Zn, $Zm$idx", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<3> Zm;
  bits<2> idx;
  let Inst{31-21} = 0b01000100101;
  let Inst{20-19} = idx;
  let Inst{18-16} = Zm;
  let Inst{15-11} = 0b00011;
  let Inst{10}    = U;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = ZPR32.ElementSize;
}

multiclass sve_int_dot_mixed_indexed<bit U, string asm, SDPatternOperator op> {
  def NAME : sve_int_dot_mixed_indexed<U, asm>;

  def : SVE_4_Op_Imm_Pat<nxv4i32, op, nxv4i32, nxv16i8, nxv16i8, i32, VectorIndexS32b_timm, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE Floating Point Matrix Multiply Accumulate Group
//===----------------------------------------------------------------------===//

class sve_fp_matrix_mla<bit sz, string asm, ZPRRegOp zprty>
: I<(outs zprty:$Zda), (ins zprty:$_Zda, zprty:$Zn, zprty:$Zm),
    asm, "\t$Zda, $Zn, $Zm", "", []>, Sched<[]> {
  bits<5> Zda;
  bits<5> Zn;
  bits<5> Zm;
  let Inst{31-23} = 0b011001001;
  let Inst{22}    = sz;
  let Inst{21}    = 1;
  let Inst{20-16} = Zm;
  let Inst{15-10} = 0b111001;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zda;

  let Constraints = "$Zda = $_Zda";
  let DestructiveInstType = DestructiveOther;
  let ElementSize = zprty.ElementSize;
}

multiclass sve_fp_matrix_mla<bit sz, string asm, ZPRRegOp zprty, SDPatternOperator op, ValueType vt> {
  def NAME : sve_fp_matrix_mla<sz, asm, zprty>;

  def : SVE_3_Op_Pat<vt, op , vt, vt, vt, !cast<Instruction>(NAME)>;
}

//===----------------------------------------------------------------------===//
// SVE Memory - Contiguous Load And Replicate 256-bit Group
//===----------------------------------------------------------------------===//

class sve_mem_ldor_si<bits<2> sz, string asm, RegisterOperand VecList>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, simm4s32:$imm4),
  asm, "\t$Zt, $Pg/z, [$Rn, $imm4]", "", []>, Sched<[]> {
  bits<5> Zt;
  bits<5> Rn;
  bits<3> Pg;
  bits<4> imm4;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = sz;
  let Inst{22-20} = 0b010;
  let Inst{19-16} = imm4;
  let Inst{15-13} = 0b001;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_ldor_si<bits<2> sz, string asm, RegisterOperand listty,
                           ZPRRegOp zprty, ValueType Ty, ValueType PredTy, SDNode Ld1ro> {
  def NAME : sve_mem_ldor_si<sz, asm, listty>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $imm4]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4s32:$imm4), 0>;

  // Base addressing mode
  def : Pat<(Ty (Ld1ro (PredTy PPR3bAny:$Pg), GPR64sp:$base)),
            (!cast<Instruction>(NAME) PPR3bAny:$Pg, GPR64sp:$base, (i64 0))>;
  let AddedComplexity = 2 in {
    // Reg + Imm addressing mode
    def : Pat<(Ty (Ld1ro (PredTy PPR3bAny:$Pg), (add GPR64:$base, (i64 simm4s32:$imm)))),
              (!cast<Instruction>(NAME) $Pg, $base, simm4s32:$imm)>;
  }
}

class sve_mem_ldor_ss<bits<2> sz, string asm, RegisterOperand VecList,
                      RegisterOperand gprty>
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm),
  asm, "\t$Zt, $Pg/z, [$Rn, $Rm]", "", []>, Sched<[]> {
  bits<5> Zt;
  bits<3> Pg;
  bits<5> Rn;
  bits<5> Rm;
  let Inst{31-25} = 0b1010010;
  let Inst{24-23} = sz;
  let Inst{22-21} = 0b01;
  let Inst{20-16} = Rm;
  let Inst{15-13} = 0;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zt;

  let mayLoad = 1;
}

multiclass sve_mem_ldor_ss<bits<2> sz, string asm, RegisterOperand listty,
                           ZPRRegOp zprty, RegisterOperand gprty, ValueType Ty,
                           ValueType PredTy, SDNode Ld1ro, ComplexPattern AddrCP> {
  def NAME : sve_mem_ldor_ss<sz, asm, listty, gprty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Rm]",
                  (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, gprty:$Rm), 0>;

  def : Pat<(Ty (Ld1ro (PredTy PPR3bAny:$gp), (AddrCP GPR64sp:$base, gprty:$offset))),
            (!cast<Instruction>(NAME) PPR3bAny:$gp, GPR64sp:$base, gprty:$offset)>;
}

//===----------------------------------------------------------------------===//
// SVE Interleave 128-bit Elements Group
//===----------------------------------------------------------------------===//

class sve_int_perm_bin_perm_128_zz<bits<2> opc, bit P, string asm>
: I<(outs ZPR128:$Zd), (ins ZPR128:$Zn, ZPR128:$Zm),
  asm, "\t$Zd, $Zn, $Zm",
  "",
  []>, Sched<[]> {
  bits<5> Zd;
  bits<5> Zm;
  bits<5> Zn;
  let Inst{31-21} = 0b00000101101;
  let Inst{20-16} = Zm;
  let Inst{15-13} = 0b000;
  let Inst{12-11} = opc;
  let Inst{10}    = P;
  let Inst{9-5}   = Zn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_bin_perm_128_zz<bits<2> opc, bit P, string asm, SDPatternOperator op> {
  def NAME : sve_int_perm_bin_perm_128_zz<opc, P, asm>;

  def : SVE_2_Op_Pat<nxv16i8,  op, nxv16i8,  nxv16i8,  !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv8i16,  op, nxv8i16,  nxv8i16,  !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv8f16,  op, nxv8f16,  nxv8f16,  !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv4i32,  op, nxv4i32,  nxv4i32,  !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv4f32,  op, nxv4f32,  nxv4f32,  !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv2i64,  op, nxv2i64,  nxv2i64,  !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv2f64,  op, nxv2f64,  nxv2f64,  !cast<Instruction>(NAME)>;
  def : SVE_2_Op_Pat<nxv8bf16, op, nxv8bf16, nxv8bf16, !cast<Instruction>(NAME)>;
}

/// Addressing modes
def am_sve_indexed_s4 :ComplexPattern<iPTR, 2, "SelectAddrModeIndexedSVE<-8,7>", [], [SDNPWantRoot]>;
def am_sve_indexed_s6 :ComplexPattern<iPTR, 2, "SelectAddrModeIndexedSVE<-32,31>", [], [SDNPWantRoot]>;

def am_sve_regreg_lsl0 : ComplexPattern<iPTR, 2, "SelectSVERegRegAddrMode<0>", []>;
def am_sve_regreg_lsl1 : ComplexPattern<iPTR, 2, "SelectSVERegRegAddrMode<1>", []>;
def am_sve_regreg_lsl2 : ComplexPattern<iPTR, 2, "SelectSVERegRegAddrMode<2>", []>;
def am_sve_regreg_lsl3 : ComplexPattern<iPTR, 2, "SelectSVERegRegAddrMode<3>", []>;

// Predicated pseudo floating point two operand instructions.
multiclass sve_fp_bin_pred_hfd<SDPatternOperator op> {
  def _UNDEF_H : PredTwoOpPseudo<NAME # _H, ZPR16, FalseLanesUndef>;
  def _UNDEF_S : PredTwoOpPseudo<NAME # _S, ZPR32, FalseLanesUndef>;
  def _UNDEF_D : PredTwoOpPseudo<NAME # _D, ZPR64, FalseLanesUndef>;

  def : SVE_3_Op_Pat<nxv8f16, op, nxv8i1, nxv8f16, nxv8f16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  def : SVE_3_Op_Pat<nxv4f16, op, nxv4i1, nxv4f16, nxv4f16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  def : SVE_3_Op_Pat<nxv2f16, op, nxv2i1, nxv2f16, nxv2f16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  def : SVE_3_Op_Pat<nxv4f32, op, nxv4i1, nxv4f32, nxv4f32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  def : SVE_3_Op_Pat<nxv2f32, op, nxv2i1, nxv2f32, nxv2f32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  def : SVE_3_Op_Pat<nxv2f64, op, nxv2i1, nxv2f64, nxv2f64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

// Predicated pseudo integer two operand instructions.
multiclass sve_int_bin_pred_bhsd<SDPatternOperator op> {
  def _UNDEF_B : PredTwoOpPseudo<NAME # _B, ZPR8, FalseLanesUndef>;
  def _UNDEF_H : PredTwoOpPseudo<NAME # _H, ZPR16, FalseLanesUndef>;
  def _UNDEF_S : PredTwoOpPseudo<NAME # _S, ZPR32, FalseLanesUndef>;
  def _UNDEF_D : PredTwoOpPseudo<NAME # _D, ZPR64, FalseLanesUndef>;

  def : SVE_3_Op_Pat<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Pseudo>(NAME # _UNDEF_B)>;
  def : SVE_3_Op_Pat<nxv8i16, op, nxv8i1,  nxv8i16, nxv8i16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1,  nxv4i32, nxv4i32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1,  nxv2i64, nxv2i64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

// As sve_int_bin_pred but when only i32 and i64 vector types are required.
multiclass sve_int_bin_pred_sd<SDPatternOperator op> {
  def _UNDEF_S : PredTwoOpPseudo<NAME # _S, ZPR32, FalseLanesUndef>;
  def _UNDEF_D : PredTwoOpPseudo<NAME # _D, ZPR64, FalseLanesUndef>;

  def : SVE_3_Op_Pat<nxv4i32, op, nxv4i1, nxv4i32, nxv4i32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  def : SVE_3_Op_Pat<nxv2i64, op, nxv2i1, nxv2i64, nxv2i64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

// Predicated pseudo integer two operand instructions. Second operand is an
// immediate specified by imm_[bhsd].
multiclass sve_int_shift_pred_bhsd<SDPatternOperator op,
                                   ComplexPattern imm_b, ComplexPattern imm_h,
                                   ComplexPattern imm_s, ComplexPattern imm_d> {
  def _UNDEF_B : PredTwoOpImmPseudo<NAME # _B, ZPR8,  Operand<i32>, FalseLanesUndef>;
  def _UNDEF_H : PredTwoOpImmPseudo<NAME # _H, ZPR16, Operand<i32>, FalseLanesUndef>;
  def _UNDEF_S : PredTwoOpImmPseudo<NAME # _S, ZPR32, Operand<i32>, FalseLanesUndef>;
  def _UNDEF_D : PredTwoOpImmPseudo<NAME # _D, ZPR64, Operand<i32>, FalseLanesUndef>;

  def : SVE_Shift_DupImm_Pred_Pat<nxv16i8, op, nxv16i1, i32, imm_b, !cast<Instruction>(NAME # _UNDEF_B)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv8i16, op, nxv8i1,  i32, imm_h, !cast<Instruction>(NAME # _UNDEF_H)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv4i32, op, nxv4i1,  i32, imm_s, !cast<Instruction>(NAME # _UNDEF_S)>;
  def : SVE_Shift_DupImm_Pred_Pat<nxv2i64, op, nxv2i1,  i64, imm_d, !cast<Instruction>(NAME # _UNDEF_D)>;
}

multiclass sve_int_bin_pred_all_active_bhsd<SDPatternOperator op> {
  def _UNDEF_B : PredTwoOpPseudo<NAME # _B, ZPR8, FalseLanesUndef>;
  def _UNDEF_H : PredTwoOpPseudo<NAME # _H, ZPR16, FalseLanesUndef>;
  def _UNDEF_S : PredTwoOpPseudo<NAME # _S, ZPR32, FalseLanesUndef>;
  def _UNDEF_D : PredTwoOpPseudo<NAME # _D, ZPR64, FalseLanesUndef>;

  def : SVE_2_Op_Pred_All_Active_Pt<nxv16i8, op, nxv16i1, nxv16i8, nxv16i8, !cast<Pseudo>(NAME # _UNDEF_B)>;
  def : SVE_2_Op_Pred_All_Active_Pt<nxv8i16, op, nxv8i1,  nxv8i16, nxv8i16, !cast<Pseudo>(NAME # _UNDEF_H)>;
  def : SVE_2_Op_Pred_All_Active_Pt<nxv4i32, op, nxv4i1,  nxv4i32, nxv4i32, !cast<Pseudo>(NAME # _UNDEF_S)>;
  def : SVE_2_Op_Pred_All_Active_Pt<nxv2i64, op, nxv2i1,  nxv2i64, nxv2i64, !cast<Pseudo>(NAME # _UNDEF_D)>;
}

