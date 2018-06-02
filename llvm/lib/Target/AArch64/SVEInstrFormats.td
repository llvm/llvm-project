//=-- SVEInstrFormats.td -  AArch64 SVE Instruction classes -*- tablegen -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AArch64 Scalable Vector Extension (SVE) Instruction Class Definitions.
//
//===----------------------------------------------------------------------===//

def SVEPatternOperand : AsmOperandClass {
  let Name = "SVEPattern";
  let ParserMethod = "tryParseSVEPattern";
  let PredicateMethod = "isSVEPattern";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidSVEPattern";
}

def sve_pred_enum : Operand<i32>, ImmLeaf<i32, [{
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

def sve_prfop : Operand<i32>, ImmLeaf<i32, [{
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
                   AsmOperandClass OpndClass, code Predicate>
    : Operand<i32>, ImmLeaf<i32, Predicate> {
  let EncoderMethod = "getImm8OptLsl";
  let DecoderMethod = "DecodeImm8OptLsl<" # ElementWidth # ">";
  let PrintMethod = "printImm8OptLsl<" # printType # ">";
  let ParserMatchClass = OpndClass;
  let MIOperandInfo = (ops i32imm, i32imm);
}

def cpy_imm8_opt_lsl_i8  : imm8_opt_lsl<8,  "int8_t",  SVECpyImmOperand8,  [{
  return AArch64_AM::isSVECpyImm<int8_t>(Imm);
}]>;
def cpy_imm8_opt_lsl_i16 : imm8_opt_lsl<16, "int16_t", SVECpyImmOperand16, [{
  return AArch64_AM::isSVECpyImm<int16_t>(Imm);
}]>;
def cpy_imm8_opt_lsl_i32 : imm8_opt_lsl<32, "int32_t", SVECpyImmOperand32, [{
  return AArch64_AM::isSVECpyImm<int32_t>(Imm);
}]>;
def cpy_imm8_opt_lsl_i64 : imm8_opt_lsl<64, "int64_t", SVECpyImmOperand64, [{
  return AArch64_AM::isSVECpyImm<int64_t>(Imm);
}]>;

def addsub_imm8_opt_lsl_i8  : imm8_opt_lsl<8,  "uint8_t",  SVEAddSubImmOperand8,  [{
  return AArch64_AM::isSVEAddSubImm<int8_t>(Imm);
}]>;
def addsub_imm8_opt_lsl_i16 : imm8_opt_lsl<16, "uint16_t", SVEAddSubImmOperand16, [{
  return AArch64_AM::isSVEAddSubImm<int16_t>(Imm);
}]>;
def addsub_imm8_opt_lsl_i32 : imm8_opt_lsl<32, "uint32_t", SVEAddSubImmOperand32, [{
  return AArch64_AM::isSVEAddSubImm<int32_t>(Imm);
}]>;
def addsub_imm8_opt_lsl_i64 : imm8_opt_lsl<64, "uint64_t", SVEAddSubImmOperand64, [{
  return AArch64_AM::isSVEAddSubImm<int64_t>(Imm);
}]>;

//===----------------------------------------------------------------------===//
// SVE PTrue - These are used extensively throughout the pattern matching so
//             it's important we define them first.
//===----------------------------------------------------------------------===//

class sve_int_ptrue<bits<2> sz8_64, bits<3> opc, string asm, PPRRegOp pprty>
: I<(outs pprty:$Pd), (ins sve_pred_enum:$pattern),
  asm, "\t$Pd, $pattern",
  "",
  []>, Sched<[]> {
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
}

multiclass sve_int_ptrue<bits<3> opc, string asm> {
  def _B : sve_int_ptrue<0b00, opc, asm, PPR8>;
  def _H : sve_int_ptrue<0b01, opc, asm, PPR16>;
  def _S : sve_int_ptrue<0b10, opc, asm, PPR32>;
  def _D : sve_int_ptrue<0b11, opc, asm, PPR64>;

  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _B) PPR8:$Pd, 0b11111), 1>;
  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _H) PPR16:$Pd, 0b11111), 1>;
  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _S) PPR32:$Pd, 0b11111), 1>;
  def : InstAlias<asm # "\t$Pd",
                  (!cast<Instruction>(NAME # _D) PPR64:$Pd, 0b11111), 1>;
}

let Predicates = [HasSVE] in {
  defm PTRUE  : sve_int_ptrue<0b000, "ptrue">;
  defm PTRUES : sve_int_ptrue<0b001, "ptrues">;
}

//===----------------------------------------------------------------------===//
// SVE Permute - Cross Lane Group
//===----------------------------------------------------------------------===//

class sve_int_perm_dup_r<bits<2> sz8_64, string asm, ZPRRegOp zprty,
                         RegisterClass srcRegType>
: I<(outs zprty:$Zd), (ins srcRegType:$Rn),
  asm, "\t$Zd, $Rn",
  "",
  []>, Sched<[]> {
  bits<5> Rn;
  bits<5> Zd;
  let Inst{31-24} = 0b00000101;
  let Inst{23-22} = sz8_64;
  let Inst{21-10} = 0b100000001110;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = Zd;
}

multiclass sve_int_perm_dup_r<string asm> {
  def _B : sve_int_perm_dup_r<0b00, asm, ZPR8, GPR32sp>;
  def _H : sve_int_perm_dup_r<0b01, asm, ZPR16, GPR32sp>;
  def _S : sve_int_perm_dup_r<0b10, asm, ZPR32, GPR32sp>;
  def _D : sve_int_perm_dup_r<0b11, asm, ZPR64, GPR64sp>;

  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _B) ZPR8:$Zd, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _H) ZPR16:$Zd, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _S) ZPR32:$Zd, GPR32sp:$Rn), 1>;
  def : InstAlias<"mov $Zd, $Rn",
                  (!cast<Instruction>(NAME # _D) ZPR64:$Zd, GPR64sp:$Rn), 1>;
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
}

multiclass sve_int_log_imm<bits<2> opc, string asm, string alias> {
  def NAME : sve_int_log_imm<opc, asm>;

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

multiclass sve_int_bin_cons_arit_0<bits<3> opc, string asm> {
  def _B : sve_int_bin_cons_arit_0<0b00, opc, asm, ZPR8>;
  def _H : sve_int_bin_cons_arit_0<0b01, opc, asm, ZPR16>;
  def _S : sve_int_bin_cons_arit_0<0b10, opc, asm, ZPR32>;
  def _D : sve_int_bin_cons_arit_0<0b11, opc, asm, ZPR64>;
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

multiclass sve_int_perm_bin_perm_zz<bits<3> opc, string asm> {
  def _B : sve_int_perm_bin_perm_zz<opc, 0b00, asm, ZPR8>;
  def _H : sve_int_perm_bin_perm_zz<opc, 0b01, asm, ZPR16>;
  def _S : sve_int_perm_bin_perm_zz<opc, 0b10, asm, ZPR32>;
  def _D : sve_int_perm_bin_perm_zz<opc, 0b11, asm, ZPR64>;
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
}

multiclass sve_int_bin_pred_log<bits<3> opc, string asm> {
  def _B : sve_int_bin_pred_arit_log<0b00, 0b11, opc, asm, ZPR8>;
  def _H : sve_int_bin_pred_arit_log<0b01, 0b11, opc, asm, ZPR16>;
  def _S : sve_int_bin_pred_arit_log<0b10, 0b11, opc, asm, ZPR32>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b11, opc, asm, ZPR64>;
}

multiclass sve_int_bin_pred_arit_0<bits<3> opc, string asm> {
  def _B : sve_int_bin_pred_arit_log<0b00, 0b00, opc, asm, ZPR8>;
  def _H : sve_int_bin_pred_arit_log<0b01, 0b00, opc, asm, ZPR16>;
  def _S : sve_int_bin_pred_arit_log<0b10, 0b00, opc, asm, ZPR32>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b00, opc, asm, ZPR64>;
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
}

multiclass sve_int_arith_imm0<bits<3> opc, string asm> {
  def _B : sve_int_arith_imm0<0b00, opc, asm, ZPR8, addsub_imm8_opt_lsl_i8>;
  def _H : sve_int_arith_imm0<0b01, opc, asm, ZPR16, addsub_imm8_opt_lsl_i16>;
  def _S : sve_int_arith_imm0<0b10, opc, asm, ZPR32, addsub_imm8_opt_lsl_i32>;
  def _D : sve_int_arith_imm0<0b11, opc, asm, ZPR64, addsub_imm8_opt_lsl_i64>;
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

//===----------------------------------------------------------------------===//
//SVE Index Generation Group
//===----------------------------------------------------------------------===//

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
  def _B : sve_int_index_ii<0b00, asm, ZPR8, simm5_32b>;
  def _H : sve_int_index_ii<0b01, asm, ZPR16, simm5_32b>;
  def _S : sve_int_index_ii<0b10, asm, ZPR32, simm5_32b>;
  def _D : sve_int_index_ii<0b11, asm, ZPR64, simm5_64b>;
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

multiclass sve_int_index_ir<string asm> {
  def _B : sve_int_index_ir<0b00, asm, ZPR8, GPR32, simm5_32b>;
  def _H : sve_int_index_ir<0b01, asm, ZPR16, GPR32, simm5_32b>;
  def _S : sve_int_index_ir<0b10, asm, ZPR32, GPR32, simm5_32b>;
  def _D : sve_int_index_ir<0b11, asm, ZPR64, GPR64, simm5_64b>;
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
  def _B : sve_int_index_ri<0b00, asm, ZPR8, GPR32, simm5_32b>;
  def _H : sve_int_index_ri<0b01, asm, ZPR16, GPR32, simm5_32b>;
  def _S : sve_int_index_ri<0b10, asm, ZPR32, GPR32, simm5_32b>;
  def _D : sve_int_index_ri<0b11, asm, ZPR64, GPR64, simm5_64b>;
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

multiclass sve_int_index_rr<string asm> {
  def _B : sve_int_index_rr<0b00, asm, ZPR8, GPR32>;
  def _H : sve_int_index_rr<0b01, asm, ZPR16, GPR32>;
  def _S : sve_int_index_rr<0b10, asm, ZPR32, GPR32>;
  def _D : sve_int_index_rr<0b11, asm, ZPR64, GPR64>;
}
//
//===----------------------------------------------------------------------===//
// SVE Bitwise Shift - Predicated Group
//===----------------------------------------------------------------------===//

class sve_int_bin_pred_shift_1<bits<2> sz8_64, bits<3> opc, string asm,
                               ZPRRegOp zprty>
: I<(outs zprty:$Zdn), (ins PPR3bAny:$Pg, zprty:$_Zdn, zprty:$Zm),
  asm, "\t$Zdn, $Pg/m, $_Zdn, $Zm",
  "",
  []>, Sched<[]> {
  bits<3> Pg;
  bits<5> Zdn;
  bits<5> Zm;
  let Inst{31-24} = 0b00000100;
  let Inst{23-22} = sz8_64;
  let Inst{21-19} = 0b010;
  let Inst{18-16} = opc;
  let Inst{15-13} = 0b100;
  let Inst{12-10} = Pg;
  let Inst{9-5}   = Zm;
  let Inst{4-0}   = Zdn;

  let Constraints = "$Zdn = $_Zdn";
}

multiclass sve_int_bin_pred_shift_1<bits<3> opc, string asm> {
  def _B : sve_int_bin_pred_shift_1<0b00, opc, asm, ZPR8>;
  def _H : sve_int_bin_pred_shift_1<0b01, opc, asm, ZPR16>;
  def _S : sve_int_bin_pred_shift_1<0b10, opc, asm, ZPR32>;
  def _D : sve_int_bin_pred_shift_1<0b11, opc, asm, ZPR64>;
}


//===----------------------------------------------------------------------===//
// SVE Shift by Immediate - Unpredicated Group
//===----------------------------------------------------------------------===//

class sve_int_bin_cons_shift_b<bits<4> tsz8_64, bits<2> opc, string asm,
                               ZPRRegOp zprty, Operand immtype>
: I<(outs zprty:$Zd), (ins zprty:$Zn, immtype:$imm),
  asm, "\t$Zd, $Zn, $imm",
  "", []>, Sched<[]> {
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

multiclass sve_int_bin_cons_shift_b_left<bits<2> opc, string asm> {
  def _B : sve_int_bin_cons_shift_b<{0,0,0,1}, opc, asm, ZPR8, vecshiftL8>;
  def _H : sve_int_bin_cons_shift_b<{0,0,1,?}, opc, asm, ZPR16, vecshiftL16> {
    let Inst{19} = imm{3};
  }
  def _S : sve_int_bin_cons_shift_b<{0,1,?,?}, opc, asm, ZPR32, vecshiftL32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve_int_bin_cons_shift_b<{1,?,?,?}, opc, asm, ZPR64, vecshiftL64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }
}

multiclass sve_int_bin_cons_shift_b_right<bits<2> opc, string asm> {
  def _B : sve_int_bin_cons_shift_b<{0,0,0,1}, opc, asm, ZPR8, vecshiftR8>;
  def _H : sve_int_bin_cons_shift_b<{0,0,1,?}, opc, asm, ZPR16, vecshiftR16> {
    let Inst{19} = imm{3};
  }
  def _S : sve_int_bin_cons_shift_b<{0,1,?,?}, opc, asm, ZPR32, vecshiftR32> {
    let Inst{20-19} = imm{4-3};
  }
  def _D : sve_int_bin_cons_shift_b<{1,?,?,?}, opc, asm, ZPR64, vecshiftR64> {
    let Inst{22}    = imm{5};
    let Inst{20-19} = imm{4-3};
  }
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

multiclass sve_mem_sst_sv_32_scaled<bits<3> opc, string asm,
                                    RegisterOperand listty,
                                    ZPRRegOp zprty,
                                    RegisterOperand sxtw_opnd,
                                    RegisterOperand uxtw_opnd > {
  def _UXTW_SCALED : sve_mem_sst_sv<opc, 0, 1, asm, listty, uxtw_opnd>;
  def _SXTW_SCALED : sve_mem_sst_sv<opc, 1, 1, asm, listty, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _UXTW_SCALED) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _SXTW_SCALED) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;
}

multiclass sve_mem_sst_sv_32_unscaled<bits<3> opc, string asm,
                                      RegisterOperand listty,
                                      ZPRRegOp zprty,
                                      RegisterOperand sxtw_opnd,
                                      RegisterOperand uxtw_opnd> {
  def _UXTW : sve_mem_sst_sv<opc, 0, 0, asm, listty, uxtw_opnd>;
  def _SXTW : sve_mem_sst_sv<opc, 1, 0, asm, listty, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _UXTW) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME # _SXTW) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;
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
                                    RegisterOperand zprext> {
  def "" : sve_mem_sst_sv2<msz, 1, asm, zprext>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm), 0>;

}

multiclass sve_mem_sst_sv_64_unscaled<bits<2> msz, string asm> {
  def "" : sve_mem_sst_sv2<msz, 0, asm, ZPR64ExtLSL8>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Rn, $Zm]",
                 (!cast<Instruction>(NAME) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, ZPR64ExtLSL8:$Zm), 0>;
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

multiclass sve_mem_sst_vi_ptrs<bits<3> opc, string asm, RegisterOperand listty,
                               ZPRRegOp zprty, Operand imm_ty> {
  def _IMM : sve_mem_sst_vi<opc, asm, zprty, listty, imm_ty>;

  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM) zprty:$Zt, PPR3bAny:$Pg, zprty:$Zn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn, $imm5]",
                  (!cast<Instruction>(NAME # _IMM) zprty:$Zt, PPR3bAny:$Pg, zprty:$Zn, imm_ty:$imm5), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM) listty:$Zt, PPR3bAny:$Pg, zprty:$Zn, 0), 1>;
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
  "",
  []>, Sched<[]> {
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

multiclass sve_int_perm_bin_perm_pp<bits<3> opc, string asm> {
  def _B : sve_int_perm_bin_perm_pp<opc, 0b00, asm, PPR8>;
  def _H : sve_int_perm_bin_perm_pp<opc, 0b01, asm, PPR16>;
  def _S : sve_int_perm_bin_perm_pp<opc, 0b10, asm, PPR32>;
  def _D : sve_int_perm_bin_perm_pp<opc, 0b11, asm, PPR64>;
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
  let Uses = !if(!eq(nf, 1), [FFR], []);
  let Defs = !if(!eq(nf, 1), [FFR], []);
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
  let Uses = !if(!eq(ff, 1), [FFR], []);
  let Defs = !if(!eq(ff, 1), [FFR], []);
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
                                        RegisterOperand sxtw_opnd,
                                        RegisterOperand uxtw_opnd> {
  def _UXTW_SCALED_REAL : sve_mem_32b_gld_sv<opc, 0, 1, asm, uxtw_opnd>;
  def _SXTW_SCALED_REAL : sve_mem_32b_gld_sv<opc, 1, 1, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_SCALED_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_SCALED_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;
}

multiclass sve_mem_32b_gld_vs_32_unscaled<bits<4> opc, string asm,
                                          RegisterOperand sxtw_opnd,
                                          RegisterOperand uxtw_opnd> {
  def _UXTW_REAL : sve_mem_32b_gld_sv<opc, 0, 0, asm, uxtw_opnd>;
  def _SXTW_REAL : sve_mem_32b_gld_sv<opc, 1, 0, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_REAL) ZPR32:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;
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

multiclass sve_mem_32b_gld_vi_32_ptrs<bits<4> opc, string asm, Operand imm_ty> {
  def _IMM_REAL : sve_mem_32b_gld_vi<opc, asm, imm_ty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn, $imm5]",
                  (!cast<Instruction>(NAME # _IMM_REAL) ZPR32:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, imm_ty:$imm5), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) Z_s:$Zt, PPR3bAny:$Pg, ZPR32:$Zn, 0), 1>;
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
                                      RegisterOperand uxtw_opnd> {
  def _UXTW_SCALED : sve_mem_32b_prfm_sv<msz, 0, asm, uxtw_opnd>;
  def _SXTW_SCALED : sve_mem_32b_prfm_sv<msz, 1, asm, sxtw_opnd>;
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

multiclass sve_mem_32b_prfm_vi<bits<2> msz, string asm, Operand imm_ty> {
  def NAME : sve_mem_32b_prfm_vi<msz, asm, imm_ty>;

  def : InstAlias<asm # "\t$prfop, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR3bAny:$Pg, ZPR32:$Zn, 0), 1>;
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
                                        RegisterOperand sxtw_opnd,
                                        RegisterOperand uxtw_opnd> {
  def _UXTW_SCALED_REAL : sve_mem_64b_gld_sv<opc, 0, 1, 0, asm, uxtw_opnd>;
  def _SXTW_SCALED_REAL : sve_mem_64b_gld_sv<opc, 1, 1, 0, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_SCALED_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_SCALED_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;
}

multiclass sve_mem_64b_gld_vs_32_unscaled<bits<4> opc, string asm,
                                          RegisterOperand sxtw_opnd,
                                          RegisterOperand uxtw_opnd> {
  def _UXTW_REAL : sve_mem_64b_gld_sv<opc, 0, 0, 0, asm, uxtw_opnd>;
  def _SXTW_REAL : sve_mem_64b_gld_sv<opc, 1, 0, 0, asm, sxtw_opnd>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _UXTW_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, uxtw_opnd:$Zm), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SXTW_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, sxtw_opnd:$Zm), 0>;
}

multiclass sve_mem_64b_gld_sv2_64_scaled<bits<4> opc, string asm,
                                         RegisterOperand zprext> {
  def _SCALED_REAL : sve_mem_64b_gld_sv<opc, 1, 1, 1, asm, zprext>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _SCALED_REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, zprext:$Zm), 0>;
}

multiclass sve_mem_64b_gld_vs2_64_unscaled<bits<4> opc, string asm> {
  def _REAL : sve_mem_64b_gld_sv<opc, 1, 0, 1, asm, ZPR64ExtLSL8>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn, $Zm]",
                  (!cast<Instruction>(NAME # _REAL) ZPR64:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, ZPR64ExtLSL8:$Zm), 0>;
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

multiclass sve_mem_64b_gld_vi_64_ptrs<bits<4> opc, string asm, Operand imm_ty> {
  def _IMM_REAL : sve_mem_64b_gld_vi<opc, asm, imm_ty>;

  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, 0), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn, $imm5]",
                 (!cast<Instruction>(NAME # _IMM_REAL) ZPR64:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, imm_ty:$imm5), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Zn]",
                  (!cast<Instruction>(NAME # _IMM_REAL) Z_d:$Zt, PPR3bAny:$Pg, ZPR64:$Zn, 0), 1>;
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
                                          RegisterOperand uxtw_opnd> {
  def _UXTW_SCALED : sve_mem_64b_prfm_sv<msz, 0, 0, asm, uxtw_opnd>;
  def _SXTW_SCALED : sve_mem_64b_prfm_sv<msz, 1, 0, asm, sxtw_opnd>;
}

multiclass sve_mem_64b_prfm_sv_lsl_scaled<bits<2> msz, string asm,
                                          RegisterOperand zprext> {
  def NAME : sve_mem_64b_prfm_sv<msz, 1, 1, asm, zprext>;
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

multiclass sve_mem_64b_prfm_vi<bits<2> msz, string asm, Operand imm_ty> {
  def NAME : sve_mem_64b_prfm_vi<msz, asm, imm_ty>;

  def : InstAlias<asm # "\t$prfop, $Pg, [$Zn]",
                  (!cast<Instruction>(NAME) sve_prfop:$prfop, PPR3bAny:$Pg, ZPR64:$Zn, 0), 1>;
}
