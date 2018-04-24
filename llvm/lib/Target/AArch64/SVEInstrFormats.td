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


class SImmMulVlOperand<int Bits, int Scale> : AsmOperandClass {
  let Name = "SImm" # Bits # "Scale" # Scale # "MulVl";
  let DiagnosticType = "InvalidMemoryIndexed" # Scale # "SImm" # Bits;
  let PredicateMethod = "isSImmScaled<" # Bits # ", " # Scale # ">";
  let RenderMethod = "addImmScaledOperands<" # Scale # ">";
}

def SImm4MulVlOperand : SImmMulVlOperand<4,1>;
def SImm4Scale2MulVlOperand : SImmMulVlOperand<4,2>;
def SImm4Scale3MulVlOperand : SImmMulVlOperand<4,3>;
def SImm4Scale4MulVlOperand : SImmMulVlOperand<4,4>;

def simm4MulVl : Operand<i64>, ImmLeaf<i64, [{ return Imm >= -8 && Imm < 8; }]> {
  let DecoderMethod = "DecodeSImm<4>";
  let ParserMatchClass = SImm4MulVlOperand;
}

def simm4Scale2MulVl : Operand<i64>, ImmLeaf<i64, [{
    return (Imm >= -16 && Imm <= 14) && ((Imm % 2) == 0x0);
  }]> {
  let DecoderMethod = "DecodeSImm<4>";
  let PrintMethod = "printImmScale<2>";
  let ParserMatchClass = SImm4Scale2MulVlOperand;
}

def simm4Scale3MulVl : Operand<i64>, ImmLeaf<i64, [{
    return (Imm >= -24 && Imm <= 21) && ((Imm % 3) == 0x0);
  }]> {
  let DecoderMethod = "DecodeSImm<4>";
  let PrintMethod = "printImmScale<3>";
  let ParserMatchClass = SImm4Scale3MulVlOperand;
}

def simm4Scale4MulVl : Operand<i64>, ImmLeaf<i64, [{
    return (Imm >= -32 && Imm <= 28) && ((Imm % 4) == 0x0);
  }]> {
  let DecoderMethod = "DecodeSImm<4>";
  let PrintMethod = "printImmScale<4>";
  let ParserMatchClass = SImm4Scale4MulVlOperand;
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

//===----------------------------------------------------------------------===//
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

multiclass sve_int_bin_pred_arit_0<bits<3> opc, string asm> {
  def _B : sve_int_bin_pred_arit_log<0b00, 0b00, opc, asm, ZPR8>;
  def _H : sve_int_bin_pred_arit_log<0b01, 0b00, opc, asm, ZPR16>;
  def _S : sve_int_bin_pred_arit_log<0b10, 0b00, opc, asm, ZPR32>;
  def _D : sve_int_bin_pred_arit_log<0b11, 0b00, opc, asm, ZPR64>;
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
: I<(outs), (ins VecList:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4MulVl:$imm4),
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
                 (!cast<Instruction>(NAME) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4MulVl:$imm4), 0>;
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
: I<(outs VecList:$Zt), (ins PPR3bAny:$Pg, GPR64sp:$Rn, simm4MulVl:$imm4),
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
                  (!cast<Instruction>(NAME # _REAL) zprty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, simm4MulVl:$imm4), 0>;
  def : InstAlias<asm # "\t$Zt, $Pg/z, [$Rn]",
                  (!cast<Instruction>(NAME # _REAL) listty:$Zt, PPR3bAny:$Pg, GPR64sp:$Rn, 0), 1>;
}

multiclass sve_mem_cld_si<bits<4> dtype, string asm, RegisterOperand listty,
                          ZPRRegOp zprty>
: sve_mem_cld_si_base<dtype, 0, asm, listty, zprty>;

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
