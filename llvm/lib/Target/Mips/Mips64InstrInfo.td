//===- Mips64InstrInfo.td - Mips64 Instruction Information -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips64 instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

// Unsigned Operand
def uimm16_64      : Operand<i64> {
  let PrintMethod = "printUnsignedImm";
}

// Signed Operand
def simm10_64 : Operand<i64>;

// Transformation Function - get Imm - 32.
def Subtract32 : SDNodeXForm<imm, [{
  return getImm(N, (unsigned)N->getZExtValue() - 32);
}]>;

// shamt must fit in 6 bits.
def immZExt6 : ImmLeaf<i32, [{return Imm == (Imm & 0x3f);}]>;

// Node immediate fits as 10-bit sign extended on target immediate.
// e.g. seqi, snei
def immSExt10_64 : PatLeaf<(i64 imm),
                           [{ return isInt<10>(N->getSExtValue()); }]>;

def immZExt16_64 : PatLeaf<(i64 imm),
                           [{ return isInt<16>(N->getZExtValue()); }]>;

def immZExt5_64 : ImmLeaf<i64, [{ return Imm == (Imm & 0x1f); }]>;

// Transformation function: get log2 of low 32 bits of immediate
def Log2LO : SDNodeXForm<imm, [{
  return getImm(N, Log2_64((unsigned) N->getZExtValue()));
}]>;

// Transformation function: get log2 of high 32 bits of immediate
def Log2HI : SDNodeXForm<imm, [{
  return getImm(N, Log2_64((unsigned) (N->getZExtValue() >> 32)));
}]>;

// Predicate: True if immediate is a power of 2 and fits 32 bits
def PowerOf2LO : PatLeaf<(imm), [{
  if (N->getValueType(0) == MVT::i64) {
    uint64_t Imm = N->getZExtValue();
    return isPowerOf2_64(Imm) && (Imm & 0xffffffff) == Imm;
  }
  else
    return false;
}]>;

// Predicate: True if immediate is a power of 2 and exceeds 32 bits
def PowerOf2HI : PatLeaf<(imm), [{
  if (N->getValueType(0) == MVT::i64) {
    uint64_t Imm = N->getZExtValue();
    return isPowerOf2_64(Imm) && (Imm & 0xffffffff00000000) == Imm;
  }
  else
    return false;
}]>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//
let usesCustomInserter = 1 in {
  def ATOMIC_LOAD_ADD_I64  : Atomic2Ops<atomic_load_add_64, GPR64>;
  def ATOMIC_LOAD_SUB_I64  : Atomic2Ops<atomic_load_sub_64, GPR64>;
  def ATOMIC_LOAD_AND_I64  : Atomic2Ops<atomic_load_and_64, GPR64>;
  def ATOMIC_LOAD_OR_I64   : Atomic2Ops<atomic_load_or_64, GPR64>;
  def ATOMIC_LOAD_XOR_I64  : Atomic2Ops<atomic_load_xor_64, GPR64>;
  def ATOMIC_LOAD_NAND_I64 : Atomic2Ops<atomic_load_nand_64, GPR64>;
  def ATOMIC_SWAP_I64      : Atomic2Ops<atomic_swap_64, GPR64>;
  def ATOMIC_CMP_SWAP_I64  : AtomicCmpSwap<atomic_cmp_swap_64, GPR64>;
}

/// Pseudo instructions for loading and storing accumulator registers.
let isPseudo = 1, isCodeGenOnly = 1 in {
  def LOAD_ACC128  : Load<"", ACC128>;
  def STORE_ACC128 : Store<"", ACC128>;
}

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
let DecoderNamespace = "Mips64" in {
/// Arithmetic Instructions (ALU Immediate)
def DADDi   : ArithLogicI<"daddi", simm16_64, GPR64Opnd>, ADDI_FM<0x18>,
              ISA_MIPS3_NOT_32R6_64R6;
def DADDiu  : ArithLogicI<"daddiu", simm16_64, GPR64Opnd, II_DADDIU,
                          immSExt16, add>,
              ADDI_FM<0x19>, IsAsCheapAsAMove, ISA_MIPS3;

let isCodeGenOnly = 1 in {
def SLTi64  : SetCC_I<"slti", setlt, simm16_64, immSExt16, GPR64Opnd>,
              SLTI_FM<0xa>;
def SLTiu64 : SetCC_I<"sltiu", setult, simm16_64, immSExt16, GPR64Opnd>,
              SLTI_FM<0xb>;
def ANDi64 : ArithLogicI<"andi", uimm16_64, GPR64Opnd, II_AND, immZExt16, and>,
             ADDI_FM<0xc>;
def ORi64   : ArithLogicI<"ori", uimm16_64, GPR64Opnd, II_OR, immZExt16, or>,
              ADDI_FM<0xd>;
def XORi64  : ArithLogicI<"xori", uimm16_64, GPR64Opnd, II_XOR, immZExt16, xor>,
              ADDI_FM<0xe>;
def LUi64   : LoadUpper<"lui", GPR64Opnd, uimm16_64>, LUI_FM;
}

/// Arithmetic Instructions (3-Operand, R-Type)
def DADD   : ArithLogicR<"dadd", GPR64Opnd, 1, II_DADD>, ADD_FM<0, 0x2c>,
             ISA_MIPS3;
def DADDu  : ArithLogicR<"daddu", GPR64Opnd, 1, II_DADDU, add>, ADD_FM<0, 0x2d>,
             ISA_MIPS3;
def DSUBu  : ArithLogicR<"dsubu", GPR64Opnd, 0, II_DSUBU, sub>, ADD_FM<0, 0x2f>,
             ISA_MIPS3;
def DSUB   : ArithLogicR<"dsub", GPR64Opnd, 0, II_DSUB>, ADD_FM<0, 0x2e>,
             ISA_MIPS3;

let isCodeGenOnly = 1 in {
def SLT64  : SetCC_R<"slt", setlt, GPR64Opnd>, ADD_FM<0, 0x2a>;
def SLTu64 : SetCC_R<"sltu", setult, GPR64Opnd>, ADD_FM<0, 0x2b>;
def AND64  : ArithLogicR<"and", GPR64Opnd, 1, II_AND, and>, ADD_FM<0, 0x24>;
def OR64   : ArithLogicR<"or", GPR64Opnd, 1, II_OR, or>, ADD_FM<0, 0x25>;
def XOR64  : ArithLogicR<"xor", GPR64Opnd, 1, II_XOR, xor>, ADD_FM<0, 0x26>;
def NOR64  : LogicNOR<"nor", GPR64Opnd>, ADD_FM<0, 0x27>;
}

/// Shift Instructions
def DSLL   : shift_rotate_imm<"dsll", uimm6, GPR64Opnd, II_DSLL, shl, immZExt6>,
             SRA_FM<0x38, 0>, ISA_MIPS3;
def DSRL   : shift_rotate_imm<"dsrl", uimm6, GPR64Opnd, II_DSRL, srl, immZExt6>,
             SRA_FM<0x3a, 0>, ISA_MIPS3;
def DSRA   : shift_rotate_imm<"dsra", uimm6, GPR64Opnd, II_DSRA, sra, immZExt6>,
             SRA_FM<0x3b, 0>, ISA_MIPS3;
def DSLLV  : shift_rotate_reg<"dsllv", GPR64Opnd, II_DSLLV, shl>,
             SRLV_FM<0x14, 0>, ISA_MIPS3;
def DSRLV  : shift_rotate_reg<"dsrlv", GPR64Opnd, II_DSRLV, srl>,
             SRLV_FM<0x16, 0>, ISA_MIPS3;
def DSRAV  : shift_rotate_reg<"dsrav", GPR64Opnd, II_DSRAV, sra>,
             SRLV_FM<0x17, 0>, ISA_MIPS3;
def DSLL32 : shift_rotate_imm<"dsll32", uimm5, GPR64Opnd, II_DSLL32>,
             SRA_FM<0x3c, 0>, ISA_MIPS3;
def DSRL32 : shift_rotate_imm<"dsrl32", uimm5, GPR64Opnd, II_DSRL32>,
             SRA_FM<0x3e, 0>, ISA_MIPS3;
def DSRA32 : shift_rotate_imm<"dsra32", uimm5, GPR64Opnd, II_DSRA32>,
             SRA_FM<0x3f, 0>, ISA_MIPS3;

// Rotate Instructions
def DROTR  : shift_rotate_imm<"drotr", uimm6, GPR64Opnd, II_DROTR, rotr,
                              immZExt6>,
             SRA_FM<0x3a, 1>, ISA_MIPS64R2;
def DROTRV : shift_rotate_reg<"drotrv", GPR64Opnd, II_DROTRV, rotr>,
             SRLV_FM<0x16, 1>, ISA_MIPS64R2;
def DROTR32 : shift_rotate_imm<"drotr32", uimm5, GPR64Opnd, II_DROTR32>,
              SRA_FM<0x3e, 1>, ISA_MIPS64R2;

/// Load and Store Instructions
///  aligned
let isCodeGenOnly = 1 in {
def LB64  : Load<"lb", GPR64Opnd, sextloadi8, II_LB>, LW_FM<0x20>;
def LBu64 : Load<"lbu", GPR64Opnd, zextloadi8, II_LBU>, LW_FM<0x24>;
def LH64  : Load<"lh", GPR64Opnd, sextloadi16, II_LH>, LW_FM<0x21>;
def LHu64 : Load<"lhu", GPR64Opnd, zextloadi16, II_LHU>, LW_FM<0x25>;
def LW64  : Load<"lw", GPR64Opnd, sextloadi32, II_LW>, LW_FM<0x23>;
def SB64  : Store<"sb", GPR64Opnd, truncstorei8, II_SB>, LW_FM<0x28>;
def SH64  : Store<"sh", GPR64Opnd, truncstorei16, II_SH>, LW_FM<0x29>;
def SW64  : Store<"sw", GPR64Opnd, truncstorei32, II_SW>, LW_FM<0x2b>;
}

def LWu   : Load<"lwu", GPR64Opnd, zextloadi32, II_LWU>, LW_FM<0x27>, ISA_MIPS3;
def LD    : Load<"ld", GPR64Opnd, load, II_LD>, LW_FM<0x37>, ISA_MIPS3;
def SD    : Store<"sd", GPR64Opnd, store, II_SD>, LW_FM<0x3f>, ISA_MIPS3;

/// load/store left/right
let isCodeGenOnly = 1 in {
def LWL64 : LoadLeftRight<"lwl", MipsLWL, GPR64Opnd, II_LWL>, LW_FM<0x22>;
def LWR64 : LoadLeftRight<"lwr", MipsLWR, GPR64Opnd, II_LWR>, LW_FM<0x26>;
def SWL64 : StoreLeftRight<"swl", MipsSWL, GPR64Opnd, II_SWL>, LW_FM<0x2a>;
def SWR64 : StoreLeftRight<"swr", MipsSWR, GPR64Opnd, II_SWR>, LW_FM<0x2e>;
}

def LDL   : LoadLeftRight<"ldl", MipsLDL, GPR64Opnd, II_LDL>, LW_FM<0x1a>,
            ISA_MIPS3_NOT_32R6_64R6;
def LDR   : LoadLeftRight<"ldr", MipsLDR, GPR64Opnd, II_LDR>, LW_FM<0x1b>,
            ISA_MIPS3_NOT_32R6_64R6;
def SDL   : StoreLeftRight<"sdl", MipsSDL, GPR64Opnd, II_SDL>, LW_FM<0x2c>,
            ISA_MIPS3_NOT_32R6_64R6;
def SDR   : StoreLeftRight<"sdr", MipsSDR, GPR64Opnd, II_SDR>, LW_FM<0x2d>,
            ISA_MIPS3_NOT_32R6_64R6;

/// Load-linked, Store-conditional
def LLD : LLBase<"lld", GPR64Opnd>, LW_FM<0x34>, ISA_MIPS3_NOT_32R6_64R6;
def SCD : SCBase<"scd", GPR64Opnd>, LW_FM<0x3c>, ISA_MIPS3_NOT_32R6_64R6;

/// Jump and Branch Instructions
let isCodeGenOnly = 1 in {
  def JR64   : IndirectBranch<"jr", GPR64Opnd>, MTLO_FM<8>;
  def BEQ64  : CBranch<"beq", brtarget, seteq, GPR64Opnd>, BEQ_FM<4>;
  def BNE64  : CBranch<"bne", brtarget, setne, GPR64Opnd>, BEQ_FM<5>;
  def BGEZ64 : CBranchZero<"bgez", brtarget, setge, GPR64Opnd>, BGEZ_FM<1, 1>;
  def BGTZ64 : CBranchZero<"bgtz", brtarget, setgt, GPR64Opnd>, BGEZ_FM<7, 0>;
  def BLEZ64 : CBranchZero<"blez", brtarget, setle, GPR64Opnd>, BGEZ_FM<6, 0>;
  def BLTZ64 : CBranchZero<"bltz", brtarget, setlt, GPR64Opnd>, BGEZ_FM<1, 0>;
  def JALR64 : JumpLinkReg<"jalr", GPR64Opnd>, JALR_FM;
  def JALR64Pseudo : JumpLinkRegPseudo<GPR64Opnd, JALR, RA, GPR32Opnd>;
  def TAILCALL64_R : TailCallReg<GPR64Opnd, JR, GPR32Opnd>;
}

def PseudoReturn64 : PseudoReturnBase<GPR64Opnd>;
def PseudoIndirectBranch64 : PseudoIndirectBranchBase<GPR64Opnd>;

/// Multiply and Divide Instructions.
def DMULT  : Mult<"dmult", II_DMULT, GPR64Opnd, [HI0_64, LO0_64]>,
             MULT_FM<0, 0x1c>, ISA_MIPS3_NOT_32R6_64R6;
def DMULTu : Mult<"dmultu", II_DMULTU, GPR64Opnd, [HI0_64, LO0_64]>,
             MULT_FM<0, 0x1d>, ISA_MIPS3_NOT_32R6_64R6;
def PseudoDMULT  : MultDivPseudo<DMULT, ACC128, GPR64Opnd, MipsMult,
                                 II_DMULT>, ISA_MIPS3_NOT_32R6_64R6;
def PseudoDMULTu : MultDivPseudo<DMULTu, ACC128, GPR64Opnd, MipsMultu,
                                 II_DMULTU>, ISA_MIPS3_NOT_32R6_64R6;
def DSDIV : Div<"ddiv", II_DDIV, GPR64Opnd, [HI0_64, LO0_64]>,
            MULT_FM<0, 0x1e>, ISA_MIPS3_NOT_32R6_64R6;
def DUDIV : Div<"ddivu", II_DDIVU, GPR64Opnd, [HI0_64, LO0_64]>,
            MULT_FM<0, 0x1f>, ISA_MIPS3_NOT_32R6_64R6;
def PseudoDSDIV : MultDivPseudo<DSDIV, ACC128, GPR64Opnd, MipsDivRem,
                                II_DDIV, 0, 1, 1>, ISA_MIPS3_NOT_32R6_64R6;
def PseudoDUDIV : MultDivPseudo<DUDIV, ACC128, GPR64Opnd, MipsDivRemU,
                                II_DDIVU, 0, 1, 1>, ISA_MIPS3_NOT_32R6_64R6;

let isCodeGenOnly = 1 in {
def MTHI64 : MoveToLOHI<"mthi", GPR64Opnd, [HI0_64]>, MTLO_FM<0x11>,
             ISA_MIPS3_NOT_32R6_64R6;
def MTLO64 : MoveToLOHI<"mtlo", GPR64Opnd, [LO0_64]>, MTLO_FM<0x13>,
             ISA_MIPS3_NOT_32R6_64R6;
def MFHI64 : MoveFromLOHI<"mfhi", GPR64Opnd, AC0_64>, MFLO_FM<0x10>,
             ISA_MIPS3_NOT_32R6_64R6;
def MFLO64 : MoveFromLOHI<"mflo", GPR64Opnd, AC0_64>, MFLO_FM<0x12>,
             ISA_MIPS3_NOT_32R6_64R6;
def PseudoMFHI64 : PseudoMFLOHI<GPR64, ACC128, MipsMFHI>,
                   ISA_MIPS3_NOT_32R6_64R6;
def PseudoMFLO64 : PseudoMFLOHI<GPR64, ACC128, MipsMFLO>,
                   ISA_MIPS3_NOT_32R6_64R6;
def PseudoMTLOHI64 : PseudoMTLOHI<ACC128, GPR64>, ISA_MIPS3_NOT_32R6_64R6;

/// Sign Ext In Register Instructions.
def SEB64 : SignExtInReg<"seb", i8, GPR64Opnd, II_SEB>, SEB_FM<0x10, 0x20>,
            ISA_MIPS32R2;
def SEH64 : SignExtInReg<"seh", i16, GPR64Opnd, II_SEH>, SEB_FM<0x18, 0x20>,
            ISA_MIPS32R2;
}

/// Count Leading
def DCLZ : CountLeading0<"dclz", GPR64Opnd>, CLO_FM<0x24>, ISA_MIPS64_NOT_64R6;
def DCLO : CountLeading1<"dclo", GPR64Opnd>, CLO_FM<0x25>, ISA_MIPS64_NOT_64R6;

/// Double Word Swap Bytes/HalfWords
def DSBH : SubwordSwap<"dsbh", GPR64Opnd>, SEB_FM<2, 0x24>, ISA_MIPS64R2;
def DSHD : SubwordSwap<"dshd", GPR64Opnd>, SEB_FM<5, 0x24>, ISA_MIPS64R2;

def LEA_ADDiu64 : EffectiveAddress<"daddiu", GPR64Opnd>, LW_FM<0x19>;

let isCodeGenOnly = 1 in
def RDHWR64 : ReadHardware<GPR64Opnd, HWRegsOpnd>, RDHWR_FM;

let AdditionalPredicates = [NotInMicroMips] in {
  // TODO: Add 'pos + size' constraint check to dext* instructions
  //       DEXT: 0 < pos + size <= 63
  //       DEXTM, DEXTU: 32 < pos + size <= 64
  def DEXT : ExtBase<"dext", GPR64Opnd, uimm5, uimm5_plus1, MipsExt>,
             EXT_FM<3>;
  def DEXTM : ExtBase<"dextm", GPR64Opnd, uimm5, uimm5_plus33, MipsExt>,
              EXT_FM<1>;
  def DEXTU : ExtBase<"dextu", GPR64Opnd, uimm5_plus32, uimm5_plus1,
                      MipsExt>, EXT_FM<2>;
}

def DINS : InsBase<"dins", GPR64Opnd, uimm6, MipsIns>, EXT_FM<7>;
def DINSU : InsBase<"dinsu", GPR64Opnd, uimm5_plus32>, EXT_FM<6>;
def DINSM : InsBase<"dinsm", GPR64Opnd, uimm5>, EXT_FM<5>;

let isCodeGenOnly = 1, rs = 0, shamt = 0 in {
  def DSLL64_32 : FR<0x00, 0x3c, (outs GPR64:$rd), (ins GPR32:$rt),
                     "dsll\t$rd, $rt, 32", [], II_DSLL>;
  def SLL64_32 : FR<0x0, 0x00, (outs GPR64:$rd), (ins GPR32:$rt),
                    "sll\t$rd, $rt, 0", [], II_SLL>;
  def SLL64_64 : FR<0x0, 0x00, (outs GPR64:$rd), (ins GPR64:$rt),
                    "sll\t$rd, $rt, 0", [], II_SLL>;
}

// We need the following pseudo instruction to avoid offset calculation for
// long branches.  See the comment in file MipsLongBranch.cpp for detailed
// explanation.

// Expands to: daddiu $dst, $src, %PART($tgt - $baltgt)
// where %PART may be %hi or %lo, depending on the relocation kind
// that $tgt is annotated with.
def LONG_BRANCH_DADDiu : PseudoSE<(outs GPR64Opnd:$dst),
  (ins GPR64Opnd:$src, brtarget:$tgt, brtarget:$baltgt), []>;

// Cavium Octeon cnMIPS instructions
let DecoderNamespace = "CnMips",
    EncodingPredicates = []<Predicate>, // FIXME: The lack of HasStdEnc is probably a bug
    AdditionalPredicates = [HasCnMips] in {

class Count1s<string opstr, RegisterOperand RO>:
  InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
         [(set RO:$rd, (ctpop RO:$rs))], II_POP, FrmR, opstr> {
  let TwoOperandAliasConstraint = "$rd = $rs";
}

class ExtsCins<string opstr, SDPatternOperator Op = null_frag>:
  InstSE<(outs GPR64Opnd:$rt), (ins GPR64Opnd:$rs, uimm5:$pos, uimm5:$lenm1),
         !strconcat(opstr, " $rt, $rs, $pos, $lenm1"),
         [(set GPR64Opnd:$rt, (Op GPR64Opnd:$rs, imm:$pos, imm:$lenm1))],
         NoItinerary, FrmR, opstr> {
  let TwoOperandAliasConstraint = "$rt = $rs";
}

class SetCC64_R<string opstr, PatFrag cond_op> :
  InstSE<(outs GPR64Opnd:$rd), (ins GPR64Opnd:$rs, GPR64Opnd:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set GPR64Opnd:$rd, (zext (cond_op GPR64Opnd:$rs,
                                             GPR64Opnd:$rt)))],
         II_SEQ_SNE, FrmR, opstr> {
  let TwoOperandAliasConstraint = "$rd = $rs";
}

class SetCC64_I<string opstr, PatFrag cond_op>:
  InstSE<(outs GPR64Opnd:$rt), (ins GPR64Opnd:$rs, simm10_64:$imm10),
         !strconcat(opstr, "\t$rt, $rs, $imm10"),
         [(set GPR64Opnd:$rt, (zext (cond_op GPR64Opnd:$rs,
                                             immSExt10_64:$imm10)))],
         II_SEQI_SNEI, FrmI, opstr> {
  let TwoOperandAliasConstraint = "$rt = $rs";
}

class CBranchBitNum<string opstr, DAGOperand opnd, PatFrag cond_op,
                    RegisterOperand RO, Operand ImmOp, bits<64> shift = 1> :
  InstSE<(outs), (ins RO:$rs, ImmOp:$p, opnd:$offset),
         !strconcat(opstr, "\t$rs, $p, $offset"),
         [(brcond (i32 (cond_op (and RO:$rs, (shl shift, immZExt5_64:$p)), 0)),
                  bb:$offset)], II_BBIT, FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
}

class MFC2OP<string asmstr, RegisterOperand RO> :
  InstSE<(outs RO:$rt, uimm16:$imm16), (ins),
         !strconcat(asmstr, "\t$rt, $imm16"), [], NoItinerary, FrmFR>;

// Unsigned Byte Add
let Pattern = [(set GPR64Opnd:$rd,
                    (and (add GPR64Opnd:$rs, GPR64Opnd:$rt), 255))] in
def BADDu  : ArithLogicR<"baddu", GPR64Opnd, 1, II_BADDU>,
                              ADD_FM<0x1c, 0x28>;

// Branch on Bit Clear /+32
def BBIT0  : CBranchBitNum<"bbit0", brtarget, seteq, GPR64Opnd,
                           uimm5_64_report_uimm6>, BBIT_FM<0x32>;
def BBIT032: CBranchBitNum<"bbit032", brtarget, seteq, GPR64Opnd, uimm5_64,
                           0x100000000>,
                           BBIT_FM<0x36>;

// Branch on Bit Set /+32
def BBIT1  : CBranchBitNum<"bbit1", brtarget, setne, GPR64Opnd,
                           uimm5_64_report_uimm6>, BBIT_FM<0x3a>;
def BBIT132: CBranchBitNum<"bbit132", brtarget, setne, GPR64Opnd, uimm5_64,
                           0x100000000>, BBIT_FM<0x3e>;

// Multiply Doubleword to GPR
let Defs = [HI0, LO0, P0, P1, P2] in
def DMUL  : ArithLogicR<"dmul", GPR64Opnd, 1, II_DMUL, mul>,
                              ADD_FM<0x1c, 0x03>;

// Extract a signed bit field /+32
def EXTS  : ExtsCins<"exts">, EXTS_FM<0x3a>;
def EXTS32: ExtsCins<"exts32">, EXTS_FM<0x3b>;

// Clear and insert a bit field /+32
def CINS  : ExtsCins<"cins">, EXTS_FM<0x32>;
def CINS32: ExtsCins<"cins32">, EXTS_FM<0x33>;

// Move to multiplier/product register
def MTM0   : MoveToLOHI<"mtm0", GPR64Opnd, [MPL0, P0, P1, P2]>, MTMR_FM<0x08>;
def MTM1   : MoveToLOHI<"mtm1", GPR64Opnd, [MPL1, P0, P1, P2]>, MTMR_FM<0x0c>;
def MTM2   : MoveToLOHI<"mtm2", GPR64Opnd, [MPL2, P0, P1, P2]>, MTMR_FM<0x0d>;
def MTP0   : MoveToLOHI<"mtp0", GPR64Opnd, [P0]>, MTMR_FM<0x09>;
def MTP1   : MoveToLOHI<"mtp1", GPR64Opnd, [P1]>, MTMR_FM<0x0a>;
def MTP2   : MoveToLOHI<"mtp2", GPR64Opnd, [P2]>, MTMR_FM<0x0b>;

// Count Ones in a Word/Doubleword
def POP   : Count1s<"pop", GPR32Opnd>, POP_FM<0x2c>;
def DPOP  : Count1s<"dpop", GPR64Opnd>, POP_FM<0x2d>;

// Set on equal/not equal
def SEQ   : SetCC64_R<"seq", seteq>, SEQ_FM<0x2a>;
def SEQi  : SetCC64_I<"seqi", seteq>, SEQI_FM<0x2e>;
def SNE   : SetCC64_R<"sne", setne>, SEQ_FM<0x2b>;
def SNEi  : SetCC64_I<"snei", setne>, SEQI_FM<0x2f>;

// 192-bit x 64-bit Unsigned Multiply and Add
let Defs = [P0, P1, P2] in
def V3MULU: ArithLogicR<"v3mulu", GPR64Opnd, 0, II_DMUL>,
                                  ADD_FM<0x1c, 0x11>;

// 64-bit Unsigned Multiply and Add Move
let Defs = [MPL0, P0, P1, P2] in
def VMM0  : ArithLogicR<"vmm0", GPR64Opnd, 0, II_DMUL>,
                                ADD_FM<0x1c, 0x10>;

// 64-bit Unsigned Multiply and Add
let Defs = [MPL1, MPL2, P0, P1, P2] in
def VMULU : ArithLogicR<"vmulu", GPR64Opnd, 0, II_DMUL>,
                                 ADD_FM<0x1c, 0x0f>;

// Move between CPU and coprocessor registers
def DMFC2_OCTEON : MFC2OP<"dmfc2", GPR64Opnd>, MFC2OP_FM<0x12, 1>;
def DMTC2_OCTEON : MFC2OP<"dmtc2", GPR64Opnd>, MFC2OP_FM<0x12, 5>;
}

}

/// Move between CPU and coprocessor registers
let DecoderNamespace = "Mips64", Predicates = [HasMips64] in {
def DMFC0 : MFC3OP<"dmfc0", GPR64Opnd, COP0Opnd>, MFC3OP_FM<0x10, 1>, ISA_MIPS3;
def DMTC0 : MTC3OP<"dmtc0", COP0Opnd, GPR64Opnd>, MFC3OP_FM<0x10, 5>, ISA_MIPS3;
def DMFC2 : MFC3OP<"dmfc2", GPR64Opnd, COP2Opnd>, MFC3OP_FM<0x12, 1>, ISA_MIPS3;
def DMTC2 : MTC3OP<"dmtc2", COP2Opnd, GPR64Opnd>, MFC3OP_FM<0x12, 5>, ISA_MIPS3;
}

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// extended loads
def : MipsPat<(i64 (extloadi1  addr:$src)), (LB64 addr:$src)>;
def : MipsPat<(i64 (extloadi8  addr:$src)), (LB64 addr:$src)>;
def : MipsPat<(i64 (extloadi16 addr:$src)), (LH64 addr:$src)>;
def : MipsPat<(i64 (extloadi32 addr:$src)), (LW64 addr:$src)>;

// hi/lo relocs
def : MipsPat<(MipsHi tglobaladdr:$in), (LUi64 tglobaladdr:$in)>;
def : MipsPat<(MipsHi tblockaddress:$in), (LUi64 tblockaddress:$in)>;
def : MipsPat<(MipsHi tjumptable:$in), (LUi64 tjumptable:$in)>;
def : MipsPat<(MipsHi tconstpool:$in), (LUi64 tconstpool:$in)>;
def : MipsPat<(MipsHi tglobaltlsaddr:$in), (LUi64 tglobaltlsaddr:$in)>;
def : MipsPat<(MipsHi texternalsym:$in), (LUi64 texternalsym:$in)>;

def : MipsPat<(MipsLo tglobaladdr:$in), (DADDiu ZERO_64, tglobaladdr:$in)>;
def : MipsPat<(MipsLo tblockaddress:$in), (DADDiu ZERO_64, tblockaddress:$in)>;
def : MipsPat<(MipsLo tjumptable:$in), (DADDiu ZERO_64, tjumptable:$in)>;
def : MipsPat<(MipsLo tconstpool:$in), (DADDiu ZERO_64, tconstpool:$in)>;
def : MipsPat<(MipsLo tglobaltlsaddr:$in),
              (DADDiu ZERO_64, tglobaltlsaddr:$in)>;
def : MipsPat<(MipsLo texternalsym:$in), (DADDiu ZERO_64, texternalsym:$in)>;

def : MipsPat<(add GPR64:$hi, (MipsLo tglobaladdr:$lo)),
              (DADDiu GPR64:$hi, tglobaladdr:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tblockaddress:$lo)),
              (DADDiu GPR64:$hi, tblockaddress:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tjumptable:$lo)),
              (DADDiu GPR64:$hi, tjumptable:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tconstpool:$lo)),
              (DADDiu GPR64:$hi, tconstpool:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tglobaltlsaddr:$lo)),
              (DADDiu GPR64:$hi, tglobaltlsaddr:$lo)>;

def : WrapperPat<tglobaladdr, DADDiu, GPR64>;
def : WrapperPat<tconstpool, DADDiu, GPR64>;
def : WrapperPat<texternalsym, DADDiu, GPR64>;
def : WrapperPat<tblockaddress, DADDiu, GPR64>;
def : WrapperPat<tjumptable, DADDiu, GPR64>;
def : WrapperPat<tglobaltlsaddr, DADDiu, GPR64>;

defm : BrcondPats<GPR64, BEQ64, BNE64, SLT64, SLTu64, SLTi64, SLTiu64,
                  ZERO_64>;

def : MipsPat<(brcond (i32 (setlt i64:$lhs, 1)), bb:$dst),
              (BLEZ64 i64:$lhs, bb:$dst)>;
def : MipsPat<(brcond (i32 (setgt i64:$lhs, -1)), bb:$dst),
              (BGEZ64 i64:$lhs, bb:$dst)>;

// setcc patterns
defm : SeteqPats<GPR64, SLTiu64, XOR64, SLTu64, ZERO_64>;
defm : SetlePats<GPR64, SLT64, SLTu64>;
defm : SetgtPats<GPR64, SLT64, SLTu64>;
defm : SetgePats<GPR64, SLT64, SLTu64>;
defm : SetgeImmPats<GPR64, SLTi64, SLTiu64>;

// truncate
def : MipsPat<(trunc (assertsext GPR64:$src)),
              (EXTRACT_SUBREG GPR64:$src, sub_32)>;
def : MipsPat<(trunc (assertzext GPR64:$src)),
              (EXTRACT_SUBREG GPR64:$src, sub_32)>;
def : MipsPat<(i32 (trunc GPR64:$src)),
              (SLL (EXTRACT_SUBREG GPR64:$src, sub_32), 0)>;

// variable shift instructions patterns
def : MipsPat<(shl GPR64:$rt, (i32 (trunc GPR64:$rs))),
              (DSLLV GPR64:$rt, (EXTRACT_SUBREG GPR64:$rs, sub_32))>;
def : MipsPat<(srl GPR64:$rt, (i32 (trunc GPR64:$rs))),
              (DSRLV GPR64:$rt, (EXTRACT_SUBREG GPR64:$rs, sub_32))>;
def : MipsPat<(sra GPR64:$rt, (i32 (trunc GPR64:$rs))),
              (DSRAV GPR64:$rt, (EXTRACT_SUBREG GPR64:$rs, sub_32))>;
def : MipsPat<(rotr GPR64:$rt, (i32 (trunc GPR64:$rs))),
              (DROTRV GPR64:$rt, (EXTRACT_SUBREG GPR64:$rs, sub_32))>;

// 32-to-64-bit extension
def : MipsPat<(i64 (anyext GPR32:$src)),
              (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPR32:$src, sub_32)>;
def : MipsPat<(i64 (zext GPR32:$src)), (DSRL (DSLL64_32 GPR32:$src), 32)>;
def : MipsPat<(i64 (sext GPR32:$src)), (SLL64_32 GPR32:$src)>;

// Sign extend in register
def : MipsPat<(i64 (sext_inreg GPR64:$src, i32)),
              (SLL64_64 GPR64:$src)>;

// bswap MipsPattern
def : MipsPat<(bswap GPR64:$rt), (DSHD (DSBH GPR64:$rt))>;

// Carry pattern
def : MipsPat<(subc GPR64:$lhs, GPR64:$rhs),
              (DSUBu GPR64:$lhs, GPR64:$rhs)>;
let AdditionalPredicates = [NotDSP] in {
  def : MipsPat<(addc GPR64:$lhs, GPR64:$rhs),
                (DADDu GPR64:$lhs, GPR64:$rhs)>;
  def : MipsPat<(addc GPR64:$lhs, immSExt16:$imm),
                (DADDiu GPR64:$lhs, imm:$imm)>;
}

// Octeon bbit0/bbit1 MipsPattern
let Predicates = [HasMips64, HasCnMips] in {
def : MipsPat<(brcond (i32 (seteq (and i64:$lhs, PowerOf2LO:$mask), 0)), bb:$dst),
              (BBIT0 i64:$lhs, (Log2LO PowerOf2LO:$mask), bb:$dst)>;
def : MipsPat<(brcond (i32 (seteq (and i64:$lhs, PowerOf2HI:$mask), 0)), bb:$dst),
              (BBIT032 i64:$lhs, (Log2HI PowerOf2HI:$mask), bb:$dst)>;
def : MipsPat<(brcond (i32 (setne (and i64:$lhs, PowerOf2LO:$mask), 0)), bb:$dst),
              (BBIT1 i64:$lhs, (Log2LO PowerOf2LO:$mask), bb:$dst)>;
def : MipsPat<(brcond (i32 (setne (and i64:$lhs, PowerOf2HI:$mask), 0)), bb:$dst),
              (BBIT132 i64:$lhs, (Log2HI PowerOf2HI:$mask), bb:$dst)>;
}

// Atomic load patterns.
def : MipsPat<(atomic_load_8 addr:$a), (LB64 addr:$a)>;
def : MipsPat<(atomic_load_16 addr:$a), (LH64 addr:$a)>;
def : MipsPat<(atomic_load_32 addr:$a), (LW64 addr:$a)>;
def : MipsPat<(atomic_load_64 addr:$a), (LD addr:$a)>;

// Atomic store patterns.
def : MipsPat<(atomic_store_8 addr:$a, GPR64:$v), (SB64 GPR64:$v, addr:$a)>;
def : MipsPat<(atomic_store_16 addr:$a, GPR64:$v), (SH64 GPR64:$v, addr:$a)>;
def : MipsPat<(atomic_store_32 addr:$a, GPR64:$v), (SW64 GPR64:$v, addr:$a)>;
def : MipsPat<(atomic_store_64 addr:$a, GPR64:$v), (SD GPR64:$v, addr:$a)>;

//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
def : MipsInstAlias<"move $dst, $src",
                    (OR64 GPR64Opnd:$dst,  GPR64Opnd:$src, ZERO_64), 1>,
      GPR_64;
def : MipsInstAlias<"move $dst, $src",
                    (DADDu GPR64Opnd:$dst,  GPR64Opnd:$src, ZERO_64), 1>,
      GPR_64;
def : MipsInstAlias<"daddu $rs, $rt, $imm",
                    (DADDiu GPR64Opnd:$rs, GPR64Opnd:$rt, simm16_64:$imm),
                    0>, ISA_MIPS3;
def : MipsInstAlias<"dadd $rs, $rt, $imm",
                    (DADDi GPR64Opnd:$rs, GPR64Opnd:$rt, simm16_64:$imm),
                    0>, ISA_MIPS3_NOT_32R6_64R6;
def : MipsInstAlias<"daddu $rs, $imm",
                    (DADDiu GPR64Opnd:$rs, GPR64Opnd:$rs, simm16_64:$imm),
                    0>, ISA_MIPS3;
def : MipsInstAlias<"dadd $rs, $imm",
                    (DADDi GPR64Opnd:$rs, GPR64Opnd:$rs, simm16_64:$imm),
                    0>, ISA_MIPS3_NOT_32R6_64R6;
def : MipsInstAlias<"dsll $rd, $rt, $rs",
                    (DSLLV GPR64Opnd:$rd, GPR64Opnd:$rt, GPR32Opnd:$rs), 0>,
                    ISA_MIPS3;
def : MipsInstAlias<"dneg $rt, $rs",
                    (DSUB GPR64Opnd:$rt, ZERO_64, GPR64Opnd:$rs), 1>,
                    ISA_MIPS3;
def : MipsInstAlias<"dneg $rt",
                    (DSUB GPR64Opnd:$rt, ZERO_64, GPR64Opnd:$rt), 0>,
                    ISA_MIPS3;
def : MipsInstAlias<"dnegu $rt, $rs",
                    (DSUBu GPR64Opnd:$rt, ZERO_64, GPR64Opnd:$rs), 1>,
                    ISA_MIPS3;
def : MipsInstAlias<"dsubu $rt, $rs, $imm",
                    (DADDiu GPR64Opnd:$rt, GPR64Opnd:$rs,
                            InvertedImOperand64:$imm), 0>, ISA_MIPS3;
def : MipsInstAlias<"dsubi $rs, $rt, $imm",
                    (DADDi GPR64Opnd:$rs, GPR64Opnd:$rt,
                           InvertedImOperand64:$imm),
                    0>, ISA_MIPS3_NOT_32R6_64R6;
def : MipsInstAlias<"dsubi $rs, $imm",
                    (DADDi GPR64Opnd:$rs, GPR64Opnd:$rs,
                           InvertedImOperand64:$imm),
                    0>, ISA_MIPS3_NOT_32R6_64R6;
def : MipsInstAlias<"dsub $rs, $rt, $imm",
                    (DADDi GPR64Opnd:$rs, GPR64Opnd:$rt,
                           InvertedImOperand64:$imm),
                    0>, ISA_MIPS3_NOT_32R6_64R6;
def : MipsInstAlias<"dsub $rs, $imm",
                    (DADDi GPR64Opnd:$rs, GPR64Opnd:$rs,
                           InvertedImOperand64:$imm),
                    0>, ISA_MIPS3_NOT_32R6_64R6;
def : MipsInstAlias<"dsubu $rs, $imm",
                    (DADDiu GPR64Opnd:$rs, GPR64Opnd:$rs,
                            InvertedImOperand64:$imm),
                    0>, ISA_MIPS3;
def : MipsInstAlias<"dsra $rd, $rt, $rs",
                    (DSRAV GPR64Opnd:$rd, GPR64Opnd:$rt, GPR32Opnd:$rs), 0>,
                    ISA_MIPS3;
def : MipsInstAlias<"dsrl $rd, $rt, $rs",
                    (DSRLV GPR64Opnd:$rd, GPR64Opnd:$rt, GPR32Opnd:$rs), 0>,
                    ISA_MIPS3;

// Two operand (implicit 0 selector) versions:
def : MipsInstAlias<"dmfc0 $rt, $rd", (DMFC0 GPR64Opnd:$rt, COP0Opnd:$rd, 0), 0>;
def : MipsInstAlias<"dmtc0 $rt, $rd", (DMTC0 COP0Opnd:$rd, GPR64Opnd:$rt, 0), 0>;
def : MipsInstAlias<"dmfc2 $rt, $rd", (DMFC2 GPR64Opnd:$rt, COP2Opnd:$rd, 0), 0>;
def : MipsInstAlias<"dmtc2 $rt, $rd", (DMTC2 COP2Opnd:$rd, GPR64Opnd:$rt, 0), 0>;

let Predicates = [HasMips64, HasCnMips] in {
def : MipsInstAlias<"synciobdma", (SYNC 0x2), 0>;
def : MipsInstAlias<"syncs",      (SYNC 0x6), 0>;
def : MipsInstAlias<"syncw",      (SYNC 0x4), 0>;
def : MipsInstAlias<"syncws",     (SYNC 0x5), 0>;
}

// cnMIPS Aliases.

// bbit* with $p 32-63 converted to bbit*32 with $p 0-31
def : MipsInstAlias<"bbit0 $rs, $p, $offset",
                    (BBIT032 GPR64Opnd:$rs, uimm5_plus32_normalize_64:$p,
                             brtarget:$offset), 0>,
      ASE_CNMIPS;
def : MipsInstAlias<"bbit1 $rs, $p, $offset",
                    (BBIT132 GPR64Opnd:$rs, uimm5_plus32_normalize_64:$p,
                             brtarget:$offset), 0>,
      ASE_CNMIPS;

// exts with $pos 32-63 in converted to exts32 with $pos 0-31
def : MipsInstAlias<"exts $rt, $rs, $pos, $lenm1",
                    (EXTS32 GPR64Opnd:$rt, GPR64Opnd:$rs,
                            uimm5_plus32_normalize:$pos, uimm5:$lenm1), 0>,
      ASE_CNMIPS;
def : MipsInstAlias<"exts $rt, $pos, $lenm1",
                    (EXTS32 GPR64Opnd:$rt, GPR64Opnd:$rt,
                            uimm5_plus32_normalize:$pos, uimm5:$lenm1), 0>,
      ASE_CNMIPS;

// cins with $pos 32-63 in converted to cins32 with $pos 0-31
def : MipsInstAlias<"cins $rt, $rs, $pos, $lenm1",
                    (CINS32 GPR64Opnd:$rt, GPR64Opnd:$rs,
                            uimm5_plus32_normalize:$pos, uimm5:$lenm1), 0>,
      ASE_CNMIPS;
def : MipsInstAlias<"cins $rt, $pos, $lenm1",
                    (CINS32 GPR64Opnd:$rt, GPR64Opnd:$rt,
                            uimm5_plus32_normalize:$pos, uimm5:$lenm1), 0>,
      ASE_CNMIPS;

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//

class LoadImmediate64<string instr_asm, Operand Od, RegisterOperand RO> :
  MipsAsmPseudoInst<(outs RO:$rt), (ins Od:$imm64),
                     !strconcat(instr_asm, "\t$rt, $imm64")> ;
def LoadImm64 : LoadImmediate64<"dli", imm64, GPR64Opnd>;

def LoadAddrReg64 : MipsAsmPseudoInst<(outs GPR64Opnd:$rt), (ins mem:$addr),
                                       "dla\t$rt, $addr">;
def LoadAddrImm64 : MipsAsmPseudoInst<(outs GPR64Opnd:$rt), (ins imm64:$imm64),
                                       "dla\t$rt, $imm64">;
