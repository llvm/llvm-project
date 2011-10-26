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

// Instruction operand types
def shamt_64       : Operand<i64>;

// Unsigned Operand
def uimm16_64      : Operand<i64> {
  let PrintMethod = "printUnsignedImm";
}

// Transformation Function - get Imm - 32.
def Subtract32 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() - 32);
}]>;

// imm32_63 predicate - True if imm is in range [32, 63].
def imm32_63 : ImmLeaf<i64,
                       [{return (int32_t)Imm >= 32 && (int32_t)Imm < 64;}],
                       Subtract32>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//
// Shifts
class LogicR_shift_rotate_imm64<bits<6> func, bits<5> _rs, string instr_asm,
                                SDNode OpNode, PatFrag PF>:
  FR<0x00, func, (outs CPU64Regs:$rd), (ins CPU64Regs:$rt, shamt_64:$shamt),
     !strconcat(instr_asm, "\t$rd, $rt, $shamt"),
     [(set CPU64Regs:$rd, (OpNode CPU64Regs:$rt, (i64 PF:$shamt)))],
     IIAlu> {
  let rs = _rs;
}

class LogicR_shift_rotate_reg64<bits<6> func, bits<5> _shamt, string instr_asm,
                                SDNode OpNode>:
  FR<0x00, func, (outs CPU64Regs:$rd), (ins CPU64Regs:$rs, CPU64Regs:$rt),
     !strconcat(instr_asm, "\t$rd, $rt, $rs"),
     [(set CPU64Regs:$rd, (OpNode CPU64Regs:$rt, CPU64Regs:$rs))], IIAlu> {
  let shamt = _shamt;
}

// Mul, Div
let rd = 0, shamt = 0, Defs = [HI64, LO64] in {
  let isCommutable = 1 in
  class Mul64<bits<6> func, string instr_asm, InstrItinClass itin>:
    FR<0x00, func, (outs), (ins CPU64Regs:$rs, CPU64Regs:$rt),
       !strconcat(instr_asm, "\t$rs, $rt"), [], itin>;

  class Div64<SDNode op, bits<6> func, string instr_asm, InstrItinClass itin>:
              FR<0x00, func, (outs), (ins CPU64Regs:$rs, CPU64Regs:$rt),
              !strconcat(instr_asm, "\t$$zero, $rs, $rt"),
              [(op CPU64Regs:$rs, CPU64Regs:$rt)], itin>;
}

// Move from Hi/Lo
let shamt = 0 in {
let rs = 0, rt = 0 in
class MoveFromLOHI64<bits<6> func, string instr_asm>:
  FR<0x00, func, (outs CPU64Regs:$rd), (ins),
     !strconcat(instr_asm, "\t$rd"), [], IIHiLo>;

let rt = 0, rd = 0 in
class MoveToLOHI64<bits<6> func, string instr_asm>:
  FR<0x00, func, (outs), (ins CPU64Regs:$rs),
     !strconcat(instr_asm, "\t$rs"), [], IIHiLo>;
}

// Count Leading Ones/Zeros in Word
class CountLeading64<bits<6> func, string instr_asm, list<dag> pattern>:
  FR<0x1c, func, (outs CPU64Regs:$rd), (ins CPU64Regs:$rs),
     !strconcat(instr_asm, "\t$rd, $rs"), pattern, IIAlu>,
     Requires<[HasBitCount]> {
  let shamt = 0;
  let rt = rd;
}

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
def DADDiu   : ArithLogicI<0x19, "daddiu", add, simm16_64, immSExt16,
                           CPU64Regs>;
def DANDi    : ArithLogicI<0x0c, "andi", and, uimm16_64, immZExt16, CPU64Regs>;
def SLTi64   : SetCC_I<0x0a, "slti", setlt, simm16_64, immSExt16, CPU64Regs>;
def SLTiu64  : SetCC_I<0x0b, "sltiu", setult, simm16_64, immSExt16, CPU64Regs>;
def ORi64    : ArithLogicI<0x0d, "ori", or, uimm16_64, immZExt16, CPU64Regs>;
def XORi64   : ArithLogicI<0x0e, "xori", xor, uimm16_64, immZExt16, CPU64Regs>;

/// Arithmetic Instructions (3-Operand, R-Type)
def DADDu    : ArithLogicR<0x00, 0x2d, "daddu", add, IIAlu, CPU64Regs, 1>;
def DSUBu    : ArithLogicR<0x00, 0x2f, "dsubu", sub, IIAlu, CPU64Regs>;
def SLT64    : SetCC_R<0x00, 0x2a, "slt", setlt, CPU64Regs>;
def SLTu64   : SetCC_R<0x00, 0x2b, "sltu", setult, CPU64Regs>;
def AND64    : ArithLogicR<0x00, 0x24, "and", and, IIAlu, CPU64Regs, 1>;
def OR64     : ArithLogicR<0x00, 0x25, "or", or, IIAlu, CPU64Regs, 1>;
def XOR64    : ArithLogicR<0x00, 0x26, "xor", xor, IIAlu, CPU64Regs, 1>;
def NOR64    : LogicNOR<0x00, 0x27, "nor", CPU64Regs>;

/// Shift Instructions
def DSLL     : LogicR_shift_rotate_imm64<0x38, 0x00, "dsll", shl, immZExt5>;
def DSRL     : LogicR_shift_rotate_imm64<0x3a, 0x00, "dsrl", srl, immZExt5>;
def DSRA     : LogicR_shift_rotate_imm64<0x3b, 0x00, "dsra", sra, immZExt5>;
def DSLL32   : LogicR_shift_rotate_imm64<0x3c, 0x00, "dsll32", shl, imm32_63>;
def DSRL32   : LogicR_shift_rotate_imm64<0x3e, 0x00, "dsrl32", srl, imm32_63>;
def DSRA32   : LogicR_shift_rotate_imm64<0x3f, 0x00, "dsra32", sra, imm32_63>;
def DSLLV    : LogicR_shift_rotate_reg64<0x24, 0x00, "dsllv", shl>;
def DSRLV    : LogicR_shift_rotate_reg64<0x26, 0x00, "dsrlv", srl>;
def DSRAV    : LogicR_shift_rotate_reg64<0x27, 0x00, "dsrav", sra>;

// Rotate Instructions
let Predicates = [HasMips64r2] in {
  def DROTR    : LogicR_shift_rotate_imm64<0x3a, 0x01, "drotr", rotr, immZExt5>;
  def DROTR32  : LogicR_shift_rotate_imm64<0x3e, 0x01, "drotr32", rotr,
                                           imm32_63>;
  def DROTRV   : LogicR_shift_rotate_reg64<0x16, 0x01, "drotrv", rotr>;
}

/// Load and Store Instructions
///  aligned 
defm LB64    : LoadM64<0x20, "lb",  sextloadi8>;
defm LBu64   : LoadM64<0x24, "lbu", zextloadi8>;
defm LH64    : LoadM64<0x21, "lh",  sextloadi16_a>;
defm LHu64   : LoadM64<0x25, "lhu", zextloadi16_a>;
defm LW64    : LoadM64<0x23, "lw",  sextloadi32_a>;
defm LWu64   : LoadM64<0x27, "lwu", zextloadi32_a>;
defm SB64    : StoreM64<0x28, "sb", truncstorei8>;
defm SH64    : StoreM64<0x29, "sh", truncstorei16_a>;
defm SW64    : StoreM64<0x2b, "sw", truncstorei32_a>;
defm LD      : LoadM64<0x37, "ld",  load_a>;
defm SD      : StoreM64<0x3f, "sd", store_a>;

///  unaligned
defm ULH64     : LoadM64<0x21, "ulh",  sextloadi16_u, 1>;
defm ULHu64    : LoadM64<0x25, "ulhu", zextloadi16_u, 1>;
defm ULW64     : LoadM64<0x23, "ulw",  sextloadi32_u, 1>;
defm USH64     : StoreM64<0x29, "ush", truncstorei16_u, 1>;
defm USW64     : StoreM64<0x2b, "usw", truncstorei32_u, 1>;
defm ULD       : LoadM64<0x37, "uld",  load_u, 1>;
defm USD       : StoreM64<0x3f, "usd", store_u, 1>;

/// Jump and Branch Instructions
def BEQ64  : CBranch<0x04, "beq", seteq, CPU64Regs>;
def BNE64  : CBranch<0x05, "bne", setne, CPU64Regs>;
def BGEZ64 : CBranchZero<0x01, 1, "bgez", setge, CPU64Regs>;
def BGTZ64 : CBranchZero<0x07, 0, "bgtz", setgt, CPU64Regs>;
def BLEZ64 : CBranchZero<0x07, 0, "blez", setle, CPU64Regs>;
def BLTZ64 : CBranchZero<0x01, 0, "bltz", setlt, CPU64Regs>;

/// Multiply and Divide Instructions.
def DMULT    : Mul64<0x1c, "dmult", IIImul>;
def DMULTu   : Mul64<0x1d, "dmultu", IIImul>;
def DSDIV    : Div64<MipsDivRem, 0x1e, "ddiv", IIIdiv>;
def DUDIV    : Div64<MipsDivRemU, 0x1f, "ddivu", IIIdiv>;

let Defs = [HI64] in
  def MTHI64  : MoveToLOHI64<0x11, "mthi">;
let Defs = [LO64] in
  def MTLO64  : MoveToLOHI64<0x13, "mtlo">;

let Uses = [HI64] in
  def MFHI64  : MoveFromLOHI64<0x10, "mfhi">;
let Uses = [LO64] in
  def MFLO64  : MoveFromLOHI64<0x12, "mflo">;

/// Count Leading
def DCLZ : CountLeading64<0x24, "dclz",
                          [(set CPU64Regs:$rd, (ctlz CPU64Regs:$rs))]>;
def DCLO : CountLeading64<0x25, "dclo",
                          [(set CPU64Regs:$rd, (ctlz (not CPU64Regs:$rs)))]>;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : Pat<(i64 immSExt16:$in),
          (DADDiu ZERO_64, imm:$in)>;
def : Pat<(i64 immZExt16:$in),
          (ORi64 ZERO_64, imm:$in)>;

// zextloadi32_u
def : Pat<(zextloadi32_u addr:$a), (DSRL (DSLL (ULW64_P8 addr:$a), 32), 32)>,
      Requires<[IsN64]>;
def : Pat<(zextloadi32_u addr:$a), (DSRL (DSLL (ULW64 addr:$a), 32), 32)>,
      Requires<[NotN64]>;

// hi/lo relocs
def : Pat<(i64 (MipsLo tglobaladdr:$in)), (DADDiu ZERO_64, tglobaladdr:$in)>;

defm : BrcondPats<CPU64Regs, BEQ64, BNE64, SLT64, SLTu64, SLTi64, SLTiu64,
                  ZERO_64>;

// setcc patterns
defm : SeteqPats<CPU64Regs, SLTiu64, XOR64, SLTu64, ZERO_64>;
defm : SetlePats<CPU64Regs, SLT64, SLTu64>;
defm : SetgtPats<CPU64Regs, SLT64, SLTu64>;
defm : SetgePats<CPU64Regs, SLT64, SLTu64>;
defm : SetgeImmPats<CPU64Regs, SLTi64, SLTiu64>;
