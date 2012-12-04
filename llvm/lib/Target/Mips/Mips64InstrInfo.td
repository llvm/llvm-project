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
  return getImm(N, (unsigned)N->getZExtValue() - 32);
}]>;

// shamt must fit in 6 bits.
def immZExt6 : ImmLeaf<i32, [{return Imm == (Imm & 0x3f);}]>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//
// Shifts
// 64-bit shift instructions.
let DecoderNamespace = "Mips64" in {
class shift_rotate_imm64<bits<6> func, bits<5> isRotate, string instr_asm,
                         SDNode OpNode>:
  shift_rotate_imm<func, isRotate, instr_asm, OpNode, immZExt6, shamt,
                   CPU64Regs>;

// Mul, Div
class Mult64<bits<6> func, string instr_asm, InstrItinClass itin>:
  Mult<func, instr_asm, itin, CPU64Regs, [HI64, LO64]>;
class Div64<SDNode op, bits<6> func, string instr_asm, InstrItinClass itin>:
  Div<op, func, instr_asm, itin, CPU64Regs, [HI64, LO64]>;

multiclass Atomic2Ops64<PatFrag Op, string Opstr> {
  def #NAME# : Atomic2Ops<Op, Opstr, CPU64Regs, CPURegs>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : Atomic2Ops<Op, Opstr, CPU64Regs, CPU64Regs>,
               Requires<[IsN64, HasStandardEncoding]> {
    let isCodeGenOnly = 1;
  }
}

multiclass AtomicCmpSwap64<PatFrag Op, string Width>  {
  def #NAME# : AtomicCmpSwap<Op, Width, CPU64Regs, CPURegs>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : AtomicCmpSwap<Op, Width, CPU64Regs, CPU64Regs>,
               Requires<[IsN64, HasStandardEncoding]> {
    let isCodeGenOnly = 1;
  }
}
}
let usesCustomInserter = 1, Predicates = [HasMips64, HasStandardEncoding],
  DecoderNamespace = "Mips64" in {
  defm ATOMIC_LOAD_ADD_I64  : Atomic2Ops64<atomic_load_add_64, "load_add_64">;
  defm ATOMIC_LOAD_SUB_I64  : Atomic2Ops64<atomic_load_sub_64, "load_sub_64">;
  defm ATOMIC_LOAD_AND_I64  : Atomic2Ops64<atomic_load_and_64, "load_and_64">;
  defm ATOMIC_LOAD_OR_I64   : Atomic2Ops64<atomic_load_or_64, "load_or_64">;
  defm ATOMIC_LOAD_XOR_I64  : Atomic2Ops64<atomic_load_xor_64, "load_xor_64">;
  defm ATOMIC_LOAD_NAND_I64 : Atomic2Ops64<atomic_load_nand_64, "load_nand_64">;
  defm ATOMIC_SWAP_I64      : Atomic2Ops64<atomic_swap_64, "swap_64">;
  defm ATOMIC_CMP_SWAP_I64  : AtomicCmpSwap64<atomic_cmp_swap_64, "64">;
}

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
let DecoderNamespace = "Mips64" in {
/// Arithmetic Instructions (ALU Immediate)
def DADDi    : ArithOverflowI<0x18, "daddi", add, simm16_64, immSExt16,
                           CPU64Regs>;
def DADDiu   : ArithLogicI<0x19, "daddiu", add, simm16_64, immSExt16,
                           CPU64Regs>, IsAsCheapAsAMove;
def DANDi    : ArithLogicI<0x0c, "andi", and, uimm16_64, immZExt16, CPU64Regs>;
def SLTi64   : SetCC_I<0x0a, "slti", setlt, simm16_64, immSExt16, CPU64Regs>;
def SLTiu64  : SetCC_I<0x0b, "sltiu", setult, simm16_64, immSExt16, CPU64Regs>;
def ORi64    : ArithLogicI<0x0d, "ori", or, uimm16_64, immZExt16, CPU64Regs>;
def XORi64   : ArithLogicI<0x0e, "xori", xor, uimm16_64, immZExt16, CPU64Regs>;
def LUi64    : LoadUpper<0x0f, "lui", CPU64Regs, uimm16_64>;

/// Arithmetic Instructions (3-Operand, R-Type)
def DADD     : ArithOverflowR<0x00, 0x2C, "dadd", IIAlu, CPU64Regs, 1>;
def DADDu    : ArithLogicR<0x00, 0x2d, "daddu", add, IIAlu, CPU64Regs, 1>;
def DSUBu    : ArithLogicR<0x00, 0x2f, "dsubu", sub, IIAlu, CPU64Regs>;
def SLT64    : SetCC_R<0x00, 0x2a, "slt", setlt, CPU64Regs>;
def SLTu64   : SetCC_R<0x00, 0x2b, "sltu", setult, CPU64Regs>;
def AND64    : ArithLogicR<0x00, 0x24, "and", and, IIAlu, CPU64Regs, 1>;
def OR64     : ArithLogicR<0x00, 0x25, "or", or, IIAlu, CPU64Regs, 1>;
def XOR64    : ArithLogicR<0x00, 0x26, "xor", xor, IIAlu, CPU64Regs, 1>;
def NOR64    : LogicNOR<0x00, 0x27, "nor", CPU64Regs>;

/// Shift Instructions
def DSLL     : shift_rotate_imm64<0x38, 0x00, "dsll", shl>;
def DSRL     : shift_rotate_imm64<0x3a, 0x00, "dsrl", srl>;
def DSRA     : shift_rotate_imm64<0x3b, 0x00, "dsra", sra>;
def DSLLV    : shift_rotate_reg<0x14, 0x00, "dsllv", shl, CPU64Regs>;
def DSRLV    : shift_rotate_reg<0x16, 0x00, "dsrlv", srl, CPU64Regs>;
def DSRAV    : shift_rotate_reg<0x17, 0x00, "dsrav", sra, CPU64Regs>;
let Pattern = []<dag> in {
  def DSLL32   : shift_rotate_imm64<0x3c, 0x00, "dsll32", shl>;
  def DSRL32   : shift_rotate_imm64<0x3e, 0x00, "dsrl32", srl>;
  def DSRA32   : shift_rotate_imm64<0x3f, 0x00, "dsra32", sra>;
}
}
// Rotate Instructions
let Predicates = [HasMips64r2, HasStandardEncoding],
    DecoderNamespace = "Mips64" in {
  def DROTR    : shift_rotate_imm64<0x3a, 0x01, "drotr", rotr>;
  def DROTRV   : shift_rotate_reg<0x16, 0x01, "drotrv", rotr, CPU64Regs>;
}

let DecoderNamespace = "Mips64" in {
/// Load and Store Instructions
///  aligned
defm LB64    : LoadM64<0x20, "lb",  sextloadi8>;
defm LBu64   : LoadM64<0x24, "lbu", zextloadi8>;
defm LH64    : LoadM64<0x21, "lh",  sextloadi16>;
defm LHu64   : LoadM64<0x25, "lhu", zextloadi16>;
defm LW64    : LoadM64<0x23, "lw",  sextloadi32>;
defm LWu64   : LoadM64<0x27, "lwu", zextloadi32>;
defm SB64    : StoreM64<0x28, "sb", truncstorei8>;
defm SH64    : StoreM64<0x29, "sh", truncstorei16>;
defm SW64    : StoreM64<0x2b, "sw", truncstorei32>;
defm LD      : LoadM64<0x37, "ld",  load>;
defm SD      : StoreM64<0x3f, "sd", store>;

/// load/store left/right
let isCodeGenOnly = 1 in {
  defm LWL64 : LoadLeftRightM64<0x22, "lwl", MipsLWL>;
  defm LWR64 : LoadLeftRightM64<0x26, "lwr", MipsLWR>;
  defm SWL64 : StoreLeftRightM64<0x2a, "swl", MipsSWL>;
  defm SWR64 : StoreLeftRightM64<0x2e, "swr", MipsSWR>;
}
defm LDL   : LoadLeftRightM64<0x1a, "ldl", MipsLDL>;
defm LDR   : LoadLeftRightM64<0x1b, "ldr", MipsLDR>;
defm SDL   : StoreLeftRightM64<0x2c, "sdl", MipsSDL>;
defm SDR   : StoreLeftRightM64<0x2d, "sdr", MipsSDR>;

/// Load-linked, Store-conditional
def LLD    : LLBase<0x34, "lld", CPU64Regs, mem>,
             Requires<[NotN64, HasStandardEncoding]>;
def LLD_P8 : LLBase<0x34, "lld", CPU64Regs, mem64>,
             Requires<[IsN64, HasStandardEncoding]> {
  let isCodeGenOnly = 1;
}
def SCD    : SCBase<0x3c, "scd", CPU64Regs, mem>,
             Requires<[NotN64, HasStandardEncoding]>;
def SCD_P8 : SCBase<0x3c, "scd", CPU64Regs, mem64>,
             Requires<[IsN64, HasStandardEncoding]> {
  let isCodeGenOnly = 1;
}

/// Jump and Branch Instructions
def JR64   : IndirectBranch<CPU64Regs>;
def BEQ64  : CBranch<0x04, "beq", seteq, CPU64Regs>;
def BNE64  : CBranch<0x05, "bne", setne, CPU64Regs>;
def BGEZ64 : CBranchZero<0x01, 1, "bgez", setge, CPU64Regs>;
def BGTZ64 : CBranchZero<0x07, 0, "bgtz", setgt, CPU64Regs>;
def BLEZ64 : CBranchZero<0x06, 0, "blez", setle, CPU64Regs>;
def BLTZ64 : CBranchZero<0x01, 0, "bltz", setlt, CPU64Regs>;
}
let DecoderNamespace = "Mips64" in
def JALR64 : JumpLinkReg<0x00, 0x09, "jalr", CPU64Regs>;
def TAILCALL64_R : JumpFR<CPU64Regs, MipsTailCall>, IsTailCall;

let DecoderNamespace = "Mips64" in {
/// Multiply and Divide Instructions.
def DMULT    : Mult64<0x1c, "dmult", IIImul>;
def DMULTu   : Mult64<0x1d, "dmultu", IIImul>;
def DSDIV    : Div64<MipsDivRem, 0x1e, "ddiv", IIIdiv>;
def DUDIV    : Div64<MipsDivRemU, 0x1f, "ddivu", IIIdiv>;

def MTHI64 : MoveToLOHI<0x11, "mthi", CPU64Regs, [HI64]>;
def MTLO64 : MoveToLOHI<0x13, "mtlo", CPU64Regs, [LO64]>;
def MFHI64 : MoveFromLOHI<0x10, "mfhi", CPU64Regs, [HI64]>;
def MFLO64 : MoveFromLOHI<0x12, "mflo", CPU64Regs, [LO64]>;

/// Sign Ext In Register Instructions.
def SEB64 : SignExtInReg<0x10, "seb", i8, CPU64Regs>;
def SEH64 : SignExtInReg<0x18, "seh", i16, CPU64Regs>;

/// Count Leading
def DCLZ : CountLeading0<0x24, "dclz", CPU64Regs>;
def DCLO : CountLeading1<0x25, "dclo", CPU64Regs>;

/// Double Word Swap Bytes/HalfWords
def DSBH : SubwordSwap<0x24, 0x2, "dsbh", CPU64Regs>;
def DSHD : SubwordSwap<0x24, 0x5, "dshd", CPU64Regs>;

def LEA_ADDiu64 : EffectiveAddress<0x19,"daddiu\t$rt, $addr", CPU64Regs, mem_ea_64>;
}
let Uses = [SP_64], DecoderNamespace = "Mips64" in
def DynAlloc64 : EffectiveAddress<0x19,"daddiu\t$rt, $addr", CPU64Regs, mem_ea_64>,
                 Requires<[IsN64, HasStandardEncoding]>;
let DecoderNamespace = "Mips64" in {
def RDHWR64 : ReadHardware<CPU64Regs, HWRegs64>;

def DEXT : ExtBase<3, "dext", CPU64Regs>;
let Pattern = []<dag> in {
  def DEXTU : ExtBase<2, "dextu", CPU64Regs>;
  def DEXTM : ExtBase<1, "dextm", CPU64Regs>;
}
def DINS : InsBase<7, "dins", CPU64Regs>;
let Pattern = []<dag> in {
  def DINSU : InsBase<6, "dinsu", CPU64Regs>;
  def DINSM : InsBase<5, "dinsm", CPU64Regs>;
}

let isCodeGenOnly = 1, rs = 0, shamt = 0 in {
  def DSLL64_32 : FR<0x00, 0x3c, (outs CPU64Regs:$rd), (ins CPURegs:$rt),
                     "dsll\t$rd, $rt, 32", [], IIAlu>;
  def SLL64_32 : FR<0x0, 0x00, (outs CPU64Regs:$rd), (ins CPURegs:$rt),
                    "sll\t$rd, $rt, 0", [], IIAlu>;
  def SLL64_64 : FR<0x0, 0x00, (outs CPU64Regs:$rd), (ins CPU64Regs:$rt),
                    "sll\t$rd, $rt, 0", [], IIAlu>;
}
}
//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// extended loads
let Predicates = [NotN64, HasStandardEncoding] in {
  def : MipsPat<(i64 (extloadi1  addr:$src)), (LB64 addr:$src)>;
  def : MipsPat<(i64 (extloadi8  addr:$src)), (LB64 addr:$src)>;
  def : MipsPat<(i64 (extloadi16 addr:$src)), (LH64 addr:$src)>;
  def : MipsPat<(i64 (extloadi32 addr:$src)), (LW64 addr:$src)>;
}
let Predicates = [IsN64, HasStandardEncoding] in {
  def : MipsPat<(i64 (extloadi1  addr:$src)), (LB64_P8 addr:$src)>;
  def : MipsPat<(i64 (extloadi8  addr:$src)), (LB64_P8 addr:$src)>;
  def : MipsPat<(i64 (extloadi16 addr:$src)), (LH64_P8 addr:$src)>;
  def : MipsPat<(i64 (extloadi32 addr:$src)), (LW64_P8 addr:$src)>;
}

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

def : MipsPat<(add CPU64Regs:$hi, (MipsLo tglobaladdr:$lo)),
              (DADDiu CPU64Regs:$hi, tglobaladdr:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tblockaddress:$lo)),
              (DADDiu CPU64Regs:$hi, tblockaddress:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tjumptable:$lo)),
              (DADDiu CPU64Regs:$hi, tjumptable:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tconstpool:$lo)),
              (DADDiu CPU64Regs:$hi, tconstpool:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tglobaltlsaddr:$lo)),
              (DADDiu CPU64Regs:$hi, tglobaltlsaddr:$lo)>;

def : WrapperPat<tglobaladdr, DADDiu, CPU64Regs>;
def : WrapperPat<tconstpool, DADDiu, CPU64Regs>;
def : WrapperPat<texternalsym, DADDiu, CPU64Regs>;
def : WrapperPat<tblockaddress, DADDiu, CPU64Regs>;
def : WrapperPat<tjumptable, DADDiu, CPU64Regs>;
def : WrapperPat<tglobaltlsaddr, DADDiu, CPU64Regs>;

defm : BrcondPats<CPU64Regs, BEQ64, BNE64, SLT64, SLTu64, SLTi64, SLTiu64,
                  ZERO_64>;

// setcc patterns
defm : SeteqPats<CPU64Regs, SLTiu64, XOR64, SLTu64, ZERO_64>;
defm : SetlePats<CPU64Regs, SLT64, SLTu64>;
defm : SetgtPats<CPU64Regs, SLT64, SLTu64>;
defm : SetgePats<CPU64Regs, SLT64, SLTu64>;
defm : SetgeImmPats<CPU64Regs, SLTi64, SLTiu64>;

// select MipsDynAlloc
def : MipsPat<(MipsDynAlloc addr:$f), (DynAlloc64 addr:$f)>,
      Requires<[IsN64, HasStandardEncoding]>;

// truncate
def : MipsPat<(i32 (trunc CPU64Regs:$src)),
              (SLL (EXTRACT_SUBREG CPU64Regs:$src, sub_32), 0)>,
      Requires<[IsN64, HasStandardEncoding]>;

// 32-to-64-bit extension
def : MipsPat<(i64 (anyext CPURegs:$src)), (SLL64_32 CPURegs:$src)>;
def : MipsPat<(i64 (zext CPURegs:$src)), (DSRL (DSLL64_32 CPURegs:$src), 32)>;
def : MipsPat<(i64 (sext CPURegs:$src)), (SLL64_32 CPURegs:$src)>;

// Sign extend in register
def : MipsPat<(i64 (sext_inreg CPU64Regs:$src, i32)),
              (SLL64_64 CPU64Regs:$src)>;

// bswap MipsPattern
def : MipsPat<(bswap CPU64Regs:$rt), (DSHD (DSBH CPU64Regs:$rt))>;

//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
def : InstAlias<"move $dst,$src", (DADD CPU64Regs:$dst,CPU64Regs:$src,ZERO_64)>;

/// Move between CPU and coprocessor registers
let DecoderNamespace = "Mips64" in {
def MFC0_3OP64  : MFC3OP<0x10, 0, (outs CPU64Regs:$rt), 
                       (ins CPU64Regs:$rd, uimm16:$sel),"mfc0\t$rt, $rd, $sel">;
def MTC0_3OP64  : MFC3OP<0x10, 4, (outs CPU64Regs:$rd, uimm16:$sel),
                       (ins CPU64Regs:$rt),"mtc0\t$rt, $rd, $sel">;
def MFC2_3OP64  : MFC3OP<0x12, 0, (outs CPU64Regs:$rt),
                       (ins CPU64Regs:$rd, uimm16:$sel),"mfc2\t$rt, $rd, $sel">;
def MTC2_3OP64  : MFC3OP<0x12, 4, (outs CPU64Regs:$rd, uimm16:$sel),
                       (ins CPU64Regs:$rt),"mtc2\t$rt, $rd, $sel">;
def DMFC0_3OP64  : MFC3OP<0x10, 1, (outs CPU64Regs:$rt), 
                       (ins CPU64Regs:$rd, uimm16:$sel),"dmfc0\t$rt, $rd, $sel">;
def DMTC0_3OP64  : MFC3OP<0x10, 5, (outs CPU64Regs:$rd, uimm16:$sel),
                       (ins CPU64Regs:$rt),"dmtc0\t$rt, $rd, $sel">;
def DMFC2_3OP64  : MFC3OP<0x12, 1, (outs CPU64Regs:$rt),
                       (ins CPU64Regs:$rd, uimm16:$sel),"dmfc2\t$rt, $rd, $sel">;
def DMTC2_3OP64  : MFC3OP<0x12, 5, (outs CPU64Regs:$rd, uimm16:$sel),
                       (ins CPU64Regs:$rt),"dmtc2\t$rt, $rd, $sel">;
}
// Two operand (implicit 0 selector) versions:
def : InstAlias<"mfc0 $rt, $rd", (MFC0_3OP64 CPU64Regs:$rt, CPU64Regs:$rd, 0)>;
def : InstAlias<"mtc0 $rt, $rd", (MTC0_3OP64 CPU64Regs:$rd, 0, CPU64Regs:$rt)>;
def : InstAlias<"mfc2 $rt, $rd", (MFC2_3OP64 CPU64Regs:$rt, CPU64Regs:$rd, 0)>;
def : InstAlias<"mtc2 $rt, $rd", (MTC2_3OP64 CPU64Regs:$rd, 0, CPU64Regs:$rt)>;
def : InstAlias<"dmfc0 $rt, $rd", (DMFC0_3OP64 CPU64Regs:$rt, CPU64Regs:$rd, 0)>;
def : InstAlias<"dmtc0 $rt, $rd", (DMTC0_3OP64 CPU64Regs:$rd, 0, CPU64Regs:$rt)>;
def : InstAlias<"dmfc2 $rt, $rd", (DMFC2_3OP64 CPU64Regs:$rt, CPU64Regs:$rd, 0)>;
def : InstAlias<"dmtc2 $rt, $rd", (DMTC2_3OP64 CPU64Regs:$rd, 0, CPU64Regs:$rt)>;

