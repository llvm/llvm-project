//===-- M88kInstrInfo.td - M88k Instructions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the M88k instructions in TableGen format.
//
//===----------------------------------------------------------------------===//


def IsMC881100 : Predicate<"Subtarget->getCPU() == \"mc88110\"">;
class MC881100 { list<Predicate> Predicates = [IsMC881100]; }

// ---------------------------------------------------------------------------//
// Selection DAG Nodes.
// ---------------------------------------------------------------------------//

// Selection DAG types.

// These are target-independent nodes, but have target-specific formats.
def SDT_CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_CallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_Call         : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;

// Selection DAG nodes.

def call             : SDNode<"M88kISD::CALL", SDT_Call,
                              [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                              SDNPVariadic]>;
def retflag          : SDNode<"M88kISD::RET_FLAG", SDTNone,
                              [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

def m88k_clr : SDNode<"M88kISD::CLR", SDTIntBinOp>;
def m88k_set : SDNode<"M88kISD::SET", SDTIntBinOp>;
def m88k_ext : SDNode<"M88kISD::EXT", SDTIntBinOp>;
def m88k_extu : SDNode<"M88kISD::EXTU", SDTIntBinOp>;
def m88k_mak : SDNode<"M88kISD::MAK", SDTIntBinOp>;
def m88k_rot : SDNode<"M88kISD::ROT", SDTIntBinOp>;
def m88k_ff1 : SDNode<"M88kISD::FF1", SDTIntBitCountUnaryOp>;
def m88k_ff0 : SDNode<"M88kISD::FF0", SDTIntBitCountUnaryOp>;

def m88k_bb0 : SDNode<"M88kISD::BB0", SDTIntBinOp>;
def m88k_bb1 : SDNode<"M88kISD::BB0", SDTIntBinOp>;
def m88k_bcond : SDNode<"M88kISD::BB0", SDTIntBinOp>;

// Hi16 and Lo16 nodes are used to handle global addresses.
def m88k_hi : SDNode<"M88kISD::Hi16", SDTIntUnaryOp>;
def m88k_lo : SDNode<"M88kISD::Lo16", SDTIntUnaryOp>;

// ---------------------------------------------------------------------------//
// Operands.
// ---------------------------------------------------------------------------//

class ImmediateAsmOperand<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addImmOperands";
}

class ImmediateOp<ValueType vt, string asmop> : Operand<vt> {
  let PrintMethod = "print"#asmop#"Operand";
  let DecoderMethod = "decode"#asmop#"Operand";
  let ParserMatchClass = !cast<AsmOperandClass>(asmop);
  let OperandType = "OPERAND_IMMEDIATE";
}

class ImmOpWithPattern<ValueType vt, string asmop, code pred, SDNodeXForm xform,
      SDNode ImmNode = imm> :
  ImmediateOp<vt, asmop>, PatLeaf<(vt ImmNode), pred, xform>;

multiclass Immediate<ValueType vt, code pred, SDNodeXForm xform, string asmop> {
  // def "" : ImmediateOp<vt, asmop>,
  //          PatLeaf<(vt imm), pred, xform>;
  def "" : ImmOpWithPattern<vt, asmop, pred, xform>;

//  def _timm : PatLeaf<(vt timm), pred, xform>;
  def _timm : ImmOpWithPattern<vt, asmop, pred, xform, timm>;
}

// Constructs an asm operand for a PC-relative address.  SIZE says how
// many bits there are.
class PCRelAsmOperand<string size> : ImmediateAsmOperand<"PCRel"#size> {
  let PredicateMethod = "isImm";
  let ParserMethod = "parsePCRel"#size;
}

// Constructs an operand for a PC-relative address with address type VT.
// ASMOP is the associated asm operand.
class PCRelOperand<ValueType vt, AsmOperandClass asmop> : Operand<vt> {
  let PrintMethod = "printPCRelOperand";
  let ParserMatchClass = asmop;
  let OperandType = "OPERAND_PCREL";
}


// Signed and unsigned operands.
def U5Imm : ImmediateAsmOperand<"U5Imm">;
def U5ImmO : ImmediateAsmOperand<"U5ImmO">  {
  let ParserMethod = "parseImmWO";
}
def U10ImmWO : ImmediateAsmOperand<"U10ImmWO"> {
  let ParserMethod = "parseImmWO";
}
def S16Imm : ImmediateAsmOperand<"S16Imm">;
def U16Imm : ImmediateAsmOperand<"U16Imm">;

// PC-relative asm operands.
def PCRel16 : PCRelAsmOperand<"16">;
def PCRel26 : PCRelAsmOperand<"26">;

// PC-relative offsets of a basic block.  The offset is sign-extended
// and shifted left by 2 bits.
def brtarget16 : PCRelOperand<OtherVT, PCRel16> {
  let EncoderMethod = "getPC16Encoding";
  let DecoderMethod = "decodePC16BranchOperand";
}

def brtarget26 : PCRelOperand<OtherVT, PCRel26> {
  let EncoderMethod = "getPC26Encoding";
  let DecoderMethod = "decodePC26BranchOperand";
}


// Extracting immediate operands from nodes.

// Bits 0-15.
def LO16 : SDNodeXForm<imm, [{
  uint32_t Value = N->getZExtValue() & 0x000000000000FFFFULL;
  return CurDAG->getTargetConstant(Value, SDLoc(N), MVT::i32);
}]>;

// Bits 16-31 (counting from the lsb).
def HI16 : SDNodeXForm<imm, [{
  uint64_t Value = (N->getZExtValue() & 0x00000000FFFF0000ULL) >> 16;
  return CurDAG->getTargetConstant(Value, SDLoc(N), MVT::i32);
}]>;


// Immediates for the lower and upper 16 bits of an i32, with the other
// bits of the i32 being zero.
defm imm32lo16 : Immediate<i32, [{
  return (N->getZExtValue() & ~0x000000000000ffffULL) == 0;
}], LO16, "U16Imm">;

defm imm32hi16 : Immediate<i32, [{
  return (N->getZExtValue() & ~0x00000000ffff0000ULL) == 0;
}], HI16, "U16Imm">;

// Immediates for the lower and upper 16 bits of an i32, with the other
// bits of the i32 being one.
defm imm32lo16c : Immediate<i32, [{
  return (uint32_t(~N->getZExtValue()) & ~0x000000000000ffffULL) == 0;
}], LO16, "U16Imm">;

defm imm32hi16c : Immediate<i32, [{
  return (uint32_t(~N->getZExtValue()) & ~0x00000000ffff0000ULL) == 0;
}], HI16, "U16Imm">;

defm imm32zx5 : Immediate<i32, [{
  return (N->getZExtValue() & ~0x000000000000001fULL) == 0;
}], NOOP_SDNodeXForm, "U5Imm">;

defm imm32zx16 : Immediate<i32, [{
  return (N->getZExtValue() & ~0x00000000000000ffULL) == 0;
}], NOOP_SDNodeXForm, "U16Imm">;

defm imm32zx5O : Immediate<i32, [{
  return (N->getZExtValue() & ~0x000000000000001fULL) == 0;
}], NOOP_SDNodeXForm, "U5ImmO">;

defm imm32zx10WO : Immediate<i32, [{
  return (N->getZExtValue() & ~0x00000000000003ffULL) == 0;
}], NOOP_SDNodeXForm, "U10ImmWO">;

// Predicate: Arbitrary 32 bit value.
def uimm32 : PatLeaf<(imm), [{
  uint64_t Val = N->getZExtValue();
  return isUInt<32>(Val) && (Val & 0xffff);
}]>;

// ---------------------------------------------------------------------------//
// Generic patterns.
// ---------------------------------------------------------------------------//

multiclass BinaryOpPat<string Inst, SDPatternOperator OpNode> {
  def : Pat<(OpNode GPR:$rs1, GPR:$rs2),
            (!cast<InstM88k>(Inst#"rr") GPR:$rs1, GPR:$rs2)>;
  def : Pat<(OpNode GPR:$rs1, imm32zx16:$imm16),
            (!cast<InstM88k>(Inst#"ri") GPR:$rs1, imm32zx16:$imm16)>;
}

class UnaryOpPat<InstM88k Inst, SDPatternOperator OpNode>
  : Pat<(OpNode GPR:$rs1), (Inst GPR:$rs1)>;

// ---------------------------------------------------------------------------//
// Logic and bit field instructions.
// ---------------------------------------------------------------------------//

// Multiclass for logical instructions with immediates.
// The pattern for "and" is slightly different.
multiclass LogicImm<bits<2> FuncI, string OpcStr, SDNode OpNode,
                  InstrItinClass itin = NoItinerary> {
  def ri  : F_LI<FuncI, 0b0,
                 (outs GPR:$rd),
                 !if(!eq(OpcStr, "and"),
                   (ins GPR:$rs1, imm32lo16c:$imm16),
                   (ins GPR:$rs1, imm32lo16:$imm16)
                 ),
                 OpcStr,
                 !if(!eq(OpcStr, "and"),
                   [(set i32:$rd, (OpNode GPR:$rs1, imm32lo16c:$imm16))],
                   [(set i32:$rd, (OpNode GPR:$rs1, imm32lo16:$imm16))]
                 ),
                 itin>;
  def riu  : F_LI<FuncI, 0b1,
                 (outs GPR:$rd),
                 !if(!eq(OpcStr, "and"),
                   (ins GPR:$rs1, imm32hi16c:$imm16),
                   (ins GPR:$rs1, imm32hi16:$imm16)
                 ),
                 !strconcat(OpcStr, ".u"),
                 !if(!eq(OpcStr, "and"),
                   [(set i32:$rd, (OpNode GPR:$rs1, imm32hi16c:$imm16))],
                   [(set i32:$rd, (OpNode GPR:$rs1, imm32hi16:$imm16))]
                 ),
                 itin>;
}

// Multiclass for logical instructions with triadic registers or immediates.
multiclass Logic<bits<5> FuncR, bits<2> FuncI, string OpcStr, SDNode OpNode,
                 InstrItinClass itin = NoItinerary>
  : LogicImm<FuncI, OpcStr, OpNode, itin> {
  let isCommutable = 1 in
    def rr  : F_LR<FuncR, 0b0, OpcStr,
                   [(set i32:$rd, (OpNode GPR:$rs1, GPR:$rs2))],
                   itin>;
  def rrc : F_LR<FuncR, 0b1, !strconcat(OpcStr, ".c"),
                 [(set i32:$rd, (OpNode GPR:$rs1, (not GPR:$rs2)))],
                 itin>;
}

defm MASK : LogicImm<0b01, "mask", and>;
defm AND : Logic<0b01000, 0b00, "and", and>;
defm XOR : Logic<0b01010, 0b10, "xor", xor>;
defm OR  : Logic<0b01011, 0b11, "or", or>;

// Pattern for 32 bit constants.
def : Pat<(and GPR:$rs1, uimm32:$imm),
          (ANDri (ANDriu GPR:$rs1, (HI16 i32:$imm)), (LO16 i32:$imm))>;
def : Pat<(or GPR:$rs1, uimm32:$imm),
          (ORri (ORriu GPR:$rs1, (HI16 i32:$imm)), (LO16 i32:$imm))>;
def : Pat<(xor GPR:$rs1, uimm32:$imm),
          (XORri (XORriu GPR:$rs1, (HI16 i32:$imm)), (LO16 i32:$imm))>;

// Arbitrary constants.
// 16-bit immediate.
def : Pat<(i32 imm32lo16:$imm), (ORri (i32 R0), imm32lo16:$imm)>;

// Arbitrary immediate.
def : Pat<(i32 imm:$imm),
          (ORri (ORriu (i32 R0), (HI16 i32:$imm)), (LO16 i32:$imm))>;

// Multiclass for bit-field instructions with triadic registers or immediates.
multiclass Bitfield<bits<6> Func, string OpcStr, SDNode OpNode, PatLeaf ImmOp,
                 InstrItinClass itin = NoItinerary> {
  def rr  : F_BR<Func, OpcStr,
                 [(set GPR:$rd, (OpNode GPR:$rs1, GPR:$rs2))],
                 itin>;
  def rwo : F_BI<Func,
                 (outs GPR:$rd), (ins GPR:$rs1, ImmOp:$w5o5),
                 OpcStr,
                 [(set GPR:$rd, (OpNode GPR:$rs1, ImmOp:$w5o5))],
                 itin>;
}

defm CLR  : Bitfield<0b100000, "clr", m88k_clr, imm32zx10WO>;
defm SET  : Bitfield<0b100010, "set", m88k_set, imm32zx10WO>;
defm EXT  : Bitfield<0b100100, "ext", m88k_ext, imm32zx10WO>;
defm EXTU : Bitfield<0b100110, "extu", m88k_extu, imm32zx10WO>;
defm MAK  : Bitfield<0b101000, "mak", m88k_mak, imm32zx10WO>;
defm ROT  : Bitfield<0b101010, "rot", m88k_rot, imm32zx5O>;

// Pattern for shifts
def : Pat<(sra GPR:$rs1, GPR:$rs2), (EXTrr GPR:$rs1, GPR:$rs2)>;
def : Pat<(srl GPR:$rs1, GPR:$rs2), (EXTUrr GPR:$rs1, GPR:$rs2)>;
def : Pat<(shl GPR:$rs1, GPR:$rs2), (MAKrr GPR:$rs1, GPR:$rs2)>;
def : Pat<(rotr GPR:$rs1, GPR:$rs2), (ROTrr GPR:$rs1, GPR:$rs2)>;
def : Pat<(sra GPR:$rs1, imm32zx5O:$o5), (EXTrwo GPR:$rs1, imm32zx5O:$o5)>;
def : Pat<(srl GPR:$rs1, imm32zx5O:$o5), (EXTUrwo GPR:$rs1, imm32zx5O:$o5)>;
def : Pat<(shl GPR:$rs1, imm32zx5O:$o5), (MAKrwo GPR:$rs1, imm32zx5O:$o5)>;
def : Pat<(rotr GPR:$rs1, imm32zx5O:$o5), (ROTrwo GPR:$rs1, imm32zx5O:$o5)>;

class FindBF<bits<6> Func, string OpcStr, SDNode OpNode,
            InstrItinClass itin = NoItinerary> :
  F_BF<Func, (outs GPR:$rd), (ins GPR:$rs2),
             OpcStr,
             [(set GPR:$rd, (OpNode GPR:$rs2))],
             itin>;

def FF1rr : FindBF<0b111010, "ff1", m88k_ff1>;
def FF0rr : FindBF<0b111011, "ff0", m88k_ff0>;

// ctlz = 32 - ff1
//def : Pat<(ctlz GPR:$rs1, GPR:$rs2), (SUBri (i32 32), (FF1rr GPR:$rs1, GPR:$rs2))>;

// ---------------------------------------------------------------------------//
// Integer arithmetic instructions.
// ---------------------------------------------------------------------------//

// Multiclass for arithmetic instructions with triadic registers or immediates.
multiclass ArithTri<bits<6> Func, string OpcStr,
                    InstrItinClass itin = NoItinerary> {
  def rr : F_IRC<Func, 0b0, 0b0, OpcStr, [], itin>;
  def ri : F_II<Func, (ins GPR:$rs1, imm32zx16:$imm16),OpcStr, [], itin>;
}

// Multiclass for arithmetic instructions with triadic registers or immediates.
multiclass ArithTriCarry<bits<6> Func, string OpcStr,
                         InstrItinClass itin = NoItinerary>
  : ArithTri<Func, OpcStr, itin> {
  def rrci  : F_IRC<Func, 0b1, 0b0, !strconcat(OpcStr, ".ci"), [], itin>;
  def rrco  : F_IRC<Func, 0b0, 0b1, !strconcat(OpcStr, ".co"), [], itin>;
  def rrcio : F_IRC<Func, 0b1, 0b1, !strconcat(OpcStr, ".cio"), [], itin>;
}

let isCommutable = 1 in {
  defm ADDU : ArithTriCarry<0b011000, "addu">;
  defm MULU : ArithTri<0b011011, "mulu">;
  defm ADD : ArithTriCarry<0b011100, "add">;
}
defm SUBU : ArithTriCarry<0b011001, "subu">;
defm DIVU : ArithTri<0b011010, "divu">;
defm SUB : ArithTriCarry<0b011101, "sub">;
defm DIVS : ArithTri<0b011110, "divs">;
defm CMP : ArithTri<0b011111, "cmp">;

let Predicates = [IsMC881100] in {
  def DIVUrrd : F_IRCD<0b011010, "divu.d">;
  let isCommutable = 1 in
    def MULUrrd : F_IRCD<0b011011, "mulu.d">;
}

// TODO Aliases div -> divs, mul -> mulu. Renames from 88100

defm : BinaryOpPat<"MULU", mul>;
defm : BinaryOpPat<"DIVS", sdiv>; // TODO Correct?
//def : Pat<(umullohi GPR:$rs1, GPR:$rs2), (MULUrrd GPR:$rs1, GPR:$rs2)>;


multiclass SetCCPat<CondCode CC, bits<10> w5o5> {
  def : Pat<(setcc (i32 GPR:$lhs), (i32 GPR:$rhs), CC),
            (EXTUrwo (CMPrr GPR:$lhs, GPR:$rhs), w5o5)>;
  def : Pat<(setcc (i32 GPR:$lhs), imm32zx16:$rhs, CC),
            (EXTUrwo (CMPrr GPR:$lhs, imm32zx16:$rhs), w5o5)>;
}

defm : SetCCPat<SETEQ, 34>;
defm : SetCCPat<SETNE, 35>;
defm : SetCCPat<SETGT, 36>;
defm : SetCCPat<SETLE, 37>;
defm : SetCCPat<SETLT, 38>;
defm : SetCCPat<SETGE, 39>;
defm : SetCCPat<SETUGT, 40>;
defm : SetCCPat<SETULE, 41>;
defm : SetCCPat<SETULT, 42>;
defm : SetCCPat<SETUGE, 43>;

/*
Addressing modes:
Register indirect with immediate index: ld rD, rS1, si16
Register indirect with index:           ld rD, rS1, rS2
Register indirect with scaled index:    ld rD, rS1, [rS2]
*/
def ADDRrr : ComplexPattern<i32, 2, "selectAddrRr", [], []>;
def ADDRri : ComplexPattern<i32, 2, "selectAddrRi", [], []>;
def ADDRrs : ComplexPattern<i32, 1, "selectAddrRs", [], []>;

// Multiclass for load and store instructions.
multiclass LoadIndexImm<bits<4> Func, string OpcStr, SDPatternOperator OpNode,
                 InstrItinClass itin = NoItinerary> {
  def riw  : F_LS<Func, /*ty=*/ 0b01,
                  (outs GPR:$rd), (ins GPR:$rs1, imm32zx16:$si16),
                  OpcStr,
                  [(set (i32 GPR:$rd), (load (ADDRri (i32 GPR:$rs1), imm32zx16:$si16)))],
                  itin>;
  def rib  : F_LS<Func, /*ty=*/ 0b11,
                  (outs GPR:$rd), (ins GPR:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".b"),
                  [(set (i32 GPR:$rd), (sextloadi8 (ADDRri (i32 GPR:$rs1), imm32zx16:$si16)))],
                  itin>;
  def rih  : F_LS<Func, /*ty=*/ 0b10,
                  (outs GPR:$rd), (ins GPR:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".h"),
                  [(set (i32 GPR:$rd), (sextloadi16 (ADDRri (i32 GPR:$rs1), imm32zx16:$si16)))],
                  itin>;
  def rid  : F_LS<Func, /*ty=*/ 0b00,
                  (outs GPR64:$rd), (ins GPR:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".d"),
                  [(set (i64 GPR64:$rd), (load (ADDRri (i32 GPR:$rs1), imm32zx16:$si16)))],
                  itin>;
}

multiclass StoreIndexImm<bits<4> Func, string OpcStr, SDPatternOperator OpNode,
                 InstrItinClass itin = NoItinerary> {
  def riw  : F_LS<Func, /*ty=*/ 0b01,
                  (outs), (ins GPR:$rd, GPR:$rs1, imm32zx16:$si16),
                  OpcStr,
                  [(store (i32 GPR:$rd), (ADDRri (i32 GPR:$rs1), imm32zx16:$si16))],
                  itin>;
  def rib  : F_LS<Func, /*ty=*/ 0b11,
                  (outs), (ins GPR:$rd, GPR:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".b"),
                  [(truncstorei8 (i32 GPR:$rd), (ADDRri (i32 GPR:$rs1), imm32zx16:$si16))],
                  itin>;
  def rih  : F_LS<Func, /*ty=*/ 0b10,
                  (outs), (ins GPR:$rd, GPR:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".h"),
                  [(truncstorei16 (i32 GPR:$rd), (ADDRri (i32 GPR:$rs1), imm32zx16:$si16))],
                  itin>;
  def rid  : F_LS<Func, /*ty=*/ 0b00,
                  (outs), (ins GPR64:$rd, GPR:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".d"),
                  [(store (i64 GPR64:$rd), (ADDRri (i32 GPR:$rs1), imm32zx16:$si16))],
                  itin>;
}

multiclass LoadUnsigned<InstrItinClass itin = NoItinerary> {
  def urih : F_LU<0b0, (outs GPR:$rd), (ins GPR:$rs1, imm32zx16:$si16),
                  "ld.hu",
                  [(set (i32 GPR:$rd), (zextloadi16 (ADDRri (i32 GPR:$rs1), imm32zx16:$si16)))],
                  itin>;
  def urib : F_LU<0b1, (outs GPR:$rd), (ins GPR:$rs1, imm32zx16:$si16),
                  "ld.bu",
                  [(set (i32 GPR:$rd), (zextloadi8 (ADDRri (i32 GPR:$rs1), imm32zx16:$si16)))],
                  itin>;
}

class LoadXR<bits<6> Func, string OpcStr, ValueType VT, InstrItinClass itin = NoItinerary>
  :  F_LX<Func, (outs XR:$rd), (ins GPR:$rs1, imm32zx16:$si16),
          OpcStr,
          [(set (VT XR:$rd), (zextloadi8 (ADDRri (i32 GPR:$rs1), imm32zx16:$si16)))],
          itin>;

multiclass LoadUnscaled<bits<4> Func, string OpcStr, SDPatternOperator OpNode,
                 InstrItinClass itin = NoItinerary> {
  let regfile = 1 in {
    def rrud  : F_LSINDUSC<Func, /*ty=*/ 0b00, /*user=*/ 0b0,
                    (outs GPR64:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".d"),
                    [(set (i64 GPR64:$rd), (load (ADDRrr (i32 GPR:$rs1), (i32 GPR:$rs2))))],
                    itin>;
    def rruw  : F_LSINDUSC<Func, /*ty=*/ 0b01, /*user=*/ 0b0,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    OpcStr,
                    [(set (i32 GPR:$rd), (load (ADDRrr (i32 GPR:$rs1), (i32 GPR:$rs2))))],
                    itin>;
    def rrub  : F_LSINDUSC<Func, /*ty=*/ 0b11, /*user=*/ 0b0,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".b"),
                    [(set (i32 GPR:$rd), (sextloadi8 (ADDRrr (i32 GPR:$rs1), (i32 GPR:$rs2))))],
                    itin>;
    def rruh  : F_LSINDUSC<Func, /*ty=*/ 0b10, /*user=*/ 0b0,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".h"),
                    [(set (i32 GPR:$rd), (sextloadi16 (ADDRrr (i32 GPR:$rs1), (i32 GPR:$rs2))))],
                    itin>;
    def rrudu : F_LSINDUSC<Func, /*ty=*/ 0b00, /*user=*/ 0b1,
                    (outs GPR64:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".d.usr"),
                    [],
                    itin>;
    def rruwu : F_LSINDUSC<Func, /*ty=*/ 0b01, /*user=*/ 0b1,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".usr"),
                    [],
                    itin>;
    def rrubu : F_LSINDUSC<Func, /*ty=*/ 0b11, /*user=*/ 0b1,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".b.usr"),
                    [],
                    itin>;
    def rruhu : F_LSINDUSC<Func, /*ty=*/ 0b10, /*user=*/ 0b1,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".h.usr"),
                    [],
                    itin>;
  }
  let regfile = 0, Predicates = [IsMC881100] in {
    def xxuw  : F_LSINDUSC<Func, /*ty=*/ 0b01, /*user=*/ 0b0,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    OpcStr,
                    [],
                    itin>;
    def xxud  : F_LSINDUSC<Func, /*ty=*/ 0b00, /*user=*/ 0b0,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".d"),
                    [],
                    itin>;
    def xxuq  : F_LSINDUSC<Func, /*ty=*/ 0b10, /*user=*/ 0b0,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".x"),
                    [],
                    itin>;
    def xxuwu : F_LSINDUSC<Func, /*ty=*/ 0b01, /*user=*/ 0b1,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".usr"),
                    [],
                    itin>;
    def xxudu : F_LSINDUSC<Func, /*ty=*/ 0b00, /*user=*/ 0b1,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".d.usr"),
                    [],
                    itin>;
    def xxuqu : F_LSINDUSC<Func, /*ty=*/ 0b10, /*user=*/ 0b1,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".x.usr"),
                    [],
                    itin>;
  }
}

multiclass LoadScaled<bits<4> Func, string OpcStr, SDPatternOperator OpNode,
                 InstrItinClass itin = NoItinerary> {
  let regfile = 1 in {
    def rrsd  : F_LSINDSC<Func, /*ty=*/ 0b00, /*user=*/ 0b0,
                    (outs GPR64:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".d"),
                    [],
                    itin>;
    def rrsw  : F_LSINDSC<Func, /*ty=*/ 0b01, /*user=*/ 0b0,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    OpcStr,
                    [],
                    itin>;
    def rrsb  : F_LSINDSC<Func, /*ty=*/ 0b11, /*user=*/ 0b0,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".b"),
                    [],
                    itin>;
    def rrsh  : F_LSINDSC<Func, /*ty=*/ 0b10, /*user=*/ 0b0,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".h"),
                    [],
                    itin>;
    def rrsdu : F_LSINDSC<Func, /*ty=*/ 0b00, /*user=*/ 0b1,
                    (outs GPR64:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".d.usr"),
                    [],
                    itin>;
    def rrswu : F_LSINDSC<Func, /*ty=*/ 0b01, /*user=*/ 0b1,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".usr"),
                    [],
                    itin>;
    def rrsbu : F_LSINDSC<Func, /*ty=*/ 0b11, /*user=*/ 0b1,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".b.usr"),
                    [],
                    itin>;
    def rrshu : F_LSINDSC<Func, /*ty=*/ 0b10, /*user=*/ 0b1,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    !strconcat(OpcStr, ".h.usr"),
                    [],
                    itin>;
  }
  let regfile = 0, Predicates = [IsMC881100] in {
    def xxsw  : F_LSINDSC<Func, /*ty=*/ 0b01, /*user=*/ 0b0,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    OpcStr,
                    [],
                    itin>;
    def xxsd  : F_LSINDSC<Func, /*ty=*/ 0b00, /*user=*/ 0b0,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".d"),
                    [],
                    itin>;
    def xxsq  : F_LSINDSC<Func, /*ty=*/ 0b10, /*user=*/ 0b0,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".x"),
                    [],
                    itin>;
    def xxswu : F_LSINDSC<Func, /*ty=*/ 0b01, /*user=*/ 0b1,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".usr"),
                    [],
                    itin>;
    def xxsdu : F_LSINDSC<Func, /*ty=*/ 0b00, /*user=*/ 0b1,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".d.usr"),
                    [],
                    itin>;
    def xxsqu : F_LSINDSC<Func, /*ty=*/ 0b10, /*user=*/ 0b1,
                    (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                    !strconcat(OpcStr, ".x.usr"),
                    [],
                    itin>;
  }
}

defm LD : LoadIndexImm<0b0001, "ld", load>;
defm ST : StoreIndexImm<0b0010, "st", store>;

defm LD : LoadUnsigned<>;

defm LD : LoadUnscaled<0b0001, "ld", load>;
defm LD : LoadScaled<0b0001, "ld", load>;

let Predicates = [IsMC881100] in {
  def LDxri : LoadXR<0b000001, "ld", f32>;
  def LDxrid : LoadXR<0b000000, "ld.d", f64>;
  def LDxrix : LoadXR<0b001111, "ld.x", f80>;
}

//def : Pat<(load (ADDRri GPR:$rs1, imm32zx16:$si16)),
//          (LDriw GPR:$rs1, imm32zx16:$si16)>;



// GlobalAddress, ExternalSymbol, Jumptable, ConstantPool
def : Pat<(m88k_hi tglobaladdr:$dst), (ORriu (i32 R0), tglobaladdr:$dst)>;
//def : Pat<(m88k_lo tglobaladdr:$dst), (OR_I_LO (i32 R0), tglobaladdr:$dst)>;

// ---------------------------------------------------------------------------//
// Flow control
// ---------------------------------------------------------------------------//

let isTerminator = 1, isBarrier = 1 in {
  def JMP : F_JMP<0b11000, (outs), (ins GPR:$rs2),
                  "jmp", [(brind GPR:$rs2)]>;
  let hasDelaySlot = 1 in
    def JMPn : F_JMP<0b11000, (outs), (ins GPR:$rs2),
                     "jmp.n", [(brind GPR:$rs2)]>;
}

let isCall = 1, isTerminator = 1, isBarrier = 1 in {
  def JSR : F_JMP<0b11001, (outs), (ins GPR:$rs2),
                  "jsr", []>;
  let hasDelaySlot = 1 in
    def JSRn : F_JMP<0b11001, (outs), (ins GPR:$rs2),
                     "jsr.n", []>;
}

let isReturn = 1, isTerminator = 1, isBarrier = 1, Uses = [R1] in {
  def RET : Pseudo<(outs), (ins), [(retflag)]>;
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  def BR : F_BRANCH<0b11000, (outs), (ins brtarget26:$d26), "br",
                    [(br bb:$d26)]>;

  let hasDelaySlot = 1 in
    def BRn : F_BRANCH<0b11000, (outs), (ins brtarget26:$d26), "br.n",
                       [(br bb:$d26)]>;

  def BB0 : F_BCOND<0b11010,
                    (outs), (ins imm32zx5:$m5, GPR:$rs1, brtarget16:$d16),
                    "bb0">;
  def BB1 : F_BCOND<0b11011,
                    (outs), (ins imm32zx5:$m5, GPR:$rs1, brtarget16:$d16),
                    "bb1">;
  def BCOND : F_BCOND<0b11101,
                      (outs), (ins imm32zx5:$m5, GPR:$rs1, brtarget16:$d16),
                      "bcnd">;
  let hasDelaySlot = 1 in {
    def BB0n : F_BCOND<0b11010,
                       (outs), (ins imm32zx5:$m5, GPR:$rs1, brtarget16:$d16),
                       "bb0.n">;
    def BB1n : F_BCOND<0b11011,
                      (outs), (ins imm32zx5:$m5, GPR:$rs1, brtarget16:$d16),
                      "bb1.n">;
    def BCONDn : F_BCOND<0b11101,
                        (outs), (ins imm32zx5:$m5, GPR:$rs1, brtarget16:$d16),
                        "bcnd.n">;
  }
}

// Compares against zero use bcond instruction.
def : Pat<(brcc SETEQ, GPR:$rs1, 0, bb:$dst),
          (BCOND 0b00010, GPR:$rs1, bb:$dst)>;
def : Pat<(brcc SETNE, GPR:$rs1, 0, bb:$dst),
          (BCOND 0b01101, GPR:$rs1, bb:$dst)>;
def : Pat<(brcc SETGT, GPR:$rs1, 0, bb:$dst),
          (BCOND 0b00001, GPR:$rs1, bb:$dst)>;
def : Pat<(brcc SETLT, GPR:$rs1, 0, bb:$dst),
          (BCOND 0b01100, GPR:$rs1, bb:$dst)>;
def : Pat<(brcc SETGE, GPR:$rs1, 0, bb:$dst),
          (BCOND 0b00011, GPR:$rs1, bb:$dst)>;
def : Pat<(brcc SETLE, GPR:$rs1, 0, bb:$dst),
          (BCOND 0b01110, GPR:$rs1, bb:$dst)>;

// Generate patterns for brcc and bcond nodes.
multiclass BranchPat<CondCode CC, PatFrag ccop, bits<5> m5> {
  def : Pat<(brcc CC, (i32 GPR:$rs1), (i32 GPR:$rs2), bb:$dst),
             (BB1 m5, (CMPrr GPR:$rs1, GPR:$rs2), bb:$dst)>;
  def : Pat<(brcc CC, (i32 GPR:$rs1), imm32zx16:$imm16, bb:$dst),
             (BB1 m5, (CMPri GPR:$rs1, imm32zx16:$imm16), bb:$dst)>;
  def : Pat<(brcond (i32 (ccop (i32 GPR:$lhs), (i32 GPR:$rhs))), bb:$dst),
             (BB1 m5, (CMPrr GPR:$lhs, GPR:$rhs), bb:$dst)>;
  def : Pat<(brcond (i32 (ccop (i32 GPR:$lhs), imm32zx16:$rhs)), bb:$dst),
             (BB1 m5, (CMPri GPR:$lhs, imm32zx16:$rhs), bb:$dst)>;
}

// Compare integer registers / integer register and immediate.
defm : BranchPat<SETEQ, seteq, 2>;
defm : BranchPat<SETNE, setne, 3>;
defm : BranchPat<SETGT, setgt, 4>;
defm : BranchPat<SETLE, setle, 5>;
defm : BranchPat<SETLT, setlt, 6>;
defm : BranchPat<SETGE, setge, 7>;
defm : BranchPat<SETUGT, setugt, 8>;
defm : BranchPat<SETULE, setule, 9>;
defm : BranchPat<SETULT, setult, 10>;
defm : BranchPat<SETUGE, setuge, 11>;

let isTerminator = 1, isBarrier = 1 in {
  def RTE : F_RTEILLOP<0b00, "rte">;
  def ILLOP1 : F_RTEILLOP<0b01, "illop1">;
  def ILLOP2 : F_RTEILLOP<0b10, "illop2">;
  def ILLOP3 : F_RTEILLOP<0b11, "illop3">;
}

// Alias for no operation.
def : InstAlias<"nop", (ORrr R0, R0, R0)>;

// ---------------------------------------------------------------------------//
// Floating point instructions.
// ---------------------------------------------------------------------------//

// Helper classes to select floating point operands.

class FT<string p, RegisterOperand FReg> {
  bits<1> regfile = 0;
  string prec = p;
  RegisterOperand opnd = FReg;
}

def FT32 : FT<"s", GPR>;
def FT64 : FT<"d", GPR64>;
def FT80 : FT<"x", ?>;

class getFT<int n> {
  FT ret = !if(!eq(n, 0), FT32, !if(!eq(n, 1), FT64, FT80));
}

// Multiclass for triadic register floating point instructions.
multiclass FArithG<bits<4> Func, string OpcStr,
                   InstrItinClass itin = NoItinerary> {
  foreach D = 0-1 in {
    foreach S1 = 0-1 in {
      foreach S2 = 0-1 in {
        let regfile = 0 in {
          defvar DPrec = getFT<D>.ret.prec;
          defvar DOpnd = getFT<D>.ret.opnd;
          defvar S1Prec = getFT<S1>.ret.prec;
          defvar S1Opnd = getFT<S1>.ret.opnd;
          defvar S2Prec = getFT<S2>.ret.prec;
          defvar S2Opnd = getFT<S2>.ret.opnd;
          def g # DPrec # S1Prec # S2Prec :
                  F_SFU1<Func, D, S1, S2,
                      (outs DOpnd:$rd), (ins S1Opnd:$rs1, S2Opnd:$rs2),
                      !strconcat(OpcStr, ".", DPrec, S1Prec, S2Prec),
                          [], itin>;
        }
      }
    }
  }
}

multiclass FArithX<bits<4> Func, string OpcStr,
                   InstrItinClass itin = NoItinerary> {
  foreach D = 0-2 in {
    foreach S1 = 0-2 in {
      foreach S2 = 0-2 in {
        let regfile = 1, Predicates = [IsMC881100] in {
          defvar DPrec = getFT<D>.ret.prec;
          defvar S1Prec = getFT<S1>.ret.prec;
          defvar S2Prec = getFT<S2>.ret.prec;
          def x # DPrec # S1Prec # S2Prec :
                  F_SFU1<Func, D, S1, S2,
                      (outs XR:$rd), (ins XR:$rs1, XR:$rs2),
                      !strconcat(OpcStr, ".", DPrec, S1Prec, S2Prec),
                          [], itin>;
        }
      }
    }
  }
}

multiclass FArith<bits<4> Func, string OpcStr,
                  InstrItinClass itin = NoItinerary>
                  : FArithG<Func, OpcStr, itin>, FArithX<Func, OpcStr, itin>;

// Multiclass for dyadic register floating point instructions.
multiclass FArith2<bits<4> Func, string OpcStr,
                   InstrItinClass itin = NoItinerary> {
  foreach D = 0-1 in {
    foreach S2 = 0-1 in {
      defvar IsFPR = !and(!ne(D, 2), !ne(S2, 2));
      if IsFPR then {
        let regfile = 0 in {
          defvar DPrec = getFT<D>.ret.prec;
          defvar DOpnd = getFT<D>.ret.opnd;
          defvar S2Prec = getFT<S2>.ret.prec;
          defvar S2Opnd = getFT<S2>.ret.opnd;
          def DPrec # S2Prec :
              F_SFU1D<Func, D, 0b00, S2,
                      (outs DOpnd:$rd), (ins S2Opnd:$rs2),
                      !strconcat(OpcStr, ".", DPrec, S2Prec),
                      [], itin>;
        }
      }
    }
  }
}

// TODO Why does ftrunc not work?
multiclass FloatBinPat<string Inst, SDPatternOperator OpNode> {
  def : Pat<(OpNode GPR:$rs1, GPR:$rs2),
            (!cast<InstM88k>(Inst#"gsss") GPR:$rs1, GPR:$rs2)>;
//  def : Pat<(ftrunc (OpNode (fpextend GPR:$rs1), GPR64:$rs2)),
//            (!cast<InstM88k>(Inst#"ssd") GPR:$rs1, GPR:$rs2)>;
  def : Pat<(fpextend (OpNode GPR:$rs1, GPR:$rs2)),
            (!cast<InstM88k>(Inst#"gdss") GPR:$rs1, GPR:$rs2)>;
  def : Pat<(OpNode (fpextend GPR:$rs1), GPR64:$rs2),
            (!cast<InstM88k>(Inst#"gdsd") GPR:$rs1, GPR64:$rs2)>;
  def : Pat<(OpNode GPR64:$rs1, (fpextend GPR:$rs2)),
            (!cast<InstM88k>(Inst#"gdds") GPR64:$rs1, GPR:$rs2)>;
  def : Pat<(OpNode GPR64:$rs1, GPR64:$rs2),
            (!cast<InstM88k>(Inst#"gddd") GPR64:$rs1, GPR64:$rs2)>;

  def : Pat<(OpNode (f32 XR:$rs1), (f32 XR:$rs2)),
            (!cast<InstM88k>(Inst#"xsss") XR:$rs1, XR:$rs2)>;
  def : Pat<(OpNode (f64 XR:$rs1), (fpextend (f32 XR:$rs2))),
            (!cast<InstM88k>(Inst#"gdds") XR:$rs1, XR:$rs2)>;
  def : Pat<(OpNode (f64 XR:$rs1), (f64 XR:$rs2)),
            (!cast<InstM88k>(Inst#"xddd") XR:$rs1, XR:$rs2)>;
  def : Pat<(OpNode (f80 XR:$rs1), (f80 XR:$rs2)),
            (!cast<InstM88k>(Inst#"xxxx") XR:$rs1, XR:$rs2)>;
}

defm FMUL : FArith<0b0000, "fmul">;
defm FADD : FArith<0b0101, "fadd">;
defm FSUB : FArith<0b0110, "fsub">;
defm FDIV : FArith<0b1110, "fdiv">;

defm : FloatBinPat<"FMUL", fmul>;
defm : FloatBinPat<"FADD", fadd>;
defm : FloatBinPat<"FSUB", fsub>;
defm : FloatBinPat<"FDIV", fdiv>;

defm INT : FArith2<0b1001, "int">;
defm NINT : FArith2<0b1010, "nint">;
defm TRNC : FArith2<0b1011, "trnc">;

let Predicates = [IsMC881100] in {
defm FCVT : FArith2<0b0001, "fcvt">;
defm FSQRT : FArith2<0b1111, "fsqrt">;
}

let regfile = 1, Predicates = [IsMC881100] in {
def MOVrxs : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b00, /*t2*/0b00,
                     (outs GPR:$rd), (ins XR:$rs2),
                     "mov.s",
                     [(set GPR:$rd, (f32 XR:$rs2))]>;
def MOVrxd : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b00, /*t2*/0b01,
                     (outs GPR64:$rd), (ins XR:$rs2),
                     "mov.d",
                     [(set GPR64:$rd, (f64 XR:$rs2))]>;
def MOVxx : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b01, /*t2*/0b11,
                    (outs XR:$rd), (ins XR:$rs2),
                    "mov",
                    []>;
//                    [(set XR:$rd, XR:$rs2)]>;
}

let regfile = 0, Predicates = [IsMC881100] in {
def MOVxrs : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b01, /*t2*/0b00,
                     (outs XR:$rd), (ins GPR:$rs2),
                     "mov.s",
                     [(set (f32 XR:$rd), GPR:$rs2)]>;
def MOVxrd : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b01, /*t2*/0b01,
                     (outs XR:$rd), (ins GPR64:$rs2),
                     "mov.d",
                     [(set (f64 XR:$rd), GPR64:$rs2)]>;
}

// ---------------------------------------------------------------------------//
// Control register instructions.
// ---------------------------------------------------------------------------//

let rs1 = 0, rs2 = 0 in {
def LDCR : F_SFU1CR<0b01000, (outs GPR:$rd), (ins CR:$cr),
                    "ldcr", "$rd, $cr">;
def FLDCR : F_SFU1CR<0b01001, (outs GPR:$rd), (ins FCR:$cr),
                     "fldcr", "$rd, $cr">;
}

let rd = 0 in {
def STCR : F_SFU1CR2<0b10000, (outs CR:$cr), (ins GPR:$rs1),
                     "stcr", "$rs1, $cr">;
def FSTCR : F_SFU1CR2<0b10001, (outs FCR:$cr), (ins GPR:$rs1),
                      "fstcr", "$rs1, $cr">;
}

def XCR : F_SFU1CR2<0b11000, (outs GPR:$rd), (ins GPR:$rs1, CR:$cr),
                    "xcr", "$rd, $rs1, $cr">;
def FXCR : F_SFU1CR2<0b11001, (outs GPR:$rd), (ins GPR:$rs1, FCR:$cr),
                     "fxcr", "$rd, $rs1, $cr">;

// ---------------------------------------------------------------------------//
// Graphics/vector instructions. 881100 only.
// ---------------------------------------------------------------------------//

multiclass PArithPixel<bits<5> func, string opc> {
  def : F_SFU2<func, 0b00, 0b11,
                  (outs GPR64:$rd), (ins GPR64:$rs1, GPR64:$rs2),
                  opc>;
  def b : F_SFU2<func, 0b00, 0b01,
                  (outs GPR64:$rd), (ins GPR64:$rs1, GPR64:$rs2),
                  !strconcat(opc, ".b")>;
  def h : F_SFU2<func, 0b00, 0b10,
                  (outs GPR64:$rd), (ins GPR64:$rs1, GPR64:$rs2),
                  !strconcat(opc, ".h")>;
}

let Predicates = [IsMC881100] in {

def PMUL : F_SFU2<0b00000, 0b00, 0b00,
                  (outs XR:$rd), (ins GPR:$rs1, GPR:$rs2),
                  "pmul">;
def PCMP : F_SFU2<0b00111, 0b00, 0b11,
                  (outs XR:$rd), (ins GPR:$rs1, GPR:$rs2),
                  "pcmp">;
defm PADD : PArithPixel<0b00100, "padd">;
defm PSUB : PArithPixel<0b00110, "psub">;
}
