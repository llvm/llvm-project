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


// Multiclass for logical instructions with immediates.
// The pattern for "and" is slightly different.
multiclass LogicImm<bits<2> FuncI, string OpcStr, SDNode OpNode,
                  InstrItinClass itin = NoItinerary> {
  def ri  : F_LI<FuncI, 0b0,
                 (outs GPROpnd:$rd),
                 !if(!eq(OpcStr, "and"),
                   (ins GPROpnd:$rs1, imm32lo16c:$imm16),
                   (ins GPROpnd:$rs1, imm32lo16:$imm16)
                 ),
                 OpcStr,
                 !if(!eq(OpcStr, "and"),
                   [(set i32:$rd, (OpNode GPROpnd:$rs1, imm32lo16c:$imm16))],
                   [(set i32:$rd, (OpNode GPROpnd:$rs1, imm32lo16:$imm16))]
                 ),
                 itin>;
  def riu  : F_LI<FuncI, 0b1,
                 (outs GPROpnd:$rd),
                 !if(!eq(OpcStr, "and"),
                   (ins GPROpnd:$rs1, imm32hi16c:$imm16),
                   (ins GPROpnd:$rs1, imm32hi16:$imm16)
                 ),
                 !strconcat(OpcStr, ".u"),
                 !if(!eq(OpcStr, "and"),
                   [(set i32:$rd, (OpNode GPROpnd:$rs1, imm32hi16c:$imm16))],
                   [(set i32:$rd, (OpNode GPROpnd:$rs1, imm32hi16:$imm16))]
                 ),
                 itin>;
}

// Multiclass for logical instructions with triadic registers or immediates.
multiclass Logic<bits<5> FuncR, bits<2> FuncI, string OpcStr, SDNode OpNode,
                 InstrItinClass itin = NoItinerary>
                 : LogicImm<FuncI, OpcStr, OpNode, itin> {
  let isCommutable = 1 in
    def rr  : F_LR<FuncR, 0b0,
                   (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                   OpcStr,
                   [(set i32:$rd, (OpNode GPROpnd:$rs1, GPROpnd:$rs2))],
                   itin>;
  def rrc : F_LR<FuncR, 0b1,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                 !strconcat(OpcStr, ".c"),
                 [(set i32:$rd, (OpNode GPROpnd:$rs1, (not GPROpnd:$rs2)))],
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
  def rr  : F_BR<Func,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                 OpcStr,
                 [(set GPROpnd:$rd, (OpNode GPROpnd:$rs1, GPROpnd:$rs2))],
                 itin>;
  def rwo : F_BI<Func,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, ImmOp:$w5o5),
                 OpcStr,
                 [(set GPROpnd:$rd, (OpNode GPROpnd:$rs1, ImmOp:$w5o5))],
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
  F_BF<Func, (outs GPROpnd:$rd), (ins GPROpnd:$rs2),
             OpcStr,
             [(set GPROpnd:$rd, (OpNode GPROpnd:$rs2))],
             itin>;

def FF1rr : FindBF<0b111010, "ff1", m88k_ff1>;
def FF0rr : FindBF<0b111011, "ff0", m88k_ff0>;

// ctlz = 32 - ff1
//def : Pat<(ctlz GPR:$rs1, GPR:$rs2), (SUBri (i32 32), (FF1rr GPR:$rs1, GPR:$rs2))>;

// Multiclass for arithmetic instructions with triadic registers or immediates.
multiclass Arith<bits<6> Func, string OpcStr, SDNode OpNode,
                 InstrItinClass itin = NoItinerary> {
  def rr  : F_IRC<Func, 0b0, 0b0,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                 OpcStr,
                 [(set GPROpnd:$rd, (OpNode GPROpnd:$rs1, GPROpnd:$rs2))],
                 itin>;
  def rrci  : F_IRC<Func, 0b1, 0b0,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                 !strconcat(OpcStr, ".ci"),
                 [(set GPROpnd:$rd, (OpNode GPROpnd:$rs1, GPROpnd:$rs2))],
                 itin>;
  def rrco  : F_IRC<Func, 0b0, 0b1,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                 !strconcat(OpcStr, ".co"),
                 [(set GPROpnd:$rd, (OpNode GPROpnd:$rs1, GPROpnd:$rs2))],
                 itin>;
  def rrcio : F_IRC<Func, 0b1, 0b1,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                 !strconcat(OpcStr, ".cio"),
                 [(set GPROpnd:$rd, (OpNode GPROpnd:$rs1, GPROpnd:$rs2))],
                 itin>;
  def ri  : F_II<Func,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, imm32zx16:$imm16),
                 OpcStr,
                 [(set GPROpnd:$rd, (OpNode GPROpnd:$rs1, imm32zx16:$imm16))],
                 itin>;
}

let isCommutable = 1 in
defm ADDU : Arith<0b011000, "addu", add>;
defm SUBU : Arith<0b011001, "subu", sub>;

def CMPrr : F_IRC<0b011111, 0b0, 0b0,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                 "cmp", [], NoItinerary>;
def CMPri : F_II<0b011111,
                 (outs GPROpnd:$rd), (ins GPROpnd:$rs1, imm32zx16:$imm16),
                 "cmp", [], NoItinerary>;

multiclass SetCCPat<CondCode CC, bits<10> w5o5> {
  def : Pat<(setcc GPR:$lhs, GPR:$rhs, CC),
            (EXTUrwo (CMPrr GPR:$lhs, GPROpnd:$rhs), w5o5)>;
  def : Pat<(setcc GPR:$lhs, imm32zx16:$rhs, CC),
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

//def : Pat<(load GPR:$rd, (ADDRri GPR:$rs1, imm32zx16:$si16)),
//          (LDriw GPR:$rd, GPR:$rs1, imm32zx16:$si16)>;


// Multiclass for load and store instructions.
multiclass LoadIndexImm<bits<4> Func, string OpcStr, SDPatternOperator OpNode,
                 InstrItinClass itin = NoItinerary> {
  def riw  : F_LS<Func, /*ty=*/ 0b01,
                  (outs GPROpnd:$rd), (ins GPROpnd:$rs1, imm32zx16:$si16),
                  OpcStr,
                  [(set (i32 GPROpnd:$rd), (load (ADDRri GPROpnd:$rs1, imm32zx16:$si16)))],
                  itin>;
  def rib  : F_LS<Func, /*ty=*/ 0b11,
                  (outs GPROpnd:$rd), (ins GPROpnd:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".b"),
                  [(set (i32 GPROpnd:$rd), (sextloadi8 (ADDRri GPROpnd:$rs1, imm32zx16:$si16)))],
                  itin>;
  def rih  : F_LS<Func, /*ty=*/ 0b10,
                  (outs GPROpnd:$rd), (ins GPROpnd:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".h"),
                  [(set (i32 GPROpnd:$rd), (sextloadi16 (ADDRri GPROpnd:$rs1, imm32zx16:$si16)))],
                  itin>;
  def rid  : F_LS<Func, /*ty=*/ 0b00,
                  (outs GPROpnd:$rd), (ins GPROpnd:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".d"),
                  [],
                  itin>;
}

multiclass StoreIndexImm<bits<4> Func, string OpcStr, SDPatternOperator OpNode,
                 InstrItinClass itin = NoItinerary> {
  def riw  : F_LS<Func, /*ty=*/ 0b01,
                  (outs), (ins GPROpnd:$rd, GPROpnd:$rs1, imm32zx16:$si16),
                  OpcStr,
                  [(store (i32 GPROpnd:$rd), (ADDRri GPROpnd:$rs1, imm32zx16:$si16))],
                  itin>;
  def rib  : F_LS<Func, /*ty=*/ 0b11,
                  (outs), (ins GPROpnd:$rd, GPROpnd:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".b"),
                  [(truncstorei8 (i32 GPROpnd:$rd), (ADDRri GPROpnd:$rs1, imm32zx16:$si16))],
                  itin>;
  def rih  : F_LS<Func, /*ty=*/ 0b10,
                  (outs), (ins GPROpnd:$rd, GPROpnd:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".h"),
                  [(truncstorei16 (i32 GPROpnd:$rd), (ADDRri GPROpnd:$rs1, imm32zx16:$si16))],
                  itin>;
  def rid  : F_LS<Func, /*ty=*/ 0b00,
                  (outs), (ins GPR64Opnd:$rd, GPROpnd:$rs1, imm32zx16:$si16),
                  !strconcat(OpcStr, ".d"),
                  [(store (i64 GPR64Opnd:$rd), (ADDRri GPROpnd:$rs1, imm32zx16:$si16))],
                  itin>;
}

class LoadUnsigned<bits<1> b, string OpcStr, PatFrag OpNode,
                 InstrItinClass itin = NoItinerary> :
  F_LU<b, (outs GPROpnd:$rd), (ins GPROpnd:$rs1, imm32zx16:$si16),
       OpcStr,
       [(set (i32 GPROpnd:$rd), (OpNode (ADDRri GPROpnd:$rs1, imm32zx16:$si16)))],
       itin>;

multiclass LoadUnscaled<bits<4> Func, string OpcStr, SDPatternOperator OpNode,
                 InstrItinClass itin = NoItinerary> {
  let regfile = 0 in {
    def rruw  : F_LSINDIRECT<Func, /*ty=*/ 0b01, /*scaled=*/ 0b0, /*user=*/ 0b0,
                    (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                    !strconcat(OpcStr, " $rd, $rs1, $rs2"),
                    [(set (i32 GPROpnd:$rd), (load (ADDRrr GPROpnd:$rs1, GPROpnd:$rs2)))],
                    itin>;
    def rrub  : F_LSINDIRECT<Func, /*ty=*/ 0b11, /*scaled=*/ 0b0, /*user=*/ 0b0,
                    (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                    !strconcat(OpcStr, ".b $rd, $rs1, $rs2"),
                    [(set (i32 GPROpnd:$rd), (sextloadi8 (ADDRrr GPROpnd:$rs1, GPROpnd:$rs2)))],
                    itin>;
    def rruh  : F_LSINDIRECT<Func, /*ty=*/ 0b10, /*scaled=*/ 0b0, /*user=*/ 0b0,
                    (outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
                    !strconcat(OpcStr, ".h $rd, $rs1, $rs2"),
                    [(set (i32 GPROpnd:$rd), (sextloadi16 (ADDRrr GPROpnd:$rs1, GPROpnd:$rs2)))],
                    itin>;
  }
}

defm LD : LoadIndexImm<0b0001, "ld", load>;
defm ST : StoreIndexImm<0b0010, "st", store>;
def LDurih : LoadUnsigned<0b0, "ld.hu", zextloadi16>;
def LDurib : LoadUnsigned<0b1, "ld.bu", zextloadi8>;

defm LD : LoadUnscaled<0b0001, "ld", load>;


// GlobalAddress, ExternalSymbol, Jumptable, ConstantPool
def : Pat<(m88k_hi tglobaladdr:$dst), (ORriu R0, tglobaladdr:$dst)>;
//def : Pat<(m88k_lo tglobaladdr:$dst), (OR_I_LO (i32 R0), tglobaladdr:$dst)>;


let isTerminator = 1, isBarrier = 1 in {
  def JMP : F_JMP<0b11000, 0, (outs), (ins GPROpnd:$rs2),
                  "jmp", [(brind GPROpnd:$rs2)]>;
  let hasDelaySlot = 1 in
    def JMPn : F_JMP<0b11000, 1, (outs), (ins GPROpnd:$rs2),
                     "jmp.n", [(brind GPROpnd:$rs2)]>;
}

let isCall = 1, isTerminator = 1, isBarrier = 1 in {
  def JSR : F_JMP<0b11001, 0, (outs), (ins GPROpnd:$rs2),
                  "jsr", []>;
  let hasDelaySlot = 1 in
    def JSRn : F_JMP<0b11001, 1, (outs), (ins GPROpnd:$rs2),
                     "jsr.n", []>;
}

let isReturn = 1, isTerminator = 1, isBarrier = 1, Uses = [R1] in {
  def RET : Pseudo<(outs), (ins), [(retflag)]>;
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  def BR : F_BRANCH<0b11000, 0, (outs), (ins brtarget26:$d26), "br",
                    [(br bb:$d26)]>;

  let hasDelaySlot = 1 in
    def BRn : F_BRANCH<0b11000, 1, (outs), (ins brtarget26:$d26), "br.n",
                       [(br bb:$d26)]>;

  def BB0 : F_BCOND<0b11010, /*next*/ 0,
                    (outs), (ins imm32zx5:$m5, GPROpnd:$rs1, brtarget16:$d16),
                    "bb0", []>;
  def BB1 : F_BCOND<0b11011, /*next*/ 0,
                    (outs), (ins imm32zx5:$m5, GPROpnd:$rs1, brtarget16:$d16),
                    "bb1", []>;
  def BCOND : F_BCOND<0b11101, /*next*/ 0,
                      (outs), (ins imm32zx5:$m5, GPROpnd:$rs1, brtarget16:$d16),
                      "bcnd", []>;
  let hasDelaySlot = 1 in {
    def BB0n : F_BCOND<0b11010, /*next*/ 1,
                       (outs), (ins imm32zx5:$m5, GPROpnd:$rs1, brtarget16:$d16),
                       "bb0.n", []>;
    def BB1n : F_BCOND<0b11011, /*next*/ 1,
                      (outs), (ins imm32zx5:$m5, GPROpnd:$rs1, brtarget16:$d16),
                      "bb1.n", []>;
    def BCONDn : F_BCOND<0b11101, /*next*/ 1,
                        (outs), (ins imm32zx5:$m5, GPROpnd:$rs1, brtarget16:$d16),
                        "bcnd.n", []>;
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
  def : Pat<(brcc CC, GPR:$rs1, GPR:$rs2, bb:$dst),
             (BB1 m5, (CMPrr GPR:$rs1, GPR:$rs2), bb:$dst)>;
  def : Pat<(brcc CC, GPR:$rs1, imm32zx16:$imm16, bb:$dst),
             (BB1 m5, (CMPri GPR:$rs1, imm32zx16:$imm16), bb:$dst)>;
  def : Pat<(brcond (i32 (ccop GPR:$lhs, GPR:$rhs)), bb:$dst),
             (BB1 m5, (CMPrr GPR:$lhs, GPR:$rhs), bb:$dst)>;
  def : Pat<(brcond (i32 (ccop GPR:$lhs, imm32zx16:$rhs)), bb:$dst),
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

// Alias for no operation.
def : InstAlias<"nop", (ORrr R0, R0, R0)>;

// Multiclass for triadic register floating point instructions.
multiclass FArith<bits<4> Func, string OpcStr, SDNode OpNode,
                 InstrItinClass itin = NoItinerary> {
  foreach D = 0-1 in {
    defvar S1 = D;
    defvar S2 = D;
    defvar DPrec = !if(D, "d", "s");
    defvar DOpnd = !if(D, FPR64Opnd, FPR32Opnd);
    defvar S1Prec = !if(S1, "d", "s");
    defvar S1Opnd = !if(S1, FPR64Opnd, FPR32Opnd);
    defvar S2Prec = !if(S2, "d", "s");
    defvar S2Opnd = !if(S2, FPR64Opnd, FPR32Opnd);
    let regfile = 0 in //!cast<M88kRegisterClass>(DOpnd).HWRegFile in
    def DPrec # S1Prec #S2Prec :
             F_SFU1<Func, D, S1, S2,
                    (outs DOpnd:$rd), (ins S1Opnd:$rs1, S2Opnd:$rs2),
                    !strconcat(OpcStr, ".", DPrec, S1Prec, S2Prec),
                    [(set DOpnd:$rd, (OpNode S1Opnd:$rs1, S2Opnd:$rs2))],
                    itin>;
  }
}

defm FMUL : FArith<0b0000, "fmul", fmul>;
defm FADD : FArith<0b0101, "fadd", fadd>;
defm FSUB : FArith<0b0110, "fsub", fsub>;
defm FDIV : FArith<0b1110, "fdiv", fdiv>;

let regfile = 1, Predicates = [IsMC881100] in {
def MOVrxs : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b00, /*t2*/0b00,
                     (outs FPR32Opnd:$rd), (ins XROpnd:$rs2),
                     "mov.s",
                     [(set FPR32Opnd:$rd, (f32 XROpnd:$rs2))]>;
def MOVrxd : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b00, /*t2*/0b01,
                     (outs FPR64Opnd:$rd), (ins XROpnd:$rs2),
                     "mov.d",
                     [(set FPR64Opnd:$rd, (f64 XROpnd:$rs2))]>;
def MOVxx : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b01, /*t2*/0b11,
                    (outs XROpnd:$rd), (ins XROpnd:$rs2),
                    "mov",
                    []>;
//                    [(set XROpnd:$rd, XROpnd:$rs2)]>;
}

let regfile = 0, Predicates = [IsMC881100] in {
def MOVxrs : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b01, /*t2*/0b00,
                     (outs XROpnd:$rd), (ins FPR32Opnd:$rs2),
                     "mov.s",
                     [(set (f32 XROpnd:$rd), FPR32Opnd:$rs2)]>;
def MOVxrd : F_SFU1D<0b1000, /*td*/ 0b00, /*t1*/0b01, /*t2*/0b01,
                     (outs XROpnd:$rd), (ins FPR64Opnd:$rs2),
                     "mov.d",
                     [(set (f64 XROpnd:$rd), FPR64Opnd:$rs2)]>;
}

let rs1 = 0, rs2 = 0 in
de FLDCR : F_SFU1CR<0b01001, (outs GPROpnd:$rd), (ins FCROpnd:$fcr),
                    "fldcr", " $rd, $fcr">;

let rd = 0 in
def FSTCR : F_SFU1CR2<0b10001, (outs FCROpnd:$fcr), (ins GPROpnd:$rs1),
                      "fstcr", " $rs1, $fcr">;

def FXCR : F_SFU1CR2<0b11001, (outs GPROpnd:$rd), (ins GPROpnd:$rs1, FCROpnd:$fcr),
                     "fxcr", " $rd, $rs1, $fcr">;
