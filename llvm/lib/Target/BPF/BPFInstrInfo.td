//===-- BPFInstrInfo.td - Target Description for BPF Target ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the BPF instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "BPFInstrFormats.td"

// Instruction Operands and Patterns

// These are target-independent nodes, but have target-specific formats.
def SDT_BPFCallSeqStart : SDCallSeqStart<[SDTCisVT<0, iPTR>,
                                          SDTCisVT<1, iPTR>]>;
def SDT_BPFCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;
def SDT_BPFCall         : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SDT_BPFSetFlag      : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>]>;
def SDT_BPFSelectCC     : SDTypeProfile<1, 5, [SDTCisSameAs<1, 2>,
                                               SDTCisSameAs<0, 4>,
                                               SDTCisSameAs<4, 5>]>;
def SDT_BPFBrCC         : SDTypeProfile<0, 4, [SDTCisSameAs<0, 1>,
                                               SDTCisVT<3, OtherVT>]>;
def SDT_BPFWrapper      : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                               SDTCisPtrTy<0>]>;

def BPFcall         : SDNode<"BPFISD::CALL", SDT_BPFCall,
                             [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                              SDNPVariadic]>;
def BPFretflag      : SDNode<"BPFISD::RET_FLAG", SDTNone,
                             [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def BPFcallseq_start: SDNode<"ISD::CALLSEQ_START", SDT_BPFCallSeqStart,
                             [SDNPHasChain, SDNPOutGlue]>;
def BPFcallseq_end  : SDNode<"ISD::CALLSEQ_END",   SDT_BPFCallSeqEnd,
                             [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def BPFbrcc         : SDNode<"BPFISD::BR_CC", SDT_BPFBrCC,
                             [SDNPHasChain, SDNPOutGlue, SDNPInGlue]>;

def BPFselectcc     : SDNode<"BPFISD::SELECT_CC", SDT_BPFSelectCC, [SDNPInGlue]>;
def BPFWrapper      : SDNode<"BPFISD::Wrapper", SDT_BPFWrapper>;

def brtarget : Operand<OtherVT>;
def calltarget : Operand<i64>;

def u64imm   : Operand<i64> {
  let PrintMethod = "printImm64Operand";
}

def i64immSExt32 : PatLeaf<(i64 imm),
                [{return isInt<32>(N->getSExtValue()); }]>;

// Addressing modes.
def ADDRri : ComplexPattern<i64, 2, "SelectAddr", [], []>;
def FIri : ComplexPattern<i64, 2, "SelectFIAddr", [add, or], []>;

// Address operands
def MEMri : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let EncoderMethod = "getMemoryOpValue";
  let DecoderMethod = "decodeMemoryOpValue";
  let MIOperandInfo = (ops GPR, i16imm);
}

// Conditional code predicates - used for pattern matching for jump instructions
def BPF_CC_EQ  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETEQ);}]>;
def BPF_CC_NE  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETNE);}]>;
def BPF_CC_GE  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETGE);}]>;
def BPF_CC_GT  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETGT);}]>;
def BPF_CC_GTU : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETUGT);}]>;
def BPF_CC_GEU : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETUGE);}]>;
def BPF_CC_LE  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETLE);}]>;
def BPF_CC_LT  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETLT);}]>;
def BPF_CC_LTU : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETULT);}]>;
def BPF_CC_LEU : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETULE);}]>;

// jump instructions
class JMP_RR<bits<4> Opc, string OpcodeStr, PatLeaf Cond>
    : InstBPF<(outs), (ins GPR:$dst, GPR:$src, brtarget:$BrDst),
              "if $dst "#OpcodeStr#" $src goto $BrDst",
              [(BPFbrcc i64:$dst, i64:$src, Cond, bb:$BrDst)]> {
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<4> src;
  bits<16> BrDst;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;
  let Inst{47-32} = BrDst;

  let op = Opc;
  let BPFSrc = 1;
  let BPFClass = 5; // BPF_JMP
}

class JMP_RI<bits<4> Opc, string OpcodeStr, PatLeaf Cond>
    : InstBPF<(outs), (ins GPR:$dst, i64imm:$imm, brtarget:$BrDst),
              "if $dst "#OpcodeStr#" $imm goto $BrDst",
              [(BPFbrcc i64:$dst, i64immSExt32:$imm, Cond, bb:$BrDst)]> {
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<16> BrDst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{51-48} = dst;
  let Inst{47-32} = BrDst;
  let Inst{31-0} = imm;

  let op = Opc;
  let BPFSrc = 0;
  let BPFClass = 5; // BPF_JMP
}

multiclass J<bits<4> Opc, string OpcodeStr, PatLeaf Cond> {
  def _rr : JMP_RR<Opc, OpcodeStr, Cond>;
  def _ri : JMP_RI<Opc, OpcodeStr, Cond>;
}

let isBranch = 1, isTerminator = 1, hasDelaySlot=0 in {
// cmp+goto instructions
defm JEQ  : J<0x1, "==",  BPF_CC_EQ>;
defm JUGT : J<0x2, ">", BPF_CC_GTU>;
defm JUGE : J<0x3, ">=", BPF_CC_GEU>;
defm JNE  : J<0x5, "!=",  BPF_CC_NE>;
defm JSGT : J<0x6, "s>", BPF_CC_GT>;
defm JSGE : J<0x7, "s>=", BPF_CC_GE>;
defm JULT : J<0xa, "<", BPF_CC_LTU>;
defm JULE : J<0xb, "<=", BPF_CC_LEU>;
defm JSLT : J<0xc, "s<", BPF_CC_LT>;
defm JSLE : J<0xd, "s<=", BPF_CC_LE>;
}

// ALU instructions
class ALU_RI<bits<4> Opc, string OpcodeStr, SDNode OpNode>
    : InstBPF<(outs GPR:$dst), (ins GPR:$src2, i64imm:$imm),
              "$dst "#OpcodeStr#" $imm",
              [(set GPR:$dst, (OpNode GPR:$src2, i64immSExt32:$imm))]> {
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{51-48} = dst;
  let Inst{31-0} = imm;

  let op = Opc;
  let BPFSrc = 0;
  let BPFClass = 7; // BPF_ALU64
}

class ALU_RR<bits<4> Opc, string OpcodeStr, SDNode OpNode>
    : InstBPF<(outs GPR:$dst), (ins GPR:$src2, GPR:$src),
              "$dst "#OpcodeStr#" $src",
              [(set GPR:$dst, (OpNode i64:$src2, i64:$src))]> {
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<4> src;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;

  let op = Opc;
  let BPFSrc = 1;
  let BPFClass = 7; // BPF_ALU64
}

multiclass ALU<bits<4> Opc, string OpcodeStr, SDNode OpNode> {
  def _rr : ALU_RR<Opc, OpcodeStr, OpNode>;
  def _ri : ALU_RI<Opc, OpcodeStr, OpNode>;
}

let Constraints = "$dst = $src2" in {
let isAsCheapAsAMove = 1 in {
  defm ADD : ALU<0x0, "+=", add>;
  defm SUB : ALU<0x1, "-=", sub>;
  defm OR  : ALU<0x4, "|=", or>;
  defm AND : ALU<0x5, "&=", and>;
  defm SLL : ALU<0x6, "<<=", shl>;
  defm SRL : ALU<0x7, ">>=", srl>;
  defm XOR : ALU<0xa, "^=", xor>;
  defm SRA : ALU<0xc, "s>>=", sra>;
}
  defm MUL : ALU<0x2, "*=", mul>;
  defm DIV : ALU<0x3, "/=", udiv>;
}

class MOV_RR<string OpcodeStr>
    : InstBPF<(outs GPR:$dst), (ins GPR:$src),
              "$dst "#OpcodeStr#" $src",
              []> {
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<4> src;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;

  let op = 0xb;     // BPF_MOV
  let BPFSrc = 1;   // BPF_X
  let BPFClass = 7; // BPF_ALU64
}

class MOV_RI<string OpcodeStr>
    : InstBPF<(outs GPR:$dst), (ins i64imm:$imm),
              "$dst "#OpcodeStr#" $imm",
              [(set GPR:$dst, (i64 i64immSExt32:$imm))]> {
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{51-48} = dst;
  let Inst{31-0} = imm;

  let op = 0xb;     // BPF_MOV
  let BPFSrc = 0;   // BPF_K
  let BPFClass = 7; // BPF_ALU64
}

class LD_IMM64<bits<4> Pseudo, string OpcodeStr>
    : InstBPF<(outs GPR:$dst), (ins u64imm:$imm),
              "$dst "#OpcodeStr#" ${imm} ll",
              [(set GPR:$dst, (i64 imm:$imm))]> {

  bits<3> mode;
  bits<2> size;
  bits<4> dst;
  bits<64> imm;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = dst;
  let Inst{55-52} = Pseudo;
  let Inst{47-32} = 0;
  let Inst{31-0} = imm{31-0};

  let mode = 0;     // BPF_IMM
  let size = 3;     // BPF_DW
  let BPFClass = 0; // BPF_LD
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def LD_imm64 : LD_IMM64<0, "=">;
def MOV_rr : MOV_RR<"=">;
def MOV_ri : MOV_RI<"=">;
}

def FI_ri
    : InstBPF<(outs GPR:$dst), (ins MEMri:$addr),
               "lea\t$dst, $addr",
               [(set i64:$dst, FIri:$addr)]> {
  // This is a tentative instruction, and will be replaced
  // with MOV_rr and ADD_ri in PEI phase
  let Inst{63-61} = 0;
  let Inst{60-59} = 3;
  let Inst{51-48} = 0;
  let Inst{55-52} = 2;
  let Inst{47-32} = 0;
  let Inst{31-0} = 0;
  let BPFClass = 0;
}


def LD_pseudo
    : InstBPF<(outs GPR:$dst), (ins i64imm:$pseudo, u64imm:$imm),
              "ld_pseudo\t$dst, $pseudo, $imm",
              [(set GPR:$dst, (int_bpf_pseudo imm:$pseudo, imm:$imm))]> {

  bits<3> mode;
  bits<2> size;
  bits<4> dst;
  bits<64> imm;
  bits<4> pseudo;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = dst;
  let Inst{55-52} = pseudo;
  let Inst{47-32} = 0;
  let Inst{31-0} = imm{31-0};

  let mode = 0;     // BPF_IMM
  let size = 3;     // BPF_DW
  let BPFClass = 0; // BPF_LD
}

// STORE instructions
class STORE<bits<2> SizeOp, string OpcodeStr, list<dag> Pattern>
    : InstBPF<(outs), (ins GPR:$src, MEMri:$addr),
              "*("#OpcodeStr#" *)($addr) = $src", Pattern> {
  bits<3> mode;
  bits<2> size;
  bits<4> src;
  bits<20> addr;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = addr{19-16}; // base reg
  let Inst{55-52} = src;
  let Inst{47-32} = addr{15-0}; // offset

  let mode = 3;     // BPF_MEM
  let size = SizeOp;
  let BPFClass = 3; // BPF_STX
}

class STOREi64<bits<2> Opc, string OpcodeStr, PatFrag OpNode>
    : STORE<Opc, OpcodeStr, [(OpNode i64:$src, ADDRri:$addr)]>;

def STW : STOREi64<0x0, "u32", truncstorei32>;
def STH : STOREi64<0x1, "u16", truncstorei16>;
def STB : STOREi64<0x2, "u8", truncstorei8>;
def STD : STOREi64<0x3, "u64", store>;

// LOAD instructions
class LOAD<bits<2> SizeOp, string OpcodeStr, list<dag> Pattern>
    : InstBPF<(outs GPR:$dst), (ins MEMri:$addr),
              "$dst = *("#OpcodeStr#" *)($addr)", Pattern> {
  bits<3> mode;
  bits<2> size;
  bits<4> dst;
  bits<20> addr;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = dst;
  let Inst{55-52} = addr{19-16};
  let Inst{47-32} = addr{15-0};

  let mode = 3;     // BPF_MEM
  let size = SizeOp;
  let BPFClass = 1; // BPF_LDX
}

class LOADi64<bits<2> SizeOp, string OpcodeStr, PatFrag OpNode>
    : LOAD<SizeOp, OpcodeStr, [(set i64:$dst, (OpNode ADDRri:$addr))]>;

def LDW : LOADi64<0x0, "u32", zextloadi32>;
def LDH : LOADi64<0x1, "u16", zextloadi16>;
def LDB : LOADi64<0x2, "u8", zextloadi8>;
def LDD : LOADi64<0x3, "u64", load>;

class BRANCH<bits<4> Opc, string OpcodeStr, list<dag> Pattern>
    : InstBPF<(outs), (ins brtarget:$BrDst),
              !strconcat(OpcodeStr, " $BrDst"), Pattern> {
  bits<4> op;
  bits<16> BrDst;
  bits<1> BPFSrc;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{47-32} = BrDst;

  let op = Opc;
  let BPFSrc = 0;
  let BPFClass = 5; // BPF_JMP
}

class CALL<string OpcodeStr>
    : InstBPF<(outs), (ins calltarget:$BrDst),
              !strconcat(OpcodeStr, " $BrDst"), []> {
  bits<4> op;
  bits<32> BrDst;
  bits<1> BPFSrc;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{31-0} = BrDst;

  let op = 8;       // BPF_CALL
  let BPFSrc = 0;
  let BPFClass = 5; // BPF_JMP
}

// Jump always
let isBranch = 1, isTerminator = 1, hasDelaySlot=0, isBarrier = 1 in {
  def JMP : BRANCH<0x0, "goto", [(br bb:$BrDst)]>;
}

// Jump and link
let isCall=1, hasDelaySlot=0, Uses = [R11],
    // Potentially clobbered registers
    Defs = [R0, R1, R2, R3, R4, R5] in {
  def JAL  : CALL<"call">;
}

class NOP_I<string OpcodeStr>
    : InstBPF<(outs), (ins i32imm:$imm),
              !strconcat(OpcodeStr, "\t$imm"), []> {
  // mov r0, r0 == nop
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<4> src;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;

  let op = 0xb;     // BPF_MOV
  let BPFSrc = 1;   // BPF_X
  let BPFClass = 7; // BPF_ALU64
  let src = 0;      // R0
  let dst = 0;      // R0
}

let hasSideEffects = 0 in
  def NOP : NOP_I<"nop">;

class RET<string OpcodeStr>
    : InstBPF<(outs), (ins),
              !strconcat(OpcodeStr, ""), [(BPFretflag)]> {
  bits<4> op;

  let Inst{63-60} = op;
  let Inst{59} = 0;
  let Inst{31-0} = 0;

  let op = 9;       // BPF_EXIT
  let BPFClass = 5; // BPF_JMP
}

let isReturn = 1, isTerminator = 1, hasDelaySlot=0, isBarrier = 1,
    isNotDuplicable = 1 in {
  def RET : RET<"exit">;
}

// ADJCALLSTACKDOWN/UP pseudo insns
let Defs = [R11], Uses = [R11] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              "#ADJCALLSTACKDOWN $amt1 $amt2",
                              [(BPFcallseq_start timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              "#ADJCALLSTACKUP $amt1 $amt2",
                              [(BPFcallseq_end timm:$amt1, timm:$amt2)]>;
}

let usesCustomInserter = 1 in {
  def Select : Pseudo<(outs GPR:$dst),
                      (ins GPR:$lhs, GPR:$rhs, i64imm:$imm, GPR:$src, GPR:$src2),
                      "# Select PSEUDO $dst = $lhs $imm $rhs ? $src : $src2",
                      [(set i64:$dst,
                       (BPFselectcc i64:$lhs, i64:$rhs, (i64 imm:$imm), i64:$src, i64:$src2))]>;
  def Select_Ri : Pseudo<(outs GPR:$dst),
                      (ins GPR:$lhs, i64imm:$rhs, i64imm:$imm, GPR:$src, GPR:$src2),
                      "# Select PSEUDO $dst = $lhs $imm $rhs ? $src : $src2",
                      [(set i64:$dst,
                       (BPFselectcc i64:$lhs, (i64 imm:$rhs), (i64 imm:$imm), i64:$src, i64:$src2))]>;
}

// load 64-bit global addr into register
def : Pat<(BPFWrapper tglobaladdr:$in), (LD_imm64 tglobaladdr:$in)>;

// 0xffffFFFF doesn't fit into simm32, optimize common case
def : Pat<(i64 (and (i64 GPR:$src), 0xffffFFFF)),
          (SRL_ri (SLL_ri (i64 GPR:$src), 32), 32)>;

// Calls
def : Pat<(BPFcall tglobaladdr:$dst), (JAL tglobaladdr:$dst)>;
def : Pat<(BPFcall texternalsym:$dst), (JAL texternalsym:$dst)>;
def : Pat<(BPFcall imm:$dst), (JAL imm:$dst)>;

// Loads
def : Pat<(extloadi8  ADDRri:$src), (i64 (LDB ADDRri:$src))>;
def : Pat<(extloadi16 ADDRri:$src), (i64 (LDH ADDRri:$src))>;
def : Pat<(extloadi32 ADDRri:$src), (i64 (LDW ADDRri:$src))>;

// Atomics
class XADD<bits<2> SizeOp, string OpcodeStr, PatFrag OpNode>
    : InstBPF<(outs GPR:$dst), (ins MEMri:$addr, GPR:$val),
              "lock *("#OpcodeStr#" *)($addr) += $val",
              [(set GPR:$dst, (OpNode ADDRri:$addr, GPR:$val))]> {
  bits<3> mode;
  bits<2> size;
  bits<4> dst;
  bits<20> addr;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = addr{19-16}; // base reg
  let Inst{55-52} = dst;
  let Inst{47-32} = addr{15-0}; // offset

  let mode = 6;     // BPF_XADD
  let size = SizeOp;
  let BPFClass = 3; // BPF_STX
}

let Constraints = "$dst = $val" in {
def XADD32 : XADD<0, "u32", atomic_load_add_32>;
def XADD64 : XADD<3, "u64", atomic_load_add_64>;
// undefined def XADD16 : XADD<1, "xadd16", atomic_load_add_16>;
// undefined def XADD8  : XADD<2, "xadd8", atomic_load_add_8>;
}

// bswap16, bswap32, bswap64
class BSWAP<bits<32> SizeOp, string OpcodeStr, list<dag> Pattern>
    : InstBPF<(outs GPR:$dst), (ins GPR:$src),
              !strconcat(OpcodeStr, "\t$dst"),
              Pattern> {
  bits<4> op;
  bits<1> BPFSrc;
  bits<4> dst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = BPFSrc;
  let Inst{51-48} = dst;
  let Inst{31-0} = imm;

  let op = 0xd;     // BPF_END
  let BPFSrc = 1;   // BPF_TO_BE (TODO: use BPF_TO_LE for big-endian target)
  let BPFClass = 4; // BPF_ALU
  let imm = SizeOp;
}

let Constraints = "$dst = $src" in {
def BSWAP16 : BSWAP<16, "bswap16", [(set GPR:$dst, (srl (bswap GPR:$src), (i64 48)))]>;
def BSWAP32 : BSWAP<32, "bswap32", [(set GPR:$dst, (srl (bswap GPR:$src), (i64 32)))]>;
def BSWAP64 : BSWAP<64, "bswap64", [(set GPR:$dst, (bswap GPR:$src))]>;
}

let Defs = [R0, R1, R2, R3, R4, R5], Uses = [R6], hasSideEffects = 1,
    hasExtraDefRegAllocReq = 1, hasExtraSrcRegAllocReq = 1, mayLoad = 1 in {
class LOAD_ABS<bits<2> SizeOp, string OpcodeStr, Intrinsic OpNode>
    : InstBPF<(outs), (ins GPR:$skb, i64imm:$imm),
              "r0 = *("#OpcodeStr#" *)skb[$imm]",
              [(set R0, (OpNode GPR:$skb, i64immSExt32:$imm))]> {
  bits<3> mode;
  bits<2> size;
  bits<32> imm;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{31-0} = imm;

  let mode = 1;     // BPF_ABS
  let size = SizeOp;
  let BPFClass = 0; // BPF_LD
}

class LOAD_IND<bits<2> SizeOp, string OpcodeStr, Intrinsic OpNode>
    : InstBPF<(outs), (ins GPR:$skb, GPR:$val),
              "r0 = *("#OpcodeStr#" *)skb[$val]",
              [(set R0, (OpNode GPR:$skb, GPR:$val))]> {
  bits<3> mode;
  bits<2> size;
  bits<4> val;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{55-52} = val;

  let mode = 2;     // BPF_IND
  let size = SizeOp;
  let BPFClass = 0; // BPF_LD
}
}

def LD_ABS_B : LOAD_ABS<2, "u8", int_bpf_load_byte>;
def LD_ABS_H : LOAD_ABS<1, "u16", int_bpf_load_half>;
def LD_ABS_W : LOAD_ABS<0, "u32", int_bpf_load_word>;

def LD_IND_B : LOAD_IND<2, "u8", int_bpf_load_byte>;
def LD_IND_H : LOAD_IND<1, "u16", int_bpf_load_half>;
def LD_IND_W : LOAD_IND<0, "u32", int_bpf_load_word>;
