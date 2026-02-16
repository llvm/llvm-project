//=- ConnexInstrInfoScalar.td - Scalar Target Description for Connex Target -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Connex scalar instructions in TableGen format.
//    It basically implements the BPF ISA.
//
//===----------------------------------------------------------------------===//

// Instruction Operands and Patterns (64 bits operands BPF)

// These are target-independent nodes, but have target-specific formats.
def SDT_ConnexCallSeqStart : SDCallSeqStart<[SDTCisVT<0, iPTR>,
                                          SDTCisVT<1, iPTR>]>;
def SDT_ConnexCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, iPTR>,
                                           SDTCisVT<1, iPTR>]>;
def SDT_ConnexCall         : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SDT_ConnexSetFlag      : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>]>;
def SDT_ConnexSelectCC     : SDTypeProfile<1, 5, [SDTCisSameAs<1, 2>,
                                               SDTCisSameAs<0, 4>,
                                               SDTCisSameAs<4, 5>]>;
def SDT_ConnexBrCC         : SDTypeProfile<0, 4, [SDTCisSameAs<0, 1>,
                                               SDTCisVT<3, OtherVT>]>;
def SDT_ConnexWrapper      : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                               SDTCisPtrTy<0>]>;

def Connexcall         : SDNode<"ConnexISD::CALL", SDT_ConnexCall,
                             [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                              SDNPVariadic]>;
def Connexretflag      : SDNode<"ConnexISD::RET_FLAG", SDTNone,
                             [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def Connexcallseq_start: SDNode<"ISD::CALLSEQ_START", SDT_ConnexCallSeqStart,
                             [SDNPHasChain, SDNPOutGlue]>;
def Connexcallseq_end  : SDNode<"ISD::CALLSEQ_END",   SDT_ConnexCallSeqEnd,
                             [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def Connexbrcc         : SDNode<"ConnexISD::BR_CC", SDT_ConnexBrCC,
                             [SDNPHasChain, SDNPOutGlue, SDNPInGlue]>;

def Connexselectcc     : SDNode<"ConnexISD::SELECT_CC", SDT_ConnexSelectCC,
                                [SDNPInGlue]>;
def ConnexWrapper      : SDNode<"ConnexISD::Wrapper", SDT_ConnexWrapper>;

def brtarget : Operand<OtherVT>;
def calltarget : Operand<i64>;

def u64imm   : Operand<i64> {
  let PrintMethod = "printImm64Operand";
}

/* Added type qualifier i64 to avoid error
    "Could not infer all types in pattern!" - type ambiguity since the
    variable name ("in dag operator") does not have a type and this poses
    issues to the Type inference algorithm, since I added in
    ConnexRegisterInfo.td a second RegisterClass with type v128i16... (or v8i64)
*/
def i64immSExt32 : PatLeaf<(i64 imm),
                [{return isInt<32>(N->getSExtValue()); }]>;

// Addressing modes.
def ADDRri : ComplexPattern<i64, 2, "SelectAddr", [], []>;
def FIri : ComplexPattern<i64, 2, "SelectFIAddr", [add, or], []>;

// Address operands
def MEMri : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let EncoderMethod = "getMemoryOpValue";
  let MIOperandInfo = (ops GPR, i16imm);
}


/* Added type qualifier i64 to avoid error
            "Could not infer all types in pattern!"
    - type ambiguity since the variable name ("in dag operator") does not have a
   type and this poses issues to the Type inference algorithm, since I added
   in ConnexRegisterInfo.td a second RegisterClass with type v2i64). */
// Conditional code predicates - used for pattern matching for jump instructions
def Connex_CC_EQ  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETEQ);}]>;
def Connex_CC_NE  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETNE);}]>;
def Connex_CC_GE  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETGE);}]>;
def Connex_CC_GT  : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETGT);}]>;
def Connex_CC_GTU : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETUGT);}]>;
def Connex_CC_GEU : PatLeaf<(i64 imm),
                         [{return (N->getZExtValue() == ISD::SETUGE);}]>;

// jump instructions
class JMP_RR<bits<4> Opc, string OpcodeStr, PatLeaf Cond>
    : InstConnex<(outs), (ins GPR:$dst, GPR:$src, brtarget:$BrDst),
              !strconcat(OpcodeStr, "\t$dst, $src goto $BrDst"),
              [(Connexbrcc i64:$dst, i64:$src, Cond, bb:$BrDst)]> {
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<4> src;
  bits<16> BrDst;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;
  let Inst{47-32} = BrDst;

  let op = Opc;
  let ConnexSrc = 1;
  // let ConnexClass = 5; // Connex_JMP
  let ConnexClass = Connex_JMP; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

class JMP_RI<bits<4> Opc, string OpcodeStr, PatLeaf Cond>
    : InstConnex<(outs), (ins GPR:$dst, i64imm:$imm, brtarget:$BrDst),
              !strconcat(OpcodeStr, "i\t$dst, $imm goto $BrDst"),
              [(Connexbrcc i64:$dst, i64immSExt32:$imm, Cond, bb:$BrDst)]> {
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<16> BrDst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{51-48} = dst;
  let Inst{47-32} = BrDst;
  let Inst{31-0} = imm;

  let op = Opc;
  let ConnexSrc = 0;
  // let ConnexClass = 5; // Connex_JMP
  let ConnexClass = Connex_JMP; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

multiclass J<bits<4> Opc, string OpcodeStr, PatLeaf Cond> {
  def _rr : JMP_RR<Opc, OpcodeStr, Cond>;
  def _ri : JMP_RI<Opc, OpcodeStr, Cond>;
}

let isBranch = 1, isTerminator = 1, hasDelaySlot=0 in {
// cmp+goto instructions
defm JEQ  : J<0x1, "jeq",  Connex_CC_EQ>;
defm JUGT : J<0x2, "jgt", Connex_CC_GTU>;
defm JUGE : J<0x3, "jge", Connex_CC_GEU>;
defm JNE  : J<0x5, "jne",  Connex_CC_NE>;
defm JSGT : J<0x6, "jsgt", Connex_CC_GT>;
defm JSGE : J<0x7, "jsge", Connex_CC_GE>;
}

// Inspired from def : Pat<(f32 (load addr:$src)), (LDRAM addr:$src)>;
//Pat<(f32 (load addr:$src)), (JEQ_ri addr:$src)>;

// ALU instructions
class ALU_RI<bits<4> Opc, string OpcodeStr, SDNode OpNode>
    : InstConnex<(outs GPR:$dst), (ins GPR:$src2, i64imm:$imm),
              !strconcat(OpcodeStr, "i\t$dst, $imm"),
              [(set GPR:$dst, (OpNode GPR:$src2, i64immSExt32:$imm))]> {
              //[(set GPR:$dst, (OpNode GPR:$src2, (i64 i64immSExt32:$imm)))]> {
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{51-48} = dst;
  let Inst{31-0} = imm;

  let op = Opc;
  let ConnexSrc = 0;
  // let ConnexClass = 7; // Connex_ALU64
  let ConnexClass = Connex_ALU64; // Inspired from BPFInstrInfo.td (Oct 2025)
}

class ALU_RR<bits<4> Opc, string OpcodeStr, SDNode OpNode>
    : InstConnex<(outs GPR:$dst), (ins GPR:$src2, GPR:$src),
              !strconcat(OpcodeStr, "\t$dst, $src"),
              [(set GPR:$dst, (OpNode i64:$src2, i64:$src))]> {
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<4> src;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;

  let op = Opc;
  let ConnexSrc = 1;
  // let ConnexClass = 7; // Connex_ALU64
  let ConnexClass = Connex_ALU64; // Inspired from BPFInstrInfo.td (Oct 2025)
}

multiclass ALU<bits<4> Opc, string OpcodeStr, SDNode OpNode> {
  def _rr : ALU_RR<Opc, OpcodeStr, OpNode>;
  def _ri : ALU_RI<Opc, OpcodeStr, OpNode>;
}

let Constraints = "$dst = $src2" in {
let isAsCheapAsAMove = 1 in {
  defm ADD : ALU<0x0, "add", add>;
  defm SUB : ALU<0x1, "sub", sub>;
  defm OR  : ALU<0x4, "or", or>;
  defm AND : ALU<0x5, "and", and>;
  defm SLL : ALU<0x6, "sll", shl>;
  defm SRL : ALU<0x7, "srl", srl>;
  defm XOR : ALU<0xa, "xor", xor>;
  defm SRA : ALU<0xc, "sra", sra>;
}
  defm MUL : ALU<0x2, "mul", mul>;
  defm DIV : ALU<0x3, "div", udiv>;
}

class MOV_RR<string OpcodeStr>
    : InstConnex<(outs GPR:$dst), (ins GPR:$src),
              !strconcat(OpcodeStr, "\t$dst, $src"),
              []> {
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<4> src;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;

  let op = 0xb;     // Connex_MOV
  let ConnexSrc = 1;   // Connex_X
  // let ConnexClass = 7; // Connex_ALU64
  let ConnexClass = Connex_ALU64; // Inspired from BPFInstrInfo.td (Oct 2025)
}

class MOV_RI<string OpcodeStr>
    : InstConnex<(outs GPR:$dst), (ins i64imm:$imm),
              !strconcat(OpcodeStr, "\t$dst, $imm"),
              [(set GPR:$dst, (i64 i64immSExt32:$imm))]> {
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{51-48} = dst;
  let Inst{31-0} = imm;

  let op = 0xb;     // Connex_MOV
  let ConnexSrc = 0;   // Connex_K
  // let ConnexClass = 7; // Connex_ALU64
  let ConnexClass = Connex_ALU64; // Inspired from BPFInstrInfo.td (Oct 2025)
}

class LD_IMM64<bits<4> Pseudo, string OpcodeStr>
    : InstConnex<(outs GPR:$dst), (ins u64imm:$imm),
              !strconcat(OpcodeStr, "\t$dst, $imm"),
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

  let mode = 0;     // Connex_IMM
  let size = 3;     // Connex_DW
  // let ConnexClass = 0; // Connex_LD
  let ConnexClass = Connex_LD; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def LD_imm64 : LD_IMM64<0, "ld_64">;
def MOV_rr : MOV_RR<"mov">;
def MOV_ri : MOV_RI<"mov">;
}

def FI_ri
    : InstConnex2<(outs GPR:$dst), (ins MEMri:$addr),
               "lea\t$dst, $addr",
               [(set i64:$dst, FIri:$addr)]> {
  // This is a tentative instruction, and will be replaced
  // with MOV_rr and ADD_ri in PEI phase
}


def LD_pseudo
    : InstConnex<(outs GPR:$dst), (ins i64imm:$pseudo, u64imm:$imm),
              "ld_pseudo\t$dst, $pseudo, $imm",
              [(set GPR:$dst, (int_connex_pseudo imm:$pseudo, imm:$imm))]> {

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

  let mode = 0;     // Connex_IMM
  let size = 3;     // Connex_DW
  // let ConnexClass = 0; // Connex_LD
  let ConnexClass = Connex_LD; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

// STORE instructions
class STORE<bits<2> SizeOp, string OpcodeStr, list<dag> Pattern>
    : InstConnex<(outs), (ins GPR:$src, MEMri:$addr),
              !strconcat(OpcodeStr, "\t$addr, $src"), Pattern> {
  bits<3> mode;
  bits<2> size;
  bits<4> src;
  bits<20> addr;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = addr{19-16}; // base reg
  let Inst{55-52} = src;
  let Inst{47-32} = addr{15-0}; // offset

  let mode = 3;     // Connex_MEM
  let size = SizeOp;
  // let ConnexClass = 3; // Connex_STX
  let ConnexClass = Connex_STX; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

class STOREi64<bits<2> Opc, string OpcodeStr, PatFrag OpNode>
    : STORE<Opc, OpcodeStr, [(OpNode i64:$src, ADDRri:$addr)]>;

def STW : STOREi64<0x0, "stw", truncstorei32>;
def STH : STOREi64<0x1, "sth", truncstorei16>;
def STB : STOREi64<0x2, "stb", truncstorei8>;
def STD : STOREi64<0x3, "std", store>;

// LOAD instructions
class LOAD<bits<2> SizeOp, string OpcodeStr, list<dag> Pattern>
    : InstConnex<(outs GPR:$dst), (ins MEMri:$addr),
              !strconcat(OpcodeStr, "\t$dst, $addr"), Pattern> {
  bits<3> mode;
  bits<2> size;
  bits<4> dst;
  bits<20> addr;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = dst;
  let Inst{55-52} = addr{19-16};
  let Inst{47-32} = addr{15-0};

  let mode = 3;     // Connex_MEM
  let size = SizeOp;
  // let ConnexClass = 1; // Connex_LDX
  let ConnexClass = Connex_LDX; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

class LOADi64<bits<2> SizeOp, string OpcodeStr, PatFrag OpNode>
    : LOAD<SizeOp, OpcodeStr, [(set i64:$dst, (OpNode ADDRri:$addr))]>;

def LDW : LOADi64<0x0, "ldw", zextloadi32>;
def LDH : LOADi64<0x1, "ldh", zextloadi16>;
def LDB : LOADi64<0x2, "ldb", zextloadi8>;
def LDD : LOADi64<0x3, "ldd", load>;

class BRANCH<bits<4> Opc, string OpcodeStr, list<dag> Pattern>
    : InstConnex<(outs), (ins brtarget:$BrDst),
              !strconcat(OpcodeStr, "\t$BrDst"), Pattern> {
  bits<4> op;
  bits<16> BrDst;
  bits<1> ConnexSrc;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{47-32} = BrDst;

  let op = Opc;
  let ConnexSrc = 0;
  // let ConnexClass = 5; // Connex_JMP
  let ConnexClass = Connex_JMP; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

/*
// Inspired from BPFInstrInfo.td (from Oct 2025)
class BRANCH_LONG<BPFJumpOp Opc, string OpcodeStr, list<dag> Pattern>
    : TYPE_ALU_JMP<Opc.Value, BPF_K.Value,
                   (outs),
                   (ins brtarget:$BrDst),
                   !strconcat(OpcodeStr, " $BrDst"),
                   Pattern> {
  bits<32> BrDst;

  let Inst{31-0} = BrDst;
  let BPFClass = BPF_JMP32;
}
*/

class CALL<string OpcodeStr>
    : InstConnex<(outs), (ins calltarget:$BrDst),
              !strconcat(OpcodeStr, "\t$BrDst"), []> {
  bits<4> op;
  bits<32> BrDst;
  bits<1> ConnexSrc;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{31-0} = BrDst;

  let op = 8;       // Connex_CALL
  let ConnexSrc = 0;
  // let ConnexClass = 5; // Connex_JMP
  let ConnexClass = Connex_JMP; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

// Jump always
let isBranch = 1, isTerminator = 1, hasDelaySlot=0, isBarrier = 1 in {
  def JMP : BRANCH<0x0, "jmp", [(br bb:$BrDst)]>;
  // def JMPL : BRANCH_LONG<BPF_JA, "gotol", []>;
}

// Jump and link
let isCall=1, hasDelaySlot=0, Uses = [R11],
    // Potentially clobbered registers
    Defs = [R0, R1, R2, R3, R4, R5] in {
  def JAL  : CALL<"call">;
}

class NOP_I<string OpcodeStr>
    : InstConnex<(outs), (ins i32imm:$imm),
          //!strconcat(OpcodeStr, "\t$imm ; // scalar or vector NOP"), []>
          !strconcat(OpcodeStr,
                     " ; // (immOperand = $imm ) scalar or vector NOP"), []>
                                                                              {
  // mov r0, r0 == nop
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<4> src;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{55-52} = src;
  let Inst{51-48} = dst;

  let op = 0xb;     // Connex_MOV
  let ConnexSrc = 1;   // Connex_X
  // let ConnexClass = 7; // Connex_ALU64
  let ConnexClass = Connex_ALU64; // Inspired from BPFInstrInfo.td (Oct 2025)
  let src = 0;      // R0
  let dst = 0;      // R0
}

/* If we manually generate NOP (for delay slots) it means we want to keep it,
  otherwise we should NOT have it generated
  (hasSideEffects - The instruction has side effects that are not captured
    by any operands of the instruction or other flags.)
*/
let  hasSideEffects = 1 in
  def NOP_BPF : NOP_I<"NOP">;


class RET<string OpcodeStr>
    : InstConnex<(outs), (ins),
              !strconcat(OpcodeStr, ""), [(Connexretflag)]> {
  bits<4> op;

  let Inst{63-60} = op;
  let Inst{59} = 0;
  let Inst{31-0} = 0;

  let op = 9;       // Connex_EXIT
  // let ConnexClass = 5; // Connex_JMP
  let ConnexClass = Connex_JMP; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

let isReturn = 1, isTerminator = 1, hasDelaySlot=0, isBarrier = 1,
    isNotDuplicable = 1 in {
  def RET : RET<"ret">;
}

// ADJCALLSTACKDOWN/UP pseudo insns
let Defs = [R11], Uses = [R11], isCodeGenOnly = 1 in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              "#ADJCALLSTACKDOWN $amt1 $amt2",
                              [(Connexcallseq_start timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              "#ADJCALLSTACKUP $amt1 $amt2",
                              [(Connexcallseq_end timm:$amt1, timm:$amt2)]>;
}

let usesCustomInserter = 1 in {
  def Select : Pseudo<
                 (outs GPR:$dst),
                 (ins GPR:$lhs, GPR:$rhs, i64imm:$imm, GPR:$src, GPR:$src2),
                 "# Select PSEUDO $dst = $lhs $imm $rhs ? $src : $src2",
                 [(set i64:$dst,
                   (Connexselectcc i64:$lhs, i64:$rhs, (i64 imm:$imm),
                    i64:$src, i64:$src2))]>;
}

// load 64-bit global addr into register
def : Pat<(ConnexWrapper tglobaladdr:$in), (LD_imm64 tglobaladdr:$in)>;

// 0xffffFFFF doesn't fit into simm32, optimize common case
def : Pat<(i64 (and (i64 GPR:$src), 0xffffFFFF)),
          (SRL_ri (SLL_ri (i64 GPR:$src), 32), 32)>;

// Calls
def : Pat<(Connexcall tglobaladdr:$dst), (JAL tglobaladdr:$dst)>;
def : Pat<(Connexcall imm:$dst), (JAL imm:$dst)>;

// Loads
def : Pat<(extloadi8  ADDRri:$src), (i64 (LDB ADDRri:$src))>;
def : Pat<(extloadi16 ADDRri:$src), (i64 (LDH ADDRri:$src))>;
def : Pat<(extloadi32 ADDRri:$src), (i64 (LDW ADDRri:$src))>;

// Atomics
class XADD<bits<2> SizeOp, string OpcodeStr, PatFrag OpNode>
    : InstConnex<(outs GPR:$dst), (ins MEMri:$addr, GPR:$val),
              !strconcat(OpcodeStr, "\t$dst, $addr, $val"),
              [(set GPR:$dst, (OpNode ADDRri:$addr, GPR:$val))]> {
  bits<3> mode;
  bits<2> size;
  // bits<4> src;
  bits<20> addr;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{51-48} = addr{19-16}; // base reg
  // let Inst{55-52} = src;
  let Inst{47-32} = addr{15-0}; // offset

  let mode = 6;     // Connex_XADD
  let size = SizeOp;
  // let ConnexClass = 3; // Connex_STX
  let ConnexClass = Connex_STX; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

// Inspired from BPFInstrInfo.td (from Oct 2025)
// Atomic add, and, or, xor
class ATOMIC_NOFETCH<ConnexWidthModifer SizeOp, string OpType, RegisterClass RegTp,
                     ConnexArithOp Opc, string Opstr>
    : TYPE_LD_ST<Connex_ATOMIC.Value, SizeOp.Value,
                 (outs RegTp:$dst),
                 (ins MEMri:$addr, RegTp:$val),
                 "lock *(" #OpType# " *)($addr ) " #Opstr# "= $val",
                   // added extra space at ($addr ) to avoid
                   //      error: unable to find operand: 'addr)'
                 []> {
  bits<4> dst;
  bits<20> addr;

  let Inst{51-48} = addr{19-16}; // base reg
  let Inst{55-52} = dst;
  let Inst{47-32} = addr{15-0}; // offset
  let Inst{7-4} = Opc.Value;
  let ConnexClass = Connex_STX;
}

let Constraints = "$dst = $val" in {
  def XADD32  : ATOMIC_NOFETCH<Connex_W,  "u32", GPR, Connex_ADD, "+">;
  def XADD64  : ATOMIC_NOFETCH<Connex_DW, "u64", GPR, Connex_ADD, "+">;

  // undefined def XADD16 : XADD<1, "xadd16", atomic_load_add_16>;
  // undefined def XADD8  : XADD<2, "xadd8", atomic_load_add_8>;
}

// bswap16, bswap32, bswap64
class BSWAP<bits<32> SizeOp, string OpcodeStr, list<dag> Pattern>
    : InstConnex<(outs GPR:$dst), (ins GPR:$src),
              !strconcat(OpcodeStr, "\t$dst"),
              Pattern> {
  bits<4> op;
  bits<1> ConnexSrc;
  bits<4> dst;
  bits<32> imm;

  let Inst{63-60} = op;
  let Inst{59} = ConnexSrc;
  let Inst{51-48} = dst;
  let Inst{31-0} = imm;

  let op = 0xd;     // Connex_END
  let ConnexSrc = 1;   // Connex_TO_LE (TODO: use Connex_TO_BE for big-endian target)
  // let ConnexClass = 4; // Connex_ALU
  let ConnexClass = Connex_ALU; // Inspired from BPFInstrInfo.td (from Oct 2025)
  let imm = SizeOp;
}

let Constraints = "$dst = $src" in {
def BSWAP16 : BSWAP<16, "bswap16",
                    [(set GPR:$dst, (srl (bswap GPR:$src), (i64 48)))]>;
def BSWAP32 : BSWAP<32, "bswap32",
                    [(set GPR:$dst, (srl (bswap GPR:$src), (i64 32)))]>;
def BSWAP64 : BSWAP<64, "bswap64",
                    [(set GPR:$dst, (bswap GPR:$src))]>;
}

let Defs = [R0, R1, R2, R3, R4, R5], Uses = [R6], hasSideEffects = 1,
    hasExtraDefRegAllocReq = 1, hasExtraSrcRegAllocReq = 1, mayLoad = 1 in {
class LOAD_ABS<bits<2> SizeOp, string OpcodeStr, Intrinsic OpNode>
    : InstConnex<(outs), (ins GPR:$skb, i64imm:$imm),
              !strconcat(OpcodeStr, "\tr0, $skb.data + $imm"),
              [(set R0, (OpNode GPR:$skb, i64immSExt32:$imm))]> {
              //[(set R0, (OpNode GPR:$skb, (i64immSExt32:$imm)))]> {
  bits<3> mode;
  bits<2> size;
  bits<32> imm;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{31-0} = imm;

  let mode = 1;     // Connex_ABS
  let size = SizeOp;
  // let ConnexClass = 0; // Connex_LD
  let ConnexClass = Connex_LD; // Inspired from BPFInstrInfo.td (from Oct 2025)
}

class LOAD_IND<bits<2> SizeOp, string OpcodeStr, Intrinsic OpNode>
    : InstConnex<(outs), (ins GPR:$skb, GPR:$val),
              !strconcat(OpcodeStr, "\tr0, $skb.data + $val"),
              [(set R0, (OpNode GPR:$skb, GPR:$val))]> {
  bits<3> mode;
  bits<2> size;
  bits<4> val;

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
  let Inst{55-52} = val;

  let mode = 2;     // Connex_IND
  let size = SizeOp;
  // let ConnexClass = 0; // Connex_LD
  let ConnexClass = Connex_LD; // Inspired from BPFInstrInfo.td (from Oct 2025)
}
}

def LD_ABS_B : LOAD_ABS<2, "ldabs_b", int_connex_load_byte>;
def LD_ABS_H : LOAD_ABS<1, "ldabs_h", int_connex_load_half>;
def LD_ABS_W : LOAD_ABS<0, "ldabs_w", int_connex_load_word>;

def LD_IND_B : LOAD_IND<2, "ldind_b", int_connex_load_byte>;
def LD_IND_H : LOAD_IND<1, "ldind_h", int_connex_load_half>;
def LD_IND_W : LOAD_IND<0, "ldind_w", int_connex_load_word>;

