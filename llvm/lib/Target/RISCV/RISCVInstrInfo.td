//===-- RISCVInstrInfo.td - Target Description for RISCV ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormats.td"

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCVCall         : SDTypeProfile<0, -1, [SDTCisVT<0, XLenVT>]>;
def SDT_RISCVCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>,
                                            SDTCisVT<1, i32>]>;
def SDT_RISCVCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                          SDTCisVT<1, i32>]>;
def SDT_RISCVSelectCC     : SDTypeProfile<1, 5, [SDTCisSameAs<1, 2>,
                                                 SDTCisSameAs<0, 4>,
                                                 SDTCisSameAs<4, 5>]>;


def Call         : SDNode<"RISCVISD::CALL", SDT_RISCVCall,
                          [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                           SDNPVariadic]>;
def CallSeqStart : SDNode<"ISD::CALLSEQ_START", SDT_RISCVCallSeqStart,
                          [SDNPHasChain, SDNPOutGlue]>;
def CallSeqEnd   : SDNode<"ISD::CALLSEQ_END", SDT_RISCVCallSeqEnd,
                          [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def RetFlag      : SDNode<"RISCVISD::RET_FLAG", SDTNone,
                          [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def SelectCC     : SDNode<"RISCVISD::SELECT_CC", SDT_RISCVSelectCC,
                          [SDNPInGlue]>;

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

class ImmAsmOperand<string prefix, int width, string suffix> : AsmOperandClass {
  let Name = prefix # "Imm" # width # suffix;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}

class SImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"S", width, suffix> {
}

class UImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"U", width, suffix> {
}

def FenceArg : AsmOperandClass {
  let Name = "FenceArg";
  let RenderMethod = "addFenceArgOperands";
  let DiagnosticType = "InvalidFenceArg";
}

def fencearg : Operand<XLenVT> {
  let ParserMatchClass = FenceArg;
  let PrintMethod = "printFenceArg";
  let DecoderMethod = "decodeUImmOperand<4>";
}

def UImmLog2XLenAsmOperand : AsmOperandClass {
  let Name = "UImmLog2XLen";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImmLog2XLen";
}

def uimmlog2xlen : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return isUInt<6>(Imm);
  return isUInt<5>(Imm);
}]> {
  let ParserMatchClass = UImmLog2XLenAsmOperand;
  // TODO: should ensure invalid shamt is rejected when decoding.
  let DecoderMethod = "decodeUImmOperand<6>";
}

def uimm5 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5>;
  let DecoderMethod = "decodeUImmOperand<5>";
}

def simm12 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<12>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<12>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<12>";
}

def uimm12 : Operand<XLenVT> {
  let ParserMatchClass = UImmAsmOperand<12>;
  let DecoderMethod = "decodeUImmOperand<12>";
}

// A 13-bit signed immediate where the least significant bit is zero.
def simm13_lsb0 : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<13, "Lsb0">;
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<13>";
}

def uimm20 : Operand<XLenVT> {
  let ParserMatchClass = UImmAsmOperand<20>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<20>";
}

// A 21-bit signed immediate where the least significant bit is zero.
def simm21_lsb0 : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<21, "Lsb0">;
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<21>";
}

// A parameterized register class alternative to i32imm/i64imm from Target.td.
def ixlenimm : Operand<XLenVT>;

// Standalone (codegen-only) immleaf patterns.
def simm32 : ImmLeaf<XLenVT, [{return isInt<32>(Imm);}]>;

// Addressing modes.
// Necessary because a frameindex can't be matched directly in a pattern.
def AddrFI : ComplexPattern<iPTR, 1, "SelectAddrFI", [frameindex], []>;

// Extract least significant 12 bits from an immediate value and sign extend
// them.
def LO12Sext : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(SignExtend64<12>(N->getZExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;

// Extract the most significant 20 bits from an immediate value. Add 1 if bit
// 11 is 1, to compensate for the low 12 bits in the matching immediate addi
// or ld/st being negative.
def HI20 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(((N->getZExtValue()+0x800) >> 12) & 0xfffff,
                                   SDLoc(N), N->getValueType(0));
}]>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class BranchCC_rri<bits<3> funct3, string opcodestr>
    : RVInstB<funct3, OPC_BRANCH, (outs),
              (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12),
              opcodestr, "$rs1, $rs2, $imm12"> {
  let isBranch = 1;
  let isTerminator = 1;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class Load_ri<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_LOAD, (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
              opcodestr, "$rd, ${imm12}(${rs1})">;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class Store_rri<bits<3> funct3, string opcodestr>
    : RVInstS<funct3, OPC_STORE, (outs),
              (ins GPR:$rs2, GPR:$rs1, simm12:$imm12),
              opcodestr, "$rs2, ${imm12}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ALU_ri<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
              opcodestr, "$rd, $rs1, $imm12">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Shift_ri<bit arithshift, bits<3> funct3, string opcodestr>
    : RVInstIShift<arithshift, funct3, OPC_OP_IMM, (outs GPR:$rd),
                   (ins GPR:$rs1, uimmlog2xlen:$shamt), opcodestr,
                   "$rd, $rs1, $shamt">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ALU_rr<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP, (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
              opcodestr, "$rd, $rs1, $rs2">;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class CSR_ir<bits<3> funct3, string opcodestr> :
      RVInstI<funct3, OPC_SYSTEM, (outs GPR:$rd), (ins uimm12:$imm12, GPR:$rs1),
              opcodestr, "$rd, $imm12, $rs1">;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class CSR_ii<bits<3> funct3, string opcodestr> :
      RVInstI<funct3, OPC_SYSTEM, (outs GPR:$rd),
              (ins uimm12:$imm12, uimm5:$rs1),
              opcodestr, "$rd, $imm12, $rs1">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ShiftW_ri<bit arithshift, bits<3> funct3, string opcodestr>
    : RVInstIShiftW<arithshift, funct3, OPC_OP_IMM_32, (outs GPR:$rd),
                    (ins GPR:$rs1, uimm5:$shamt), opcodestr,
                    "$rd, $rs1, $shamt">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ALUW_rr<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_32, (outs GPR:$rd),
              (ins GPR:$rs1, GPR:$rs2), opcodestr, "$rd, $rs1, $rs2">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
def LUI : RVInstU<OPC_LUI, (outs GPR:$rd), (ins uimm20:$imm20),
                  "lui", "$rd, $imm20">;

def AUIPC : RVInstU<OPC_AUIPC, (outs GPR:$rd), (ins uimm20:$imm20),
                    "auipc", "$rd, $imm20">;

let isCall = 1 in
def JAL : RVInstJ<OPC_JAL, (outs GPR:$rd), (ins simm21_lsb0:$imm20),
                  "jal", "$rd, $imm20">;

let isCall = 1 in
def JALR : RVInstI<0b000, OPC_JALR, (outs GPR:$rd),
                   (ins GPR:$rs1, simm12:$imm12),
                   "jalr", "$rd, $rs1, $imm12">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

def BEQ  : BranchCC_rri<0b000, "beq">;
def BNE  : BranchCC_rri<0b001, "bne">;
def BLT  : BranchCC_rri<0b100, "blt">;
def BGE  : BranchCC_rri<0b101, "bge">;
def BLTU : BranchCC_rri<0b110, "bltu">;
def BGEU : BranchCC_rri<0b111, "bgeu">;

def LB  : Load_ri<0b000, "lb">;
def LH  : Load_ri<0b001, "lh">;
def LW  : Load_ri<0b010, "lw">;
def LBU : Load_ri<0b100, "lbu">;
def LHU : Load_ri<0b101, "lhu">;

def SB : Store_rri<0b000, "sb">;
def SH : Store_rri<0b001, "sh">;
def SW : Store_rri<0b010, "sw">;

def ADDI  : ALU_ri<0b000, "addi">;
def SLTI  : ALU_ri<0b010, "slti">;
def SLTIU : ALU_ri<0b011, "sltiu">;
def XORI  : ALU_ri<0b100, "xori">;
def ORI   : ALU_ri<0b110, "ori">;
def ANDI  : ALU_ri<0b111, "andi">;

def SLLI : Shift_ri<0, 0b001, "slli">;
def SRLI : Shift_ri<0, 0b101, "srli">;
def SRAI : Shift_ri<1, 0b101, "srai">;

def ADD  : ALU_rr<0b0000000, 0b000, "add">;
def SUB  : ALU_rr<0b0100000, 0b000, "sub">;
def SLL  : ALU_rr<0b0000000, 0b001, "sll">;
def SLT  : ALU_rr<0b0000000, 0b010, "slt">;
def SLTU : ALU_rr<0b0000000, 0b011, "sltu">;
def XOR  : ALU_rr<0b0000000, 0b100, "xor">;
def SRL  : ALU_rr<0b0000000, 0b101, "srl">;
def SRA  : ALU_rr<0b0100000, 0b101, "sra">;
def OR   : ALU_rr<0b0000000, 0b110, "or">;
def AND  : ALU_rr<0b0000000, 0b111, "and">;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
def FENCE : RVInstI<0b000, OPC_MISC_MEM, (outs),
                    (ins fencearg:$pred, fencearg:$succ),
                    "fence", "$pred, $succ"> {
  bits<4> pred;
  bits<4> succ;

  let rs1 = 0;
  let rd = 0;
  let imm12 = {0b0000,pred,succ};
}

def FENCE_I : RVInstI<0b001, OPC_MISC_MEM, (outs), (ins), "fence.i", ""> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 0;
}

def ECALL : RVInstI<0b000, OPC_SYSTEM, (outs), (ins), "ecall", ""> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 0;
}

def EBREAK : RVInstI<0b000, OPC_SYSTEM, (outs), (ins), "ebreak", ""> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 1;
}
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0

def CSRRW : CSR_ir<0b001, "csrrw">;
def CSRRS : CSR_ir<0b010, "csrrs">;
def CSRRC : CSR_ir<0b011, "csrrc">;

def CSRRWI : CSR_ii<0b101, "csrrwi">;
def CSRRSI : CSR_ii<0b110, "csrrsi">;
def CSRRCI : CSR_ii<0b111, "csrrci">;

/// RV64I instructions

let Predicates = [IsRV64] in {
def LWU   : Load_ri<0b110, "lwu">;
def LD    : Load_ri<0b011, "ld">;
def SD    : Store_rri<0b011, "sd">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def ADDIW : RVInstI<0b000, OPC_OP_IMM_32, (outs GPR:$rd),
                    (ins GPR:$rs1, simm12:$imm12),
                    "addiw", "$rd, $rs1, $imm12">;

def SLLIW : ShiftW_ri<0, 0b001, "slliw">;
def SRLIW : ShiftW_ri<0, 0b101, "srliw">;
def SRAIW : ShiftW_ri<1, 0b101, "sraiw">;

def ADDW  : ALUW_rr<0b0000000, 0b000, "addw">;
def SUBW  : ALUW_rr<0b0100000, 0b000, "subw">;
def SLLW  : ALUW_rr<0b0000000, 0b001, "sllw">;
def SRLW  : ALUW_rr<0b0000000, 0b101, "srlw">;
def SRAW  : ALUW_rr<0b0100000, 0b101, "sraw">;
} // Predicates = [IsRV64]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//
// Naming convention: For 'generic' pattern classes, we use the naming
// convention PatTy1Ty2. For pattern classes which offer a more complex
// expension, prefix the class name, e.g. BccPat.
//===----------------------------------------------------------------------===//

/// Generic pattern classes

class PatGprGpr<SDPatternOperator OpNode, RVInstR Inst>
    : Pat<(OpNode GPR:$rs1, GPR:$rs2), (Inst GPR:$rs1, GPR:$rs2)>;
class PatGprSimm12<SDPatternOperator OpNode, RVInstI Inst>
    : Pat<(OpNode GPR:$rs1, simm12:$imm12), (Inst GPR:$rs1, simm12:$imm12)>;
class PatGprUimmLog2XLen<SDPatternOperator OpNode, RVInstIShift Inst>
    : Pat<(OpNode GPR:$rs1, uimmlog2xlen:$shamt),
          (Inst GPR:$rs1, uimmlog2xlen:$shamt)>;

/// Predicates

def IsOrAdd: PatFrag<(ops node:$A, node:$B), (or node:$A, node:$B), [{
  return isOrEquivalentToAdd(N);
}]>;

/// Immediates

def : Pat<(simm12:$imm), (ADDI X0, simm12:$imm)>;
// TODO: Add a pattern for immediates with all zeroes in the lower 12 bits.
def : Pat<(simm32:$imm), (ADDI (LUI (HI20 imm:$imm)), (LO12Sext imm:$imm))>;

/// Simple arithmetic operations

def : PatGprGpr<add, ADD>;
def : PatGprSimm12<add, ADDI>;
def : PatGprGpr<sub, SUB>;
def : PatGprGpr<or, OR>;
def : PatGprSimm12<or, ORI>;
def : PatGprGpr<and, AND>;
def : PatGprSimm12<and, ANDI>;
def : PatGprGpr<xor, XOR>;
def : PatGprSimm12<xor, XORI>;
def : PatGprGpr<shl, SLL>;
def : PatGprUimmLog2XLen<shl, SLLI>;
def : PatGprGpr<srl, SRL>;
def : PatGprUimmLog2XLen<srl, SRLI>;
def : PatGprGpr<sra, SRA>;
def : PatGprUimmLog2XLen<sra, SRAI>;

/// FrameIndex calculations

def : Pat<(add (i32 AddrFI:$Rs), simm12:$imm12),
          (ADDI (i32 AddrFI:$Rs), simm12:$imm12)>;
def : Pat<(IsOrAdd (i32 AddrFI:$Rs), simm12:$imm12),
          (ADDI (i32 AddrFI:$Rs), simm12:$imm12)>;

/// Setcc

def : PatGprGpr<setlt, SLT>;
def : PatGprSimm12<setlt, SLTI>;
def : PatGprGpr<setult, SLTU>;
def : PatGprSimm12<setult, SLTIU>;

// Define pattern expansions for setcc operations that aren't directly
// handled by a RISC-V instruction.
def : Pat<(seteq GPR:$rs1, GPR:$rs2), (SLTIU (XOR GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(setne GPR:$rs1, GPR:$rs2), (SLTU X0, (XOR GPR:$rs1, GPR:$rs2))>;
def : Pat<(setugt GPR:$rs1, GPR:$rs2), (SLTU GPR:$rs2, GPR:$rs1)>;
def : Pat<(setuge GPR:$rs1, GPR:$rs2), (XORI (SLTU GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(setule GPR:$rs1, GPR:$rs2), (XORI (SLTU GPR:$rs2, GPR:$rs1), 1)>;
def : Pat<(setgt GPR:$rs1, GPR:$rs2), (SLT GPR:$rs2, GPR:$rs1)>;
def : Pat<(setge GPR:$rs1, GPR:$rs2), (XORI (SLT GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(setle GPR:$rs1, GPR:$rs2), (XORI (SLT GPR:$rs2, GPR:$rs1), 1)>;

let usesCustomInserter = 1 in
def Select_GPR_Using_CC_GPR
    : Pseudo<(outs GPR:$dst),
             (ins GPR:$lhs, GPR:$rhs, ixlenimm:$imm, GPR:$src, GPR:$src2),
             [(set XLenVT:$dst, (SelectCC GPR:$lhs, GPR:$rhs,
              (XLenVT imm:$imm), GPR:$src, GPR:$src2))]>;

/// Branches and jumps

// Match `(brcond (CondOp ..), ..)` and lower to the appropriate RISC-V branch
// instruction.
class BccPat<PatFrag CondOp, RVInstB Inst>
    : Pat<(brcond (i32 (CondOp GPR:$rs1, GPR:$rs2)), bb:$imm12),
          (Inst GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12)>;

def : BccPat<seteq, BEQ>;
def : BccPat<setne, BNE>;
def : BccPat<setlt, BLT>;
def : BccPat<setge, BGE>;
def : BccPat<setult, BLTU>;
def : BccPat<setuge, BGEU>;

class BccSwapPat<PatFrag CondOp, RVInst InstBcc>
    : Pat<(brcond (i32 (CondOp GPR:$rs1, GPR:$rs2)), bb:$imm12),
          (InstBcc GPR:$rs2, GPR:$rs1, bb:$imm12)>;

// Condition codes that don't have matching RISC-V branch instructions, but
// are trivially supported by swapping the two input operands
def : BccSwapPat<setgt, BLT>;
def : BccSwapPat<setle, BGE>;
def : BccSwapPat<setugt, BLTU>;
def : BccSwapPat<setule, BGEU>;

// An extra pattern is needed for a brcond without a setcc (i.e. where the
// condition was calculated elsewhere).
def : Pat<(brcond GPR:$cond, bb:$imm12), (BNE GPR:$cond, X0, bb:$imm12)>;

let isBarrier = 1, isBranch = 1, isTerminator = 1 in
def PseudoBR : Pseudo<(outs), (ins simm21_lsb0:$imm20), [(br bb:$imm20)]>,
               PseudoInstExpansion<(JAL X0, simm21_lsb0:$imm20)>;

let isCall = 1, Defs=[X1] in
let isBarrier = 1, isBranch = 1, isIndirectBranch = 1, isTerminator = 1 in
def PseudoBRIND : Pseudo<(outs), (ins GPR:$rs1, simm12:$imm12), []>,
                  PseudoInstExpansion<(JALR X0, GPR:$rs1, simm12:$imm12)>;

def : Pat<(brind GPR:$rs1), (PseudoBRIND GPR:$rs1, 0)>;
def : Pat<(brind (add GPR:$rs1, simm12:$imm12)),
          (PseudoBRIND GPR:$rs1, simm12:$imm12)>;

let isCall = 1, Defs = [X1] in
def PseudoCALL : Pseudo<(outs), (ins GPR:$rs1), [(Call GPR:$rs1)]>,
                 PseudoInstExpansion<(JALR X1, GPR:$rs1, 0)>;

let isBarrier = 1, isReturn = 1, isTerminator = 1 in
def PseudoRET : Pseudo<(outs), (ins), [(RetFlag)]>,
                PseudoInstExpansion<(JALR X0, X1, 0)>;

/// Loads

multiclass LdPat<PatFrag LoadOp, RVInst Inst> {
  def : Pat<(LoadOp GPR:$rs1), (Inst GPR:$rs1, 0)>;
  def : Pat<(LoadOp AddrFI:$rs1), (Inst AddrFI:$rs1, 0)>;
  def : Pat<(LoadOp (add GPR:$rs1, simm12:$imm12)),
            (Inst GPR:$rs1, simm12:$imm12)>;
  def : Pat<(LoadOp (add AddrFI:$rs1, simm12:$imm12)),
            (Inst AddrFI:$rs1, simm12:$imm12)>;
  def : Pat<(LoadOp (IsOrAdd AddrFI:$rs1, simm12:$imm12)),
            (Inst AddrFI:$rs1, simm12:$imm12)>;
}

defm : LdPat<sextloadi8, LB>;
defm : LdPat<extloadi8, LB>;
defm : LdPat<sextloadi16, LH>;
defm : LdPat<extloadi16, LH>;
defm : LdPat<load, LW>;
defm : LdPat<zextloadi8, LBU>;
defm : LdPat<zextloadi16, LHU>;

/// Stores

multiclass StPat<PatFrag StoreOp, RVInst Inst> {
  def : Pat<(StoreOp GPR:$rs2, GPR:$rs1), (Inst GPR:$rs2, GPR:$rs1, 0)>;
  def : Pat<(StoreOp GPR:$rs2, AddrFI:$rs1), (Inst GPR:$rs2, AddrFI:$rs1, 0)>;
  def : Pat<(StoreOp GPR:$rs2, (add GPR:$rs1, simm12:$imm12)),
            (Inst GPR:$rs2, GPR:$rs1, simm12:$imm12)>;
  def : Pat<(StoreOp GPR:$rs2, (add AddrFI:$rs1, simm12:$imm12)),
            (Inst GPR:$rs2, AddrFI:$rs1, simm12:$imm12)>;
  def : Pat<(StoreOp GPR:$rs2, (IsOrAdd AddrFI:$rs1, simm12:$imm12)),
            (Inst GPR:$rs2, AddrFI:$rs1, simm12:$imm12)>;
}

defm : StPat<truncstorei8, SB>;
defm : StPat<truncstorei16, SH>;
defm : StPat<store, SW>;

/// Other pseudo-instructions

// Pessimistically assume the stack pointer will be clobbered
let Defs = [X2], Uses = [X2] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(CallSeqStart timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(CallSeqEnd timm:$amt1, timm:$amt2)]>;
} // Defs = [X2], Uses = [X2]

//===----------------------------------------------------------------------===//
// Standard extensions
//===----------------------------------------------------------------------===//

include "RISCVInstrInfoM.td"
include "RISCVInstrInfoA.td"
include "RISCVInstrInfoF.td"
include "RISCVInstrInfoD.td"
include "RISCVInstrInfoC.td"
