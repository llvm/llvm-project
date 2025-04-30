//===-- Next32InstrFormats.td - NS Instruction Formats -------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// constraints
def SDTNext32BRCC
    : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>, SDTCisVT<2, OtherVT>]>;

def SDTNext32ParallelOperations : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDTNext32SetTID : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDTNext32GMemWrite : SDTypeProfile<1, -1, []>;

def SDTNext32Sym : SDTypeProfile<0, 1, [SDTCisVT<0, iPTR>]>;

def SDTNext32WRAPPER
    : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

def SDTNext32FrameOffsetWrapper
    : SDTypeProfile<1, 2, [SDTCisPtrTy<1>, SDTCisInt<2>]>;

def SDTCondIntBinOp : SDTypeProfile<1, 4, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>, SDTCisInt<3>,
  SDTCisInt<4>
]>;

def SDTCondUnaryOp : SDTypeProfile<1, 3, [SDTCisInt<1>, SDTCisInt<2>]>;

def SDTCondIntShiftOp : SDTypeProfile<1, 4, [
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>
]>;

def SDTCondIntExtendOp : SDTypeProfile<1, 3, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisOpSmallerThanOp<1, 0>,
  SDTCisSameNumEltsAs<0, 1>, SDTCisInt<2>, SDTCisInt<3>
]>;

def SDTCondIntBinHiLoOp : SDTypeProfile<2, 4, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>,
  SDTCisSameAs<0, 4>, SDTCisSameAs<0, 5>, SDTCisInt<0>
]>;

def SDTIntTernaryOp
    : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def SDTCondIntTernaryOp : SDTypeProfile<1, 5, [
  SDTCisSameAs<1, 2>, SDTCisSameAs<1, 3>, SDTCisSameAs<1, 4>,
  SDTCisSameAs<1, 5>, SDTCisInt<0>, SDTCisInt<1>
]>;

def SDTNext32FrameOp : SDTypeProfile<3, 2, [
  SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>,
  SDTCisSameAs<0, 4>
]>;

def SDTNext32Shift64Op : SDTypeProfile<2, 3, [
  SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>,
  SDTCisSameAs<0, 4>
]>;

def SDTNext32MemOp
    : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def SDTNext32CondMemOp : SDTypeProfile<0, 5, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>
]>;

def SDTSizedIntUnaryOp
    : SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def SDTSizedIntBinOp
    : SDTypeProfile<1, 3,
                    [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>]>;

def SDTNext32GMemRead1 : SDTypeProfile<1, 4, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>
]>;
def SDTNext32GMemRead2 : SDTypeProfile<2, 4, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>
]>;
def SDTNext32GMemRead4 : SDTypeProfile<4, 4, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>
]>;
def SDTNext32GMemRead8 : SDTypeProfile<8, 4, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>, SDTCisInt<8>, SDTCisInt<9>,
  SDTCisInt<10>, SDTCisInt<11>
]>;
def SDTNext32GMemRead16 : SDTypeProfile<16, 4, [
  SDTCisInt<0>,  SDTCisInt<1>,  SDTCisInt<2>,  SDTCisInt<3>,  SDTCisInt<4>,
  SDTCisInt<5>,  SDTCisInt<6>,  SDTCisInt<7>,  SDTCisInt<8>,  SDTCisInt<9>,
  SDTCisInt<10>, SDTCisInt<11>, SDTCisInt<12>, SDTCisInt<13>, SDTCisInt<14>,
  SDTCisInt<15>, SDTCisInt<16>, SDTCisInt<17>, SDTCisInt<18>, SDTCisInt<19>
]>;

def SDTNext32GVMemRead1 : SDTypeProfile<1, 5, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>
]>;
def SDTNext32GVMemRead2 : SDTypeProfile<2, 5, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>
]>;
def SDTNext32GVMemRead4 : SDTypeProfile<4, 5, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>, SDTCisInt<8>
]>;
def SDTNext32GVMemRead8 : SDTypeProfile<8, 5, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>, SDTCisInt<8>, SDTCisInt<9>,
  SDTCisInt<10>, SDTCisInt<11>, SDTCisInt<12>
]>;
def SDTNext32GVMemRead16 : SDTypeProfile<16, 5, [
  SDTCisInt<0>,  SDTCisInt<1>,  SDTCisInt<2>,  SDTCisInt<3>,  SDTCisInt<4>,
  SDTCisInt<5>,  SDTCisInt<6>,  SDTCisInt<7>,  SDTCisInt<8>,  SDTCisInt<9>,
  SDTCisInt<10>, SDTCisInt<11>, SDTCisInt<12>, SDTCisInt<13>, SDTCisInt<14>,
  SDTCisInt<15>, SDTCisInt<16>, SDTCisInt<17>, SDTCisInt<18>, SDTCisInt<19>,
  SDTCisInt<20>
]>;
def SDTNext32GMemPrefetch
    : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def SDTNext32GMemFaOpS : SDTypeProfile<1, 5, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>
]>;
def SDTNext32GMemFaOpD : SDTypeProfile<2, 6, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>
]>;

def SDTNext32GMemCasS : SDTypeProfile<1, 6, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>
]>;
def SDTNext32GMemCasD : SDTypeProfile<2, 8, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>, SDTCisInt<8>, SDTCisInt<9>
]>;

def SDTNext32PseudoLEA : SDTypeProfile<2, 8, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>, SDTCisInt<8>, SDTCisInt<9>
]>;

// Arithmetic operations that emit both value and flags
def SDTNext32BinaryIntFlagsOp : SDTypeProfile<2, 2, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>
]>;
def SDTNext32TernaryIntFlagsOp : SDTypeProfile<2, 3, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>,
  SDTCisSameAs<0, 4>
]>;

// next32 opcode nodes
def next32inc : SDNode<"Next32ISD::INC", SDTUnaryOp>;
def next32dec : SDNode<"Next32ISD::DEC", SDTUnaryOp>;

def next32addflags : SDNode<"Next32ISD::ADDFLAGS", SDTNext32BinaryIntFlagsOp>;
def next32subflags : SDNode<"Next32ISD::SUBFLAGS", SDTNext32BinaryIntFlagsOp>;
def next32adcflags : SDNode<"Next32ISD::ADCFLAGS", SDTNext32TernaryIntFlagsOp>;
def next32sbbflags : SDNode<"Next32ISD::SBBFLAGS", SDTNext32TernaryIntFlagsOp>;
def next32frameoffsetwrapper
    : SDNode<"Next32ISD::FRAME_OFFSET_WRAPPER", SDTNext32FrameOffsetWrapper>;

def next32dup : SDNode<"Next32ISD::DUP", SDTIntBinOp>;
def next32brcc : SDNode<"Next32ISD::BR_CC", SDTNext32BRCC, [SDNPHasChain]>;
def next32select : SDNode<"Next32ISD::SELECT", SDTIntBinOp>;
def next32chain : SDNode<"Next32ISD::CHAIN", SDTIntBinOp>;
def next32chainp : SDNode<"Next32ISD::CHAINP", SDTIntBinOp>;
def next32writer : SDNode<"Next32ISD::WRITER", SDTSizedIntBinOp>;
def next32pseudolea : SDNode<"Next32ISD::PSEUDO_LEA", SDTNext32PseudoLEA, []>;
def next32gmemwrite : SDNode<"Next32ISD::G_MEM_WRITE", SDTNext32GMemWrite,
                             [SDNPHasChain, SDNPVariadic]>;
def next32gmemread1
    : SDNode<"Next32ISD::G_MEM_READ_1", SDTNext32GMemRead1, [SDNPHasChain]>;
def next32gmemread2
    : SDNode<"Next32ISD::G_MEM_READ_2", SDTNext32GMemRead2, [SDNPHasChain]>;
def next32gmemread4
    : SDNode<"Next32ISD::G_MEM_READ_4", SDTNext32GMemRead4, [SDNPHasChain]>;
def next32gmemread8
    : SDNode<"Next32ISD::G_MEM_READ_8", SDTNext32GMemRead8, [SDNPHasChain]>;
def next32gmemread16
    : SDNode<"Next32ISD::G_MEM_READ_16", SDTNext32GMemRead16, [SDNPHasChain]>;
def next32gvmemwrite : SDNode<"Next32ISD::G_VMEM_WRITE", SDTNext32GMemWrite,
                              [SDNPHasChain, SDNPVariadic]>;
def next32gvmemread1
    : SDNode<"Next32ISD::G_VMEM_READ_1", SDTNext32GVMemRead1, [SDNPHasChain]>;
def next32gvmemread2
    : SDNode<"Next32ISD::G_VMEM_READ_2", SDTNext32GVMemRead2, [SDNPHasChain]>;
def next32gvmemread4
    : SDNode<"Next32ISD::G_VMEM_READ_4", SDTNext32GVMemRead4, [SDNPHasChain]>;
def next32gvmemread8
    : SDNode<"Next32ISD::G_VMEM_READ_8", SDTNext32GVMemRead8, [SDNPHasChain]>;
def next32gvmemread16
    : SDNode<"Next32ISD::G_VMEM_READ_16", SDTNext32GVMemRead16, [SDNPHasChain]>;
def next32prefetch
    : SDNode<"Next32ISD::PREFETCH", SDTNext32GMemPrefetch, [SDNPHasChain]>;
def next32gmemfaops
    : SDNode<"Next32ISD::G_MEM_FAOP_S", SDTNext32GMemFaOpS, [SDNPHasChain]>;
def next32gmemfaopd
    : SDNode<"Next32ISD::G_MEM_FAOP_D", SDTNext32GMemFaOpD, [SDNPHasChain]>;
def next32gmemcass
    : SDNode<"Next32ISD::G_MEM_CAS_S", SDTNext32GMemCasS, [SDNPHasChain]>;
def next32gmemcasd
    : SDNode<"Next32ISD::G_MEM_CAS_D", SDTNext32GMemCasD, [SDNPHasChain]>;
def Next32RetFlag : SDNode<"Next32ISD::RET_FLAG", SDTNone,
                           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def Next32FeederArgs : SDNode<"Next32ISD::FEEDER_ARGS", SDTNone,
                              [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def Next32CallPtrWrapper : SDNode<"Next32ISD::CALLPTR_WRAPPER", SDTNone,
                                  [SDNPHasChain, SDNPOutGlue, SDNPVariadic]>;

def Next32CallPtr : SDNode<"Next32ISD::CALLPTR", SDTIntBinOp>;
def Next32CondCallPtr : SDNode<"Next32ISD::CALLPTR", SDTCondIntBinOp>;
def Next32Call : SDNode<"Next32ISD::CALL", SDTNone,
                        [SDNPHasChain, SDNPOutGlue, SDNPVariadic]>;
def Next32CallTerminatorTID
    : SDNode<"Next32ISD::CALL_TERMINATOR_TID", SDTNone,
             [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;
def Next32Sym : SDNode<"Next32ISD::SYM", SDTNext32Sym,
                       [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def Next32Barrier : SDNode<"Next32ISD::BARRIER", SDTIntBinOp, [SDNPHasChain]>;
def Next32SetTID
    : SDNode<"Next32ISD::SET_TID", SDTNext32SetTID, [SDNPHasChain]>;

// next32 conditional opcode nodes
def next32condadd : SDNode<"Next32ISD::ADDc", SDTCondIntBinOp>;
def next32condxor : SDNode<"Next32ISD::XORc", SDTCondIntBinOp>;
def next32condor : SDNode<"Next32ISD::ORc", SDTCondIntBinOp>;
def next32condand : SDNode<"Next32ISD::ANDc", SDTCondIntBinOp>;
def next32condsub : SDNode<"Next32ISD::SUBc", SDTCondIntBinOp>;
def next32condshl : SDNode<"Next32ISD::SHLc", SDTCondIntShiftOp, []>;
def next32condshr : SDNode<"Next32ISD::SHRc", SDTCondIntShiftOp, []>;
def next32condshri : SDNode<"Next32ISD::SHRIc", SDTCondIntShiftOp, []>;
def next32condselect : SDNode<"Next32ISD::SELECTc", SDTCondIntShiftOp>;
def next32condchain : SDNode<"Next32ISD::CHAINc", SDTCondIntBinOp>;
def next32condchainp : SDNode<"Next32ISD::CHAINPc", SDTCondIntBinOp>;
def next32condnot : SDNode<"Next32ISD::NOTc", SDTCondUnaryOp>;
def next32condneg : SDNode<"Next32ISD::NEGc", SDTCondUnaryOp>;
def next32condinc : SDNode<"Next32ISD::INCc", SDTCondUnaryOp>;
def next32conddec : SDNode<"Next32ISD::DECc", SDTCondUnaryOp>;
def next32condctlz : SDNode<"Next32ISD::CTLZc", SDTCondUnaryOp>;
def next32condcttz : SDNode<"Next32ISD::CTTZc", SDTCondUnaryOp>;
def next32wrapper : SDNode<"Next32ISD::WRAPPER", SDTNext32WRAPPER>;
def next32setframe
    : SDNode<"Next32ISD::SET_FRAME", SDTNext32FrameOp, [SDNPHasChain]>;
def next32resetframe
    : SDNode<"Next32ISD::RESET_FRAME", SDTIntTernaryOp, [SDNPHasChain]>;
def next32alloca
    : SDNode<"Next32ISD::ALLOCA", SDTNext32FrameOp, [SDNPHasChain]>;

def next32shl64 : SDNode<"Next32ISD::SHL64", SDTNext32Shift64Op>;
def next32shr64 : SDNode<"Next32ISD::SHR64", SDTNext32Shift64Op>;
def next32shri64 : SDNode<"Next32ISD::SHRI64", SDTNext32Shift64Op>;

def next32branchtarget : Operand<OtherVT>;
def next32mcsymbol : Operand<iPTR>;
def conditionvalue : Operand<i32> { let PrintMethod = "printCondition"; }
def rri_attribute : Operand<i32> { let PrintMethod = "printRRIAttribute"; }

def next32relocimm
    : ComplexPattern<i32, 1, "selectRelocImm", [imm, next32wrapper], [], 0>;

def next32sizefield : Operand<i32> { let PrintMethod = "printInstructionSize"; }

def next32vecelemcountfield : Operand<i32> {
  let PrintMethod = "printVectorElementCount";
}

def next32addrspacefield : Operand<i32> {
  let PrintMethod = "printMemoryAddressSpace";
}

def next32alignsize : Operand<i32> { let PrintMethod = "printAlignSize"; }
// base instruction
class Next32Instruction<bits<8> opcode, dag outs, dag Ins, string opcodestr,
                        string operands, list<dag> pattern> : Instruction {
  field bits<64> Inst;

  let Namespace = "Next32";
  let DecoderNamespace = "Next32";
  let Size = 8;
  field bits<64> SoftFail = 0;

  bits<3> DataTypeSize = 0;

  bit isWriterChain = 0;
  bit is128BitRRRRInstruction = 0;
  bit is128BitRRRRInstructionSrcReg1In = 0;
  bit is128BitRRRRInstructionSrcReg1Out = 0;
  bit is128BitRRRRInstructionSrcReg2In = 0;
  bit is128BitRRRRInstructionSrcReg2Out = 0;
  bit is128BitRRRRInstructionWithCount = 0;
  bit is128BitRRRRInstructionWithAddrSpace = 0;

  let TSFlags{0} = isWriterChain;
  let TSFlags{1} = is128BitRRRRInstruction;
  let TSFlags{2} = is128BitRRRRInstructionSrcReg1In;
  let TSFlags{3} = is128BitRRRRInstructionSrcReg1Out;
  let TSFlags{4} = is128BitRRRRInstructionSrcReg2In;
  let TSFlags{5} = is128BitRRRRInstructionSrcReg2Out;
  let TSFlags{6} = is128BitRRRRInstructionWithCount;
  let TSFlags{7} = is128BitRRRRInstructionWithAddrSpace;

  let Inst{58 - 56} = DataTypeSize;
  let Inst{55 - 48} = opcode;

  dag OutOperandList = outs;
  dag InOperandList = Ins;
  string AsmString = !strconcat(opcodestr, operands);
  let Pattern = pattern;
}

class Next32MOVInstruction
    : Next32Instruction<0x80,
                        (outs GPR32
                         : $dst),
                        (ins i32imm
                         : $imm),
                        "movl", "\t$dst, $imm", [(set GPR32
                                                  : $dst, next32relocimm
                                                  : $imm)]> {
  bits<16> dst;
  bits<32> imm;

  let Inst{47 - 32} = dst;
  let Inst{31 - 0} = imm;
}

// base RRI instruction
class Next32RRIInstruction<bits<8> opcode, string opcodestr,
                           SDPatternOperator OpNode>
    : Next32Instruction<opcode,
                        (outs GPR32
                         : $dst),
                        (ins rri_attribute
                         : $attr, i32imm
                         : $imm),
                        !strconcat(opcodestr, "$attr"), "\t$dst, $imm",
                        [(set i32
                          : $dst, (OpNode i32
                                   : $attr, i32
                                   : $imm))]> {
  bits<2> attr;
  bits<14> imm;
  bits<16> dst;

  let Inst{47 - 32} = 0; // NoCondition
  let Inst{31 - 30} = attr;
  let Inst{29 - 16} = imm;
  let Inst{15 - 0} = dst;
}

// base RRI conditional instruction
class Next32RRICondInstruction<bits<8> opcode, string opcodestr,
                               SDPatternOperator OpNode>
    : Next32Instruction<opcode,
                        (outs GPR32
                         : $dst),
                        (ins rri_attribute
                         : $attr, i32imm
                         : $imm, conditionvalue
                         : $condval, GPR32
                         : $condreg),
                        !strconcat(opcodestr, "$attr$condval"),
                        "\t$dst, $imm [$condreg]", [(set i32
                                                     : $dst, (OpNode i32
                                                              : $attr, i32
                                                              : $imm, i32
                                                              : $condval, i32
                                                              : $condreg))]> {
  bits<4> condval;
  bits<16> condreg;
  bits<2> attr;
  bits<14> imm;
  bits<16> dst;

  let Inst{63 - 60} = condval;
  let Inst{47 - 32} = condreg;
  let Inst{31 - 30} = attr;
  let Inst{29 - 16} = imm;
  let Inst{15 - 0} = dst;
}

multiclass Next32RRIInstruction<bits<8> opcode, string opcodestr,
                                SDPatternOperator OpNode,
                                SDPatternOperator CondOpNode> {
  def NAME : Next32RRIInstruction<opcode, opcodestr, OpNode> {}
  def c : Next32RRICondInstruction<opcode, opcodestr, CondOpNode> {}
}

// base RR instruction
class Next32RRInstruction<bits<8> opcode, dag outs, dag ins, string opcodestr,
                          string operands, list<dag> pattern>
    : Next32Instruction<opcode, outs, ins, opcodestr, operands, pattern> {
  bits<16> src;
  bits<16> dst;

  let Inst{31 - 16} = src;
  let Inst{15 - 0} = dst;
}

// unconditional
class Next32RRRInstruction<bits<8> opcode, dag outs, dag ins, string opcodestr,
                           string operands, list<dag> pattern>
    : Next32RRInstruction<opcode, outs, ins, opcodestr, operands, pattern> {
  bits<16> src2;
  let Inst{47 - 32} = src2;
}

// unconditional
class Next32RRBinaryInstruction<bits<8> opcode, string opcodestr,
                                SDPatternOperator OpNode>
    : Next32RRInstruction<opcode,
                          (outs GPR32
                           : $dst),
                          (ins GPR32
                           : $tempsrc, GPR32
                           : $src),
                          opcodestr, "\t$dst, $src", [(set GPR32
                                                       : $dst, (OpNode i32
                                                                : $tempsrc, i32
                                                                : $src))]> {
  let Inst{47 - 32} = 0; // NoCondition
  let Constraints = "$dst = $tempsrc";
}

class Next32RRSizedBinaryInstruction<bits<8> opcode, string opcodestr,
                                     SDPatternOperator OpNode>
    : Next32RRInstruction<opcode,
                          (outs GPR32
                           : $dst),
                          (ins GPR32
                           : $tempsrc, GPR32
                           : $src, next32sizefield
                           : $size),
                          !strconcat(opcodestr, "$size"), "\t$dst, $src",
                          [(set GPR32
                            : $dst, (OpNode i32
                                     : $tempsrc, i32
                                     : $src, i32
                                     : $size))]> {
  bits<3> size;
  let DataTypeSize = size;
  let Inst{47 - 32} = 0;
  let Constraints = "$dst = $tempsrc";
}

// unconditional + conditional
multiclass Next32RRBinaryInstruction<bits<8> opcode, string opcodestr,
                                     SDPatternOperator OpNode,
                                     SDPatternOperator CondOpNode> {
  def NAME : Next32RRBinaryInstruction<opcode, opcodestr, OpNode> {}
  def c : Next32RRInstruction<opcode,
                              (outs GPR32
                               : $dst),
                              (ins GPR32
                               : $tempsrc, GPR32
                               : $src, conditionvalue
                               : $condval, GPR32
                               : $condreg),
                              !strconcat(opcodestr, "$condval"),
                              "\t$dst, $src [$condreg]",
                              [(set GPR32
                                : $dst, (CondOpNode i32
                                         : $tempsrc, i32
                                         : $src, i32
                                         : $condval, i32
                                         : $condreg))]> {
    bits<4> condval;
    bits<16> condreg;
    let Inst{63 - 60} = condval;
    let Inst{47 - 32} = condreg;
    let Constraints = "$dst = $tempsrc";
  }
}

class Next32RRLoHiInstruction<bits<8> opcode, string opcodestr,
                              SDPatternOperator OpNode>
    : Next32RRInstruction<opcode,
                          (outs GPR32
                           : $dst_lo, GPR32
                           : $dst_hi),
                          (ins GPR32
                           : $src1, GPR32
                           : $src2),
                          opcodestr, "\t$dst_lo, $dst_hi",
                          [(set GPR32
                            : $dst_lo, GPR32
                            : $dst_hi, (OpNode i32
                                        : $src1, i32
                                        : $src2))]> {
  bits<16> dst_lo;
  bits<16> dst_hi;

  let dst = dst_lo;
  let src = dst_hi;
  let Inst{47 - 32} = 0; // NoCondition

  let Constraints = "$dst_lo = $src1, $dst_hi = $src2";
}

// unconditional
class Next32RRUnaryInstruction<bits<8> opcode, string opcodestr,
                               SDPatternOperator OpNode>
    : Next32RRInstruction<opcode,
                          (outs GPR32
                           : $dst),
                          (ins GPR32
                           : $src),
                          opcodestr, "\t$dst", [(set GPR32
                                                 : $dst, (OpNode i32
                                                          : $src))]> {
  let Inst{47 - 32} = 0; // NoCondition
  let Constraints = "$dst = $src";
}

class Next32MemDataInstruction<bits<8> opcode, dag outs, dag ins,
                               string opcodestr, string operands,
                               list<dag> pattern>
    : Next32Instruction<opcode, outs, ins, opcodestr, operands, pattern> {
  bits<16> dst;

  let Inst{15 - 0} = dst;
}

class Next32RRSizedUnaryInstruction<bits<8> opcode, string opcodestr>
    : Next32RRInstruction<opcode,
                          (outs GPR32
                           : $dst),
                          (ins GPR32
                           : $src, next32sizefield
                           : $size),
                          !strconcat(opcodestr, "$size"), "\t$dst", []> {
  bits<3> size;
  let DataTypeSize = size;
  let Inst{47 - 32} = 0; // NoCondition
  let Constraints = "$dst = $src";
}

multiclass Next32Size8RRUnaryInstruction<bits<8> opcode, string opcodestr> {
  let DataTypeSize = 0 in {
    def _8 : Next32RRUnaryInstruction<opcode, !strconcat(opcodestr, ".8"),
                                      null_frag>;
  }
}

multiclass Next32Size16RRUnaryInstruction<bits<8> opcode, string opcodestr> {
  let DataTypeSize = 1 in {
    def _16 : Next32RRUnaryInstruction<opcode, !strconcat(opcodestr, ".16"),
                                       null_frag>;
  }
  defm NAME : Next32Size8RRUnaryInstruction<opcode, opcodestr>;
}

// unconditional + conditional
multiclass Next32RRUnaryInstruction<bits<8> opcode, string opcodestr,
                                    SDPatternOperator OpNode,
                                    SDPatternOperator CondOpNode> {
  def NAME : Next32RRUnaryInstruction<opcode, opcodestr, OpNode> {}
  def c : Next32RRInstruction<opcode,
                              (outs GPR32
                               : $dst),
                              (ins GPR32
                               : $src, conditionvalue
                               : $condval, GPR32
                               : $condreg),
                              !strconcat(opcodestr, "$condval"),
                              "\t$dst, [$condreg]", [(set GPR32
                                                      : $dst, (CondOpNode i32
                                                               : $src, i32
                                                               : $condval, i32
                                                               : $condreg))]> {
    bits<4> condval;
    bits<16> condreg;
    let Inst{63 - 60} = condval;
    let Inst{47 - 32} = condreg;
    let Constraints = "$dst = $src";
  }
}

// For setframe and alloca
class Next32RRRRFrameInstruction<bits<8> opcode, string opcodestr,
                                 SDPatternOperator OpNode>
    : Next32Instruction<opcode,
                        (outs GPR32
                         : $dst1, GPR32
                         : $dst2, GPR32
                         : $src1),
                        (ins GPR32
                         : $src1_in, GPR32
                         : $src2),
                        opcodestr, "\t $dst1, $dst2, $src1, $src2",
                        [(set GPR32
                          : $dst1, GPR32
                          : $dst2, GPR32
                          : $src1, (OpNode i32
                                    : $src1_in, i32
                                    : $src2))]> {
  bits<16> dst1;
  bits<16> dst2;

  let Inst{47 - 32} = 0;
  let Inst{31 - 16} = dst1;
  let Inst{15 - 0} = dst2;

  let is128BitRRRRInstruction = 1;
  let is128BitRRRRInstructionSrcReg1In = 1;
  let is128BitRRRRInstructionSrcReg1Out = 1;
  let is128BitRRRRInstructionSrcReg2In = 1;
  let Constraints = "$src1_in = $src1";
}

class Next32RRRRShiftInstruction<bits<8> opcode, string opcodestr,
                                 SDPatternOperator OpNode>
    : Next32Instruction<opcode,
                        (outs GPR32
                         : $dst_lo_out, GPR32
                         : $dst_hi_out),
                        (ins GPR32
                         : $dst_lo, GPR32
                         : $dst_hi, GPR32
                         : $src1),
                        opcodestr, "\t$dst_lo, $dst_hi, $src1",
                        [(set GPR32
                          : $dst_lo_out, GPR32
                          : $dst_hi_out, (OpNode i32
                                          : $dst_lo, i32
                                          : $dst_hi, i32
                                          : $src1))]> {
  bits<16> dst_lo;
  bits<16> dst_hi;

  let Inst{47 - 32} = 0;
  let Inst{31 - 16} = dst_lo;
  let Inst{15 - 0} = dst_hi;

  let is128BitRRRRInstruction = 1;
  let is128BitRRRRInstructionSrcReg1In = 1;
  let Constraints = "$dst_lo_out = $dst_lo, $dst_hi_out = $dst_hi";
}

// base instruction
class Next32RRRRTernaryInstruction<bits<8> opcode, string opcodestr,
                                   SDPatternOperator OpNode>
    : Next32Instruction<opcode,
                        (outs GPR32
                         : $dst1_out),
                        (ins GPR32
                         : $dst1, GPR32
                         : $dst2, GPR32
                         : $src1),
                        opcodestr, "\t $dst1, $dst2, $src1",
                        [(set GPR32
                          : $dst1_out, (OpNode i32
                                        : $dst1, i32
                                        : $dst2, i32
                                        : $src1))]> {
  bits<16> dst1;
  bits<16> dst2;

  let Inst{47 - 32} = 0;
  let Inst{31 - 16} = dst1;
  let Inst{15 - 0} = dst2;

  let is128BitRRRRInstruction = 1;
  let is128BitRRRRInstructionSrcReg1In = 1;
  let Constraints = "$dst1_out = $dst1";
}

class Next32VMemInstruction<bits<8> opcode, string opcodestr>
    : Next32Instruction<opcode, (outs),
                        (ins next32alignsize
                         : $align, next32sizefield
                         : $type, GPR32
                         : $dst1, GPR32
                         : $dst2, GPR32
                         : $src1, GPR32
                         : $src2, next32vecelemcountfield
                         : $count, next32addrspacefield
                         : $addrspace),
                        !strconcat(opcodestr, "$count", "$type", "$align",
                                   "$addrspace"),
                        "\t$dst1, $dst2, $src1, $src2", []> {
  bits<16> dst1;
  bits<16> dst2;
  bits<3> align;
  bits<3> type;

  let DataTypeSize = type;
  let Inst{63 - 61} = align;
  let Inst{47 - 32} = 0;
  let Inst{31 - 16} = dst1;
  let Inst{15 - 0} = dst2;

  let is128BitRRRRInstruction = 1;
  let is128BitRRRRInstructionSrcReg1In = 1;
  let is128BitRRRRInstructionSrcReg2In = 1;
  let is128BitRRRRInstructionWithCount = 1;
  let is128BitRRRRInstructionWithAddrSpace = 1;
}

// base instruction
class Next32MemInstruction<bits<8> opcode, string opcodestr>
    : Next32Instruction<opcode, (outs),
                        (ins next32alignsize
                         : $align, next32sizefield
                         : $size, GPR32
                         : $dst1, GPR32
                         : $dst2, GPR32
                         : $src1, next32vecelemcountfield
                         : $count, next32addrspacefield
                         : $addrspace),
                        !strconcat(opcodestr, "$count", "$size", "$align",
                                   "$addrspace"),
                        "\t$dst1, $dst2, $src1", []> {
  bits<16> dst1;
  bits<16> dst2;
  bits<3> align;
  bits<3> size;

  let DataTypeSize = size;
  let Inst{63 - 61} = align;
  let Inst{47 - 32} = 0;
  let Inst{31 - 16} = dst1;
  let Inst{15 - 0} = dst2;

  let is128BitRRRRInstruction = 1;
  let is128BitRRRRInstructionSrcReg1In = 1;
  let is128BitRRRRInstructionWithCount = 1;
  let is128BitRRRRInstructionWithAddrSpace = 1;
}
