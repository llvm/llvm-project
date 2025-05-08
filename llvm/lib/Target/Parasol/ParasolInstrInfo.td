//== ParasolInstrInfo.td - Target Description for Parasol Target -*- tablegen -*-=//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the Parasol implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

// Procedure return
def parasol_ret : SDNode<"ParasolISD::Ret", SDTNone,
                       [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

// Procedure calling
def SDT_CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>,
                                       SDTCisVT<1, i32>]>;
def SDT_CallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                    SDTCisVT<1, i32>]>;

// These are target-independent nodes, but have target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_CallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_CallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

include "ParasolInstrFormats.td"

let Defs = [X2], Uses = [X2] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(callseq_start timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(callseq_end timm:$amt1, timm:$amt2)]>;
} // end Defs = [X2], Uses = [X2]

//===----------------------------------------------------------------------===//
// Move Instructions
//===----------------------------------------------------------------------===//

def MOVrr : ParasolInst<(outs IR:$dst), (ins IR:$src), "mov", "$dst, $src", []> {
    let hasSideEffects = 0;
    let mayLoad = 0;
    let mayStore = 0;
    let Opcode = OPC_MOV.Value;

    bits<6> dst;
    bits<6> src;

    let Inst{19-14} = src;
    let Inst{13-8} = dst;
}


//===----------------------------------------------------------------------===//
// Load Immediate Instructions
//===----------------------------------------------------------------------===//

def LoadImmediate1 : ParasolLoadImm<Parasolimm1, i1>;
def LoadImmediate8 : ParasolLoadImm<Parasolimm8, i8>;
def LoadImmediate16 : ParasolLoadImm<Parasolimm16, i16>;
def LoadImmediate32 : ParasolLoadImm<Parasolimm32, i32>;


//===----------------------------------------------------------------------===//
// Load/Store Instructions
//===----------------------------------------------------------------------===//

def LOAD : ParasolInst<(outs IR:$val), (ins PR:$addr, Parasolimm32:$size), "ldr", "$val, $addr, $size", []> {
    bits<3> size;
    bits<6> val;
    bits<6> addr;

    let hasSideEffects = 0;
    let mayLoad = 1;
    let mayStore = 0;

    // 2 ^ size bits
    let Inst{22-20} = size;
    let Inst{19-14} = addr;
    let Inst{13-8} = val;
    let Opcode = OPC_LOAD.Value;
}

def : Pat<(i8 (load PR:$addr)), (LOAD $addr, 3)>;
def : Pat<(i16 (load PR:$addr)), (LOAD $addr, 4)>;
def : Pat<(i32 (load PR:$addr)), (LOAD $addr, 5)>;

def STORE : ParasolInst<(outs), (ins IR:$val, PR:$addr, Parasolimm32:$size), "str", "$addr, $val, $size", []> {
    bits<3> size;
    bits<6> val;
    bits<6> addr;

    let hasSideEffects = 0;
    let mayLoad = 0;
    let mayStore = 1;

    // 2 ^ size bits
    let Inst{22-20} = size;
    let Inst{19-14} = val;
    let Inst{13-8} = addr;
    let Opcode = OPC_STORE.Value;
}

def : Pat<(store i8:$val , PR:$addr), (STORE $val, $addr, 3)>;
def : Pat<(store i16:$val, PR:$addr), (STORE $val, $addr, 4)>;
def : Pat<(store i32:$val, PR:$addr), (STORE $val, $addr, 5)>;


//===----------------------------------------------------------------------===//
// Truncation and zero extension
//===----------------------------------------------------------------------===//

class ParasolTruncInst<ValueType to, ValueType from>
    : ParasolInst<(outs IR:$output), (ins IR:$input), "trunc", "$output, $input, " # log2ValueType<to>.value # ", " # log2ValueType<from>.value,
                   [(set to:$output, (trunc from:$input))]> {
    bits<6> output;
    bits<6> input;

    let hasSideEffects = 0;
    let mayLoad = 0;
    let mayStore = 0;

    let Inst{25-23} = log2ValueType<from>.value;
    let Inst{22-20} = log2ValueType<to>.value;
    let Inst{19-14} = input;
    let Inst{13-8} = output;
    let Opcode = OPC_TRUNC.Value;
}

def TRUNC_1_8 : ParasolTruncInst<i1, i8>;
def TRUNC_1_16 : ParasolTruncInst<i1, i16>;
def TRUNC_1_32 : ParasolTruncInst<i1, i32>;
def TRUNC_8_16 : ParasolTruncInst<i8, i16>;
def TRUNC_8_32 : ParasolTruncInst<i8, i32>;
def TRUNC_16_32 : ParasolTruncInst<i16, i32>;

class ParasolZExtInst<ValueType to, ValueType from>
    : ParasolInst<(outs IR:$output), (ins IR:$input), "zext", "$output, $input, " # log2ValueType<to>.value # ", " # log2ValueType<from>.value,
                   [(set to:$output, (zext from:$input))]> {
    bits<6> output;
    bits<6> input;

    let hasSideEffects = 0;
    let mayLoad = 0;
    let mayStore = 0;

    let Inst{25-23} = log2ValueType<from>.value;
    let Inst{22-20} = log2ValueType<to>.value;
    let Inst{19-14} = input;
    let Inst{13-8} = output;
    let Opcode = OPC_ZEXT.Value;
}

def ZEXT_8_1 : ParasolZExtInst<i8, i1>;
def ZEXT_16_1 : ParasolZExtInst<i16, i1>;
def ZEXT_32_1 : ParasolZExtInst<i32, i1>;
def ZEXT_16_8 : ParasolZExtInst<i16, i8>;
def ZEXT_32_8 : ParasolZExtInst<i32, i8>;
def ZEXT_32_16 : ParasolZExtInst<i32, i16>;

// We don't have signed types at the moment so we replace them all with zero
// extends.
def : Pat<(i8 (sext_inreg IR:$input, i1)), (ZEXT_8_1 $input)>;
def : Pat<(i16 (sext_inreg IR:$input, i1)), (ZEXT_16_1 $input)>;
def : Pat<(i16 (sext_inreg IR:$input, i8)), (ZEXT_16_8 $input)>;
def : Pat<(i32 (sext_inreg IR:$input, i1)), (ZEXT_32_1 $input)>;
def : Pat<(i32 (sext_inreg IR:$input, i8)), (ZEXT_32_8 $input)>;
def : Pat<(i32 (sext_inreg IR:$input, i16)), (ZEXT_32_16 $input)>;

def : Pat<(i8 (sext i1:$input)), (ZEXT_8_1 $input)>;
def : Pat<(i16 (sext i1:$input)), (ZEXT_16_1 $input)>;
def : Pat<(i16 (sext i8:$input)), (ZEXT_16_8 $input)>;
def : Pat<(i32 (sext i1:$input)), (ZEXT_32_1 $input)>;
def : Pat<(i32 (sext i8:$input)), (ZEXT_32_8 $input)>;
def : Pat<(i32 (sext i16:$input)), (ZEXT_32_16 $input)>;

def : Pat<(i8 (anyext i1:$input)), (ZEXT_8_1 $input)>;
def : Pat<(i16 (anyext i1:$input)), (ZEXT_16_1 $input)>;
def : Pat<(i16 (anyext i8:$input)), (ZEXT_16_8 $input)>;
def : Pat<(i32 (anyext i1:$input)), (ZEXT_32_1 $input)>;
def : Pat<(i32 (anyext i8:$input)), (ZEXT_32_8 $input)>;
def : Pat<(i32 (anyext i16:$input)), (ZEXT_32_16 $input)>;


//===----------------------------------------------------------------------===//
// Arithmetic and logic instructions
//===----------------------------------------------------------------------===//

// Could add isAsCheapAsAMove = 1 to these instructions.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

    def BindReadPlain
        : ParasolBindInst<OPC_BINDREADONLY, "bpr", 0, bindreadplain>;
    def BindReadWritePlain
        : ParasolBindInst<OPC_BINDREADWRITE, "bprw", 0, bindreadwriteplain>;
    def BindReadEncrypted
        : ParasolBindInst<OPC_BINDREADONLY, "ber", 1, bindreadencrypted>;
    def BindReadWriteEncrypted
        : ParasolBindInst<OPC_BINDREADWRITE, "berw", 1, bindreadwriteencrypted>;

    def ADDrr   : ParasolBinaryALUInst<OPC_ADD, "add">,
                  Sched<[WriteIALU, ReadIALU]>;

    // SUNSCREEN TODO:
    // ISDNodes indicates that the e variants of add/sub take in three
    // arguments, but their definition in TargetSelectionDAG.td specifies that
    // they only take in two arguments. We don't know how to rectify this at the
    // moment.
    //
    // def ADDCrr  : ParasolTernaryInst<OPC_ADDC, "addc">,
    //               Sched<[WriteIALU, ReadIALU]>;

    def SUBrr   : ParasolBinaryALUInst<OPC_SUB, "sub">,
                  Sched<[WriteIALU, ReadIALU]>;

    // def SUBBrr  : ParasolTernaryInst<OPC_SUBB, "subb">,
    //               Sched<[WriteIALU, ReadIALU]>;

    def MULrr   : ParasolBinaryALUInst<OPC_MUL, "mul">,
                  Sched<[WriteIALU, ReadIALU]>;

    def SHLrr   : ParasolBinaryALUInst<OPC_SHL, "shl">,
                  Sched<[WriteIALU, ReadIALU]>;

    def ROTLrr  : ParasolBinaryALUInst<OPC_ROTL, "rotl">,
                   Sched<[WriteIALU, ReadIALU]>;

    def SHRrr   : ParasolBinaryALUInst<OPC_SHR, "shr">,
                  Sched<[WriteIALU, ReadIALU]>;

    def ROTRrr  : ParasolBinaryALUInst<OPC_ROTR, "rotr">,
                   Sched<[WriteIALU, ReadIALU]>;

    def ANDrr   : ParasolBinaryALUInst<OPC_AND, "and">,
                  Sched<[WriteIALU, ReadIALU]>;

    def ORrr    : ParasolBinaryALUInst<OPC_OR, "or">,
                  Sched<[WriteIALU, ReadIALU]>;

    def XORrr   : ParasolBinaryALUInst<OPC_XOR, "xor">,
                  Sched<[WriteIALU, ReadIALU]>;

    def NOTr    : ParasolUnaryALUInst<OPC_NOT, "not">,
                  Sched<[WriteIALU, ReadIALU]>;

    // SUSCREEN TODO: This is not defined in the selection DAG, so we need to
    // figure out how to handle this.
    def NEGr    : ParasolUnaryALUInst<OPC_NEG, "neg">,
                  Sched<[WriteIALU, ReadIALU]>;

    // Comparisons are listed here:
    // https://llvm.org/doxygen/InstrTypes_8h_source.html
    let isCompare = 1 in {

        def GTrr    : ParasolBinaryALUInst<OPC_GT, "gt">,
                    Sched<[WriteIALU, ReadIALU]>;

        def GErr    : ParasolBinaryALUInst<OPC_GE, "ge">,
                    Sched<[WriteIALU, ReadIALU]>;

        def LTrr    : ParasolBinaryALUInst<OPC_LT, "lt">,
                    Sched<[WriteIALU, ReadIALU]>;

        def LErr    : ParasolBinaryALUInst<OPC_LE, "le">,
                    Sched<[WriteIALU, ReadIALU]>;

        def EQrr    : ParasolBinaryALUInst<OPC_EQ, "eq">,
                    Sched<[WriteIALU, ReadIALU]>;

    }

    def CMUXrrr : ParasolTernaryInst<OPC_CMUX, "cmux">,
                Sched<[WriteIALU, ReadIALU]> {
        let isSelect = 1;
    }

}

// Tell globalisel the equivalent instruction is for these sd nodes
def : GINodeEquiv<BindReadPlain, bindreadplain>;
def : GINodeEquiv<BindReadWritePlain, bindreadwriteplain>;
def : GINodeEquiv<BindReadEncrypted, bindreadencrypted>;
def : GINodeEquiv<BindReadWriteEncrypted, bindreadwriteencrypted>;

// Patterns for binary operations
class Pat2Ir<SDPatternOperator OpNode, ParasolInst Inst>
    : Pat<(OpNode IR:$rs1, IR:$rs2), (Inst IR:$rs1, IR:$rs2)>;

multiclass Pat1IrMultiValueType<SDPatternOperator OpNode, ParasolInst inst> {
    def _1 : Pat<(OpNode i1:$rs1), (inst i1:$rs1)>;
    def _8 : Pat<(OpNode i8:$rs1), (inst i8:$rs1)>;
    def _16 : Pat<(OpNode i16:$rs1), (inst i16:$rs1)>;
    def _32 : Pat<(OpNode i32:$rs1), (inst i32:$rs1)>;
}

multiclass Pat2IrMultiValueType<SDPatternOperator OpNode, ParasolInst inst> {
    def _1 : Pat<(OpNode i1:$rs1, i1:$rs2), (inst i1:$rs1, i1:$rs2)>;
    def _8 : Pat<(OpNode i8:$rs1, i8:$rs2), (inst i8:$rs1, i8:$rs2)>;
    def _16 : Pat<(OpNode i16:$rs1, i16:$rs2), (inst i16:$rs1, i16:$rs2)>;
    def _32 : Pat<(OpNode i32:$rs1, i32:$rs2), (inst i32:$rs1, i32:$rs2)>;
}

multiclass PatIrI32MultiValueType<SDPatternOperator OpNode, ParasolInst inst> {
    def _1 : Pat<(OpNode i1:$rs1, i32:$rs2), (inst i1:$rs1, i32:$rs2)>;
    def _8 : Pat<(OpNode i8:$rs1, i32:$rs2), (inst i8:$rs1, i32:$rs2)>;
    def _16 : Pat<(OpNode i16:$rs1, i32:$rs2), (inst i16:$rs1, i32:$rs2)>;
    def _32 : Pat<(OpNode i32:$rs1, i32:$rs2), (inst i32:$rs1, i32:$rs2)>;
}

multiclass Pat3IrMultiValueType<SDPatternOperator OpNode, ParasolInst inst> {
    def _1 : Pat<(OpNode i1:$rs1, i1:$rs2, i1:$rs3), (inst i1:$rs1, i1:$rs2, i1:$rs3)>;
    def _8 : Pat<(OpNode i8:$rs1, i8:$rs2, i8:$rs3), (inst i8:$rs1, i8:$rs2, i8:$rs3)>;
    def _16 : Pat<(OpNode i16:$rs1, i16:$rs2, i16:$rs3), (inst i16:$rs1, i16:$rs2, i16:$rs3)>;
    def _32 : Pat<(OpNode i32:$rs1, i32:$rs2, i32:$rs3), (inst i32:$rs1, i32:$rs2, i32:$rs3)>;
}

defm : Pat1IrMultiValueType<not, NOTr>;
// Negation does not exist directly as a selection dag node, hence it is not
// here. Negation on a floating point number does exist.

defm : Pat2IrMultiValueType<add, ADDrr>;
// defm : Pat3IrMultiValueType<adde, ADDCrr>;
defm : Pat2IrMultiValueType<sub, SUBrr>;
// defm : Pat3IrMultiValueType<sube, SUBrr>;
defm : Pat2IrMultiValueType<mul, MULrr>;

defm : Pat2IrMultiValueType<and, ANDrr>;
defm : Pat2IrMultiValueType<or, ORrr>;
defm : Pat2IrMultiValueType<xor, XORrr>;

defm : Pat2IrMultiValueType<shl, SHLrr>;
defm : Pat2IrMultiValueType<rotl, ROTLrr>;
defm : Pat2IrMultiValueType<srl, SHRrr>;
defm : Pat2IrMultiValueType<rotr, ROTRrr>;

defm : PatIrI32MultiValueType<shl, SHLrr>;
defm : PatIrI32MultiValueType<rotl, ROTLrr>;
defm : PatIrI32MultiValueType<srl, SHRrr>;
defm : PatIrI32MultiValueType<rotr, ROTRrr>;

// Patterns for comparison operations
multiclass PatComparisonMultiValueType<CondCode condcode, dag pattern> {
    def _1 : Pat<(i1 (setcc i1:$rs1, i1:$rs2, condcode)), pattern>;
    def _8 : Pat<(i1 (setcc i8:$rs1, i8:$rs2, condcode)), pattern>;
    def _16 : Pat<(i1 (setcc i16:$rs1, i16:$rs2, condcode)), pattern>;
    def _32 : Pat<(i1 (setcc i32:$rs1, i32:$rs2, condcode)), pattern>;
}

defm : PatComparisonMultiValueType<SETUGT, (GTrr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETUGE, (GErr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETULT, (LTrr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETULE, (LErr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETUEQ, (EQrr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETUNE, (NOTr (EQrr $rs1, $rs2))>;

defm : PatComparisonMultiValueType<SETGT, (GTrr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETGE, (GErr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETLT, (LTrr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETLE, (LErr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETEQ, (EQrr $rs1, $rs2)>;
defm : PatComparisonMultiValueType<SETNE, (NOTr (EQrr $rs1, $rs2))>;

/* Procedure calling instructions */
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
let isCall = 1 in
def JALR : ParasolInstI<OPC_JALR, (outs GPR:$rd),
                   (ins GPR:$rs1, i32imm:$imm32),
                   "jalr", "$rd, ${imm32}(${rs1})">,
           Sched<[WriteJalr, ReadJalr]>;
}

let isReturn = 1, isBarrier = 1, isTerminator = 1 in
def PseudoRET : Pseudo<(outs), (ins), [(parasol_ret)]>,
                PseudoInstExpansion<(JALR X0, X1, 0)>;

def : InstAlias<"ret",                   (JALR      X0,      X1, 0), 8>;

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

// If we find the pattern of srl by n and then and with 1, we can replace it
// with a bit select.

// GLOBALISEL NOT IMPORTED
// warning: Skipped pattern: Dst pattern child is an unsupported kind
def : Pat<(i8 (and (srl IR:$rs1, Parasolimm32:$n), 1)),
          (ZEXT_8_1 (TRUNC_1_8 (SHRrr $rs1, $n)))>;

def : Pat<(i16 (and (srl IR:$rs1, Parasolimm32:$n), 1)),
          (ZEXT_16_1 (TRUNC_1_16 (SHRrr $rs1, $n)))>;

def : Pat<(i32 (and (srl IR:$rs1, Parasolimm32:$n), 1)),
          (ZEXT_32_1 (TRUNC_1_32 (SHRrr $rs1, $n)))>;

// If we and with 1 we just take the least significant bit.
def : Pat<(i8 (and i8:$rs1, 1)),
          (ZEXT_8_1 (TRUNC_1_8 $rs1))>;

def : Pat<(i16 (and i16:$rs1, 1)),
          (ZEXT_16_1 (TRUNC_1_16 $rs1))>;

def : Pat<(i32 (and i32:$rs1, 1)),
          (ZEXT_32_1 (TRUNC_1_32 $rs1))>;

// We have a specific pattern when a truncation is performed, which is bit selecting zero.

def : Pat<(i1 (trunc (srl i8:$rs1, Parasolimm32:$n))),
          (TRUNC_1_8 (SHRrr $rs1, $n))>;

def : Pat<(i1 (trunc (srl i16:$rs1, Parasolimm32:$n))),
            (TRUNC_1_16 (SHRrr $rs1, $n))>;

def : Pat<(i1 (trunc (srl i32:$rs1, Parasolimm32:$n))),
            (TRUNC_1_32 (SHRrr $rs1, $n))>;

// We shift by a immediate value, and with 1, and then check for equality with a
// boolean is just a bit select.

// SUNSCREEN TODO: Look into whether the optimizer ever generates a pattern with
// not equal instead of equal, in which case we should generate the patterns for
// that as well.

multiclass PatSetCCShiftAnd1Eq<CondCode condcode> {
    // These first set of cases are if a shift of zero is performed, which is
    // already optimized away by the time we get to the selection DAG.
    // GLOBALISEL NOT IMPORTED
    // warning: Skipped pattern: unsupported type for Src operand (SETUEQ:{ *:[Other] })
    def : Pat<(i1 (setcc (and i8:$rs1, 1), 0, condcode)),
              (NOTr (TRUNC_1_8 $rs1))>;

    def : Pat<(i1 (setcc (and i16:$rs1, 1), 0, condcode)),
              (NOTr (TRUNC_1_16 $rs1))>;

    def : Pat<(i1 (setcc (and i32:$rs1, 1), 0, condcode)),
              (NOTr (TRUNC_1_32 $rs1))>;

    def : Pat<(i1 (setcc 0, (and i8:$rs1, 1), condcode)),
              (NOTr (TRUNC_1_8 $rs1))>;

    def : Pat<(i1 (setcc 0, (and i16:$rs1, 1), condcode)),
              (NOTr (TRUNC_1_16 $rs1))>;

    def : Pat<(i1 (setcc 0, (and i32:$rs1, 1), condcode)),
              (NOTr (TRUNC_1_32 $rs1))>;

    def : Pat<(i1 (setcc (and i8:$rs1, 1), 1, condcode)),
              (TRUNC_1_8 $rs1)>;

    def : Pat<(i1 (setcc (and i16:$rs1, 1), 1, condcode)),
              (TRUNC_1_16 $rs1)>;

    def : Pat<(i1 (setcc (and i32:$rs1, 1), 1, condcode)),
              (TRUNC_1_32 $rs1)>;

    def : Pat<(i1 (setcc 1, (and i8:$rs1, 1), condcode)),
              (TRUNC_1_8 $rs1)>;

    def : Pat<(i1 (setcc 1, (and i16:$rs1, 1), condcode)),
              (TRUNC_1_16 $rs1)>;

    def : Pat<(i1 (setcc 1, (and i32:$rs1, 1), condcode)),
              (TRUNC_1_32 $rs1)>;


    // Patterns with shifts
    def : Pat<(i1 (setcc (and (srl i8:$rs1, Parasolimm32:$n), 1), 0, condcode)),
              (NOTr (TRUNC_1_8 (SHRrr $rs1, $n)))>;

    def : Pat<(i1 (setcc (and (srl i16:$rs1, Parasolimm32:$n), 1), 0, condcode)),
              (NOTr (TRUNC_1_16 (SHRrr $rs1, $n)))>;

    def : Pat<(i1 (setcc (and (srl i32:$rs1, Parasolimm32:$n), 1), 0, condcode)),
              (NOTr (TRUNC_1_32 (SHRrr $rs1, $n)))>;

    def : Pat<(i1 (setcc 0, (and (srl i8:$rs1, Parasolimm32:$n), 1), condcode)),
              (NOTr (TRUNC_1_8 (SHRrr $rs1, $n)))>;

    def : Pat<(i1 (setcc 0, (and (srl i16:$rs1, Parasolimm32:$n), 1), condcode)),
              (NOTr (TRUNC_1_16 (SHRrr $rs1, $n)))>;

    def : Pat<(i1 (setcc 0, (and (srl i32:$rs1, Parasolimm32:$n), 1), condcode)),
              (NOTr (TRUNC_1_32 (SHRrr $rs1, $n)))>;

    def : Pat<(i1 (setcc (and (srl i8:$rs1, Parasolimm32:$n), 1), 1, condcode)),
              (TRUNC_1_8 (SHRrr $rs1, $n))>;

    def : Pat<(i1 (setcc (and (srl i16:$rs1, Parasolimm32:$n), 1), 1, condcode)),
              (TRUNC_1_16 (SHRrr $rs1, $n))>;

    def : Pat<(i1 (setcc (and (srl i32:$rs1, Parasolimm32:$n), 1), 1, condcode)),
              (TRUNC_1_32 (SHRrr $rs1, $n))>;

    def : Pat<(i1 (setcc 1, (and (srl i8:$rs1, Parasolimm32:$n), 1), condcode)),
              (TRUNC_1_8 (SHRrr $rs1, $n))>;

    def : Pat<(i1 (setcc 1, (and (srl i16:$rs1, Parasolimm32:$n), 1), condcode)),
              (TRUNC_1_16 (SHRrr $rs1, $n))>;

    def : Pat<(i1 (setcc 1, (and (srl i32:$rs1, Parasolimm32:$n), 1), condcode)),
              (TRUNC_1_32 (SHRrr $rs1, $n))>;
}

defm : PatSetCCShiftAnd1Eq<SETEQ>;
defm : PatSetCCShiftAnd1Eq<SETUEQ>;
defm : PatSetCCShiftAnd1Eq<SETOEQ>;


//===----------------------------------------------------------------------===//
// Select patterns
//===----------------------------------------------------------------------===//

multiclass PatSelect<ValueType comparisonvt> {
    def : Pat<(comparisonvt (select i1:$bool, IR:$trueval, IR:$falseval)),
              (CMUXrrr $bool, $trueval, $falseval)>;

    // Convert when you have PR input to load the value first
    def : Pat<(comparisonvt (load (select i1:$bool, PR:$trueval, PR:$falseval))),
              (CMUXrrr $bool, (LOAD $trueval, log2ValueType<comparisonvt>.value), (LOAD $falseval, log2ValueType<comparisonvt>.value))>;
}

defm : PatSelect<i1>;
defm : PatSelect<i8>;
defm : PatSelect<i16>;
defm : PatSelect<i32>;
