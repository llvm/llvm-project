//===-- SuperHInstrInfo.td - Main SuperH Instruction Definition -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the SuperH instruction set, defining the instructions
/// and properties of the instructions which are needed for code generation,
/// machine code emission, and analysis.
///
/// The subsystems are ordered after the instruction classification that
/// The SuperH ISA manuals use.
///
/// See: https://www.shared-ptr.com/sh_insns.html
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "SuperHInstrFormats.td"

//===----------------------------------------------------------------------===//
// Instruction Patterns
//===----------------------------------------------------------------------===//

defvar SH = DefaultMode;
def sh_ptr_rc : RegClassByHwMode<[SH], [GPR]>;

// Both cases can use the same decoder method, so avoid the dispatch
// by hwmode by setting an explicit DecoderMethod
def ptr_op : RegisterOperand<sh_ptr_rc> {
  let DecoderMethod = "DecodeIntRegsRegisterClass";
}


//===----------------------------------------------------------------------===//
// SuperH Type Profiles
//===----------------------------------------------------------------------===//

def SHSDT_CallSeqStart  : SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SHSDT_CallSeqEnd  : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SHSDT_Call        : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

//===----------------------------------------------------------------------===//
// SuperH Specific Nodes
//===----------------------------------------------------------------------===//

def SHCallSeqStart  : SDNode<"ISD::CALLSEQ_START", SHSDT_CallSeqStart,
                            [SDNPHasChain, SDNPOutGlue]>;

def SHCallSeqEnd    : SDNode<"ISD::CALLSEQ_END", SHSDT_CallSeqEnd,
                            [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def SHCall          : SDNode<"SHISD::CALL", SHSDT_Call,
                            [SDNPHasChain, SDNPOutGlue,
                             SDNPOptInGlue, SDNPVariadic]>;

def SHRet           : SDNode<"SHISD::RET", SDTNone,
                            [SDNPHasChain, SDNPOptInGlue, 
                             SDNPVariadic]>;

//===----------------------------------------------------------------------===//
// Operand Classes
//===----------------------------------------------------------------------===//

// Base Operand Class
class SHOpClass<string name, 
                list<AsmOperandClass> superClasses = []> : AsmOperandClass {
  let Name = name;
  let ParserMethod = "parseOperand";
  let SuperClasses = superClasses;
}

// Immediates
class SHImmClass<int width> : SHOpClass<"Imm" # width> {
    let PredicateMethod = "isImm";
    let RenderMethod = "addImmOperands";
}

// Displacements
class SHDispClass<int width> : SHOpClass<"Disp" # width> {
    let PredicateMethod = "isDisp";
    let RenderMethod = "addDispOperands";
}

// Memory Ops
def SHMemClass : SHOpClass<"Mem"> {
  let PredicateMethod = "isMem";
  let RenderMethod = "addMemOperands";
}

//===----------------------------------------------------------------------===//
// Operands
//===----------------------------------------------------------------------===//

// Immediate
class SHImmOp<int width, ValueType vt>
      : Operand<vt> {
    let ParserMatchClass = SHImmClass<width>;
    let OperandType = "OPERAND_IMMEDIATE";
    let MCOperandPredicate = [{int64_t Imm; return MCOp.evaluateAsConstantImm(Imm) ? isInt<}] # width # [{>(Imm) : false;}];
}
def imm8 : SHImmOp<8, i32>, ImmLeaf<i32, [{ return isInt<8>(Imm); }]>;

// Displacement
class SHDispOp<int width, ValueType vt>
      : Operand<vt>, PatLeaf<(imm), [{ return isUInt<}] # width # [{>(N->getZExtValue()); }]> {
    let ParserMatchClass = SHDispClass<width>;
    let OperandType = "OPERAND_IMMEDIATE";
    let MCOperandPredicate = [{
        int64_t Imm;
        if (MCOp.evaluateAsConstantImm(Imm))
            return isUInt<#width>(Imm);
        return MCOp.isBareSymbolRef();
    }];
}
def disp4   : SHDispOp<4, i4>;
def disp8   : SHDispOp<8, i8>;
def disp12  : SHDispOp<12, i16>;



// Memory
class SHMemRegOp<RegisterClass regClass> 
      : RegisterOperand<regClass> {
    let ParserMatchClass = SHMemClass;
    let OperandType = "OPERAND_MEMORY";
}
def GPRMem : SHMemRegOp<GPR>;

// Memory Indirect (Register + Register)
def MemRRI : Operand<iPTR> {
  let ParserMatchClass = SHMemClass;
  let OperandType = "OPERAND_MEMORY";
  let MIOperandInfo = (ops GPR, GPR);
}

// Memory Indirect (Register + Disp)
def MemRD : Operand<iPTR> {
  let ParserMatchClass = SHMemClass;
  let OperandType = "OPERAND_MEMORY";
  let MIOperandInfo = (ops GPR, disp8);
}

// Memory Indirect (GBR + disp)
def MemGBRI : Operand<iPTR> {
  let ParserMatchClass = SHMemClass;
  let OperandType = "OPERAND_MEMORY";
  let MIOperandInfo = (ops R_GBR, disp8);
}

// Memory Indirect (R0 + disp)
def MemR0I : Operand<iPTR> {
  let ParserMatchClass = SHMemClass;
  let OperandType = "OPERAND_MEMORY";
  let MIOperandInfo = (ops R_R0, disp8);
}

// Addressing mode pattern reg+disp
let WantsRoot = true in
def addr : ComplexPattern<iPTR, 2, "SelectAddr">;





//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Instruction Definitions
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//




//===----------------------------------------------------------------------===//
// Data Transfer Instructions
//===----------------------------------------------------------------------===//

// mov Rm, Rn
let hasSideEffects = 0 in
def MOVRmRn       : InstRmRn<0b0110000000000011,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "mov $Rm,$Rn",
                            []>;

// mov #imm:8, Rn
let hasSideEffects = 0 in 
def MOVI8Rn       : InstI8Rn<0b1110000000000000,
                            (outs GPR:$Rn), (ins imm8:$imm),
                            "mov #$imm,$Rn",
                            [(set i32:$Rn, imm8:$imm)]>;


// mova @(disp,PC), R0


//===----------------------------------------------------------------------===//
// Load Instructions
//===----------------------------------------------------------------------===//

// Indirect Memory -> Register Load
let hasSideEffects = 0, mayLoad = 1, isReMaterializable = 1 in {
 def MOVBRmiRn          : InstRmRn<0b0110000000000000,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.b @$Rm,$Rn",
                                  [(set i32:$Rn, (sextloadi8 i32:$Rm))]>;

 def MOVWRmiRn          : InstRmRn<0b0110000000000001,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.w @$Rm,$Rn",
                                  [(set i32:$Rn, (sextloadi16 i32:$Rm))]>;

 def MOVLRmiRn          : InstRmRn<0b0110000000000010,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.l @$Rm,$Rn",
                                  [(set i32:$Rn, (load i32:$Rm))]>;
}
 
// Indirect Memory w/ Displacement -> Register Load
let hasSideEffects = 0, mayLoad = 1, isReMaterializable = 1 in {
  def MOVLD4RmiRn    : InstRmD4Rn<0b0101000000000000,
                                  (outs GPRMem:$Rn), (ins GPR:$Rm, disp4:$disp),
                                  "mov.l @($disp, $Rm),$Rn",
                                  []>;

                                  // "mov.l", [(set i32:$Rn, (load addr:$disp))]>;
}


//===----------------------------------------------------------------------===//
// Store Instructions
//===----------------------------------------------------------------------===//

// Register -> Indirect Memory Store
let hasSideEffects = 0, mayStore = 1 in {
 def MOVBRmRni          : InstRmRn<0b0010000000000000,
                                  (outs), (ins GPRMem:$Rn, GPR:$Rm),
                                  "mov.b $Rm,@$Rn",
                                  [(truncstorei8 GPRMem:$Rn, i32:$Rm)]>;

 def MOVWRmRni          : InstRmRn<0b0010000000000001,
                                  (outs), (ins GPRMem:$Rn, GPR:$Rm),
                                  "mov.w $Rm,@$Rn",
                                  [(truncstorei16 GPRMem:$Rn, i32:$Rm)]>;

 def MOVLRmRni          : InstRmRn<0b0010000000000010,
                                  (outs), (ins GPRMem:$Rn, GPR:$Rm),
                                  "mov.l $Rm,@$Rn",
                                  [(store GPRMem:$Rn, i32:$Rm)]>;
}

//===----------------------------------------------------------------------===//
// Arithmetic Instructions
//===----------------------------------------------------------------------===//

//
//      ADDITION
//
let hasSideEffects = 0, isCommutable = 1, Constraints = "$src = $Rn" in {
  
  // add Rm, Rn
  def ADDRmRn       : InstRmRn<0b0011000000001100, 
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "add $Rm,$Rn",
                              [(set i32:$Rn, (add i32:$src, i32:$Rm))]>;
  
  // add imm:8, Rn
  def ADDI8Rn       : InstI8Rn<0b0011000000001100, 
                              (outs GPR:$Rn), (ins GPR:$src, imm8:$imm),
                              "add #$imm,$Rn",
                              [(set i32:$Rn, (add i32:$src, imm8:$imm))]>;

  // addc Rm, Rn
  let Defs = [SR], Uses = [SR] in
  def ADDCRmRn      : InstRmRn<0b0011000000001110,
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "addc $Rm,$Rn",
                              [(set i32:$Rn, (adde i32:$src, i32:$Rm))]>;
  
  // addv Rm, Rn
  let Defs = [SR] in
  def ADDVRmRn      : InstRmRn<0b0011000000001111,
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "addv $Rm,$Rn",
                              [(set i32:$Rn, (addc i32:$src, i32:$Rm))]>;

}

//
//      SUBTRACTION
//
let hasSideEffects = 0, isCommutable = 1, Constraints = "$src = $Rn" in {

  // sub Rm, Rn
  def SUBRmRn       : InstRmRn<0b0011000000001000,
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "sub $Rm,$Rn",
                              [(set i32:$Rn, (sub i32:$src, i32:$Rm))]>;

  // subc Rm, Rn
  let Defs = [SR], Uses = [SR] in
  def SUBCRmRn      : InstRmRn<0b0011000000001010,
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "subc $Rm,$Rn",
                              [(set i32:$Rn, (sube i32:$src, i32:$Rm))]>;

  // subv Rm, Rn
  let Defs = [SR] in
  def SUBVRmRn      : InstRmRn<0b0011000000001011,
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "subv $Rm,$Rn",
                              [(set i32:$Rn, (subc i32:$src, i32:$Rm))]>;
  
  // dt Rn
  let Defs = [SR] in
  def DTRmRn        : InstRn<0b0011000000001000,
                            (outs GPR:$Rn), (ins GPR:$src),
                            "dt $Rn",
                            [(set i32:$Rn, (sub i32:$src, -1))]>;
}

//
//      NEGATION
//
let hasSideEffects = 0 in {

  // neg Rm, Rn
  def NEGRmRn       : InstRmRn<0b0110000000001011,
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "neg $Rm,$Rn",
                              [(set i32:$Rn, (ineg i32:$Rm))]>;

  // negc Rm, Rn
  let Defs = [SR], Uses = [SR] in
  def NEGCRmRn      : InstRmRn<0b0110000000001010,
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "negc $Rm,$Rn",
                              [(set i32:$Rn, (ineg i32:$Rm))]>;

}

//
//      MULTIPLY
//
let hasSideEffects = 0, isCommutable = 1, Constraints = "$src = $Rn" in {

  // mul.l Rm, Rn
  let Defs = [MACL] in
  def MULRmRn       : InstRmRn<0b0000000000000111, 
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "mul.l $Rm,$Rn",
                              []>;

  // mulr Rm, Rn
  let Uses = [R0] in
  def MULRR0Rn        : InstRn<0b0100000010000000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "mulr R0,$Rn",
                              []>;

}

//===----------------------------------------------------------------------===//
// Branch Instructions
//===----------------------------------------------------------------------===//

// TODO: Port these to the updated way.
// def BF_Disp      : BrOp_Disp8<0b1000101100000000, "bf">;
// def BFS_Disp    : BrOpD_Disp8<0b1000111100000000, "bf/s">;
// def BT_Disp      : BrOp_Disp8<0b1000100100000000, "bt">;
// def BTS_Disp    : BrOpD_Disp8<0b1000110100000000, "bt/s">;
// def BRA_Disp    : BrOp_Disp12<0b1010000000000000, "bra">;
// def BRAF_Rm         : BrOp_Rm<0b0000000000100011, "braf">;
// def BSR_Disp    : BrOp_Disp12<0b1011000000000000, "bsr">;
// def BSRF_Rm         : BrOp_Rm<0b0000000000000011, "bsrf">;
// def JMP_Rmi        : BrOp_Rmi<0b0100000000101011, "jmp">;


//===----------------------------------------------------------------------===//
// Call Instructions
//===----------------------------------------------------------------------===//

let isCall = 1 in {
  let Uses = [GBR] in
  def JSRRmi          : InstRm<0b0100000000001011,
                              (outs), (ins GPRMem:$Rm),
                              "jsr @$Rm",
                              [(SHCall GPRMem:$Rm)]>;
}


//===----------------------------------------------------------------------===//
// Return Instructions
//===----------------------------------------------------------------------===//

let isTerminator = 1, isReturn = 1, isBarrier = 1 in {
  def RTS             : Inst<0b0000000000001011,
                              (outs), (ins),
                              "rts",
                              []>;
}






//===----------------------------------------------------------------------===//
// Control Instructions
//===----------------------------------------------------------------------===//

// nop
let hasSideEffects = 1 in
def NOP : Inst<0b0000000000001001, (outs), (ins), "nop", []>;

// sleep
let hasSideEffects = 1 in
def SLEEP : Inst<0b0000000000011011, (outs), (ins), "sleep", []>;




//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// Helper instruction for mapping frame indices to relative frame pointer
// offsets.
let Defs = [SR], hasSideEffects = 0 in
def SHFrmIdx : SHPseudo<(outs GPR:$dst), (ins GPR:$src, disp12:$src2)>;

let Defs = [R0], Uses = [R0] in {
def ADJCALLSTACKDOWN : SHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                               "!ADJCALLSTACKDOWN $amt1, $amt2",
                               [(SHCallSeqStart timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP : SHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            "!ADJCALLSTACKUP $amt1",
                            [(SHCallSeqEnd timm:$amt1, timm:$amt2)]>;
}