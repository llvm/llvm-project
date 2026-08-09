//===-- SuperHInstrInfo.td - Main SuperH Instruction Definition -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------------===//
///
/// \file
/// This file describes the SuperH instruction set, defining the instructions
/// and properties of the instructions which are needed for code generation,
/// machine code emission, and analysis.
///
/// The instructions in this file in particular cover SH1-2.
/// later version and extensions are in their respective files.
///
/// The subsystems are ordered after the instruction classification that
/// The SuperH ISA manuals use.
///
/// See: https://www.shared-ptr.com/sh_insns.html
/// See: https://saturnopensdk.github.io/
///
//===--------------------------------------------------------------------------===//

//===--------------------------------------------------------------------------===//
// Instruction Formats
//===--------------------------------------------------------------------------===//

include "SuperHInstrFormats.td"

//===--------------------------------------------------------------------------===//
// Instruction Patterns
//===--------------------------------------------------------------------------===//

defvar SH = DefaultMode;
def sh_ptr_rc : RegClassByHwMode<[SH], [GPR]>;

// Both cases can use the same decoder method, so avoid the dispatch
// by hwmode by setting an explicit DecoderMethod
def ptr_op : RegisterOperand<sh_ptr_rc> {
  let DecoderMethod = "DecodeIntRegsRegisterClass";
}

//===--------------------------------------------------------------------------===//
// SuperH Type Profiles
//===--------------------------------------------------------------------------===//

def SHSDT_CallSeqStart  : SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SHSDT_CallSeqEnd  : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SHSDT_Call        : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

//===--------------------------------------------------------------------------===//
// SuperH Specific Nodes
//===--------------------------------------------------------------------------===//

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

//===--------------------------------------------------------------------------===//
// Operand Classes
//===--------------------------------------------------------------------------===//

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

//===--------------------------------------------------------------------------===//
// Operands
//===--------------------------------------------------------------------------===//

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

//===--------------------------------------------------------------------------===//
// Predicates
//===--------------------------------------------------------------------------===//

def HasFSCA     : Predicate<"Subtarget->hasFSCA()">,
                  AssemblerPredicate<(all_of FeatureFSCA)>;

def HasFSRRA    : Predicate<"Subtarget->hasFSRRA()">,
                  AssemblerPredicate<(all_of FeatureFSRRA)>;

def HasFPU      : Predicate<"Subtarget->hasFPU()">,
                  AssemblerPredicate<(all_of FeatureFPU)>;

def HasDSP      : Predicate<"Subtarget->hasDSP()">,
                  AssemblerPredicate<(all_of FeatureDSP)>;

def HasSH2      : Predicate<"Subtarget->hasSH2Inst()">,
                  AssemblerPredicate<(all_of FeatureSH2)>;

def HasSH2E     : Predicate<"Subtarget->hasSH2EInst()">,
                  AssemblerPredicate<(all_of FeatureSH2E)>;

def HasSH2A     : Predicate<"Subtarget->hasSH2AInst()">,
                  AssemblerPredicate<(all_of FeatureSH2A)>;

def HasSH3      : Predicate<"Subtarget->hasSH3Inst()">,
                  AssemblerPredicate<(all_of FeatureSH3)>;

def HasSH3E     : Predicate<"Subtarget->hasSH3EInst()">,
                  AssemblerPredicate<(all_of FeatureSH3E)>;

def HasSH4      : Predicate<"Subtarget->hasSH4Inst()">,
                  AssemblerPredicate<(all_of FeatureSH4)>;

def HasSH4A     : Predicate<"Subtarget->hasSH4AInst()">,
                  AssemblerPredicate<(all_of FeatureSH4A)>;




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                          Data Transfer Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

// mov Rm, Rn
let hasSideEffects = 0 in
def MOVRmRn       : InstRmRn<0b0110000000000011,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "mov $Rm,$Rn",
                            []>;

// mov #imm:8, Rn
let hasSideEffects = 0 in 
def MOVI8Rn       : InstRnI8<0b1110000000000000,
                            (outs GPR:$Rn), (ins imm8:$imm),
                            "mov #$imm,$Rn",
                            [(set i32:$Rn, imm8:$imm)]>;

// mov @(disp, PC), R0
let hasSideEffects = 0, mayLoad = 1, Defs = [R0] in
def MOVAD8PCR0      : InstD8<0b1100011100000000,
                            (outs), (ins disp8:$disp),
                            "mov @($disp,PC),R0",
                            []>;

// mov.w @(disp,PC), Rn 
let hasSideEffects = 0, mayLoad = 1 in
def MOVWD8PCRn    : InstRnD8<0b1001000000000000,
                            (outs GPR:$Rn), (ins disp8:$disp),
                            "mov.w @($disp,PC),$Rn",
                            [(set i32:$Rn, (sextloadi16 addr:$disp))]>;

// mov.l @(disp,PC), Rn 
let hasSideEffects = 0, mayLoad = 1 in
def MOVLD8PCRn    : InstRnD8<0b1101000000000000,
                            (outs GPR:$Rn), (ins disp8:$disp),
                            "mov.l @($disp,PC),$Rn",
                            [(set i32:$Rn, (load addr:$disp))]>;

// movt Rn
def MOVTRn          : InstRn<0b0000000000101001,
                            (outs GPR:$Rn), (ins),
                            "movt $Rn",
                            []>;

// swap.b Rm,Rn
def SWAPBRmRn     : InstRmRn<0b0110000000001000,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "swap.b $Rm,$Rn",
                            []>;

// swap.w Rm,Rn
def SWAPWRmRn     : InstRmRn<0b0110000000001001,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "swap.w $Rm,$Rn",
                            []>;

// xtrct Rm,Rn
def XTRCTRmRn     : InstRmRn<0b0010000000001101,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "xtrct $Rm,$Rn",
                            []>;

//===--------------------------------------------------------------------------===//
// Load Instructions
//===--------------------------------------------------------------------------===//

// Indirect Memory -> Register Load
let hasSideEffects = 0, mayLoad = 1, isReMaterializable = 1 in {

  // mov.b @Rm, Rn
  def MOVBRmiRn         : InstRmRn<0b0110000000000000,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.b @$Rm,$Rn",
                                  [(set i32:$Rn, (sextloadi8 i32:$Rm))]>;

  // mov.w @Rm, Rn
  def MOVWRmiRn         : InstRmRn<0b0110000000000001,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.w @$Rm,$Rn",
                                  [(set i32:$Rn, (sextloadi16 i32:$Rm))]>;

  // mov.l @Rm, Rn
  def MOVLRmiRn         : InstRmRn<0b0110000000000010,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.l @$Rm,$Rn",
                                  [(set i32:$Rn, (load i32:$Rm))]>;

  // mov.b @Rm+, Rn
  def MOVBRminciRn      : InstRmRn<0b0110000000000100,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.b @$Rm+,$Rn",
                                  []>;

  // mov.w @Rm+, Rn
  def MOVWRminciRn      : InstRmRn<0b0110000000000101,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.w @$Rm+,$Rn",
                                  []>;

  // mov.l @Rm+, Rn
  def MOVLRminciRn      : InstRmRn<0b0110000000000110,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.l @$Rm+,$Rn",
                                  []>;

  // mov.b @(disp, Rm), R0
  let Defs = [R0] in
  def MOVBD4RmiR0       : InstRmD4<0b1000010000000000,
                                  (outs), (ins GPR:$Rm, disp4:$disp),
                                  "mov.b @($disp,$Rm),R0",
                                  []>;
  
  // mov.w @(disp, Rm), R0
  let Defs = [R0] in
  def MOVWD4RmiR0       : InstRmD4<0b1000010100000000,
                                  (outs), (ins GPR:$Rm, disp4:$disp),
                                  "mov.w @($disp,$Rm),R0",
                                  []>;
  
  // mov.l @(disp, Rm), Rn
  def MOVLD4RmiRn     : InstRmRnD4<0b0101000000000000,
                                  (outs GPRMem:$Rn), (ins GPR:$Rm, disp4:$disp),
                                  "mov.l @($disp,$Rm),$Rn",
                                  []>;

  // mov.b @(R0,Rm),Rn
  let Uses = [R0] in
  def MOVBR0RmiRn       : InstRmRn<0b0000000000001100,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.b @(R0,Rm),Rn",
                                  []>;
  
  // mov.w @(R0,Rm),Rn
  let Uses = [R0] in
  def MOVWR0RmiRn       : InstRmRn<0b0000000000001101,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.w @(R0,$Rm),$Rn",
                                  []>;
  
  // mov.l @(R0,Rm),Rn
  let Uses = [R0] in
  def MOVLR0RmiRn       : InstRmRn<0b0000000000001110,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.l @(R0,$Rm),$Rn",
                                  []>;

  // mov.b @(disp,GBR),R0
  let Defs = [R0], Uses = [GBR] in
  def MOVBD12GBRiR0       : InstD8<0b1100010000000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.b @($disp,GBR),R0",
                                  []>;
  
  // mov.w @(disp,GBR),R0
  let Defs = [R0], Uses = [GBR] in
  def MOVWD12GBRiR0       : InstD8<0b1100010100000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.w @($disp,GBR),R0",
                                  []>;
  
  // mov.l @(disp,GBR),R0
  let Defs = [R0], Uses = [GBR] in
  def MOVLD12GBRiR0       : InstD8<0b1100011000000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.l @($disp,GBR),R0",
                                  []>;
}



//===--------------------------------------------------------------------------===//
// Store Instructions
//===--------------------------------------------------------------------------===//

// Register -> Indirect Memory Store
let hasSideEffects = 0, mayStore = 1 in {

  // mov.b Rm, @Rn
  def MOVBRmRni         : InstRmRn<0b0010000000000000,
                                  (outs), (ins GPRMem:$Rn, GPR:$Rm),
                                  "mov.b $Rm,@$Rn",
                                  [(truncstorei8 i32:$Rm, GPRMem:$Rn)]>;

  // mov.w Rm, @Rn
  def MOVWRmRni         : InstRmRn<0b0010000000000001,
                                  (outs), (ins GPR:$Rm, GPRMem:$Rn),
                                  "mov.w $Rm,@$Rn",
                                  [(truncstorei16 i32:$Rm, GPRMem:$Rn)]>;

  // mov.l Rm, @Rn
  def MOVLRmRni         : InstRmRn<0b0010000000000010,
                                  (outs), (ins GPRMem:$Rn, GPR:$Rm),
                                  "mov.l $Rm,@$Rn",
                                  [(store i32:$Rm, GPRMem:$Rn)]>;

  // mov.b Rm, @-Rn
  def MOVBRmRndeci      : InstRmRn<0b0010000000000100,
                                  (outs), (ins GPRMem:$Rn, GPR:$Rm),
                                  "mov.b $Rm,@-$Rn",
                                  []>;

  // mov.w Rm, @-Rn
  def MOVWRmRndeci      : InstRmRn<0b0010000000000101,
                                  (outs), (ins GPR:$Rm, GPRMem:$Rn),
                                  "mov.w $Rm,@-$Rn",
                                  []>;

  // mov.l Rm, @-Rn
  def MOVLRmRndeci      : InstRmRn<0b0010000000000110,
                                  (outs), (ins GPRMem:$Rn, GPR:$Rm),
                                  "mov.l $Rm,@-$Rn",
                                  []>;
  
  // mov.b R0, @(disp, Rn)
  let Uses = [R0] in
  def MOVBR0D4Rni       : InstRnD4<0b1000000000000000,
                                  (outs GPR:$Rn), (ins disp4:$disp),
                                  "mov.b R0,@($disp,$Rn)",
                                  []>;
  
  // mov.w R0, @(disp, Rn)
  let Uses = [R0] in
  def MOVWR0D4Rni       : InstRnD4<0b1000000100000000,
                                  (outs GPR:$Rn), (ins disp4:$disp),
                                  "mov.w R0,@($disp,$Rn)",
                                  []>;
  
  // mov.l Rm, @(disp, Rn)
  def MOVLRmD4Rni     : InstRmRnD4<0b0001000000000000,
                                  (outs GPR:$Rn), (ins GPR:$Rm, disp4:$disp),
                                  "mov.l $Rm,@($disp,$Rn)",
                                  []>;

  // mov.b Rm,@(R0,Rn)
  let Uses = [R0] in
  def MOVBRmR0Rni       : InstRmRn<0b0000000000000100,
                                  (outs), (ins GPR:$Rm, GPR:$Rn),
                                  "mov.b $Rm,@(R0,$Rn)",
                                  []>;
  
  // mov.w Rm,@(R0,Rn)
  let Uses = [R0] in
  def MOVWRmR0Rni       : InstRmRn<0b0000000000000101,
                                  (outs), (ins GPR:$Rm, GPR:$Rn),
                                  "mov.w $Rm,@(R0,$Rn)",
                                  []>;
  
  // mov.l Rm,@(R0,Rn)
  let Uses = [R0] in
  def MOVLRmR0Rni       : InstRmRn<0b0000000000000110,
                                  (outs), (ins GPR:$Rm, GPR:$Rn),
                                  "mov.l $Rm,@(R0,$Rn)",
                                  []>;

  // mov.b R0,@(disp,GBR)
  let Uses = [R0, GBR] in
  def MOVBR0D12GBRi       : InstD8<0b1100000000000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.b R0,@($disp,GBR)",
                                  []>;
  
  // mov.w R0,@(disp,GBR)
  let Uses = [R0, GBR] in
  def MOVWR0D12GBRi       : InstD8<0b1100000100000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.w R0,@($disp,GBR)",
                                  []>;
  
  // mov.l R0,@(disp,GBR)
  let Uses = [R0, GBR] in
  def MOVLR0D12GBRi       : InstD8<0b1100001000000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.l R0,@($disp,GBR)",
                                  []>;
}





//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                            Arithmetic Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//



//===--------------------------------------------------------------------------===//
// Addition
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, isCommutable = 1, Constraints = "$src = $Rn" in {
  
  // add Rm, Rn
  def ADDRmRn       : InstRmRn<0b0011000000001100, 
                              (outs GPR:$Rn), (ins GPR:$src, GPR:$Rm),
                              "add $Rm,$Rn",
                              [(set i32:$Rn, (add i32:$src, i32:$Rm))]>;
  
  // add imm:8, Rn
  def ADDI8Rn       : InstRnI8<0b0011000000001100, 
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



//===--------------------------------------------------------------------------===//
// Subtraction
//===--------------------------------------------------------------------------===//

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
  def DTRn            : InstRn<0b0100000000010000,
                              (outs GPR:$Rn), (ins GPR:$src),
                              "dt $Rn",
                              [(set i32:$Rn, (sub i32:$src, -1))]>;
}

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



//===--------------------------------------------------------------------------===//
// Multiplication
//===--------------------------------------------------------------------------===//

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




//===--------------------------------------------------------------------------===//
// Division
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, isCommutable = 1 in {

  // div0s Rm, Rn
  let Defs = [SR] in
  def DIV0SRmRn     : InstRmRn<0b0010000000000111, 
                              (outs), (ins GPR:$Rn, GPR:$Rm),
                              "div0s $Rm,$Rn",
                              []>;
  
  // div0u
  let Defs = [SR] in
  def DIV0U             : Inst<0b0000000000011001, 
                              (outs), (ins),
                              "div0u",
                              []>;

  // div1 Rm, Rn
  let Defs = [SR] in
  def DIV1RmRn      : InstRmRn<0b0011000000000100, 
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "div1 $Rm,$Rn",
                              []>;
}

let hasSideEffects = 0, isCommutable = 1, Constraints = "$src = $Rn" in {

  // Expand LLVM sdiv to the following sequence:
  // div0s
  // div1 Rm, Rn
  def DIVSRmRn        : SHPseudo<(outs GPR:$Rn), 
                                 (ins GPR:$Rm, GPR:$src),
                                 "; divs $Rm, $Rn",
                                 [(set i32:$Rn, (sdiv i32:$Rm, i32:$src))]>;

  // Expand LLVM udiv to the following sequence:
  // div0u
  // div1 Rm, Rn
  def DIVURmRn        : SHPseudo<(outs GPR:$Rn), 
                                 (ins GPR:$Rm, GPR:$src),
                                 "; divu $Rm, $Rn",
                                 [(set i32:$Rn, (udiv i32:$Rm, i32:$src))]>;
}




//===--------------------------------------------------------------------------===//
// Comparison
//===--------------------------------------------------------------------------===//

// cmp/eq #imm,R0
let Defs = [R0, SR] in
def CMPEQI8R0         : InstI8<0b1000100000000000,
                              (outs), (ins imm8:$imm),
                              "cmp/eq #imm,R0",
                              []>;

// cmp/eq Rm,Rn
let Defs = [SR] in
def CMPEQRmRn       : InstRmRn<0b0011000000000000,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/eq $Rm,$Rn",
                              []>;

// cmp/hs Rm,Rn
let Defs = [SR] in
def CMPHSRmRn       : InstRmRn<0b0011000000000010,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/hs $Rm,$Rn",
                              []>;

// cmp/ge Rm,Rn
let Defs = [SR] in
def CMPGERmRn       : InstRmRn<0b0011000000000011,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/ge $Rm,$Rn",
                              []>;

// cmp/hi Rm,Rn
let Defs = [SR] in
def CMPHIRmRn       : InstRmRn<0b0011000000000110,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/hi $Rm,$Rn",
                              []>;

// cmp/gt Rm,Rn
let Defs = [SR] in
def CMPGTRmRn       : InstRmRn<0b0011000000000111,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/gt $Rm,$Rn",
                              []>;

// cmp/pl Rn
let Defs = [SR] in
def CMPPLRn           : InstRn<0b0100000000010101,
                              (outs), (ins GPR:$Rn),
                              "cmp/pl $Rn",
                              []>;

// cmp/pz Rn
let Defs = [SR] in
def CMPPZRn           : InstRn<0b0100000000010001,
                              (outs), (ins GPR:$Rn),
                              "cmp/pz $Rn",
                              []>;

// cmp/str Rm,Rn
let Defs = [SR] in
def CMPSTRRmRn      : InstRmRn<0b0010000000001100,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/str $Rm,$Rn",
                              []>;






//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                        Logic Operation Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//






//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Shift Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Branch Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

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


//===--------------------------------------------------------------------------===//
// Call Instructions
//===--------------------------------------------------------------------------===//

let isCall = 1 in {
  let Uses = [GBR] in
  def JSRRmi          : InstRm<0b0100000000001011,
                              (outs), (ins GPRMem:$Rm),
                              "jsr @$Rm",
                              [(SHCall GPRMem:$Rm)]>;
}


//===--------------------------------------------------------------------------===//
// Return Instructions
//===--------------------------------------------------------------------------===//

let isTerminator = 1, isReturn = 1, isBarrier = 1 in {
  def RTS             : Inst<0b0000000000001011,
                              (outs), (ins),
                              "rts",
                              []>;
}





//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                          System Control Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//





//===--------------------------------------------------------------------------===//
// Control Instructions
//===--------------------------------------------------------------------------===//

// nop
let hasSideEffects = 1 in
def NOP : Inst<0b0000000000001001, (outs), (ins), "nop", []>;

// sleep
let hasSideEffects = 1 in
def SLEEP : Inst<0b0000000000011011, (outs), (ins), "sleep", []>;


//===--------------------------------------------------------------------------===//
// Bigger Extensions
//===--------------------------------------------------------------------------===//

include "SuperHInstrDSP.td"


//===--------------------------------------------------------------------------===//
// Pseudo instructions
//===--------------------------------------------------------------------------===//

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