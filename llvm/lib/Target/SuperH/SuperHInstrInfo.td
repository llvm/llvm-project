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
def SHSDT_CallSeqEnd    : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SHSDT_Call          : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SHSDT_PICAdd        : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisPtrTy<1>, SDTCisVT<2, i32>]>;
def SHSDT_Brcond        : SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisInt<1>]>;
def SHSDT_Cmp           : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>, SDTCisInt<2>]>;
def SHSDT_SelectCC      : SDTypeProfile<1, 3,
                          [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisInt<3>]>;

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

def SHRetGlue       : SDNode<"SHISD::RET_GLUE", SDTNone,
                            [SDNPHasChain, SDNPOptInGlue, 
                             SDNPVariadic]>;

def SHWrapper       : SDNode<"SHISD::WRAPPER", SDTIntUnaryOp>;
def SHPICAdd        : SDNode<"SHISD::PIC_ADD", SHSDT_PICAdd>;

// Conditional Branching and Comparison
def SHselectcc      : SDNode<"SHISD::SELECT_CC", SHSDT_SelectCC, [SDNPInGlue]>;
def SHbrcond        : SDNode<"SHISD::BRCOND", SHSDT_Brcond, [SDNPHasChain, SDNPInGlue]>;
def SHcmp           : SDNode<"SHISD::CMP", SHSDT_Cmp, [SDNPOutGlue]>;


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

def brtarget       : Operand<OtherVT> {
  let EncoderMethod = "getBranchTargetOpValue";
  let DecoderMethod = "DecodeBranchTarget";
  let PrintMethod = "printPCRelImm";
  let OperandType = "OPERAND_PCREL";
}

// An operand for the CONSTPOOL_ENTRY pseudo-instruction.
def cpinst_operand : Operand<i32> {
  let PrintMethod = "printCPInstOperand";
}

// Memory
class SHMemRegOp<RegisterClass regClass> 
      : RegisterOperand<regClass> {
    let ParserMatchClass = SHMemClass;
    let OperandType = "OPERAND_MEMORY";
}
def GPRMem : SHMemRegOp<GPR>;

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

// mova @(disp, PC), R0
let hasSideEffects = 0, mayLoad = 1, Defs = [R0], isDelayIllegal = 1 in
def MOVAD8PCR0      : InstD8<0b1100011100000000,
                            (outs), (ins disp8:$disp),
                            "mova @($disp,PC),R0",
                            [(set R0, (load disp8:$disp))]>;

// mov.w @(disp,PC), Rn 
let hasSideEffects = 0, mayLoad = 1, isDelayIllegal = 1 in
def MOVWD8PCRn    : InstRnD8<0b1001000000000000,
                            (outs GPR:$Rn), (ins disp8:$disp),
                            "mov.w @($disp,PC),$Rn",
                            [(set i32:$Rn, (sextloadi16 disp8:$disp))]>;

// mov.l @(disp,PC), Rn 
let hasSideEffects = 0, mayLoad = 1, isDelayIllegal = 1 in
def MOVLD8PCRn    : InstRnD8<0b1101000000000000,
                            (outs GPR:$Rn), (ins disp8:$disp),
                            "mov.l @($disp,PC),$Rn",
                            [(set i32:$Rn, (load disp8:$disp))]>;

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

let hasSideEffects = 0, Constraints = "$src = $Rn" in {
  
  // add Rm, Rn
  def ADDRmRn       : InstRmRn<0b0011000000001100, 
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "add $Rm,$Rn",
                              [(set i32:$Rn, (add i32:$Rm, i32:$src))]>;
  
  // add imm:8, Rn
  def ADDI8Rn       : InstRnI8<0b0111000000000000, 
                              (outs GPR:$Rn), (ins GPR:$src, imm8:$imm),
                              "add #$imm,$Rn",
                              [(set i32:$Rn, (add i32:$src, imm8:$imm))]>;

  // addc Rm, Rn
  let Defs = [SR], Uses = [SR], isCommutable = 1 in
  def ADDCRmRn      : InstRmRn<0b0011000000001110,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "addc $Rm,$Rn",
                              [(set i32:$Rn, (adde i32:$Rm, i32:$src))]>;
  
  // addv Rm, Rn
  let Defs = [SR], isCommutable = 1 in
  def ADDVRmRn      : InstRmRn<0b0011000000001111,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "addv $Rm,$Rn",
                              [(set i32:$Rn, (addc i32:$Rm, i32:$src))]>;

}



//===--------------------------------------------------------------------------===//
// Subtraction
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, isCommutable = 1, Constraints = "$src = $Rn" in {

  // sub Rm, Rn
  def SUBRmRn       : InstRmRn<0b0011000000001000,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "sub $Rm,$Rn",
                              [(set i32:$Rn, (sub i32:$Rm, i32:$src))]>;

  // subc Rm, Rn
  let Defs = [SR], Uses = [SR] in
  def SUBCRmRn      : InstRmRn<0b0011000000001010,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "subc $Rm,$Rn",
                              [(set i32:$Rn, (sube i32:$Rm, i32:$src))]>;

  // subv Rm, Rn
  let Defs = [SR] in
  def SUBVRmRn      : InstRmRn<0b0011000000001011,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "subv $Rm,$Rn",
                              [(set i32:$Rn, (subc i32:$Rm, i32:$src))]>;
  
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
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
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
                              (outs), (ins GPR:$Rm, GPR:$Rn),
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

// and Rm,Rn
let Constraints = "$src = $Rn" in
def ANDRmRn       : InstRmRn<0b0010000000001001, 
                            (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                            "and $Rm,$Rn",
                            [(set i32:$Rn, (and i32:$Rm, i32:$src))]>;
// and #imm,R0
let Defs = [R0], Uses = [R0] in
def ANDI8R0         : InstI8<0b1100100100000000, 
                            (outs), (ins imm8:$imm),
                            "and #$imm,R0",
                            []>;

// not Rm,Rn
def NOTRmRn       : InstRmRn<0b0110000000000111, 
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "not $Rm,$Rn",
                            [(set i32:$Rn, (not i32:$Rm))]>;

// or Rm,Rn
let Constraints = "$src = $Rn" in
def ORRmRn        : InstRmRn<0b0010000000001011, 
                            (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                            "or $Rm,$Rn",
                            [(set i32:$Rn, (or i32:$Rm, i32:$src))]>;
// or #imm,R0
let Defs = [R0], Uses = [R0] in
def ORI8R0          : InstI8<0b1100101100000000, 
                            (outs), (ins imm8:$imm),
                            "or #$imm,R0",
                            []>;

// xor Rm,Rn
let Constraints = "$src = $Rn" in
def XORRmRn       : InstRmRn<0b0010000000001010, 
                            (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                            "xor $Rm,$Rn",
                            [(set i32:$Rn, (xor i32:$Rm, i32:$src))]>;
// xor #imm,R0
let Defs = [R0], Uses = [R0] in
def XORI8R0         : InstI8<0b1100101000000000, 
                            (outs), (ins imm8:$imm),
                            "xor #$imm,R0",
                            []>;

// tst Rm,Rn
let Defs = [SR] in
def TSTRmRn       : InstRmRn<0b0010000000001000, 
                            (outs), (ins GPR:$Rm, GPR:$Rn),
                            "tst $Rm,$Rn",
                            []>;
// tst #imm,R0
let Defs = [SR], Uses = [R0] in
def TSTI8R0         : InstI8<0b1100100000000000, 
                            (outs), (ins imm8:$imm),
                            "tst #$imm,R0",
                            []>;





//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Shift Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, Constraints = "$src = $Rn" in {

  // rotcl Rn
  let Defs = [SR] in
  def ROTCLRn         : InstRn<0b0100000000100100, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotcl $Rn",
                              []>;

  // rotcr Rn
  let Defs = [SR] in
  def ROTCRRn         : InstRn<0b0100000000100101, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotcr $Rn",
                              []>;

  // rotl Rn
  let Defs = [SR] in
  def ROTLRn          : InstRn<0b0100000000000100, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotl $Rn",
                              [(set i32:$Rn, (rotl i32:$src, (i8 1)))]>;

  // rotr Rn
  let Defs = [SR] in
  def ROTRRn          : InstRn<0b0100000000000101, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotr $Rn",
                              [(set i32:$Rn, (rotr i32:$src, (i8 1)))]>;

  // shal Rn
  let Defs = [SR] in
  def SHALRn          : InstRn<0b0100000000100000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shal $Rn",
                              []>;

  // shar Rn
  let Defs = [SR] in
  def SHARRn          : InstRn<0b0100000000100001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shar $Rn",
                              [(set i32:$Rn, (sra i32:$src, (i8 1)))]>;

  // shll Rn
  let Defs = [SR] in
  def SHLLRn          : InstRn<0b0100000000000000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 1)))]>;

  // shll2 Rn
  let Defs = [SR] in
  def SHLL2Rn          : InstRn<0b0100000000001000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll2 $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 2)))]>;

  // shll8 Rn
  def SHLL8Rn          : InstRn<0b0100000000011000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll8 $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 8)))]>;

  // shll16 Rn
  def SHLL16Rn          : InstRn<0b0100000000101000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll16 $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 16)))]>;

  // shlr Rn
  let Defs = [SR] in
  def SHLRRn          : InstRn<0b0100000000000001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shlr $Rn",
                              [(set i32:$Rn, (srl i32:$src, (i8 1)))]>;

  // shlr2 Rn
  def SHLR2Rn          : InstRn<0b0100000000001001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shlr2 $Rn",
                              [(set i32:$Rn, (srl i32:$src, (i8 2)))]>;

  // shlr8 Rn
  def SHLR8Rn          : InstRn<0b0100000000011001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shlr8 $Rn",
                              [(set i32:$Rn, (srl i32:$src, (i8 8)))]>;

  // shlr16 Rn
  def SHLR16Rn          : InstRn<0b0100000000101001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shlr16 $Rn",
                              [(set i32:$Rn, (srl i32:$src, (i8 16)))]>;

}


//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Branch Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

let isTerminator = 1 in {
  let isBranch = 1, Uses = [SR] in
  def BF              : InstD8<0b1000101100000000, 
                              (outs), (ins brtarget:$disp),
                              "bf $disp",
                              []>;

  let isBranch = 1, hasDelaySlot = 1, Uses = [SR] in
  def BFS             : InstD8<0b1000111100000000, 
                              (outs), (ins brtarget:$disp),
                              "bf/s $disp",
                              []>;

  let isBranch = 1, Uses = [SR] in
  def BT              : InstD8<0b1000100100000000, 
                              (outs), (ins brtarget:$disp),
                              "bt $disp",
                              []>;

  let isBranch = 1, hasDelaySlot = 1, Uses = [SR] in
  def BTS             : InstD8<0b1000110100000000, 
                              (outs), (ins brtarget:$disp),
                              "bt/s $disp",
                              []>;

  let isBranch = 1, isBarrier = 1, hasDelaySlot = 1 in
  def BRA            : InstD12<0b1010000000000000, 
                              (outs), (ins brtarget:$disp),
                              "bra $disp",
                              [(br bb:$disp)]>;

  let isIndirectBranch = 1, isBarrier = 1, hasDelaySlot = 1 in
  def BRAF            : InstRm<0b0000000000100011, 
                              (outs), (ins GPR:$Rm),
                              "braf $Rm",
                              []>;

  let isBranch = 1, isBarrier = 1, hasDelaySlot = 1 in
  def BSR            : InstD12<0b1011000000000000, 
                              (outs), (ins brtarget:$disp),
                              "bsr $disp",
                              []>;

  let isBranch = 1, isBarrier = 1, hasDelaySlot = 1 in
  def BSRF            : InstRm<0b0000000000000011, 
                              (outs), (ins GPR:$Rm),
                              "bsrf $Rm",
                              []>;

  let isBranch = 1, isBarrier = 1, hasDelaySlot = 1 in
  def JMP             : InstRm<0b0100000000101011, 
                              (outs), (ins GPR:$Rm),
                              "jmp @$Rm",
                              []>;
}



//===--------------------------------------------------------------------------===//
// Call Instructions
//===--------------------------------------------------------------------------===//

let isCall = 1, isBarrier = 1, isTerminator = 1, hasDelaySlot = 1 in {
  let Uses = [SR, GBR] in
  def JSRRmi          : InstRm<0b0100000000001011,
                              (outs), (ins GPRMem:$Rm),
                              "jsr @$Rm",
                              [(SHCall GPRMem:$Rm)]>;
}





//===--------------------------------------------------------------------------===//
// Return Instructions
//===--------------------------------------------------------------------------===//

let isTerminator = 1, isReturn = 1, isBarrier = 1, hasDelaySlot = 1 in {
  def RTS               : Inst<0b0000000000001011,
                              (outs), (ins),
                              "rts",
                              [(SHRetGlue)]>;
}





//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                          System Control Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

// clrmac
def CLRMAC              : Inst<0b0000000000101000, (outs), (ins), "clrmac", []>;

// clrt
let Defs = [SR] in
def CLRT                : Inst<0b0000000000001000, (outs), (ins), "clrt", []>;

// ldc Rm, SR
let Defs = [SR], isDelayIllegal = 1 in
def LDCRmSR           : InstRm<0b0100000000001110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,SR",
                              []>;

// ldc.l @Rm+, SR
let Defs = [SR], isDelayIllegal = 1 in
def LDCLRminciSR      : InstRm<0b0100000000000111, 
                              (outs), (ins GPR:$Rm), 
                              "ldc.l @$Rm+,SR",
                              []>;

// ldc Rm, GBR
let Defs = [GBR] in
def LDCRmGBR          : InstRm<0b0100000000011110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,GBR",
                              []>;

// ldc.l @Rm+, GBR
let Defs = [GBR] in
def LDCLRminciGBR     : InstRm<0b0100000000010111, 
                              (outs), (ins GPR:$Rm), 
                              "ldc.l @$Rm+,GBR",
                              []>;

// ldc Rm, VBR
let Defs = [VBR], isDelayIllegal = 1 in
def LDCRmVBR          : InstRm<0b0100000000101110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,VBR",
                              []>;

// ldc.l @Rm+, VBR
let Defs = [VBR], isDelayIllegal = 1 in
def LDCLRminciVBR     : InstRm<0b0100000000100111, 
                              (outs), (ins GPR:$Rm), 
                              "ldc.l @$Rm+,VBR",
                              []>;

// lds Rm, MACH
let Defs = [MACH] in
def LDSRmMACH         : InstRm<0b0100000000001010, 
                              (outs), (ins GPR:$Rm), 
                              "lds $Rm,MACH",
                              []>;

// lds.l @Rm+, MACH
let Defs = [MACH] in
def LDSLRminciMACH    : InstRm<0b0100000000000110, 
                              (outs), (ins GPR:$Rm), 
                              "lds.l @$Rm+,MACH",
                              []>;

// lds Rm, MACL
let Defs = [MACH] in
def LDSRmMACL         : InstRm<0b0100000000011010, 
                              (outs), (ins GPR:$Rm), 
                              "lds $Rm,MACL",
                              []>;

// lds.l @Rm+, MACL
let Defs = [MACH] in
def LDSLRminciMACL    : InstRm<0b0100000000010110, 
                              (outs), (ins GPR:$Rm), 
                              "lds.l @$Rm+,MACL",
                              []>;

// lds Rm, PR
let Defs = [PR] in
def LDSRmPR           : InstRm<0b0100000000101010, 
                              (outs), (ins GPR:$Rm), 
                              "lds $Rm,PR",
                              []>;

// lds.l @Rm+, PR
let Defs = [PR] in
def LDSLRminciPR      : InstRm<0b0100000000100110, 
                              (outs), (ins GPR:$Rm), 
                              "lds.l @$Rm+,PR",
                              []>;

// nop
let hasSideEffects = 1, isTerminator = 1 in
def NOP                 : Inst<0b0000000000001001, (outs), (ins), "nop", []>;

// rte
let isTerminator = 1, isReturn = 1, isBranch = 1, hasDelaySlot = 1 in
def RTE                 : Inst<0b0000000000101011, (outs), (ins), "rte", []>;

// sett
let Defs = [SR] in
def SETT                : Inst<0b0000000000011000, (outs), (ins), "sett", []>;

// sleep
let hasSideEffects = 1, isDelayIllegal = 1 in
def SLEEP               : Inst<0b0000000000011011, (outs), (ins), "sleep", []>;

// stc SR,Rn
let Uses = [SR], isDelayIllegal = 1 in
def STCSRRn           : InstRn<0b0000000000000010, 
                              (outs GPR:$Rn), (ins), 
                              "stc SR,$Rn",
                              []>;

// stc.l SR,@-Rn
let Uses = [SR], isDelayIllegal = 1 in
def STCLSRRndeci      : InstRn<0b0100000000000011, 
                              (outs GPR:$Rn), (ins), 
                              "stc.l SR,@-$Rn",
                              []>;

// stc GBR,Rn
let Uses = [GBR] in
def STCGBRRn          : InstRn<0b0000000000010010, 
                              (outs GPR:$Rn), (ins), 
                              "stc GBR,$Rn",
                              []>;

// stc.l GBR,@-Rn
let Uses = [GBR] in
def STCLGBRRndeci     : InstRn<0b0100000000010011, 
                              (outs GPR:$Rn), (ins), 
                              "stc.l GBR,@-$Rn",
                              []>;

// stc VBR,Rn
let Uses = [VBR], isDelayIllegal = 1 in
def STCVBRRn          : InstRn<0b0000000000100010, 
                              (outs GPR:$Rn), (ins), 
                              "stc VBR,$Rn",
                              []>;

// stc.l VBR,@-Rn
let Uses = [VBR], isDelayIllegal = 1 in
def STCLVBRRndeci     : InstRn<0b0100000000100011, 
                              (outs GPR:$Rn), (ins), 
                              "stc.l VBR,@-$Rn",
                              []>;

// sts MACH,Rn
let Uses = [MACH] in
def STSMACHRn         : InstRn<0b0000000000001010, 
                              (outs GPR:$Rn), (ins), 
                              "sts MACH,$Rn",
                              []>;

// sts.l MACH,@-Rn
let Uses = [MACH] in
def STSLMACHRndeci    : InstRn<0b0100000000000010, 
                              (outs GPR:$Rn), (ins), 
                              "sts.l MACH,@-$Rn",
                              []>;

// sts MACL,Rn
let Uses = [MACL] in
def STSMACLRn         : InstRn<0b0000000000011010, 
                              (outs GPR:$Rn), (ins), 
                              "sts MACL,$Rn",
                              []>;

// sts.l MACL,@-Rn
let Uses = [MACL] in
def STSLMACLRndeci    : InstRn<0b0100000000010010, 
                              (outs GPR:$Rn), (ins), 
                              "sts.l MACL,@-$Rn",
                              []>;

// sts PR,Rn
let Uses = [PR] in
def STSPRRn           : InstRn<0b0000000000101010, 
                              (outs GPR:$Rn), (ins), 
                              "sts PR,$Rn",
                              []>;

// sts.l PR,@-Rn
let Uses = [PR] in
def STSLPRRndeci      : InstRn<0b0100000000100010, 
                              (outs GPR:$Rn), (ins), 
                              "sts.l PR,@-$Rn",
                              []>;

// trapa #imm
let Defs = [PC], isCall = 1 in
def TRAPAI8           : InstI8<0b1100001100000000, 
                              (outs), (ins imm8:$imm), 
                              "trapa #$imm",
                              []>;




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                                  Extensions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

include "SuperHInstrDSP.td"




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Pseudo instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

//===--------------------------------------------------------------------------===//
// Stack Frame
//===--------------------------------------------------------------------------===//

// Helper instruction for mapping frame indices to relative frame pointer
// offsets.
let Defs = [SR], hasSideEffects = 0 in
def SHFrmIdx : SHPseudo<(outs GPR:$dst), (ins GPR:$src, disp12:$src2)>;

/// CONSTPOOL_ENTRY - This instruction represents a floating constant pool in
/// the function.  The first operand is the ID# for this instruction, the second
/// is the index into the MachineConstantPool that this is, the third is the
/// size in bytes of this constant pool entry.
let hasSideEffects = 0, isNotDuplicable = 1, hasNoSchedulingInfo = 1 in
def CONSTPOOL_ENTRY : SHPseudo<(outs), (ins cpinst_operand:$instid, cpinst_operand:$cpidx, i32imm:$size), 
                              "!CONSTPOOL_ENTRY $instid, $cpidx, $size",
                              []>;

let Defs = [R0], Uses = [R0] in {
def ADJCALLSTACKDOWN : SHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                               "!ADJCALLSTACKDOWN $amt1, $amt2",
                               [(SHCallSeqStart timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP : SHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            "!ADJCALLSTACKUP $amt1",
                            [(SHCallSeqEnd timm:$amt1, timm:$amt2)]>;
}