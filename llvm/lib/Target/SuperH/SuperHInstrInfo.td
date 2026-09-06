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
def SHSDT_Brcond        : SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisVT<1, OtherVT>]>;
def SHSDT_Cmp           : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>, SDTCisVT<2, OtherVT>]>;
def SHSDT_SelectCC      : SDTypeProfile<1, 3, 
                          [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisVT<3, i8>]>;

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
  let ParserMethod = "parseImm";
  let RenderMethod = "addImmOperands";
}

// Displacements
class SHDispClass<int width> : SHOpClass<"Disp" # width> {
  let PredicateMethod = "isDisp";
  let ParserMethod = "parseDisp";
  let RenderMethod = "addDispOperands";
}

// PC-Relative Addressing
class SHPCRelClass : SHOpClass<"PCRel"> {
  let PredicateMethod = "isPCRel";
  let ParserMethod = "parsePCRel";
  let RenderMethod = "addPCRelOperands";
}

// Register Indirect Ops
class SHRegIndirectClass<string op> : SHOpClass<"IReg" # op> {
  let PredicateMethod = "isIReg" # op;
  let ParserMethod = "parseRegisterIndirect";
  let RenderMethod = "addIRegOperands";
}

//===--------------------------------------------------------------------------===//
// Operands
//===--------------------------------------------------------------------------===//

// Operand with Scaling
class SHScaledOp<ValueType vt, string name, int scale, bit encode = 1>
      : Operand<vt> {

  defvar Scale = !cond(!gt(scale, 0) : "<" # scale # ">", true : "");
  defvar Suffix = Scale;

  let PrintMethod = "print" # name # Suffix;
  let EncoderMethod = !cond(encode : "get" # name # "OpValue" # Suffix, true : "");
}


// Immediate
class SHImmOp<int width, ValueType vt>
      : Operand<vt>, ImmLeaf<vt, [{ return isInt<}] # width # [{>(Imm); }]> {
    let ParserMatchClass = SHImmClass<width>;
    let OperandType = "OPERAND_IMMEDIATE";
    let PrintMethod = "printImm";
    let MCOperandPredicate = [{
      int64_t Imm;
      if (MCOp.evaluateAsConstantImm(Imm))
        return isInt<}] # width # [{>(Imm)
      return false;
    }];
}

def imm8 : SHImmOp<8, i32>;

// Displacement
class SHDispOp<int width, ValueType vt, int scale = 0>
      : SHScaledOp<vt, "Disp", scale>, 
        PatLeaf<(imm), [{ return isUInt<}] # width # [{>(N->getZExtValue()); }]> {
  let ParserMatchClass = SHDispClass<width>;
  let OperandType = "OPERAND_IMMEDIATE";
  let MCOperandPredicate = [{
      int64_t Imm;
      if (MCOp.evaluateAsConstantImm(Imm))
          return isUInt<#width>(Imm);
      return MCOp.isBareSymbolRef();
  }];
}

// Displacement
def disp4     : SHDispOp<4,  i4,  1>;
def disp4x2   : SHDispOp<4,  i4,  2>;
def disp4x4   : SHDispOp<4,  i4,  4>;
def disp8     : SHDispOp<8,  i8,  1>;
def disp8x2   : SHDispOp<8,  i8,  2>;
def disp8x4   : SHDispOp<8,  i8,  4>;
def disp12    : SHDispOp<12, i16, 1>;

// Memory Operand
class SHMemriOp<RegisterClass rc, Operand dt, int scale = 0>
      : SHScaledOp<iPTR, "Memri", scale, 0>,
        PatLeaf<(imm), [{ return isUInt<32>(N->getSExtValue()); }]>  {
  let OperandType = "OPERAND_MEMORY";
  let MIOperandInfo = (ops rc, dt);
}

def memri     : SHMemriOp<GPR, i32imm,  1>;
def memri4    : SHMemriOp<GPR, disp4,   1>;
def memri4x2  : SHMemriOp<GPR, disp4x2, 2>;
def memri4x4  : SHMemriOp<GPR, disp4x4, 4>;
def memri8x2  : SHMemriOp<GPR, disp8,   2>;
def memri8x4  : SHMemriOp<GPR, disp8,   4>;

// PCRel Operand
class SHPCRelOp<ValueType vt, int scale = 0>
      : SHScaledOp<vt, "PCRel", scale>,
        PatLeaf<(imm), [{ return isUInt<}] # width # [{>(N->getZExtValue()); }]>  {
  let OperandType = "OPERAND_PCREL";
}

def pcrel8   : SHPCRelOp<i8>;
def pcrel8x2 : SHPCRelOp<i8, 2>;
def pcrel8x4 : SHPCRelOp<i8, 4>;
def pcrel12  : SHPCRelOp<i16>;

// Branch Targets
def brtarget       : Operand<OtherVT> {
  let ParserMatchClass = SHPCRelClass<>;
  let EncoderMethod = "getBranchTargetOpValue";
  let DecoderMethod = "DecodeBranchTarget";
  let PrintMethod = "printPCRel<1>";
  let OperandType = "OPERAND_PCREL";
}

// An operand for the CONSTPOOL_ENTRY pseudo-instruction.
def cpinst_operand : Operand<i32> {
  let PrintMethod = "printCPInstOperand";
}

// Register Indirect
class SHIndirectRegOp<RegisterClass regClass, string op> 
      : RegisterOperand<regClass> {
    let ParserMatchClass = SHRegIndirectClass<op>;
    let OperandType = "OPERAND_MEMORY";
    let PrintMethod = "printIReg" # op;
}
def IGPR    : SHIndirectRegOp<GPR, "">;
def IGPRInc : SHIndirectRegOp<GPR, "Inc">;
def IGPRDec : SHIndirectRegOp<GPR, "Dec">;

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
// NOTE:  The names of the instructions are taken from the SH manual's
//        Pseudocode examples for each instruction.
//        When adding new instructions, please keep this naming scheme.
//===--------------------------------------------------------------------------===//

// mov Rm, Rn
let hasSideEffects = 0 in
def MOV           : InstRmRn<0b0110000000000011,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "mov $Rm,$Rn",
                            []>;

// mov #imm:8, Rn
let hasSideEffects = 0 in 
def MOVI          : InstRnI8<0b1110000000000000,
                            (outs GPR:$Rn), (ins imm8:$imm),
                            "mov $imm,$Rn",
                            [(set i32:$Rn, imm:$imm)]>;

// mova @(disp, PC), R0
let hasSideEffects = 0, mayLoad = 1, Uses = [PC], Defs = [R0], isDelayIllegal = 1 in
def MOVA          : InstD8<0b1100011100000000,
                          (outs), (ins pcrel8x4:$disp),
                          "mova $disp,r0",
                          []>;

// mov.w @(disp,PC), Rn 
let hasSideEffects = 0, mayLoad = 1, Uses = [PC], isDelayIllegal = 1 in
def MOVWI         : InstRnD8<0b1001000000000000,
                            (outs GPR:$Rn), (ins pcrel8x2:$disp),
                            "mov.w $disp,$Rn",
                            [(set i32:$Rn, (sextloadi16 constpool:$disp))]>;

// mov.l @(disp,PC), Rn 
let hasSideEffects = 0, mayLoad = 1, Uses = [PC], isDelayIllegal = 1 in
def MOVLI         : InstRnD8<0b1101000000000000,
                            (outs GPR:$Rn), (ins pcrel8x4:$disp),
                            "mov.l $disp,$Rn",
                            [(set i32:$Rn, (load constpool:$disp))]>;

// movt Rn
def MOVT          : InstRn<0b0000000000101001,
                            (outs GPR:$Rn), (ins),
                            "movt $Rn",
                            []>;

// swap.b Rm,Rn
def SWAPB         : InstRmRn<0b0110000000001000,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "swap.b $Rm,$Rn",
                            []>;

// swap.w Rm,Rn
def SWAPW         : InstRmRn<0b0110000000001001,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "swap.w $Rm,$Rn",
                            []>;

// xtrct Rm,Rn
def XTRCT         : InstRmRn<0b0010000000001101,
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "xtrct $Rm,$Rn",
                            []>;

//===--------------------------------------------------------------------------===//
// Load Instructions
//===--------------------------------------------------------------------------===//

// Indirect Memory -> Register Load
let hasSideEffects = 0, mayLoad = 1, isReMaterializable = 1 in {

  // mov.b @Rm, Rn
  def MOVBL             : InstRmRn<0b0110000000000000,
                                  (outs GPR:$Rn), (ins IGPR:$Rm),
                                  "mov.b $Rm,$Rn",
                                  []>;

  // mov.w @Rm, Rn
  def MOVWL             : InstRmRn<0b0110000000000001,
                                  (outs GPR:$Rn), (ins IGPR:$Rm),
                                  "mov.w $Rm,$Rn",
                                  []>;

  // mov.l @Rm, Rn
  def MOVLL             : InstRmRn<0b0110000000000010,
                                  (outs GPR:$Rn), (ins IGPR:$Rm),
                                  "mov.l $Rm,$Rn",
                                  []>;

  // mov.b @Rm+, Rn
  let isDelayIllegal = 1 in
  def MOVBP             : InstRmRn<0b0110000000000100,
                                  (outs GPR:$Rn), (ins IGPRInc:$Rm),
                                  "mov.b $Rm,$Rn",
                                  []>;

  // mov.w @Rm+, Rn
  let isDelayIllegal = 1 in
  def MOVWP             : InstRmRn<0b0110000000000101,
                                  (outs GPR:$Rn), (ins IGPRInc:$Rm),
                                  "mov.w $Rm,$Rn",
                                  []>;

  // mov.l @Rm+, Rn
  let isDelayIllegal = 1 in
  def MOVLP             : InstRmRn<0b0110000000000110,
                                  (outs GPR:$Rn), (ins IGPRInc:$Rm),
                                  "mov.l $Rm,$Rn",
                                  []>;

  // mov.b @(disp, Rm), R0
  let Defs = [R0] in
  def MOVBL4            : InstRmD4<0b1000010000000000,
                                  (outs), (ins (memri4 $Rm, $disp):$src),
                                  "mov.b $src,R0",
                                  []>;
  
  // mov.w @(disp, Rm), R0
  let Defs = [R0] in
  def MOVWL4            : InstRmD4<0b1000010100000000,
                                  (outs), (ins (memri4x2 $Rm, $disp):$src),
                                  "mov.w $src,R0",
                                  []>;
  
  // mov.l @(disp, Rm), Rn
  def MOVLL4          : InstRmRnD4<0b0101000000000000,
                                  (outs GPR:$Rn), (ins (memri4x4 $Rm, $disp):$src),
                                  "mov.l $src,$Rn",
                                  []>;

  // mov.b @(R0,Rm),Rn
  let Uses = [R0] in
  def MOVBL0            : InstRmRn<0b0000000000001100,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.b @(R0,Rm),Rn",
                                  []>;
  
  // mov.w @(R0,Rm),Rn
  let Uses = [R0] in
  def MOVWL0            : InstRmRn<0b0000000000001101,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.w @(R0,$Rm),$Rn",
                                  []>;
  
  // mov.l @(R0,Rm),Rn
  let Uses = [R0] in
  def MOVLL0            : InstRmRn<0b0000000000001110,
                                  (outs GPR:$Rn), (ins GPR:$Rm),
                                  "mov.l @(R0,$Rm),$Rn",
                                  []>;

  // mov.b @(disp,GBR),R0
  let Defs = [R0], Uses = [GBR] in
  def MOVBLG              : InstD8<0b1100010000000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.b @($disp,GBR),R0",
                                  []>;
  
  // mov.w @(disp,GBR),R0
  let Defs = [R0], Uses = [GBR] in
  def MOVWLG              : InstD8<0b1100010100000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.w @($disp,GBR),R0",
                                  []>;
  
  // mov.l @(disp,GBR),R0
  let Defs = [R0], Uses = [GBR] in
  def MOVLLG              : InstD8<0b1100011000000000,
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
  def MOVBS             : InstRmRn<0b0010000000000000,
                                  (outs), (ins GPR:$Rm, IGPR:$Rn),
                                  "mov.b $Rm,$Rn",
                                  []>;

  // mov.w Rm, @Rn
  def MOVWS             : InstRmRn<0b0010000000000001,
                                  (outs), (ins GPR:$Rm, IGPR:$Rn),
                                  "mov.w $Rm,$Rn",
                                  []>;

  // mov.l Rm, @Rn
  def MOVLS             : InstRmRn<0b0010000000000010,
                                  (outs), (ins GPR:$Rm, IGPR:$Rn),
                                  "mov.l $Rm,$Rn",
                                  []>;

  // mov.b Rm, @-Rn
  let isDelayIllegal = 1 in
  def MOVBM             : InstRmRn<0b0010000000000100,
                                  (outs), (ins GPR:$Rm, IGPRDec:$Rn),
                                  "mov.b $Rm,$Rn",
                                  []>;

  // mov.w Rm, @-Rn
  let isDelayIllegal = 1 in
  def MOVWM             : InstRmRn<0b0010000000000101,
                                  (outs), (ins GPR:$Rm, IGPRDec:$Rn),
                                  "mov.w $Rm,$Rn",
                                  []>;

  // mov.l Rm, @-Rn
  let isDelayIllegal = 1 in
  def MOVLM             : InstRmRn<0b0010000000000110,
                                  (outs), (ins GPR:$Rm, IGPRDec:$Rn),
                                  "mov.l $Rm,$Rn",
                                  []>;
  
  // mov.b R0, @(disp, Rn)
  let Uses = [R0] in
  def MOVBS4            : InstRnD4<0b1000000000000000,
                                  (outs), (ins (memri4 $Rn, $disp):$dst),
                                  "mov.b r0,$dst",
                                  []>;
  
  // mov.w R0, @(disp, Rn)
  let Uses = [R0] in
  def MOVWS4            : InstRnD4<0b1000000100000000,
                                  (outs), (ins (memri4x2 $Rn, $disp):$dst),
                                  "mov.w r0,$dst",
                                  []>;
  
  // mov.l Rm, @(disp, Rn)
  def MOVLS4          : InstRmRnD4<0b0001000000000000,
                                  (outs), (ins GPR:$Rm, (memri4x4 $Rn, $disp):$dst),
                                  "mov.l $Rm,$dst",
                                  []>;

  // mov.b Rm,@(R0,Rn)
  let Uses = [R0] in
  def MOVBS0            : InstRmRn<0b0000000000000100,
                                  (outs), (ins GPR:$Rm, GPR:$Rn),
                                  "mov.b $Rm,@(R0,$Rn)",
                                  []>;
  
  // mov.w Rm,@(R0,Rn)
  let Uses = [R0] in
  def MOVWS0            : InstRmRn<0b0000000000000101,
                                  (outs), (ins GPR:$Rm, GPR:$Rn),
                                  "mov.w $Rm,@(R0,$Rn)",
                                  []>;
  
  // mov.l Rm,@(R0,Rn)
  let Uses = [R0] in
  def MOVLS0            : InstRmRn<0b0000000000000110,
                                  (outs), (ins GPR:$Rm, GPR:$Rn),
                                  "mov.l $Rm,@(R0,$Rn)",
                                  []>;

  // mov.b R0,@(disp,GBR)
  let Uses = [R0, GBR] in
  def MOVBSG              : InstD8<0b1100000000000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.b R0,@($disp,GBR)",
                                  []>;
  
  // mov.w R0,@(disp,GBR)
  let Uses = [R0, GBR] in
  def MOVWSG              : InstD8<0b1100000100000000,
                                  (outs), (ins disp12:$disp),
                                  "mov.w R0,@($disp,GBR)",
                                  []>;
  
  // mov.l R0,@(disp,GBR)
  let Uses = [R0, GBR] in
  def MOVLSG              : InstD8<0b1100001000000000,
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
  def ADD           : InstRmRn<0b0011000000001100, 
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "add $Rm,$Rn",
                              [(set i32:$Rn, (add i32:$Rm, i32:$src))]>;
  
  // add imm:8, Rn
  def ADDI          : InstRnI8<0b0111000000000000, 
                              (outs GPR:$Rn), (ins GPR:$src, imm8:$imm),
                              "add $imm,$Rn",
                              [(set i32:$Rn, (add i32:$src, imm:$imm))]>;

  // addc Rm, Rn
  let Defs = [SR], Uses = [SR], isCommutable = 1 in
  def ADDC          : InstRmRn<0b0011000000001110,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "addc $Rm,$Rn",
                              [(set i32:$Rn, (adde i32:$Rm, i32:$src))]>;
  
  // addv Rm, Rn
  let Defs = [SR], isCommutable = 1 in
  def ADDV          : InstRmRn<0b0011000000001111,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "addv $Rm,$Rn",
                              [(set i32:$Rn, (addc i32:$Rm, i32:$src))]>;

}



//===--------------------------------------------------------------------------===//
// Subtraction
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, isCommutable = 1, Constraints = "$src = $Rn" in {

  // sub Rm, Rn
  def SUB           : InstRmRn<0b0011000000001000,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "sub $Rm,$Rn",
                              [(set i32:$Rn, (sub i32:$Rm, i32:$src))]>;

  // subc Rm, Rn
  let Defs = [SR], Uses = [SR] in
  def SUBC          : InstRmRn<0b0011000000001010,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "subc $Rm,$Rn",
                              [(set i32:$Rn, (sube i32:$Rm, i32:$src))]>;

  // subv Rm, Rn
  let Defs = [SR] in
  def SUBV          : InstRmRn<0b0011000000001011,
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "subv $Rm,$Rn",
                              [(set i32:$Rn, (subc i32:$Rm, i32:$src))]>;
  
  // dt Rn
  let Defs = [SR] in
  def DT              : InstRn<0b0100000000010000,
                              (outs GPR:$Rn), (ins GPR:$src),
                              "dt $Rn",
                              [(set i32:$Rn, (sub i32:$src, -1))]>;
}

let hasSideEffects = 0 in {

  // neg Rm, Rn
  def NEG           : InstRmRn<0b0110000000001011,
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "neg $Rm,$Rn",
                              [(set i32:$Rn, (ineg i32:$Rm))]>;

  // negc Rm, Rn
  let Defs = [SR], Uses = [SR] in
  def NEGC          : InstRmRn<0b0110000000001010,
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "negc $Rm,$Rn",
                              [(set i32:$Rn, (ineg i32:$Rm))]>;

}



//===--------------------------------------------------------------------------===//
// Multiplication
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, isCommutable = 1 in {

  // mac.w @Rm+, @Rn+
  let Defs = [MACLO] in
  def MACW      : InstRmRn<0b0100000000001111, 
                              (outs), (ins IGPRInc:$Rm, IGPRInc:$Rn),
                              "mac.w $Rm,$Rn",
                              []>;

  // mac.l @Rm+, @Rn+
  let Defs = [MACLO, MACHI] in
  def MACL          : InstRmRn<0b0000000000001111, 
                              (outs), (ins IGPRInc:$Rm, IGPRInc:$Rn),
                              "mac.l $Rm,$Rn",
                              []>;

  // mul.l Rm, Rn
  let Defs = [MACLO] in
  def MULL          : InstRmRn<0b0000000000000111, 
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "mul.l $Rm,$Rn",
                              []>;

  // muls.w Rm, Rn
  let Defs = [MACLO] in
  def MULS          : InstRmRn<0b0010000000001111, 
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "muls.w $Rm,$Rn",
                              []>;

  // mulu.w Rm, Rn
  let Defs = [MACLO] in
  def MULU          : InstRmRn<0b0010000000001110, 
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "mulu.w $Rm,$Rn",
                              []>;

  // dmuls.l Rm, Rn
  let Defs = [MACLO, MACHI] in
  def DMULS         : InstRmRn<0b0011000000001101, 
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "dmuls.l $Rm,$Rn",
                              []>;

  // dmulu.l Rm, Rn
  let Defs = [MACLO, MACHI] in
  def DMULU         : InstRmRn<0b0011000000000101, 
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "dmulu.l $Rm,$Rn",
                              []>;
}




//===--------------------------------------------------------------------------===//
// Division
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, isCommutable = 1 in {

  // div0s Rm, Rn
  let Defs = [SR] in
  def DIV0S         : InstRmRn<0b0010000000000111, 
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
  def DIV1          : InstRmRn<0b0011000000000100, 
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "div1 $Rm,$Rn",
                              []>;
}




//===--------------------------------------------------------------------------===//
// Sign/Zero Extend
//===--------------------------------------------------------------------------===//
  
// exts.b Rm, Rn
def EXTSB           : InstRmRn<0b0110000000001110, 
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "exts.b $Rm,$Rn",
                              [(set i32:$Rn, (sext i8:$Rm))]>;

// exts.w Rm, Rn
def EXTSW           : InstRmRn<0b0110000000001111, 
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "exts.w $Rm,$Rn",
                              [(set i32:$Rn, (sext i16:$Rm))]>;

// extu.b Rm, Rn
def EXTUB           : InstRmRn<0b0110000000001100, 
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "extu.b $Rm,$Rn",
                              [(set i32:$Rn, (zext i8:$Rm))]>;

// extu.w Rm, Rn
def EXTUW           : InstRmRn<0b0110000000001101, 
                              (outs GPR:$Rn), (ins GPR:$Rm),
                              "extu.w $Rm,$Rn",
                              [(set i32:$Rn, (zext i16:$Rm))]>;





//===--------------------------------------------------------------------------===//
// Comparison
//===--------------------------------------------------------------------------===//

// cmp/eq #imm,R0
let Defs = [R0, SR] in
def CMPIM             : InstI8<0b1000100000000000,
                              (outs), (ins imm8:$imm),
                              "cmp/eq #imm,R0",
                              []>;

// cmp/eq Rm,Rn
let Defs = [SR] in
def CMPEQ           : InstRmRn<0b0011000000000000,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/eq $Rm,$Rn",
                              []>;

// cmp/hs Rm,Rn
let Defs = [SR] in
def CMPHS           : InstRmRn<0b0011000000000010,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/hs $Rm,$Rn",
                              []>;

// cmp/ge Rm,Rn
let Defs = [SR] in
def CMPGE           : InstRmRn<0b0011000000000011,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/ge $Rm,$Rn",
                              []>;

// cmp/hi Rm,Rn
let Defs = [SR] in
def CMPHI           : InstRmRn<0b0011000000000110,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/hi $Rm,$Rn",
                              []>;

// cmp/gt Rm,Rn
let Defs = [SR] in
def CMPGT           : InstRmRn<0b0011000000000111,
                              (outs), (ins GPR:$Rm, GPR:$Rn),
                              "cmp/gt $Rm,$Rn",
                              []>;

// cmp/pl Rn
let Defs = [SR] in
def CMPPL             : InstRn<0b0100000000010101,
                              (outs), (ins GPR:$Rn),
                              "cmp/pl $Rn",
                              []>;

// cmp/pz Rn
let Defs = [SR] in
def CMPPZ             : InstRn<0b0100000000010001,
                              (outs), (ins GPR:$Rn),
                              "cmp/pz $Rn",
                              []>;

// cmp/str Rm,Rn
let Defs = [SR] in
def CMPSTR          : InstRmRn<0b0010000000001100,
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
def AND             : InstRmRn<0b0010000000001001, 
                            (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                            "and $Rm,$Rn",
                            [(set i32:$Rn, (and i32:$Rm, i32:$src))]>;
// and #imm,R0
let Defs = [R0], Uses = [R0] in
def ANDI            : InstI8<0b1100100100000000, 
                            (outs), (ins imm8:$imm),
                            "and $imm,R0",
                            []>;
// and #imm,@(R0,GBR)
let Uses = [R0, GBR] in
def ANDM            : InstI8<0b1100110100000000, 
                            (outs), (ins imm8:$imm),
                            "and $imm,@(r0,gbr)",
                            []>;

// not Rm,Rn
def NOT           : InstRmRn<0b0110000000000111, 
                            (outs GPR:$Rn), (ins GPR:$Rm),
                            "not $Rm,$Rn",
                            [(set i32:$Rn, (not i32:$Rm))]>;

// or Rm,Rn
let Constraints = "$src = $Rn" in
def OR            : InstRmRn<0b0010000000001011, 
                            (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                            "or $Rm,$Rn",
                            [(set i32:$Rn, (or i32:$Rm, i32:$src))]>;
// or #imm,R0
let Defs = [R0], Uses = [R0] in
def ORI             : InstI8<0b1100101100000000, 
                            (outs), (ins imm8:$imm),
                            "or $imm,R0",
                            []>;
// or.b #imm,@(R0,GBR)
let Uses = [R0, GBR] in
def ORM             : InstI8<0b1100111100000000, 
                            (outs), (ins imm8:$imm),
                            "or.b $imm,@(r0,gbr)",
                            []>;

// xor Rm,Rn
let Constraints = "$src = $Rn" in
def XOR           : InstRmRn<0b0010000000001010, 
                            (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                            "xor $Rm,$Rn",
                            [(set i32:$Rn, (xor i32:$Rm, i32:$src))]>;

// xor #imm,R0
let Defs = [R0], Uses = [R0] in
def XORI            : InstI8<0b1100101000000000, 
                            (outs), (ins imm8:$imm),
                            "xor $imm,r0",
                            []>;

// xor.b #imm,@(R0,GBR)
let Defs = [R0], Uses = [R0] in
def XORM            : InstI8<0b1100111000000000, 
                            (outs), (ins imm8:$imm),
                            "xor.b $imm,@(r0,gbr)",
                            []>;

// tas.b @Rn
let Defs = [SR] in
def TAS             : InstRn<0b0010000000001000, 
                            (outs), (ins IGPR:$Rn),
                            "tas.b $Rn",
                            []>;

// tst Rm,Rn
let Defs = [SR] in
def TST           : InstRmRn<0b0010000000001000, 
                            (outs), (ins GPR:$Rm, GPR:$Rn),
                            "tst $Rm,$Rn",
                            []>;
// tst #imm,R0
let Defs = [SR], Uses = [R0] in
def TSTI            : InstI8<0b1100100000000000, 
                            (outs), (ins imm8:$imm),
                            "tst $imm,r0",
                            []>;

// tst.b #imm,@(R0,GBR)
let Defs = [SR], Uses = [R0, GBR] in
def TSTM            : InstI8<0b1100110000000000, 
                            (outs), (ins imm8:$imm),
                            "tst.b $imm,@(r0,gbr)",
                            []>;





//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Shift Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, Constraints = "$src = $Rn" in {

  // rotcl Rn
  let Defs = [SR] in
  def ROTCL           : InstRn<0b0100000000100100, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotcl $Rn",
                              []>;

  // rotcr Rn
  let Defs = [SR] in
  def ROTCR           : InstRn<0b0100000000100101, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotcr $Rn",
                              []>;

  // rotl Rn
  let Defs = [SR] in
  def ROTL            : InstRn<0b0100000000000100, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotl $Rn",
                              [(set i32:$Rn, (rotl i32:$src, (i8 1)))]>;

  // rotr Rn
  let Defs = [SR] in
  def ROTR            : InstRn<0b0100000000000101, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "rotr $Rn",
                              [(set i32:$Rn, (rotr i32:$src, (i8 1)))]>;

  // shal Rn
  let Defs = [SR] in
  def SHAL            : InstRn<0b0100000000100000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shal $Rn",
                              []>;

  // shar Rn
  let Defs = [SR] in
  def SHAR            : InstRn<0b0100000000100001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shar $Rn",
                              [(set i32:$Rn, (sra i32:$src, (i8 1)))]>;

  // shll Rn
  let Defs = [SR] in
  def SHLL            : InstRn<0b0100000000000000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 1)))]>;

  // shll2 Rn
  let Defs = [SR] in
  def SHLL2           : InstRn<0b0100000000001000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll2 $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 2)))]>;

  // shll8 Rn
  def SHLL8           : InstRn<0b0100000000011000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll8 $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 8)))]>;

  // shll16 Rn
  def SHLL16          : InstRn<0b0100000000101000, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shll16 $Rn",
                              [(set i32:$Rn, (shl i32:$src, (i8 16)))]>;

  // shlr Rn
  let Defs = [SR] in
  def SHLR            : InstRn<0b0100000000000001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shlr $Rn",
                              [(set i32:$Rn, (srl i32:$src, (i8 1)))]>;

  // shlr2 Rn
  def SHLR2           : InstRn<0b0100000000001001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shlr2 $Rn",
                              [(set i32:$Rn, (srl i32:$src, (i8 2)))]>;

  // shlr8 Rn
  def SHLR8           : InstRn<0b0100000000011001, 
                              (outs GPR:$Rn), (ins GPR:$src),
                              "shlr8 $Rn",
                              [(set i32:$Rn, (srl i32:$src, (i8 8)))]>;

  // shlr16 Rn
  def SHLR16          : InstRn<0b0100000000101001, 
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
                              (outs), (ins IGPR:$Rm),
                              "jmp $Rm",
                              []>;
}



//===--------------------------------------------------------------------------===//
// Call Instructions
//===--------------------------------------------------------------------------===//

let isCall = 1, isBarrier = 1, isTerminator = 1, hasDelaySlot = 1 in {
  let Uses = [PR, PC] in
  def JSR             : InstRm<0b0100000000001011,
                              (outs), (ins IGPR:$Rm),
                              "jsr $Rm",
                              [(SHCall IGPR:$Rm)]>;
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
def LDCSR             : InstRm<0b0100000000001110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,SR",
                              []>;

// ldc.l @Rm+, SR
let Defs = [SR], isDelayIllegal = 1 in
def LDCMSR            : InstRm<0b0100000000000111, 
                              (outs), (ins IGPRInc:$Rm), 
                              "ldc.l $Rm,SR",
                              []>;

// ldc Rm, GBR
let Defs = [GBR] in
def LDCGBR            : InstRm<0b0100000000011110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,GBR",
                              []>;

// ldc.l @Rm+, GBR
let Defs = [GBR] in
def LDCMGBR           : InstRm<0b0100000000010111, 
                              (outs), (ins IGPRInc:$Rm), 
                              "ldc.l $Rm,GBR",
                              []>;

// ldc Rm, VBR
let Defs = [VBR], isDelayIllegal = 1 in
def LDCVBR            : InstRm<0b0100000000101110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,VBR",
                              []>;

// ldc.l @Rm+, VBR
let Defs = [VBR], isDelayIllegal = 1 in
def LDCMVBR           : InstRm<0b0100000000100111, 
                              (outs), (ins IGPRInc:$Rm), 
                              "ldc.l $Rm,VBR",
                              []>;

// lds Rm, MACH
let Defs = [MACHI] in
def LDSMACH           : InstRm<0b0100000000001010, 
                              (outs), (ins GPR:$Rm), 
                              "lds $Rm,MACH",
                              []>;

// lds.l @Rm+, MACH
let Defs = [MACHI] in
def LDSMMACH           : InstRm<0b0100000000000110, 
                              (outs), (ins IGPRInc:$Rm), 
                              "lds.l $Rm,MACH",
                              []>;

// lds Rm, MACL
let Defs = [MACLO] in
def LDSMACL           : InstRm<0b0100000000011010, 
                              (outs), (ins GPR:$Rm), 
                              "lds $Rm,MACL",
                              []>;

// lds.l @Rm+, MACL
let Defs = [MACLO] in
def LDSMMACL          : InstRm<0b0100000000010110, 
                              (outs), (ins IGPRInc:$Rm), 
                              "lds.l $Rm,MACL",
                              []>;

// lds Rm, PR
let Defs = [PR] in
def LDSPR             : InstRm<0b0100000000101010, 
                              (outs), (ins GPR:$Rm), 
                              "lds $Rm,PR",
                              []>;

// lds.l @Rm+, PR
let Defs = [PR] in
def LDSMPR            : InstRm<0b0100000000100110, 
                              (outs), (ins IGPRInc:$Rm), 
                              "lds.l $Rm,PR",
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
def STCSR             : InstRn<0b0000000000000010, 
                              (outs GPR:$Rn), (ins), 
                              "stc SR,$Rn",
                              []>;

// stc.l SR,@-Rn
let Uses = [SR], isDelayIllegal = 1 in
def STCMSR            : InstRn<0b0100000000000011, 
                              (outs IGPRDec:$Rn), (ins), 
                              "stc.l SR,$Rn",
                              []>;

// stc GBR,Rn
let Uses = [GBR] in
def STCGBR          : InstRn<0b0000000000010010, 
                              (outs GPR:$Rn), (ins), 
                              "stc GBR,$Rn",
                              []>;

// stc.l GBR,@-Rn
let Uses = [GBR] in
def STCMGBR           : InstRn<0b0100000000010011, 
                              (outs IGPRDec:$Rn), (ins), 
                              "stc.l GBR,$Rn",
                              []>;

// stc VBR,Rn
let Uses = [VBR], isDelayIllegal = 1 in
def STCVBR            : InstRn<0b0000000000100010, 
                              (outs GPR:$Rn), (ins), 
                              "stc VBR,$Rn",
                              []>;

// stc.l VBR,@-Rn
let Uses = [VBR], isDelayIllegal = 1 in
def STCMVBR           : InstRn<0b0100000000100011, 
                              (outs IGPRDec:$Rn), (ins), 
                              "stc.l VBR,$Rn",
                              []>;

// sts MACH,Rn
let Uses = [MACHI] in
def STSMACH           : InstRn<0b0000000000001010, 
                              (outs GPR:$Rn), (ins), 
                              "sts MACH,$Rn",
                              []>;

// sts.l MACH,@-Rn
let Uses = [MACHI] in
def STSMMACH          : InstRn<0b0100000000000010, 
                              (outs IGPRDec:$Rn), (ins), 
                              "sts.l MACH,$Rn",
                              []>;

// sts MACL,Rn
let Uses = [MACLO] in
def STSMACL           : InstRn<0b0000000000011010, 
                              (outs GPR:$Rn), (ins), 
                              "sts MACL,$Rn",
                              []>;

// sts.l MACL,@-Rn
let Uses = [MACLO] in
def STSMMACL          : InstRn<0b0100000000010010, 
                              (outs IGPRDec:$Rn), (ins), 
                              "sts.l MACL,$Rn",
                              []>;

// sts PR,Rn
let Uses = [PR] in
def STSPR             : InstRn<0b0000000000101010, 
                              (outs GPR:$Rn), (ins), 
                              "sts PR,$Rn",
                              []>;

// sts.l PR,@-Rn
let Uses = [PR] in
def STSMPR            : InstRn<0b0100000000100010, 
                              (outs IGPRDec:$Rn), (ins), 
                              "sts.l PR,$Rn",
                              []>;

// trapa #imm
let Defs = [PC], isCall = 1 in
def TRAPA             : InstI8<0b1100001100000000, 
                              (outs), (ins imm8:$imm), 
                              "trapa $imm",
                              []>;




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                                  Extensions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

include "SuperHInstrDSP.td"





//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                            Non-Instruction Patterns
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

//===--------------------------------------------------------------------------===//
// Zero-Extend Load
//===--------------------------------------------------------------------------===//

// def : Pat<(i32 (zextloadi1  i32:$src)), (EXTUBRmRn (MOVBRmiRn i32:$src))>;
// def : Pat<(i32 (zextloadi8  i32:$src)), (EXTUBRmRn (MOVBRmiRn i32:$src))>;
// def : Pat<(i32 (zextloadi16 i32:$src)), (EXTUWRmRn (MOVWRmiRn i32:$src))>;
// def : Pat<(i32 (extloadi1   i32:$src)), (EXTSBRmRn (MOVBRmiRn i32:$src))>;
// def : Pat<(i32 (extloadi8   i32:$src)), (EXTSBRmRn (MOVBRmiRn i32:$src))>;
// def : Pat<(i32 (extloadi16  i32:$src)), (EXTSWRmRn (MOVWRmiRn i32:$src))>;
// def : Pat<(i32 (sextloadi1  i32:$src)), (EXTSBRmRn (MOVBRmiRn i32:$src))>;
// def : Pat<(i32 (sextloadi8  i32:$src)), (EXTSBRmRn (MOVBRmiRn i32:$src))>;
// def : Pat<(i32 (sextloadi16 i32:$src)), (EXTSWRmRn (MOVWRmiRn i32:$src))>;

def : Pat<(sext_inreg i32:$src, i1),  (EXTSB i32:$src)>;
def : Pat<(sext_inreg i32:$src, i8),  (EXTSB i32:$src)>;
def : Pat<(sext_inreg i32:$src, i16), (EXTSW i32:$src)>;


//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Pseudo instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

//===--------------------------------------------------------------------------===//
// SelectCC & SetCC
//===--------------------------------------------------------------------------===//

def Select8     : SHSelectPseudo<(outs GPR:$dst),
                                (ins GPR:$src, GPR:$src2, i8imm:$cc),
                                "; Select8 PSEUDO",
                                [(set i8:$dst,
                                 (SHselectcc i8:$src, i8:$src2, imm:$cc))]>;

def Select16    : SHSelectPseudo<(outs GPR:$dst),
                                (ins GPR:$src, GPR:$src2, i8imm:$cc),
                                "; Select16 PSEUDO",
                                [(set i16:$dst,
                                 (SHselectcc i16:$src, i16:$src2, imm:$cc))]>;

def Select32    : SHSelectPseudo<(outs GPR:$dst),
                                (ins GPR:$src, GPR:$src2, i8imm:$cc),
                                "; Select32 PSEUDO",
                                [(set i32:$dst,
                                 (SHselectcc i32:$src, i32:$src2, imm:$cc))]>;

//===--------------------------------------------------------------------------===//
// Frame Load/Store
//===--------------------------------------------------------------------------===//

// LOAD FROM FRAME
def MOVBLFR : SHLoadPseudo<(outs GPR:$Rn), (ins (memri $Rm, $disp):$src),
                          "! MOVBLFR $src, $Rn",
                          [(set i32:$Rn, (extloadi8 addr:$src))]>;

def MOVWLFR : SHLoadPseudo<(outs GPR:$Rn), (ins (memri $Rm, $disp):$src),
                          "! MOVWLFR $src, $Rn",
                          [(set i32:$Rn, (extloadi16 addr:$src))]>;

def MOVLLFR : SHLoadPseudo<(outs GPR:$Rn), (ins (memri $Rm, $disp):$src),
                          "! MOVLLFR $src, $Rn",
                          [(set i32:$Rn, (load addr:$src))]>;

// STORE TO FRAME
def MOVBSFR : SHStorePseudo<(outs), (ins GPR:$src, (memri $Rm, $disp):$dst),
                           "! MOVBSFR $src, $dst",
                           [(truncstorei8 i32:$src, addr:$dst)]>;

def MOVWSFR : SHStorePseudo<(outs), (ins GPR:$src, (memri $Rm, $disp):$dst),
                           "! MOVWSFR $src, $dst",
                           [(truncstorei16 i32:$src, addr:$dst)]>;

def MOVLSFR : SHStorePseudo<(outs), (ins GPR:$src, (memri $Rm, $disp):$dst),
                           "! MOVLSFR $src, $dst",
                           [(store i32:$src, addr:$dst)]>;

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